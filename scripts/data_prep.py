#!/usr/bin/env python3
"""
data_prep.py -- resumable face-crop preprocessing for DFF-Adapter reproduction.

What it does, per video:
  1. sample K frames evenly spaced over the video          (sample_frames)
  2. detect the largest face in every sampled frame (MTCNN) (detect_boxes)
  3. crop a square around the face (box side * scale) and resize to size x size
  4. save JPEGs to  <out>/<dataset>/<method>/<video_id>/<frame_idx>.jpg

Resumable / crash-safe:
  - frames are written to a hidden .tmp_<video_id> folder, then the folder is
    renamed. A video is "done" only if <video_id>/_status.json exists.
  - stop any time (Ctrl+C, shutdown). Re-running skips finished videos and
    deletes half-written temp folders.

Usage:
  python data_prep.py crop --dataset ffpp --root ffpp_data --out crops --limit 3
  python data_prep.py preview --out crops --dataset ffpp       # eyeball results
  python data_prep.py crop --dataset ffpp --root ffpp_data --out crops
  python data_prep.py manifest --dataset ffpp --out crops
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

FFPP_FAKE = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ----------------------------------------------------------------- listing
def list_ffpp(root):
    """Return [(method, video_path)] interleaved so a partial run is balanced."""
    root = Path(root)
    per_method = {}
    d = root / "original_sequences" / "youtube" / "c23" / "videos"
    per_method["original"] = sorted(d.glob("*.mp4"))
    for m in FFPP_FAKE:
        d = root / "manipulated_sequences" / m / "c23" / "videos"
        per_method[m] = sorted(d.glob("*.mp4"))
    for m, v in per_method.items():
        print(f"  {m:15s}: {len(v)} videos")
    items = []
    for i in range(max(len(v) for v in per_method.values())):
        for m, v in per_method.items():
            if i < len(v):
                items.append((m, v[i]))
    return items


def list_cdf2(root, list_file):
    """Celeb-DF v1 or v2 (same layout). method = parent folder name (Celeb-real / YouTube-real /
    Celeb-synthesis). list_file (optional) = official test list, lines like
    '1 YouTube-real/00170.mp4' (we take the last token as the relative path)."""
    root = Path(root)
    if list_file:
        rels = []
        for line in open(list_file):
            line = line.strip()
            if line:
                rels.append(line.split()[-1])
        paths = [root / r for r in rels]
    else:
        paths = sorted(root.glob("*/*.mp4"))
    items = [(p.parent.name, p) for p in paths if p.exists()]
    missing = len(paths) - len(items)
    if missing:
        print(f"  WARNING: {missing} listed videos not found under {root}")
    return items


# ------------------------------------------------------------ frame logic
def sample_frames(path, k):
    """Read k evenly spaced frames (BGR). Returns (frames, frame_indices)."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return [], []
    idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    want, last = set(idx.tolist()), int(idx[-1])
    frames, ids, i = [], [], 0
    while i <= last:
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
                ids.append(i)
        i += 1
    cap.release()
    return frames, ids


def detect_boxes(mtcnn, frames, det_max):
    """Largest-face box per frame in ORIGINAL pixel coords, or None."""
    h, w = frames[0].shape[:2]
    s = min(1.0, det_max / max(h, w))  # detect on a smaller copy for speed
    small = []
    for f in frames:
        if s < 1.0:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        small.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    boxes, _ = mtcnn.detect(np.stack(small))
    out = []
    for b in boxes:
        if b is None or len(b) == 0:
            out.append(None)
            continue
        b = np.asarray(b, dtype=float).reshape(-1, 4)
        j = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        out.append(b[j] / s)
    return out


def fill_missing(boxes):
    """Frames with no detection reuse the nearest detected box in the video."""
    found = [i for i, b in enumerate(boxes) if b is not None]
    if not found:
        return None
    filled = list(boxes)
    for i in range(len(filled)):
        if filled[i] is None:
            nearest = min(found, key=lambda j: abs(j - i))
            filled[i] = boxes[nearest]
    return filled


def crop_square(img, box, scale, size):
    """Square crop centred on the box, side = max(w,h)*scale, resized to size.
    Uses an affine warp so crops that run off the image edge are zero-padded."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    s = size / side
    left, top = cx - side / 2, cy - side / 2
    M = np.array([[s, 0, -s * left], [0, s, -s * top]], dtype=np.float32)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA)


# ------------------------------------------------------- thermal safety
def gpu_temp():
    """Current GPU temperature in C via nvidia-smi, or None if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def wait_until_cool(max_temp, resume_margin=10):
    """If the GPU is above max_temp, sleep until it is resume_margin below it."""
    t = gpu_temp()
    if t is None or t < max_temp:
        return
    print(f"\nGPU at {t}C (limit {max_temp}C) -- pausing to cool down...")
    while True:
        time.sleep(10)
        t = gpu_temp()
        if t is None or t <= max_temp - resume_margin:
            print(f"GPU at {t}C -- resuming")
            return


# ---------------------------------------------------------------- cropping
def run_crop(args):
    import torch
    from facenet_pytorch import MTCNN
    from tqdm import tqdm

    if args.dataset == "ffpp":
        print("Listing FF++:")
        items = list_ffpp(args.root)
    else:
        items = list_cdf2(args.root, args.list)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} videos in queue")

    ds_out = Path(args.out) / args.dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    for m, _ in items:  # clean up leftovers from an interrupted run
        for t in (ds_out / m).glob(".tmp_*") if (ds_out / m).exists() else []:
            shutil.rmtree(t, ignore_errors=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    mtcnn = MTCNN(keep_all=True, device=dev, thresholds=[0.6, 0.7, 0.7])

    failed_log = Path(args.out) / f"{args.dataset}_failed.txt"
    done = skipped = failed = 0
    t0 = time.time()
    bar = tqdm(items, unit="vid")
    try:
        for method, vpath in bar:
            if args.max_temp:
                wait_until_cool(args.max_temp)
            vid = vpath.stem
            final = ds_out / method / vid
            if (final / "_status.json").exists():
                skipped += 1
                continue
            tmp = ds_out / method / f".tmp_{vid}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)

            frames, ids = sample_frames(vpath, args.frames)
            boxes = None
            if frames:
                boxes = []
                for s in range(0, len(frames), args.det_batch):
                    boxes += detect_boxes(mtcnn, frames[s : s + args.det_batch], args.det_max)
            n_missing = sum(b is None for b in boxes) if boxes else 0
            filled = fill_missing(boxes) if boxes else None
            if filled is None:  # no face anywhere in the video
                with open(failed_log, "a") as fh:
                    fh.write(f"{method}/{vid}\n")
                shutil.rmtree(tmp, ignore_errors=True)
                failed += 1
                continue
            for f, i, b in zip(frames, ids, filled):
                c = crop_square(f, b, args.scale, args.size)
                cv2.imwrite(str(tmp / f"{i:05d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
            json.dump(
                {"n_saved": len(frames), "n_no_detection": n_missing,
                 "src": str(vpath), "scale": args.scale, "size": args.size},
                open(tmp / "_status.json", "w"),
            )
            os.sync()  # force data to disk first, so a power cut cannot leave empty files
            os.rename(tmp, final)  # atomic: video becomes "done" here
            done += 1
            bar.set_postfix(done=done, skip=skipped, fail=failed)
            if args.cooldown:
                time.sleep(args.cooldown)
    except KeyboardInterrupt:
        print("\nInterrupted -- safe to re-run; finished videos are kept.")
    print(f"done={done} skipped={skipped} failed={failed} "
          f"time={(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------- preview
def run_preview(args):
    """Montage of random crops so you can LOOK at them (always do this)."""
    ds_out = Path(args.out) / args.dataset
    vids = [p.parent for p in ds_out.glob("*/*/_status.json")]
    if not vids:
        sys.exit("no finished videos found")
    random.seed(0)
    picks = random.sample(vids, min(len(vids), 24))
    tiles = []
    for v in picks:
        jpgs = sorted(v.glob("*.jpg"))
        img = cv2.imread(str(random.choice(jpgs)))
        img = cv2.resize(img, (160, 160))
        cv2.putText(img, v.parent.name[:12], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]
    out = Path(args.out) / f"{args.dataset}_preview.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print("wrote", out)


# ----------------------------------------------------------------- verify
def run_verify(args):
    """After a crash/power cut: re-check the N most recently finished videos.
    A video is bad if its jpg count != n_saved or any jpg fails to decode.
    Bad videos are deleted so the next crop run redoes them."""
    ds_out = Path(args.out) / args.dataset
    sts = sorted(ds_out.glob("*/*/_status.json"), key=lambda p: p.stat().st_mtime)
    total = len(sts)
    sts = sts[-args.last:] if args.last else sts
    bad = 0
    for st in sts:
        vdir = st.parent
        jpgs = sorted(vdir.glob("*.jpg"))
        try:
            meta = json.load(open(st))
            ok = len(jpgs) == meta["n_saved"]
        except Exception:  # empty / truncated status file (power cut)
            ok = False
        if ok:
            for j in jpgs:
                if cv2.imread(str(j)) is None:
                    ok = False
                    break
        if not ok:
            print("BAD, removing:", vdir)
            shutil.rmtree(vdir, ignore_errors=True)
            bad += 1
    print(f"checked {len(sts)} of {total} finished videos, removed {bad}")


# --------------------------------------------------------------- manifest
def run_manifest(args):
    """CSV of every finished video: dataset, method, video_id, label, n_frames, path.
    label: 0 = real, 1 = fake. Split assignment is done later, from the official
    FF++ split files, not here."""
    ds_out = Path(args.out) / args.dataset
    real_methods = {"original", "Celeb-real", "YouTube-real"}
    rows = []
    for st in sorted(ds_out.glob("*/*/_status.json")):
        vdir = st.parent
        try:
            meta = json.load(open(st))
        except Exception:
            print("WARNING: unreadable status, skipping (run verify):", vdir)
            continue
        method = vdir.parent.name
        rows.append({
            "dataset": args.dataset, "method": method, "video_id": vdir.name,
            "label": 0 if method in real_methods else 1,
            "n_frames": meta["n_saved"], "path": str(vdir.relative_to(args.out)),
        })
    out = Path(args.out) / f"{args.dataset}_manifest.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("wrote", out, "-", len(rows), "videos")
    print(Counter((r["method"], r["label"]) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["crop", "preview", "manifest", "verify"])
    ap.add_argument("--dataset", choices=["ffpp", "cdf1", "cdf2"], required=True,
                    help="cdf1/cdf2 = Celeb-DF v1/v2 (same folder layout)")
    ap.add_argument("--root", help="raw video root (crop mode)")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--list", help="cdf2: official test list file")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--scale", type=float, default=1.3)
    ap.add_argument("--det_max", type=int, default=640, help="detect on frames resized to this max side")
    ap.add_argument("--det_batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only first N videos (for testing)")
    ap.add_argument("--cooldown", type=float, default=0.0, help="sleep seconds between videos (cooler laptop)")
    ap.add_argument("--max_temp", type=int, default=0, help="pause when GPU is above this temperature (C); 0 = off")
    ap.add_argument("--last", type=int, default=50, help="verify: check the N most recent videos (0 = all)")
    args = ap.parse_args()
    {"crop": run_crop, "preview": run_preview, "manifest": run_manifest, "verify": run_verify}[args.mode](args)


if __name__ == "__main__":
    main()