"""
Video -> 32 uniformly sampled frames -> MTCNN face crop -> 224x224 JPEGs.

Usage (from ~/B24DS018_ML_Project, venv active):

  # Full run
  python preprocess_videos.py --dataset ffpp --out frames
  python preprocess_videos.py --dataset celebdf --out frames

Resumable: videos are skipped only when a valid meta.json with
status="ok" is already present for that video.
Augmentation is NOT done here; apply it in the training Dataset (train split only).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

MANIFESTS = {"ffpp": "splits/ffpp_manifest.csv", "celebdf": "splits/celebdf_manifest.csv"}
PATH_CANDS = ["video_path", "path", "filepath", "file_path", "video", "file"]
ID_CANDS = ["video_id", "id", "video_name", "name", "filename"]
LABEL_CANDS = ["label", "is_fake", "binary_label", "class", "real_fake"]
SPLIT_CANDS = ["split", "subset", "partition"]


def pick(df, override, cands, what):
    if override:
        if override not in df.columns:
            sys.exit(f"--{what}-col '{override}' not in columns {list(df.columns)}")
        return override
    for c in cands:
        if c in df.columns:
            return c
    return None


def sample_indices(n_frames, k):
    if n_frames <= 0:
        return []
    return sorted(set(np.linspace(0, n_frames - 1, k).round().astype(int).tolist()))


def read_frames(path, k):
    """Sequential grab (robust vs. inaccurate seeking). Returns {idx: BGR frame}."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(n, k)
    wanted, last = set(idxs), (idxs[-1] if idxs else -1)
    frames, i = {}, 0
    while i <= last and cap.grab():
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                frames[i] = f
        i += 1
    cap.release()
    return frames, n


def square_crop(img, box, margin, size):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * margin
    X1, Y1 = int(round(cx - s / 2)), int(round(cy - s / 2))
    X2, Y2 = int(round(cx + s / 2)), int(round(cy + s / 2))
    pl, pt = max(0, -X1), max(0, -Y1)
    pr, pb = max(0, X2 - w), max(0, Y2 - h)
    if pl or pt or pr or pb:  # pad (keeps crop square, no distortion)
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        X1, X2, Y1, Y2 = X1 + pl, X2 + pl, Y1 + pt, Y2 + pt
    crop = img[Y1:Y2, X1:X2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def detect_boxes(mtcnn, frames_bgr, chunk=8):
    """Largest-face box per frame (None if no face). Frames of one video share a size."""
    out = []
    for i in range(0, len(frames_bgr), chunk):
        batch = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr[i:i + chunk]]
        boxes, _ = mtcnn.detect(batch)
        for b in boxes:
            if b is None or len(b) == 0:
                out.append(None)
            else:
                areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                out.append(b[int(np.argmax(areas))])
    return out


def process_video(mtcnn, vpath, outdir, args):
    frames, n_total = read_frames(vpath, args.frames)
    if not frames:
        return {"status": "unreadable", "n_total_frames": n_total, "n_saved": 0, "n_no_face": 0}
    idxs = sorted(frames)
    boxes = detect_boxes(mtcnn, [frames[i] for i in idxs])

    # Fallback: frames with no detection reuse nearest previous (else next) good box
    good = [b for b in boxes if b is not None]
    if not good:
        return {"status": "no_face", "n_total_frames": n_total, "n_saved": 0, "n_no_face": len(idxs)}
    n_no_face, last = 0, None
    filled = []
    for b in boxes:
        if b is not None:
            last = b
        else:
            n_no_face += 1
        filled.append(last)
    first_good = good[0]
    filled = [b if b is not None else first_good for b in filled]

    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, b in zip(idxs, filled):
        crop = square_crop(frames[i], b, args.margin, args.size)
        fn = outdir / f"{i:05d}.jpg"
        cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, args.jpg_quality])
        saved.append(i)
    return {"status": "ok", "n_total_frames": n_total, "n_saved": len(saved),
            "n_no_face": n_no_face, "frame_indices": saved}


def make_montage(outdir, dest, n=8):
    files = sorted(outdir.glob("*.jpg"))
    if not files:
        return
    sel = [files[j] for j in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
    imgs = [cv2.resize(cv2.imread(str(f)), (160, 160)) for f in sel]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), np.hstack(imgs))


def unique_video_id(path):
    """
    Create a deterministic unique ID from the full manifest path.

    FF++ contains multiple videos with the same filename stem, so
    Path(path).stem alone cannot uniquely identify a video.
    """
    path_str = str(Path(path))
    stem = Path(path).stem
    digest = hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MANIFESTS))
    ap.add_argument("--root", default=".", help="project root (video paths resolved against it)")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=1.3, help="box enlargement factor")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--path-col"); ap.add_argument("--id-col")
    ap.add_argument("--label-col"); ap.add_argument("--split-col")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    df = pd.read_csv(root / MANIFESTS[args.dataset])
    print("Manifest columns:", list(df.columns))
    pcol = pick(df, args.path_col, PATH_CANDS, "path")
    icol = pick(df, args.id_col, ID_CANDS, "id")
    lcol = pick(df, args.label_col, LABEL_CANDS, "label")
    scol = pick(df, args.split_col, SPLIT_CANDS, "split")

    if pcol is None:
        sys.exit("Could not find the video-path column; pass --path-col.")

    if icol is None:  # use a deterministic unique ID from the full path
        df["_vid"] = df[pcol].map(unique_video_id)
        icol = "_vid"

    if scol is None:
        df["_split"] = "test" if args.dataset == "celebdf" else "unknown"
        scol = "_split"

    print(f"Using columns: path={pcol} id={icol} label={lcol} split={scol}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        post_process=False,
        select_largest=False
    )

    out_root = root / args.out / args.dataset
    log_rows, t0 = [], time.time()

    n_ok = n_skip = n_fail = 0

    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        unit="vid",
        dynamic_ncols=True
    )

    for _, r in pbar:
        pbar.set_postfix(
            done=n_ok,
            skipped=n_skip,
            failed=n_fail,
            left=len(df) - pbar.n
        )

        vpath = Path(r[pcol])

        if not vpath.is_absolute():
            vpath = root / vpath

        vid = str(r[icol])
        outdir = out_root / str(r[scol]) / vid
        meta_f = outdir / "meta.json"

        # Skip only successfully preprocessed videos.
        # Missing, invalid, or failed metadata will be processed again.
        if meta_f.exists():
            try:
                meta = json.loads(meta_f.read_text())

                if meta.get("status") == "ok":
                    n_skip += 1
                    continue

            except (json.JSONDecodeError, OSError):
                pass

        try:
            res = process_video(mtcnn, vpath, outdir, args)

        except Exception as e:  # never let one bad video kill the run
            res = {
                "status": f"error: {e}",
                "n_saved": 0,
                "n_no_face": 0,
                "n_total_frames": -1
            }

        res.update(
            video_id=vid,
            split=str(r[scol]),
            video_path=str(vpath)
        )

        if lcol:
            res["label"] = r[lcol]

        if res["status"] == "ok":
            n_ok += 1
            meta_f.write_text(json.dumps(res))

        else:
            n_fail += 1
            log_rows.append(res)  # failures are logged, no meta.json -> retried next run

    # failure report
    if log_rows:
        fail = root / args.out / f"failed_{args.dataset}.csv"
        pd.DataFrame(log_rows).to_csv(fail, index=False)
        print(f"{len(log_rows)} videos failed/no-face -> {fail}")

    print(f"Done in {(time.time() - t0) / 60:.1f} min. Output: {out_root}")


if __name__ == "__main__":
    main()
