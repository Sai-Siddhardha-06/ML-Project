from pathlib import Path
import csv

PROJECT = Path.home() / "B24DS018_ML_Project"
DATASET = PROJECT / "ff_data"
SPLITS = PROJECT / "splits"
OUTPUT = SPLITS / "ffpp_manifest.csv"

split_files = {
    "train": SPLITS / "train_pairs.txt",
    "val": SPLITS / "val_pairs.txt",
    "test": SPLITS / "test_pairs.txt",
}

methods = {
    "original": DATASET / "original_sequences/youtube/c23/videos",
    "Deepfakes": DATASET / "manipulated_sequences/Deepfakes/c23/videos",
    "Face2Face": DATASET / "manipulated_sequences/Face2Face/c23/videos",
    "FaceSwap": DATASET / "manipulated_sequences/FaceSwap/c23/videos",
    "NeuralTextures": DATASET / "manipulated_sequences/NeuralTextures/c23/videos",
}

pair_to_split = {}
source_to_pair = {}

for split, path in split_files.items():
    with open(path) as f:
        for line in f:
            pair = line.strip()
            if not pair:
                continue

            a, b = pair.split("_")

            pair_to_split[pair] = split
            source_to_pair[a] = pair
            source_to_pair[b] = pair

rows = []

for method, folder in methods.items():
    for video in sorted(folder.glob("*.mp4")):
        name = video.stem

        if method == "original":
            pair = source_to_pair[name]
            label = 0
        else:
            a, b = name.split("_")
            pair = f"{min(a, b)}_{max(a, b)}"
            label = 1

        split = pair_to_split[pair]

        rows.append([
            str(video.relative_to(PROJECT)),
            split,
            label,
            method,
            pair
        ])

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "split", "label", "manipulation", "source_pair"])
    writer.writerows(rows)

print(f"Manifest written to: {OUTPUT}")
print(f"Total videos: {len(rows)}")

for split in ["train", "val", "test"]:
    count = sum(row[1] == split for row in rows)
    print(f"{split}: {count}")

for split in ["train", "val", "test"]:
    real = sum(row[1] == split and row[2] == 0 for row in rows)
    fake = sum(row[1] == split and row[2] == 1 for row in rows)
    print(f"{split}: real={real}, fake={fake}")
