from pathlib import Path
import random

PROJECT = Path.home() / "B24DS018_ML_Project"
ROOT = PROJECT / "ff_data"
OUT = PROJECT / "splits"
OUT.mkdir(parents=True, exist_ok=True)

METHOD = "Deepfakes"
VIDEO_DIR = ROOT / "manipulated_sequences" / METHOD / "c23" / "videos"

# Extract unique unordered source pairs.
pairs = set()

for path in VIDEO_DIR.glob("*.mp4"):
    a, b = map(int, path.stem.split("_"))
    pairs.add(tuple(sorted((a, b))))

pairs = sorted(pairs)

assert len(pairs) == 500, f"Expected 500 source pairs, found {len(pairs)}"

# Reproducible shuffle.
rng = random.Random(42)
rng.shuffle(pairs)

train_pairs = pairs[:400]
val_pairs = pairs[400:450]
test_pairs = pairs[450:500]

assert len(train_pairs) == 400
assert len(val_pairs) == 50
assert len(test_pairs) == 50

def write_pairs(filename, pair_list):
    with open(OUT / filename, "w") as f:
        for a, b in pair_list:
            f.write(f"{a:03d}_{b:03d}\n")

write_pairs("train_pairs.txt", train_pairs)
write_pairs("val_pairs.txt", val_pairs)
write_pairs("test_pairs.txt", test_pairs)

with open(OUT / "split_summary.txt", "w") as f:
    f.write("FF++ c23 leakage-safe source-pair split\n")
    f.write("Random seed: 42\n\n")
    f.write(f"Train source pairs: {len(train_pairs)}\n")
    f.write(f"Validation source pairs: {len(val_pairs)}\n")
    f.write(f"Test source pairs: {len(test_pairs)}\n")
    f.write(f"Total source pairs: {len(pairs)}\n")

print("Split created successfully.")
print(f"Train: {len(train_pairs)} source pairs")
print(f"Val:   {len(val_pairs)} source pairs")
print(f"Test:  {len(test_pairs)} source pairs")
print(f"Output: {OUT}")
