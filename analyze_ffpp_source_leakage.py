from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "B24DS018_ML_Project" / "ff_data"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
N_SOURCE = 1000

parent = list(range(N_SOURCE))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

pair_count = 0

for method in METHODS:
    directory = ROOT / "manipulated_sequences" / method / "c23" / "videos"

    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")

    for path in directory.glob("*.mp4"):
        try:
            a, b = map(int, path.stem.split("_"))
        except ValueError:
            raise ValueError(f"Unexpected filename: {path.name}")

        if not (0 <= a < N_SOURCE and 0 <= b < N_SOURCE):
            raise ValueError(f"Source ID outside 000-999: {path.name}")

        union(a, b)
        pair_count += 1

components = defaultdict(list)

for source_id in range(N_SOURCE):
    components[find(source_id)].append(source_id)

sizes = sorted((len(v) for v in components.values()), reverse=True)

print("=" * 60)
print("FF++ SOURCE-LEAKAGE ANALYSIS")
print("=" * 60)
print(f"Root: {ROOT}")
print(f"Manipulated videos checked: {pair_count}")
print(f"Expected manipulated videos: {len(METHODS) * N_SOURCE}")
print(f"Source videos: {N_SOURCE}")
print(f"Connected components: {len(components)}")
print()
print("Largest 20 component sizes:")
print(sizes[:20])
print()
print(f"Largest component: {sizes[0]} source videos")
print(f"Smallest component: {sizes[-1]} source videos")
print()
print("Interpretation:")
print("- Each manipulated pair links its two source videos.")
print("- Connected source videos must remain in the same split.")
print("- No videos have been moved or copied.")
print("=" * 60)
