from pathlib import Path
from collections import Counter

PROJECT = Path.home() / "B24DS018_ML_Project"
DATASET = PROJECT / "Celeb_DF_V2"
TEST_LIST = DATASET / "List_of_testing_videos.txt"

counts = Counter()
missing = []

with open(TEST_LIST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        label, relative_path = line.split(maxsplit=1)
        video_path = DATASET / relative_path

        folder = relative_path.split("/")[0]
        counts[folder] += 1

        if not video_path.exists():
            missing.append(relative_path)

print("=" * 60)
print("CELEB-DF-v2 OFFICIAL TEST SET VERIFICATION")
print("=" * 60)
print(f"Total test entries: {sum(counts.values())}")
print()
print(f"Celeb-real:       {counts['Celeb-real']}")
print(f"YouTube-real:     {counts['YouTube-real']}")
print(f"Celeb-synthesis:  {counts['Celeb-synthesis']}")
print()
print(f"Missing files: {len(missing)}")

assert sum(counts.values()) == 518
assert not missing

print()
print("✓ 518 official test entries verified")
print("✓ Every listed test video exists")
