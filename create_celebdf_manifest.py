from pathlib import Path
import csv

PROJECT = Path.home() / "B24DS018_ML_Project"
DATASET = PROJECT / "Celeb_DF_V2"
TEST_LIST = DATASET / "List_of_testing_videos.txt"
OUTPUT = PROJECT / "splits" / "celebdf_manifest.csv"

rows = []

with open(TEST_LIST) as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        label, relative_path = line.split(maxsplit=1)
        video_path = DATASET / relative_path
        folder = relative_path.split("/")[0]

        rows.append([
            str(video_path.relative_to(PROJECT)),
            "test",
            int(label),
            folder
        ])

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "split", "label", "source"])
    writer.writerows(rows)

print(f"Manifest written to: {OUTPUT}")
print(f"Total videos: {len(rows)}")

real = sum(row[2] == 1 for row in rows)
fake = sum(row[2] == 0 for row in rows)

print(f"Real: {real}")
print(f"Fake: {fake}")
