from pathlib import Path
import csv
from collections import Counter

folder = Path("data/omnifall/annotations")

for file in sorted(folder.glob("*.csv")):
    with file.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    sources = Counter(row["dataset"] for row in rows)
    videos = {(row["dataset"], row["path"]) for row in rows}

    print(f"\n=== {file.name} ===")
    print(f"Số đoạn annotation: {len(rows)}")
    print(f"Số video duy nhất: {len(videos)}")
    print("Số đoạn theo dataset:")

    for name, count in sorted(sources.items()):
        print(f"  {name}: {count}")

    if rows:
        print("Ví dụ annotation:", rows[0])