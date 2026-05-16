from pathlib import Path
import csv
import re

project = Path.home() / "rocket_project"
thesis_folder = project / "data" / "thesis"
out_folder = thesis_folder / "filtered"
out_folder.mkdir(exist_ok=True)

num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

categories = {
    "trajectory": [
        "flight", "alt", "altitude", "speed", "velocity", "acceleration", "time"
    ],
    "roll_control": [
        "roll", "roll speed", "roll rate", "angular", "gyro"
    ],
    "fin_control": [
        "fin", "deflection", "controller", "saturation", "servo"
    ],
    "comparison": [
        "stabilized", "unstabilized", "pid", "reduction"
    ],
}

all_rows = []
category_rows = {cat: [] for cat in categories}

clean_files = sorted(thesis_folder.glob("clean_*.csv"))

for file in clean_files:
    lines = file.read_text(errors="ignore").splitlines()

    for line_num, line in enumerate(lines, start=1):
        text = line.replace('"', "")
        text = re.sub(r",+", ",", text)
        text = text.strip(",")

        if not text.strip():
            continue

        lower = text.lower()
        numbers = num_re.findall(text)

        hit_categories = []

        for cat, words in categories.items():
            if any(word in lower for word in words):
                hit_categories.append(cat)

        if not hit_categories and len(numbers) < 2:
            continue

        row = {
            "file": file.name,
            "line": line_num,
            "categories": "|".join(hit_categories),
            "numbers_found": "|".join(numbers),
            "text": text,
        }

        all_rows.append(row)

        for cat in hit_categories:
            category_rows[cat].append(row)

# Save all filtered rows
all_path = out_folder / "all_filtered_candidates.csv"

with open(all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "line", "categories", "numbers_found", "text"]
    )
    writer.writeheader()
    writer.writerows(all_rows)

# Save each category
for cat, rows in category_rows.items():
    path = out_folder / f"{cat}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "line", "categories", "numbers_found", "text"]
        )
        writer.writeheader()
        writer.writerows(rows)

# Save summary
summary_path = out_folder / "summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("THESIS FILTER SUMMARY\n")
    f.write("=====================\n\n")
    f.write(f"Clean CSV files scanned: {len(clean_files)}\n")
    f.write(f"Total useful rows found: {len(all_rows)}\n\n")

    for cat, rows in category_rows.items():
        f.write(f"{cat}: {len(rows)} rows\n")

print("✅ Filtering complete")
print(f"Scanned {len(clean_files)} clean CSV files")
print(f"Saved to: {out_folder}")