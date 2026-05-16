from pathlib import Path
import csv
import re

project = Path.home() / "rocket_project"
thesis_folder = project / "data" / "thesis"
out_folder = thesis_folder / "filtered"
out_folder.mkdir(exist_ok=True)

num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

rows = []

for file in sorted(thesis_folder.glob("clean_*.csv")):
    lines = file.read_text(errors="ignore").splitlines()

    for i, line in enumerate(lines, start=1):
        clean = line.replace('"', "")
        clean = re.sub(r",+", ",", clean)
        clean = clean.strip(",")

        numbers = num_re.findall(clean)

        if len(numbers) >= 2:
            rows.append({
                "file": file.name,
                "line": i,
                "num_count": len(numbers),
                "numbers": "|".join(numbers),
                "text": clean
            })

out_csv = out_folder / "numeric_rows.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "line", "num_count", "numbers", "text"]
    )
    writer.writeheader()
    writer.writerows(rows)

summary = out_folder / "numeric_summary.txt"

file_counts = {}
for row in rows:
    file_counts[row["file"]] = file_counts.get(row["file"], 0) + 1

with open(summary, "w", encoding="utf-8") as f:
    f.write("NUMERIC DATA SUMMARY\n")
    f.write("====================\n\n")
    f.write(f"Total numeric rows found: {len(rows)}\n\n")

    for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{file}: {count} numeric rows\n")

print("✅ Numeric scan complete")
print(f"Saved: {out_csv}")
print(f"Summary: {summary}")