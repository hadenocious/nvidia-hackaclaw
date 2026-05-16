from pathlib import Path
import pandas as pd

project = Path.home() / "rocket_project"

# Put your real/digitized CSV files here
data_folder = project / "data" / "thesis" / "digitized"

# Output folder
out_folder = project / "data" / "processed"
out_folder.mkdir(exist_ok=True)

print("Looking for data in:", data_folder)

if not data_folder.exists():
    print("Missing folder. Create it first:")
    print(data_folder)
    raise SystemExit

csv_files = list(data_folder.glob("*.csv"))

if len(csv_files) == 0:
    print("No CSV files found in digitized folder.")
    raise SystemExit

print("\nFound CSV files:")
for f in csv_files:
    print("-", f.name)

all_data = []

for file in csv_files:
    df = pd.read_csv(file)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    df["source_file"] = file.name

    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

out_path = out_folder / "thesis_combined_dataset.csv"
combined.to_csv(out_path, index=False)

print("\n✅ Combined dataset saved:")
print(out_path)

print("\nColumns found:")
for col in combined.columns:
    print("-", col)

print("\nPreview:")
print(combined.head())