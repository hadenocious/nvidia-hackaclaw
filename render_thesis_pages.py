from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF is not installed.")
    print("Run: pip3 install pymupdf --break-system-packages")
    raise SystemExit

project = Path.home() / "rocket_project"
pdf_path = project / "data" / "thesis" / "thesis.pdf"
out_folder = project / "data" / "thesis" / "page_images"
out_folder.mkdir(exist_ok=True)

# Pages we care about based on the CSV/table search
pages = [84, 93, 113, 118, 120]

if not pdf_path.exists():
    print(f"Could not find PDF at: {pdf_path}")
    raise SystemExit

doc = fitz.open(pdf_path)

print(f"PDF pages: {len(doc)}")

for page_num in pages:
    index = page_num - 1

    if index < 0 or index >= len(doc):
        print(f"Skipping page {page_num}: out of range")
        continue

    page = doc[index]

    # Higher zoom = clearer image
    zoom = 3
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)
    out_path = out_folder / f"page_{page_num}.png"
    pix.save(out_path)

    print(f"Saved {out_path}")

print("Done.")
