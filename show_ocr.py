"""
show_ocr.py
Print the text CampusGPT extracts from a file (uses OCR for scanned/handwritten pages).

    python show_ocr.py data/YourFile.pdf          (summary + first page)
    python show_ocr.py data/YourFile.pdf 5        (page / slide 5)
"""

import sys

from loaders import load_file

if len(sys.argv) < 2:
    raise SystemExit("Usage: python show_ocr.py data/YourFile.pdf [page]")

docs = load_file(sys.argv[1])
print(f"{len(docs)} page(s) with text")
wanted = int(sys.argv[2]) if len(sys.argv) > 2 else None
shown = False
for d in docs:
    page = d.metadata.get("page")
    if wanted is None or page == wanted:
        print(f"\n--- page {page} ---")
        print(d.page_content[:1500])
        shown = True
        if wanted is None:
            break
if not shown:
    print(f"No text found for page {wanted}.")