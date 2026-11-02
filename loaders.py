"""
loaders.py
Extracts raw text from PDF, PPTX, and DOCX files.
Returns LangChain Document objects so they plug into ingest.py.

PDF pages with no extractable text (scanned / handwritten PDFs) fall back
to OCR using pypdfium2 (page -> image) + RapidOCR (image -> text).
Both install with pip only, no extra programs needed on Windows.
"""

import os
from langchain_core.documents import Document
from pypdf import PdfReader
from pptx import Presentation
import docx

_ocr_engine = None


def _get_ocr():
    """Load the OCR engine once (first load is slow, it downloads models)."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR  # newer package
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # older package
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_page(path, page_index):
    """Render one PDF page to an image and OCR it. Returns text or ''."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        page = pdf[page_index]
        image = page.render(scale=2).to_numpy()
    finally:
        pdf.close()

    result = _get_ocr()(image)

    # newer rapidocr returns an object with .txts
    txts = getattr(result, "txts", None)
    if txts is not None:
        return "\n".join(txts)

    # older rapidocr_onnxruntime returns (list of [box, text, score], elapsed)
    items = result[0] if isinstance(result, tuple) else result
    if not items:
        return ""
    return "\n".join(item[1] for item in items)


def load_pdf(path):
    docs = []
    reader = PdfReader(path)
    ocr_pages = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            try:
                text = _ocr_page(path, i)
                ocr_pages += 1
            except Exception as e:
                print(f"OCR failed on {os.path.basename(path)} page {i + 1}: {e}")
                text = ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i + 1},
                )
            )
    if ocr_pages:
        print(f"{os.path.basename(path)}: used OCR on {ocr_pages} page(s).")
    return docs


def load_pptx(path):
    docs = []
    prs = Presentation(path)
    for i, slide in enumerate(prs.slides):
        text_runs = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        text_runs.append(run.text)
        text = "\n".join(text_runs)
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i + 1},
                )
            )
    return docs


def load_docx(path):
    docs = []
    d = docx.Document(path)
    full_text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    if full_text.strip():
        docs.append(
            Document(
                page_content=full_text,
                metadata={"source": os.path.basename(path), "page": 1},
            )
        )
    return docs


def load_file(path):
    """Route a single file to the correct loader based on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".ppt", ".pptx"):
        return load_pptx(path)
    elif ext in (".doc", ".docx"):
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_directory(data_path="data/"):
    """Load every supported file inside a folder."""
    all_docs = []
    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path, fname)
        if os.path.isfile(fpath):
            try:
                all_docs.extend(load_file(fpath))
            except ValueError:
                print(f"Skipping unsupported file: {fname}")
    return all_docs