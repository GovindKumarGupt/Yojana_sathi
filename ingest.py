"""Step 1: read PDFs -> clean text -> split into chunks -> save chunks.json

Folder layout expected:
    data/raw/pm_kisan/guidelines.pdf
    data/raw/pm_kisan/faq.pdf
The sub-folder name decides the scheme (see SCHEMES in config.py).
"""
import os
import re
import json
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RAW_FOLDER, CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, SCHEMES, LAST_PAGE_TO_KEEP


def extract_text():
    """Return one record per PDF page: scheme, file name, page number, raw text."""
    pages = []
    for folder, scheme_name in SCHEMES.items():
        scheme_dir = os.path.join(RAW_FOLDER, folder)
        if not os.path.isdir(scheme_dir):
            print(f"[skip] folder not found: {scheme_dir}")
            continue
        for file_name in sorted(os.listdir(scheme_dir)):
            if not file_name.lower().endswith(".pdf"):
                continue
            pdf = fitz.open(os.path.join(scheme_dir, file_name))
            last_page = LAST_PAGE_TO_KEEP.get(file_name, len(pdf))
            for page_num, page in enumerate(pdf, start=1):
                if page_num > last_page:
                    break                                  # ignore the garbled Hindi half
                pages.append({
                    "scheme": scheme_name,
                    "file_name": file_name,
                    "page": page_num,
                    "text": page.get_text(),
                })
            pdf.close()
    return pages


def clean_text(text):
    """Remove blank lines, lone page numbers, 'Page x of y' and extra spaces."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue                                   # blank line
        if line.isdigit():
            continue                                   # lone page number
        if re.fullmatch(r"(?i)page\s*\d+(\s*of\s*\d+)?", line):
            continue                                   # "Page 3 of 10"
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunks(pages):
    """Split each cleaned page into overlapping chunks, keeping its labels."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = []
    for p in pages:
        text = clean_text(p["text"])
        if len(text) < 50:            # skip near-empty pages (cover pages, scans)
            continue
        for piece in splitter.split_text(text):
            chunks.append({
                "scheme": p["scheme"],
                "file_name": p["file_name"],
                "page": p["page"],
                "text": piece,
            })
    return chunks


if __name__ == "__main__":
    os.makedirs(os.path.dirname(CHUNKS_FILE), exist_ok=True)
    pages = extract_text()
    if not pages:
        raise SystemExit("No PDFs found. Put them in data/raw/pm_kisan/ and run again.")
    chunks = make_chunks(pages)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"{len(pages)} pages -> {len(chunks)} chunks saved to {CHUNKS_FILE}")
    if chunks:
        print("\nFirst chunk preview:\n", chunks[0]["text"][:300])
