"""
scripts/preprocess_pdfs.py
──────────────────────────
Extracts text and tables from maintenance/device PDF manuals,
detects fault codes and safety warnings, produces the
all_device_fault_chunks.json file consumed by build_index.py.

Usage:
    python scripts/preprocess_pdfs.py --manuals_dir <folder_with_pdfs> [--out data/all_device_fault_chunks.json]

The output JSON is a list of chunk dicts with the structure:
    {
        "chunk_id":  "<manual_stem>_p<page>_c<n>",
        "content":   "...",
        "type":      "text" | "table",
        "metadata": {
            "device":            "...",
            "manual":            "filename.pdf",
            "page":              42,
            "error_codes":       ["E37", ...],
            "has_safety_warning": true
        }
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# ── Pattern constants ────────────────────────────────────────────────────────

FAULT_RE = re.compile(
    r"\b(E\d{2,5}|ERR[-\s]?\d{2,5}|Fault\s*\d{1,5}|Code\s*\d{1,5}"
    r"|Alarm\s*\d{1,5}|F\d{2,4}|Error\s+\d{1,5})\b",
    re.IGNORECASE,
)
SAFETY_RE = re.compile(r"\b(DANGER|WARNING|CAUTION|AVERTISSEMENT|WARNUNG)\b", re.IGNORECASE)

# Chunk text targets — aim for ~500 chars, minimum 80
CHUNK_TARGET = 500
CHUNK_MIN = 80


# ── Helpers ──────────────────────────────────────────────────────────────────

def _device_name_from_pdf(path: Path) -> str:
    """Derive a human-readable device name from the PDF filename."""
    stem = path.stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+\(\d+\)$", "", stem)  # remove " (1)" duplicates
    return stem.strip().title()


def _find_error_codes(text: str) -> list[str]:
    matches = {m.upper().replace(" ", "").replace("-", "") for m in FAULT_RE.findall(text)}
    return sorted(matches) if matches else []


def _has_safety(text: str) -> bool:
    return bool(SAFETY_RE.search(text))


def _chunk_text(text: str) -> list[str]:
    """Split a page's text into overlapping chunks of ~CHUNK_TARGET chars."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 1 <= CHUNK_TARGET * 1.5:
            current += "\n" + para
        else:
            if len(current) >= CHUNK_MIN:
                chunks.append(current)
            current = para
    if current and len(current) >= CHUNK_MIN:
        chunks.append(current)
    return chunks or ([text[:CHUNK_TARGET]] if len(text) >= CHUNK_MIN else [])


def _table_to_text(rows: list[list[str | None]]) -> str:
    lines = []
    for row in rows:
        cells = [str(c).strip() if c else "" for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ── Per-PDF extraction ───────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list[dict]:
    try:
        import fitz  # PyMuPDF
        import pdfplumber
    except ImportError:
        print("ERROR: Install PyMuPDF and pdfplumber first:  pip install PyMuPDF pdfplumber")
        sys.exit(1)

    manual_name = pdf_path.name
    device_name = _device_name_from_pdf(pdf_path)
    chunks: list[dict] = []
    counter = 0

    print(f"  Processing: {manual_name}", flush=True)

    # ── Text extraction via PyMuPDF ──
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  [WARN] Skipping {manual_name}: {exc}")
        return []

    page_texts: dict[int, str] = {}
    for page_num, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        if text.strip():
            page_texts[page_num] = text
    doc.close()

    # ── Table extraction via pdfplumber ──
    page_tables: dict[int, list[list]] = {}
    try:
        with pdfplumber.open(str(pdf_path)) as plumb:
            for page_num, page in enumerate(plumb.pages, start=1):
                try:
                    tables = page.extract_tables()
                    if tables:
                        page_tables[page_num] = tables
                except Exception:
                    pass
    except Exception as exc:
        print(f"  [WARN] pdfplumber failed for {manual_name}: {exc} (using text only)")

    # ── Build chunks ──
    all_pages = sorted(set(list(page_texts.keys()) + list(page_tables.keys())))
    for page_num in all_pages:
        # Tables first
        for table in page_tables.get(page_num, []):
            table_text = _table_to_text(table)
            if len(table_text.strip()) < CHUNK_MIN:
                continue
            chunk_id = f"{pdf_path.stem}_p{page_num}_t{counter}"
            counter += 1
            chunks.append({
                "chunk_id": chunk_id,
                "content": table_text,
                "type": "table",
                "metadata": {
                    "device": device_name,
                    "manual": manual_name,
                    "page": page_num,
                    "error_codes": _find_error_codes(table_text),
                    "has_safety_warning": _has_safety(table_text),
                },
            })

        # Text chunks
        text = page_texts.get(page_num, "")
        for chunk_text in _chunk_text(text):
            chunk_id = f"{pdf_path.stem}_p{page_num}_c{counter}"
            counter += 1
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_text,
                "type": "text",
                "metadata": {
                    "device": device_name,
                    "manual": manual_name,
                    "page": page_num,
                    "error_codes": _find_error_codes(chunk_text),
                    "has_safety_warning": _has_safety(chunk_text),
                },
            })

    print(f"    -> {len(chunks)} chunks from {len(all_pages)} pages")
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PDF manuals into Fixora chunk JSON.")
    parser.add_argument("--manuals_dir", required=True, help="Folder containing PDF manuals.")
    parser.add_argument("--out", default="data/all_device_fault_chunks.json", help="Output JSON path.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip PDFs already in the output file.")
    args = parser.parse_args()

    manuals_dir = Path(args.manuals_dir)
    out_path = Path(args.out)

    if not manuals_dir.is_dir():
        print(f"ERROR: {manuals_dir} is not a directory.")
        sys.exit(1)

    pdf_files = sorted(manuals_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {manuals_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {manuals_dir}")

    # Load existing chunks if --skip_existing
    existing: list[dict] = []
    existing_manuals: set[str] = set()
    if args.skip_existing and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        existing_manuals = {c["metadata"]["manual"] for c in existing}
        print(f"Loaded {len(existing)} existing chunks from {out_path}")

    all_chunks: list[dict] = list(existing)
    for pdf in pdf_files:
        if pdf.name in existing_manuals:
            print(f"  Skipping (already indexed): {pdf.name}")
            continue
        chunks = extract_pdf(pdf)
        all_chunks.extend(chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_chunks)} total chunks -> {out_path}")


if __name__ == "__main__":
    main()
