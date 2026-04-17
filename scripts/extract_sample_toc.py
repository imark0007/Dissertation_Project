"""
Attempt to extract the Table of Contents from the Nouman sample PDF.

The sample PDF may be scanned/flattened; if so, text extraction can fail.
This script tries PyMuPDF first (if installed), then falls back to a basic
PDFMiner extraction (if installed).

Usage:
    python scripts/extract_sample_toc.py
"""
from __future__ import annotations

import re
from pathlib import Path


SAMPLE_PDF = Path(
    "docs/reference/dissertation_samples/Dissertation Sample/"
    "Muhammad_Nouman_B01654699_Dissertation_Report.docx.pdf"
)


def _clean_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def _looks_like_toc(lines: list[str]) -> bool:
    joined = "\n".join(lines[:80]).lower()
    return ("contents" in joined) or ("table of contents" in joined)


def _print_candidate(page_num: int, lines: list[str]) -> None:
    print(f"\n--- Candidate TOC page: {page_num} ---")
    for ln in lines[:140]:
        print(ln)


def try_pymupdf() -> bool:
    try:
        import fitz  # type: ignore
    except Exception:
        return False

    doc = fitz.open(str(SAMPLE_PDF))
    print(f"PyMuPDF pages: {doc.page_count}")

    # Search first 40 pages for "Contents"
    for i in range(min(40, doc.page_count)):
        text = doc.load_page(i).get_text("text")
        lines = _clean_lines(text)
        if not lines:
            continue
        if _looks_like_toc(lines):
            _print_candidate(i + 1, lines)
            return True

    # If no explicit marker, search for dense dotted leaders patterns "... 12"
    dotted = re.compile(r"\.{3,}\s*\d+\s*$")
    for i in range(min(40, doc.page_count)):
        text = doc.load_page(i).get_text("text")
        lines = _clean_lines(text)
        hits = sum(1 for ln in lines if dotted.search(ln))
        if hits >= 10:
            _print_candidate(i + 1, lines)
            return True

    print("PyMuPDF: no TOC-like text found (may be scanned).")
    return True


def try_pdfminer() -> bool:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return False

    text = extract_text(str(SAMPLE_PDF), maxpages=40)
    lines = _clean_lines(text)
    if not lines:
        print("PDFMiner: empty text (likely scanned).")
        return True

    # Print first occurrences around 'Contents'
    for idx, ln in enumerate(lines):
        if "contents" in ln.lower():
            start = max(0, idx - 10)
            end = min(len(lines), idx + 120)
            print("\n--- PDFMiner TOC window ---")
            for ln2 in lines[start:end]:
                print(ln2)
            return True

    print("PDFMiner: no 'Contents' found in extracted text.")
    return True


def main() -> None:
    if not SAMPLE_PDF.exists():
        raise SystemExit(f"Missing: {SAMPLE_PDF}")
    print(f"Sample PDF: {SAMPLE_PDF} ({SAMPLE_PDF.stat().st_size} bytes)")

    if try_pymupdf():
        return
    if try_pdfminer():
        return

    print("Neither PyMuPDF nor PDFMiner is available in this environment.")


if __name__ == "__main__":
    main()

