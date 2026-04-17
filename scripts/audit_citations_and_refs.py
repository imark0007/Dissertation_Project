"""
Audit in-text citations vs Chapter 11 reference list.

Usage (PowerShell, repo root):
    python scripts/audit_citations_and_refs.py

Outputs:
  - Missing references for in-text citations (by first-author + year)
  - Unused references (present in reference list but not cited)
  - Basic DOI/URL hygiene checks (format only; not a network validation)
"""

from __future__ import annotations

import re
from pathlib import Path


def _slice_chapter_11(md: str) -> str:
    m = re.search(r"^## Chapter 11\s+–\s+References\s*$", md, flags=re.M)
    if not m:
        raise ValueError("Could not find Chapter 11 – References header")
    rest = md[m.end() :]
    m2 = re.search(r"^## Chapter 12\b", rest, flags=re.M)
    if m2:
        rest = rest[: m2.start()]
    return rest


def _slice_body(md: str) -> str:
    m = re.search(r"^## Chapter 11\s+–\s+References\s*$", md, flags=re.M)
    if not m:
        raise ValueError("Could not find Chapter 11 – References header")
    return md[: m.start()]


def main() -> None:
    md_path = Path("Dissertation_Arka_Talukder.md")
    md = md_path.read_text(encoding="utf-8")

    refs_text = _slice_chapter_11(md)
    body = _slice_body(md)

    # Reference entries (first author surname + year).
    # Matches: "Surname, I. ... (2025) 'Title', ..."
    ref_pat = re.compile(r"^([A-Z][A-Za-z'\-]+),\s+[^\n]*?\((\d{4})\)\s+'", flags=re.M)
    refs = {(a, y) for a, y in ref_pat.findall(refs_text)}

    intext: set[tuple[str, str]] = set()

    # Allow basic Latin + common accented letters in surnames (e.g., Miège)
    name = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'\-]+"

    # 1) Narrative citations: "Surname et al. (YYYY)"
    for a, y in re.findall(rf"\b({name})\s+et\s+al\.\s*\((\d{{4}})\)", body):
        intext.add((a, y))

    # 2) Narrative citations: "Surname and Surname (YYYY)" → capture FIRST author only
    for a, y in re.findall(rf"\b({name})\s+and\s+{name}\s*\((\d{{4}})\)", body):
        if a in {"Figure", "Chapter", "Section", "Table", "Appendix"}:
            continue
        intext.add((a, y))

    # 3) Narrative citations: "Surname (YYYY)" (single author only)
    # Avoid matching the second author in "X and Y (YYYY)" by requiring that we are NOT preceded by "and ".
    for a, y in re.findall(rf"(?<!and\s)\b({name})\s*\((\d{{4}})\)", body):
        if a in {"Figure", "Chapter", "Section", "Table", "Appendix"}:
            continue
        intext.add((a, y))

    # 4) Parenthetical citations: "(Surname, YYYY)" and "(Surname et al., YYYY)"
    for a, y in re.findall(rf"\(({name})(?:\s+et\s+al\.)?,\s*(\d{{4}})\)", body):
        intext.add((a, y))

    # 5) Parenthetical citations: "(Surname and Surname, YYYY)" → capture FIRST author only
    for a, y in re.findall(rf"\(({name})\s+and\s+{name},\s*(\d{{4}})\)", body):
        intext.add((a, y))

    missing = sorted(intext - refs)
    unused = sorted(refs - intext)

    print(f"In-text citation pairs (first-author/year): {len(intext)}")
    print(f"Reference list pairs (first-author/year):   {len(refs)}")

    print("\nMissing references for in-text citations:")
    if not missing:
        print("  (none)")
    else:
        for a, y in missing:
            print(f"  - {a} {y}")

    print("\nUnused references (present but not cited):")
    if not unused:
        print("  (none)")
    else:
        for a, y in unused:
            print(f"  - {a} {y}")

    # DOI/URL hygiene checks (format only)
    bad_doi_lines: list[str] = []
    for line in refs_text.splitlines():
        if "doi.org" in line:
            if not re.search(r"https?://doi\.org/10\.[^\s]+", line):
                bad_doi_lines.append(line.strip())

    print(f"\nDOI lines failing basic pattern: {len(bad_doi_lines)}")
    for l in bad_doi_lines[:25]:
        print(f"  - {l}")

    http_non_doi = [
        line.strip()
        for line in refs_text.splitlines()
        if "http://" in line and "doi.org" not in line and line.strip()
    ]
    print(f"\nNon-DOI http:// links: {len(http_non_doi)}")
    for l in http_non_doi[:25]:
        print(f"  - {l}")


if __name__ == "__main__":
    main()

