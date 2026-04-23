"""
Count the word length of the dissertation abstract in Arka_Talukder_Dissertation_Final_DRAFT.md.

Usage:
    python scripts/count_abstract_words.py
"""
from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    md = Path("Arka_Talukder_Dissertation_Final_DRAFT.md").read_text(encoding="utf-8")
    m = re.search(r"^## 1\.\s*Abstract\s*$", md, flags=re.M)
    if not m:
        raise SystemExit("Could not find '## 1. Abstract' heading")
    after = md[m.end() :]
    m_end = re.search(r"^---\s*$", after, flags=re.M)
    if not m_end:
        raise SystemExit("Could not find end of abstract (--- separator)")
    abstract_block = after[: m_end.start()].strip()
    # If a **Keywords:** line exists, many word-count rules exclude it.
    parts = re.split(r"^\*\*Keywords:\*\*.*$", abstract_block, flags=re.M)
    abstract_body = parts[0].strip()

    body_words = re.findall(r"[A-Za-z0-9']+", abstract_body)
    all_words = re.findall(r"[A-Za-z0-9']+", abstract_block)

    print(f"Abstract words (body only, excluding Keywords line): {len(body_words)}")
    print(f"Abstract words (including Keywords line): {len(all_words)}")


if __name__ == "__main__":
    main()

