"""Approximate word counts for Dissertation_Arka_Talukder.md (quick supervisor check)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "Dissertation_Arka_Talukder.md"
text = MD.read_text(encoding="utf-8")


def count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))


# Abstract (body only, excluding the Keywords line if present)
m = re.search(r"^## 1\.\s*Abstract\s*$", text, flags=re.M)
if m:
    after = text[m.end() :]
    m_end = re.search(r"^---\s*$", after, flags=re.M)
    abstract_block = after[: m_end.start()].strip() if m_end else after.strip()
    abstract_body = re.split(r"^\*\*Keywords:\*\*.*$", abstract_block, flags=re.M)[0].strip()
    abstract_n = count_words(abstract_body)
else:
    abstract_n = 0

# Main body: Chapter 1 through end of Chapter 10 (exclude references, bib, appendices)
c1 = text.find("## Chapter 1")
c11 = text.find("## Chapter 11")
main = text[c1:c11] if c1 >= 0 and c11 > c1 else text
main_n = count_words(main)

# Chapters 1-13 excluding abstract/front (rough "full technical")
c0 = text.find("## Chapter 1")
c14 = text.find("---", text.find("## Chapter 13"))  # first --- after ch13 start - fragile
# Simpler: Ch1 through Ch13 end = before EOF or before trailing if any
ch13 = text.find("## Chapter 13")
rest = text[ch13:] if ch13 >= 0 else ""
ch13_end = len(text)
with_appendices = text[c1:ch13_end] if c1 >= 0 else text
with_app_n = count_words(with_appendices)

print("Dissertation_Arka_Talukder.md")
print("  abstract_words:", abstract_n)
print("  main_body_ch1_to_ch10_words (excl. refs, bib, appendices):", main_n)
print("  ch1_through_file_end_words:", with_app_n)
