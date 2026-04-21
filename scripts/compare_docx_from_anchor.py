"""Compare two .docx from anchor paragraph (e.g. Abstract) to end."""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

from docx import Document


def all_paras(doc_path: Path) -> list[str]:
    return [(p.text or "").strip() for p in Document(str(doc_path)).paragraphs]


def find_start(paras: list[str], must_contain: str) -> int:
    for i, t in enumerate(paras):
        if must_contain in t:
            return i
    return 0


def norm(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    return s


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    final = root / "submission" / "Arka_Talukder_Dissertation_Final_Submission.docx"
    hum = root / "submission" / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
    anchor = "1. Abstract"
    if len(sys.argv) > 1:
        anchor = sys.argv[1]

    pa, pb = all_paras(final), all_paras(hum)
    ia, ib = find_start(pa, anchor), find_start(pb, anchor)
    sa, sb = pa[ia:], pb[ib:]

    text_a = "\n".join(norm(t) for t in sa if t)
    text_b = "\n".join(norm(t) for t in sb if t)

    def wc(x: str) -> int:
        return len(re.findall(r"\b[\w'-]+\b", x))

    print(f"Anchor {anchor!r}: final para {ia}, humanized para {ib}")
    print("Words from anchor:", wc(text_a), "(final)", wc(text_b), "(humanized)")
    print("Paragraphs from anchor:", len([t for t in sa if t]), len([t for t in sb if t]))

    # Line diff (paragraph as line)
    la = [norm(t) for t in sa if t]
    lb = [norm(t) for t in sb if t]
    diff = list(difflib.unified_diff(la[:400], lb[:400], lineterm="", n=0))
    print("\nFirst 80 diff lines (unified, first 400 paras each):")
    for line in diff[:80]:
        print(line[:200])

    # Quick check: technical numbers that must match
    key_nums = ["934", "F1", "31 MB", "23ms", "128,002", "128002", "B01821011"]
    print("\nPresence in final / humanized (from anchor):")
    for k in key_nums:
        print(f"  {k!r}: {k in text_a} / {k in text_b}")


if __name__ == "__main__":
    main()
