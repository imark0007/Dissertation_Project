"""Compare two dissertation .docx files: stats + paragraph-level similarity sample."""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("pip install python-docx")
    raise


def paras_text(doc_path: Path) -> list[str]:
    d = Document(str(doc_path))
    out = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            out.append(t)
    return out


def word_count(texts: list[str]) -> int:
    return sum(len(re.findall(r"\b[\w'-]+\b", t)) for t in texts)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    a = root / "submission" / "Arka_Talukder_Dissertation_Final_Submission.docx"
    b = root / "submission" / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
    if len(sys.argv) >= 3:
        a, b = Path(sys.argv[1]), Path(sys.argv[2])

    pa, pb = paras_text(a), paras_text(b)
    print("A:", a.name)
    print("  non-empty paragraphs:", len(pa))
    print("  words:", word_count(pa))
    print("B:", b.name)
    print("  non-empty paragraphs:", len(pb))
    print("  words:", word_count(pb))

    # LCS-style rough: how many A paras have exact match in B
    set_b = set(pb)
    exact_a_in_b = sum(1 for x in pa if x in set_b)
    print("Exact paragraph matches (A in B):", exact_a_in_b, "/", len(pa))

    # First 15 mismatches by position (sequential compare, naive)
    n = min(len(pa), len(pb))
    mism = 0
    for i in range(n):
        if pa[i] != pb[i]:
            mism += 1
            if mism <= 12:
                print(f"\n--- mismatch at para index {i} ---")
                print("A:", pa[i][:220] + ("..." if len(pa[i]) > 220 else ""))
                print("B:", pb[i][:220] + ("..." if len(pb[i]) > 220 else ""))
    if len(pa) != len(pb):
        print("\nParagraph count differs; sequential alignment is approximate.")
    print("\nTotal sequential mismatches in shared prefix:", mism, "/", n)


if __name__ == "__main__":
    main()
