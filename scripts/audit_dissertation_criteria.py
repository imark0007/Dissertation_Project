"""Audit dissertation against requested chapter-level checklist."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "Arka_Talukder_Dissertation_Final_DRAFT.md"


def count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))


def chapter_blocks(text: str) -> dict[int, str]:
    parts = re.split(r"\n## Chapter (\d+)[^\n]*\n", text)
    out: dict[int, str] = {}
    i = 1
    while i < len(parts) - 1:
        ch = int(parts[i])
        out[ch] = parts[i + 1]
        i += 2
    return out


def _is_contiguous(nums: list[int], start: int = 1) -> bool:
    if not nums:
        return False
    expected = list(range(start, max(nums) + 1))
    return nums == expected


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    ch = chapter_blocks(text)
    print("Chapter word counts")
    targets = {
        1: (1700, 2000),
        2: (4200, 4800),
        3: (1100, 1400),
        4: (2600, 3000),
        5: (1400, 1700),
        6: (2800, 3200),
        7: (1400, 1700),
        8: (2300, 2700),
        9: (1800, 2200),
        10: (1400, 1700),
    }
    for i in range(1, 14):
        if i in ch:
            wc = count_words(ch[i])
            if i in targets:
                lo, hi = targets[i]
                ok = lo <= wc <= hi
                print(f"  Ch{i}: {wc}  target[{lo}-{hi}] {'PASS' if ok else 'FAIL'}")
            else:
                print(f"  Ch{i}: {wc}")

    # Serial integrity checks (chapter and caption numbering).
    chapter_heading_nums = [
        int(n) for n in re.findall(r"^## Chapter (\d+)\b", text, flags=re.MULTILINE)
    ]
    fig_body = [
        int(n)
        for n in re.findall(r"^\*\*Figure\s+(\d+):", text, flags=re.MULTILINE)
    ]
    tab_body = [
        int(n)
        for n in re.findall(r"^\*\*Table\s+(\d+):", text, flags=re.MULTILINE)
    ]
    fig_app = re.findall(r"^\*\*Figure\s+(A1-\d+)\*\*", text, flags=re.MULTILINE)
    fig_a1_paths = len(re.findall(r"fig_a1_(\d+)\.png", text))

    print("\nSerial checks")
    print("  Chapters in order:", chapter_heading_nums == list(range(1, 14)))
    print("  Figure captions contiguous (1..N):", _is_contiguous(fig_body, start=1))
    print("  Table captions contiguous (1..N):", _is_contiguous(tab_body, start=1))
    print("  Appendix A1 code images (expected 14 png refs):", fig_a1_paths == 14, f"(found {fig_a1_paths})")
    if fig_app:
        print("  **Figure A1-* formal captions in MD:", fig_app)
    if fig_a1_paths != 14:
        print("   - Appendix A1: list fig_a1_*.png paths in Final_DRAFT.md (see scripts/render_appendix1_code_figures.py).")
    if chapter_heading_nums != list(range(1, 14)):
        print("   - Found chapter headings:", chapter_heading_nums)
    if not _is_contiguous(fig_body, start=1):
        print("   - Found figure caption numbers:", fig_body)
    if not _is_contiguous(tab_body, start=1):
        print("   - Found table caption numbers:", tab_body)

    # Core checks from requested checklist
    rq = (
        "How can an explainable dynamic graph neural network, trained using federated learning, "
        "detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC "
        "operations on CPU-based edge devices?"
    )
    print("\nCore checks")
    print("  Main RQ verbatim present:", rq in text)
    print("  Sub-question 1 present:", "Does a dynamic graph model perform better" in text)
    print("  Sub-question 2 present:", "Can federated learning maintain similar performance" in text)
    print("  Sub-question 3 present:", "Can the model generate useful explanations" in text)
    print("  Section 1.6 table present:", "| Criterion (weight) |" in text)
    print("  Figure 6 ref in Ch5:", "Figure 6" in ch.get(5, ""))
    print("  Figure 7 ref in Ch5:", "Figure 7" in ch.get(5, ""))
    print("  Figure 8 ref in Ch5:", "Figure 8" in ch.get(5, ""))
    print("  k=5 in Ch5:", "k* = 5" in ch.get(5, "") or "k = 5" in ch.get(5, ""))
    print("  window=50 in Ch5:", "50 flows per window" in ch.get(5, ""))
    print("  sequence=5 in Ch5:", "sequences of five" in ch.get(5, ""))
    print("  POST /score in Ch6:", "POST /score" in ch.get(6, ""))
    print("  Figures 9-14 in Ch6:", all(f"Figure {n}" in ch.get(6, "") for n in range(9, 15)))
    print("  Metrics formulae in 7.4:", all(x in ch.get(7, "") for x in ["Precision = TP / (TP + FP)", "Recall = TP / (TP + FN)", "F1 = 2 ×"]))
    print("  Dataset stats in Ch7:", "500,000 flows" in ch.get(7, "") and "920 training sequences" in ch.get(7, ""))
    print("  Table 2 key values in Ch8:", all(v in ch.get(8, "") for v in ["0.9986", "0.9942", "1.0000"]))
    print("  Table 3 in Ch8:", "Table 3" in ch.get(8, ""))
    print("  Figures 15-24 in Ch8:", all(f"Figure {n}" in ch.get(8, "") for n in range(15, 25)))
    print("  Future work list in Ch9:", "### 9.8" in ("## Chapter 9 – " + ch.get(9, "")))
    print("  Chapter 10 first-person:", " I " in (" " + ch.get(10, "") + " "))

    # References DOI check
    if "## Chapter 11" in text and "## Chapter 12" in text:
        refs = text.split("## Chapter 11", 1)[1].split("## Chapter 12", 1)[0]
        entries = [ln.strip() for ln in refs.splitlines() if re.match(r"^[A-Z].*\(\d{4}\)", ln.strip())]
        non_doi = [e for e in entries if "Available at:" in e and "doi.org" not in e.lower()]
        print("\nReferences")
        print("  entries:", len(entries))
        print("  non-DOI entries:", len(non_doi))
        for e in non_doi:
            print("   -", e[:120])


if __name__ == "__main__":
    main()
