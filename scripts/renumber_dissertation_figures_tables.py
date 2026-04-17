"""
One-off renumber: body figures 1..24 and tables 1..7 in strict document order.
Appendix code figures stay labelled Figure A1-1 .. A1-6 in captions (unchanged).

Run from repo root:
    python scripts/renumber_dissertation_figures_tables.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "Dissertation_Arka_Talukder.md"

# old figure number -> new (document order)
FIG: dict[int, int] = {
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    10: 5,
    17: 6,
    18: 7,
    1: 8,
    25: 9,
    26: 10,
    27: 11,
    28: 12,
    29: 13,
    30: 14,
    2: 15,
    3: 16,
    6: 17,
    7: 18,
    8: 19,
    9: 20,
    4: 21,
    5: 22,
    16: 23,
    15: 24,
}

# old table number -> new (document order: lit review first)
TAB: dict[int, int] = {5: 1, 1: 2, 2: 3, 3: 4, 4: 5, 6: 6, 7: 7}


def _sub_fig(m: re.Match[str]) -> str:
    n = int(m.group(1))
    if n in FIG:
        return f"Figure {FIG[n]}"
    return m.group(0)


def _sub_figs_plural(m: re.Match[str]) -> str:
    """Figures 12, 3 -> Figures NEW, NEW"""
    nums = [int(x) for x in m.group(1).replace(" ", "").split(",")]
    out = [str(FIG[n]) for n in nums if n in FIG]
    return "Figures " + ", ".join(out)


def _sub_fig_range(m: re.Match[str]) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    if a in FIG and b in FIG:
        return f"Figures {FIG[a]}–{FIG[b]}"
    return m.group(0)


def _sub_table(m: re.Match[str]) -> str:
    n = int(m.group(1))
    if n in TAB:
        return f"Table {TAB[n]}"
    return m.group(0)


def main() -> None:
    text = MD.read_text(encoding="utf-8")

    # "Figures 2, 6, 7" style
    text = re.sub(
        r"Figures\s+((?:\d+\s*,\s*)+\d+)",
        _sub_figs_plural,
        text,
    )
    # "Figures 2–5" or "Figures 17–18"
    text = re.sub(r"Figures\s+(\d+)\s*[–-]\s*(\d+)", _sub_fig_range, text)

    text = re.sub(r"\bFigure\s+(\d+)\b", _sub_fig, text)
    text = re.sub(r"\bTable\s+(\d+)\b", _sub_table, text)

    MD.write_text(text, encoding="utf-8", newline="\n")
    print("Updated", MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
