"""
One-shot renumber of body figures from 6..24 to 7..25 to make room for a new
Figure 6 (Project Gantt chart) inserted in Chapter 3.

Safe-by-construction:
- Operates only on the literal patterns 'Figure N' and 'Figures' lists.
- Does NOT touch figure file paths or appendix labels.
- Limits Table of Figures row renumbering to the LOF block only.
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "Dissertation_Arka_Talukder.md"

text = PATH.read_text(encoding="utf-8")
original = text

# 1. 'Figure N' (single) — descending so 24->25 first to avoid double-bumps.
for n in range(24, 5, -1):
    text = re.sub(rf"\bFigure {n}\b", f"Figure {n + 1}", text)

# 2. 'Figures' multi-references like 'Figures 6-8', 'Figures 15 and 16',
#    'Figures 23-24', 'Figures 17, 18 and 19'.
def bump_numbers_in_multi(m: re.Match) -> str:
    head = m.group(1)
    body = m.group(2)
    def bump(num_match: re.Match) -> str:
        n = int(num_match.group(0))
        return str(n + 1) if 6 <= n <= 24 else str(n)
    return head + re.sub(r"\d+", bump, body)

text = re.sub(
    r"(Figures\s+)((?:\d+[\s,\-–and]+){1,5}\d+)",
    bump_numbers_in_multi,
    text,
)

# 3. Renumber the LOF table rows ONLY between the LOF header line and the
#    next blank line.  Locate the LOF header by exact match.
lines = text.splitlines()
lof_header_pat = re.compile(r"^\|\s*Figure\s*\|\s*Title")
in_lof = False
out: list[str] = []
for ln in lines:
    if lof_header_pat.match(ln):
        in_lof = True
        out.append(ln)
        continue
    if in_lof:
        if ln.strip() == "":
            in_lof = False
            out.append(ln)
            continue
        # Bump the leading | N | column if N in 6..24.
        m = re.match(r"^\|\s*(\d{1,2})\s*\|(.*)$", ln)
        if m:
            n = int(m.group(1))
            if 6 <= n <= 24:
                out.append(f"| {n + 1} |{m.group(2)}")
                continue
    out.append(ln)
text = "\n".join(out) + ("\n" if original.endswith("\n") else "")

if text == original:
    raise SystemExit("No changes were made — aborting (manual review needed).")

PATH.write_text(text, encoding="utf-8")
print(f"Renumbered figures in: {PATH}")
