"""
One-shot composite renumber to make room for four new Chapter 8 figures:

Current  →  New
  22       23       (FL convergence shifts down by 1 to make room for per-class
                     bar chart at slot 22)
  23       27       (Model comparison shifts down by 4)
  24       28       (Ablation bar shifts down by 4)
  25       29       (Sensitivity heatmap shifts down by 4)

The new figures will then be inserted manually:
  22  Per-class precision/recall/F1 bar chart        (Section 8.2)
  24  Federated GNN confusion matrix                 (Section 8.3)
  25  Federated communication cost over rounds      (Section 8.3)
  26  Central GNN training loss curve               (Section 8.4)

Safe-by-construction:
- Operates on a fixed map only (no broad regex replacement).
- Renumbers in DESCENDING new-value order so 25→29 happens before 22→23.
- Skips file paths and Appendix labels.
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "Arka_Talukder_Dissertation_Final_DRAFT.md"

# The mapping is read in order; do larger new-value first to avoid clashes.
RENUMBER_MAP = [
    (25, 29),
    (24, 28),
    (23, 27),
    (22, 23),
]

text = PATH.read_text(encoding="utf-8")
original = text

# 1. Single 'Figure N' references.
for old, new in RENUMBER_MAP:
    text = re.sub(rf"\bFigure {old}\b", f"Figure {new}", text)

# 2. 'Figures' multi-references: bump every numeric token in the listed
#    range/list according to the same map.
def remap_multi(m: re.Match) -> str:
    head = m.group(1)
    body = m.group(2)
    map_dict = dict(RENUMBER_MAP)

    def bump(num_match: re.Match) -> str:
        n = int(num_match.group(0))
        return str(map_dict.get(n, n))

    return head + re.sub(r"\d+", bump, body)


text = re.sub(
    r"(Figures\s+)((?:\d+[\s,\-–and]+){1,5}\d+)",
    remap_multi,
    text,
)

# 3. Renumber LOF table rows ONLY between the LOF header line and the next
#    blank line.
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
        m = re.match(r"^\|\s*(\d{1,2})\s*\|(.*)$", ln)
        if m:
            n = int(m.group(1))
            new_n = dict(RENUMBER_MAP).get(n, n)
            if new_n != n:
                out.append(f"| {new_n} |{m.group(2)}")
                continue
    out.append(ln)
text = "\n".join(out) + ("\n" if original.endswith("\n") else "")

if text == original:
    raise SystemExit("No changes were made (manual review needed).")

PATH.write_text(text, encoding="utf-8")
print(f"Renumbered Chapter 8 figures in: {PATH}")
