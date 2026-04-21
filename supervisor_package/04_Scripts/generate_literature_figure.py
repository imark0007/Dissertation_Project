"""
Generate Literature Review figure: positioning of related work (GNN, FL, Explainability, SIEM).
Research-based; all sources are cited in the dissertation References.

Output: assets/literature_positioning.png (for use in §3 Literature Review)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Studies and components (1 = yes, 0 = no). Order: GNN, FL, Explainability, SIEM-style alert
# Based on dissertation §3 and References
data = [
    ("Pinto et al. (2023)", 0, 0, 0, 0),   # Dataset
    ("Velickovic et al. (2018)", 1, 0, 0, 0),  # GAT
    ("McMahan et al. (2017)", 0, 1, 0, 0),    # FedAvg
    ("Sundararajan et al. (2017)", 0, 0, 1, 0),  # IG
    ("Han et al. (2025)", 1, 0, 0, 0),
    ("Ngo et al. (2025)", 1, 0, 0, 0),
    ("Basak et al. (2025)", 1, 0, 1, 0),
    ("Lusa et al. (2025)", 1, 0, 1, 0),
    ("Lazzarini et al. (2023)", 0, 1, 0, 0),
    ("Albanbay et al. (2025)", 0, 1, 0, 0),
    ("Alabbadi & Bajaber (2025)", 0, 0, 1, 0),
    ("Yang et al. (2025)", 1, 0, 0, 0),
    ("This project", 1, 1, 1, 1),
]
labels = ["GNN/Graph", "Federated\nlearning", "Explainability", "SIEM/Alert"]
studies = [r[0] for r in data]
matrix = np.array([r[1:5] for r in data])

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1, alpha=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(len(studies)))
ax.set_yticklabels(studies, fontsize=8)
ax.axhline(y=len(studies) - 1, color="darkgreen", linewidth=2)
for i in range(len(studies)):
    for j in range(4):
        text = "Yes" if matrix[i, j] else "—"
        color = "darkblue" if matrix[i, j] else "gray"
        ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="bold" if matrix[i, j] else "normal")
ax.set_title("Positioning of related work: GNN, federated learning, explainability, and SIEM-style alerts")
ax.set_xlabel("Components (sources: References §11)")
fig.tight_layout()
out = ROOT / "assets" / "literature_positioning.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)
