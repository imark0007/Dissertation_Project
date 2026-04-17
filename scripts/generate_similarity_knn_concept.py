"""
Author-original conceptual figure: flows as nodes, kNN edges in 2D feature projection.

Not copied from any paper figure. The *idea* of attribute/similarity-based graphs
for IoT IDS is discussed in Ngo et al. (2025) — cite in the dissertation caption.

Style: Matplotlib (https://matplotlib.org/stable/gallery/index.html).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "similarity_knn_concept.png"

# Local import after ROOT on path if needed
import sys

sys.path.insert(0, str(ROOT))
from src.evaluation.plot_style import apply_thesis_style


def main():
    apply_thesis_style()
    rng = np.random.default_rng(42)
    # Synthetic 2D projection of "flows": two loose clusters (benign vs attack toy layout)
    n = 7
    benign = rng.normal([0.2, 0.3], 0.12, size=(4, 2))
    attack = rng.normal([0.85, 0.75], 0.1, size=(3, 2))
    pts = np.vstack([benign, attack])
    labels = np.array([0, 0, 0, 0, 1, 1, 1])

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=150)
    colors = np.where(labels == 0, "#1f77b4", "#d62728")
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=120, edgecolors="black", linewidths=0.6, zorder=3)
    for i, (x, y) in enumerate(pts):
        ax.text(x + 0.02, y + 0.02, f"f{i+1}", fontsize=9)

    # kNN edges from node 0 (first benign) to 2 nearest by Euclidean distance
    origin = 0
    dists = np.linalg.norm(pts - pts[origin], axis=1)
    order = np.argsort(dists)[1:3]  # two neighbours
    for j in order:
        ax.plot([pts[origin, 0], pts[j, 0]], [pts[origin, 1], pts[j, 1]], "k-", alpha=0.45, lw=1.2, zorder=1)

    ax.set_xlabel("Projected feature dimension 1 (illustrative)")
    ax.set_ylabel("Projected feature dimension 2 (illustrative)")
    ax.set_title(r"Concept: kNN similarity graph within one window (each point = one flow)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=9, label="Pool A (illustr.)"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9, label="Pool B (illustr.)"),
        ],
        loc="upper left",
        frameon=True,
    )
    fig.text(
        0.5,
        0.02,
        "Author’s schematic (Matplotlib). Real implementation uses 46-D features and Euclidean kNN (Section 5.3).",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
