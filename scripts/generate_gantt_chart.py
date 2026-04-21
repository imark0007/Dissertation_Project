"""
Generate a clean Gantt chart for Chapter 3 (Project Management).

Output:
    assets/gantt_chart.png  (300 DPI, Times New Roman labels)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "gantt_chart.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


PHASES = [
    ("P1: Requirements and literature freeze",        0,  6, "#1f77b4"),
    ("P2: Dataset subset and preprocessing",          5, 10, "#1f77b4"),
    ("P3: Graph construction and central GNN",        9, 18, "#2ca02c"),
    ("P4: Baselines and federated training (Flower)",16, 26, "#2ca02c"),
    ("P5: Explainability, alerts and FastAPI",       24, 32, "#ff7f0e"),
    ("P6: Ablation, sensitivity and multi-seed",     30, 38, "#ff7f0e"),
    ("P7: Final figures and report writing",         34, 45, "#d62728"),
]

GROUP_LEGEND = [
    Patch(facecolor="#1f77b4", label="Foundation (data and pipeline)"),
    Patch(facecolor="#2ca02c", label="Modelling (central and federated)"),
    Patch(facecolor="#ff7f0e", label="Explainability and deployment"),
    Patch(facecolor="#d62728", label="Evaluation and write-up"),
]


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=300)

    y_positions = list(range(len(PHASES)))
    for y, (label, start, end, colour) in zip(y_positions, PHASES):
        ax.barh(
            y=y,
            width=end - start,
            left=start,
            height=0.55,
            color=colour,
            edgecolor="#222222",
            linewidth=0.6,
        )
        ax.text(
            start + (end - start) / 2,
            y,
            f"{end - start} d",
            va="center",
            ha="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([p[0] for p in PHASES])
    ax.invert_yaxis()
    ax.set_xlim(0, 46)
    ax.set_xticks(list(range(0, 46, 5)))
    ax.set_xlabel("Project day (45-day MSc window)")
    ax.set_title("Figure 9: Project Gantt chart (six execution phases plus write-up)", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(handles=GROUP_LEGEND, loc="lower right", frameon=True, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
