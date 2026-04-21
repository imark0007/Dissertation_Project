"""
Generate Literature Review Figure 5:
Positioning of related work (GNN, FL, Explainability, SIEM).

Output:
  assets/literature_positioning.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.plot_style import apply_thesis_style

# Studies and components (1 = yes, 0 = no). Order: GNN, FL, Explainability, SIEM-style alert.
DATA = [
    ("Pinto et al. (2023)", 0, 0, 0, 0),
    ("Velickovic et al. (2018)", 1, 0, 0, 0),
    ("McMahan et al. (2017)", 0, 1, 0, 0),
    ("Sundararajan et al. (2017)", 0, 0, 1, 0),
    ("Han et al. (2025)", 1, 0, 0, 0),
    ("Ngo et al. (2025)", 1, 0, 0, 0),
    ("Basak et al. (2025)", 1, 0, 1, 0),
    ("Lusa et al. (2025)", 1, 0, 1, 0),
    ("Lazzarini et al. (2023)", 0, 1, 0, 0),
    ("Albanbay et al. (2025)", 0, 1, 0, 0),
    ("Alabbadi and Bajaber (2025)", 0, 0, 1, 0),
    ("Yang et al. (2025)", 1, 0, 0, 0),
    ("This project", 1, 1, 1, 1),
]

COL_LABELS = [
    "GNN / Graph",
    "Federated learning",
    "Explainability",
    "SIEM-style alert",
]


def main() -> None:
    apply_thesis_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    studies = [r[0] for r in DATA]
    matrix = np.array([r[1:5] for r in DATA], dtype=int)

    # Build a clean matrix table for better readability in dissertation print/PDF.
    fig, ax = plt.subplots(figsize=(12.2, 8.8), constrained_layout=True)
    ax.axis("off")

    cell_text = [["Yes" if v == 1 else "-" for v in row] for row in matrix]
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=studies,
        colLabels=COL_LABELS,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        edges="closed",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    # Increase row height for cleaner spacing.
    tbl.scale(1.1, 1.9)

    header_color = "#E8EEF8"
    zebra_a = "#FFFFFF"
    zebra_b = "#F8FAFC"
    edge_color = "#CBD5E1"
    yes_color = "#166534"
    no_color = "#6B7280"
    project_highlight = "#DCFCE7"

    # Column header styling.
    for c in range(len(COL_LABELS)):
        cell = tbl[(0, c)]
        cell.set_facecolor(header_color)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(1.1)
        txt = cell.get_text()
        txt.set_fontweight("bold")
        txt.set_color("#111827")
        txt.set_fontsize(11)

    # Data rows and row labels.
    for r in range(1, len(studies) + 1):
        row_color = zebra_a if r % 2 else zebra_b
        if r == len(studies):
            row_color = project_highlight

        # Row label is in column -1.
        label_cell = tbl[(r, -1)]
        label_cell.set_facecolor(row_color)
        label_cell.set_edgecolor(edge_color)
        label_cell.set_linewidth(1.1 if r == len(studies) else 0.9)
        label_text = label_cell.get_text()
        label_text.set_fontsize(11)
        label_text.set_color("#111827")
        if r == len(studies):
            label_text.set_fontweight("bold")

        for c in range(len(COL_LABELS)):
            cell = tbl[(r, c)]
            cell.set_facecolor(row_color)
            cell.set_edgecolor(edge_color)
            cell.set_linewidth(1.1 if r == len(studies) else 0.9)
            txt = cell.get_text()
            is_yes = matrix[r - 1, c] == 1
            txt.set_color(yes_color if is_yes else no_color)
            txt.set_fontweight("bold" if is_yes or r == len(studies) else "normal")
            txt.set_fontsize(12 if is_yes else 11)

    # Add tiny legend line for quick interpretation.
    ax.text(
        0.01,
        0.04,
        "Yes = component covered in study, - = not reported",
        transform=ax.transAxes,
        fontsize=10,
        color="#374151",
    )

    ax.set_title(
        "Figure 5. Positioning of related work across GNN, Federated Learning, Explainability, and SIEM-style alerts",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    out = ROOT / "assets" / "literature_positioning.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    main()
