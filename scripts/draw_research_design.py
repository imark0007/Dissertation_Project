"""Draw a simple system-architecture figure for the dissertation (Chapter 5)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "research_design_system.png"


def add_box(ax, xy, wh, text, fc="#f2f2f2", ec="#333333"):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    return (x + w / 2, y + h), (x + w / 2, y)  # bottom center, top center


def arrow(ax, p0, p1):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=0.9,
        color="#444444",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Top row: pipeline
    _, b1 = add_box(ax, (0.15, 3.85), (1.35, 0.75), "CICIoT2023\nsubset (flows)")
    _, b2 = add_box(ax, (1.75, 3.85), (1.45, 0.75), "Preprocess\n(train stats)")
    _, b3 = add_box(ax, (3.45, 3.85), (1.45, 0.75), "kNN graphs\nper window")
    _, b4 = add_box(ax, (5.15, 3.85), (1.55, 0.75), "GAT + GRU\n(dynamic GNN)")
    _, b5 = add_box(ax, (6.95, 3.85), (1.45, 0.75), "Binary\nclassifier")

    arrow(ax, (0.9, 4.25), (1.75, 4.25))
    arrow(ax, (3.2, 4.25), (3.45, 4.25))
    arrow(ax, (4.9, 4.25), (5.15, 4.25))
    arrow(ax, (6.7, 4.25), (6.95, 4.25))

    # Middle: FL branch
    _, c1 = add_box(ax, (0.4, 2.35), (1.7, 0.85), "Client 1–3\nlocal data")
    _, c2 = add_box(ax, (2.45, 2.35), (1.55, 0.85), "Flower\nFedAvg")
    _, c3 = add_box(ax, (4.35, 2.35), (1.65, 0.85), "Global weights\n(no raw upload)")
    arrow(ax, (1.25, 2.78), (2.45, 2.78))
    arrow(ax, (4.0, 2.78), (4.35, 2.78))

    # Link FL to model row (conceptual)
    arrow(ax, (5.2, 3.85), (5.2, 3.2))
    arrow(ax, (3.75, 3.85), (3.75, 3.2))
    ax.text(3.75, 3.45, "non-IID split", ha="center", fontsize=7.5, color="#555555")

    # Bottom: explain + API
    _, d1 = add_box(ax, (1.0, 0.55), (1.85, 0.8), "Captum IG +\nGAT attention")
    _, d2 = add_box(ax, (3.35, 0.55), (1.85, 0.8), "FastAPI\nCPU inference")
    _, d3 = add_box(ax, (5.7, 0.55), (1.85, 0.8), "ECS-like JSON\nalerts → SIEM")

    arrow(ax, (5.2, 3.85), (1.9, 1.35))  # from classifier down toward explain
    arrow(ax, (2.9, 0.95), (3.35, 0.95))
    arrow(ax, (5.2, 0.95), (5.7, 0.95))

    ax.set_title(
        "Research design: IoT flows → model training (central / federated) → explainable edge alerts",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
