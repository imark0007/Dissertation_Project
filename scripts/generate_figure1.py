"""
Figure 1 — End-to-end research pipeline (topic: IoT flows → graph GNN → FL → XAI → SIEM JSON).

Run from repo root:
    python scripts/generate_figure1.py

Output: assets/figure1_pipeline.png (matches Arka_Talukder_Dissertation_Final_DRAFT.md and dissertation_to_docx.py).
Uses Matplotlib style sheets via src.evaluation.plot_style.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.evaluation.plot_style import apply_thesis_style

OUT_PATH = ROOT / "assets" / "figure1_pipeline.png"


def _box(ax, x, y, w, h, text, *, fontsize=9, title=None, title_fs=10, fc="#334155", ec="#1e293b"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.25,
        alpha=0.95,
    )
    ax.add_patch(p)
    yy = y + h - 0.22
    if title:
        ax.text(x + w / 2, yy, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color="white")
        yy -= 0.38
    ax.text(x + w / 2, yy, text, ha="center", va="top", fontsize=fontsize, color="white", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, *, rad=0.0):
    style = "arc3,rad=%.2f" % rad if rad else None
    kw = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color="#0f172a", shrinkA=2, shrinkB=2)
    if style:
        kw["connectionstyle"] = style
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))


def create_pipeline_figure() -> None:
    apply_thesis_style()
    plt.rcParams["font.family"] = "sans-serif"

    fig_w, fig_h = 14.0, 9.2
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    # Swimlane labels (left)
    ax.text(
        0.12,
        fig_h - 0.55,
        "Research pipeline — prototype path (CPU edge · CICIoT2023)",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        ha="left",
        va="top",
    )
    ax.text(
        0.12,
        fig_h - 0.95,
        "Software-defined IoT flow telemetry → dynamic GNN → federated learning → explainable SIEM-shaped alerts",
        fontsize=9,
        color="#475569",
        ha="left",
        va="top",
    )

    lane_left = 0.35
    ax.text(lane_left, 7.55, "Data &\ngraphs", fontsize=8, fontweight="bold", color="#64748b", ha="center", va="center")
    ax.text(lane_left, 4.85, "Models &\ntraining", fontsize=8, fontweight="bold", color="#64748b", ha="center", va="center")
    ax.text(lane_left, 2.05, "SOC output\n& eval", fontsize=8, fontweight="bold", color="#64748b", ha="center", va="center")

    # --- Row 1: data pipeline (y ~ 6.5–7.8) ---
    y1, h = 6.45, 1.35
    _box(
        ax,
        0.85,
        y1,
        2.55,
        h,
        "CICIoT2023\n46 flow features\n(benign / attack)",
        title="1  Benchmark",
        fc="#1e3a5f",
    )
    _box(
        ax,
        3.55,
        y1,
        2.65,
        h,
        "Clean · impute\nStandardScaler\nbinary labels",
        title="2  Preprocess",
        fc="#0f766e",
    )
    _box(
        ax,
        6.45,
        y1,
        3.35,
        h,
        "kNN graph / window\n50 flows · k=5\nseq. of 5 graphs → PyG",
        title="3  Graph + sequence",
        fc="#5b21b6",
    )
    _box(
        ax,
        10.05,
        y1,
        3.6,
        h,
        "Train / val / test\n(stratified)\nconfig YAML",
        title="4  Splits",
        fc="#4c1d95",
    )
    _arrow(ax, 3.4, y1 + h * 0.55, 3.55, y1 + h * 0.55)
    _arrow(ax, 6.2, y1 + h * 0.55, 6.45, y1 + h * 0.55)
    _arrow(ax, 9.8, y1 + h * 0.55, 10.05, y1 + h * 0.55)

    # --- Row 2: models (y ~ 3.9–5.2) ---
    y2, h2 = 3.85, 1.38
    _box(
        ax,
        0.85,
        y2,
        2.45,
        h2,
        "200 trees · depth 20\n46-D tabular",
        title="RF baseline",
        fc="#b91c1c",
    )
    _box(
        ax,
        3.45,
        y2,
        2.45,
        h2,
        "128→64→32\nReLU + dropout",
        title="MLP baseline",
        fc="#b91c1c",
    )
    _box(
        ax,
        6.05,
        y2,
        3.15,
        h2,
        "2× GAT · pool\nGRU · logits\n(central PyTorch)",
        title="Central GNN",
        fc="#991b1b",
    )
    _box(
        ax,
        9.45,
        y2,
        4.2,
        h2,
        "Flower FedAvg\n3 clients · non-IID α=0.5\n10 rounds · same arch",
        title="Federated GNN",
        fc="#7f1d1d",
    )
    # Arrows from graph row (bottom) down to each training column
    y_from = y1 + 0.02
    targets = (
        (0.85 + 2.45 / 2, y2 + h2),  # RF column centre, top of box
        (3.45 + 2.45 / 2, y2 + h2),
        (6.05 + 3.15 / 2, y2 + h2),
        (9.45 + 4.2 / 2, y2 + h2),
    )
    graph_cx = 6.45 + 3.35 / 2
    for tcx, ty in targets:
        _arrow(ax, graph_cx, y_from, tcx, ty, rad=0.08)

    # --- Row 3: explain → API → JSON → metrics ---
    y3, h3 = 1.05, 1.45
    _box(
        ax,
        0.85,
        y3,
        3.0,
        h3,
        "Integrated Gradients\n+ GAT attention\ntop-k features / flows",
        title="5  Explain (Captum)",
        fc="#c2410c",
    )
    _box(
        ax,
        4.05,
        y3,
        2.55,
        h3,
        "POST /score\nCPU inference\noptional explain",
        title="6  FastAPI",
        fc="#0369a1",
    )
    _box(
        ax,
        6.85,
        y3,
        3.15,
        h3,
        "ECS-like JSON\nseverity · score\nexplanation block",
        title="7  SIEM-shaped alert",
        fc="#075985",
    )
    _box(
        ax,
        10.2,
        y3,
        3.45,
        h3,
        "P, R, F1, ROC-AUC\nFPR · latency (ms)\nFL comms (MB)",
        title="8  Evaluation",
        fc="#047857",
    )
    _arrow(ax, 3.85, y3 + h3 * 0.5, 4.05, y3 + h3 * 0.5)
    _arrow(ax, 6.6, y3 + h3 * 0.5, 6.85, y3 + h3 * 0.5)
    _arrow(ax, 10.0, y3 + h3 * 0.5, 10.2, y3 + h3 * 0.5)

    # Central column: models down to explainability box
    gnn_cx = 6.05 + 3.15 / 2
    ex_cx = 0.85 + 3.0 / 2
    _arrow(ax, gnn_cx, y2 + 0.05, ex_cx, y3 + h3, rad=0.15)

    ax.text(
        7.0,
        0.42,
        "Arka Talukder · B01821011 · MSc Cyber Security · UWS   |   generated by scripts/generate_figure1.py",
        fontsize=7.5,
        color="#94a3b8",
        ha="center",
        va="bottom",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    create_pipeline_figure()
