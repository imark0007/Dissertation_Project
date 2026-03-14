"""
Generate all Literature Review figures (research-based; sources in References §11).
Run: python scripts/generate_literature_figures.py
Outputs: assets/literature_*.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = ROOT / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_ids_taxonomy():
    """§3.1: Taxonomy of intrusion detection approaches (Kolias et al., Pinto et al., Wang et al., Zhong et al.)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # Boxes
    boxes = [
        (0.5, 3.5, 2.5, 1.2, "Signature-based\n(known patterns)", "#E8F4F8"),
        (3.75, 3.5, 2.5, 1.2, "Anomaly-based\n(normal vs. deviation)", "#D4EDDA"),
        (7, 3.5, 2.5, 1.2, "ML / data-driven\n(flow, graph, FL)", "#FFF3CD"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="gray")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, wrap=True)
    ax.text(5, 2, "IoT intrusion detection approaches\n(Sources: Kolias et al., 2017; Pinto et al., 2023; Wang et al., 2025; Zhong et al., 2024)", ha="center", fontsize=8, style="italic")
    ax.text(5, 0.8, "This project: ML-based with graph + temporal + FL + explainability", ha="center", fontsize=8, fontweight="bold")
    fig.suptitle("Figure: Taxonomy of IDS approaches relevant to IoT and this project", fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "literature_ids_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_DIR / "literature_ids_taxonomy.png")


def fig_fedavg_flow():
    """§3.4: FedAvg flow — clients train locally, send updates, server aggregates (McMahan et al., 2017)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    # Server
    ax.add_patch(mpatches.FancyBboxPatch((4, 2.2), 2, 1.2, boxstyle="round,pad=0.03", facecolor="#E8E8F4", edgecolor="black"))
    ax.text(5, 2.8, "Server\n(FedAvg)", ha="center", va="center", fontsize=9)
    # Clients
    for i, x in enumerate([0.8, 3.5, 6.2, 9]):
        if i < 3:
            ax.add_patch(mpatches.FancyBboxPatch((x, 0.3), 1.4, 0.9, boxstyle="round,pad=0.02", facecolor="#D4EDDA", edgecolor="gray"))
            ax.text(x + 0.7, 0.75, f"Client {i}", ha="center", va="center", fontsize=8)
        else:
            ax.text(x + 0.5, 0.75, "...", ha="center", fontsize=12)
    # Arrows: client -> server (updates)
    ax.annotate("", xy=(5, 2.2), xytext=(1.5, 1.2), arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("", xy=(5, 2.2), xytext=(4.2, 1.2), arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("", xy=(5, 2.2), xytext=(6.9, 1.2), arrowprops=dict(arrowstyle="->", color="gray"))
    ax.text(3.2, 1.6, "model updates", fontsize=7, color="gray")
    # Arrow: server -> clients (broadcast)
    ax.annotate("", xy=(4.2, 1.2), xytext=(5, 2.2), arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5))
    ax.text(4.5, 1.75, "global model", fontsize=7, color="darkblue")
    ax.text(5, 3.6, "Federated learning (FedAvg): no raw data leaves clients\n(McMahan et al., 2017; Lazzarini et al., 2023)", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "literature_fedavg_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_DIR / "literature_fedavg_flow.png")


def fig_dynamic_gnn_concept():
    """§3.3: Dynamic GNN concept — graph snapshots → GAT → GRU → prediction (Velickovic et al., Liu et al.)."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    # Flow: t1 -> t2 -> t3 -> ... -> GAT -> GRU -> prediction
    boxes = [
        (0.3, 1, 1.1, 1, "Graph\n(t=1)", "#E8F4F8"),
        (1.6, 1, 1.1, 1, "Graph\n(t=2)", "#E8F4F8"),
        (2.9, 1, 1.1, 1, "...", "#f0f0f0"),
        (4, 1, 1.2, 1, "Graph\n(t=T)", "#E8F4F8"),
        (5.5, 1, 1.3, 1, "GAT\n(Velickovic et al.)", "#D4EDDA"),
        (7.1, 1, 1.2, 1, "GRU\n(temporal)", "#FFF3CD"),
        (8.5, 1, 1, 1, "Label", "#E8E8F4"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="gray"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0], 1.5), xytext=(boxes[i][0] + boxes[i][2], 1.5), arrowprops=dict(arrowstyle="->", color="gray"))
    ax.text(5, 0.35, "Dynamic GNN: graph snapshots over time → GAT (structure) → GRU (temporal) → prediction\n(Sources: Velickovic et al., 2018; Liu et al., 2019; Lusa et al., 2025; Basak et al., 2025)", ha="center", fontsize=7, style="italic")
    fig.suptitle("Figure: Conceptual flow of a dynamic GNN (GAT + GRU) for temporal graph classification", fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "literature_dynamic_gnn_concept.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_DIR / "literature_dynamic_gnn_concept.png")


def fig_explainability_types():
    """§3.5: Types of explainability used in this project (Sundararajan et al., Lundberg & Lee, Kokhlikyan et al.)."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        (0.5, 0.8, 2.8, 1.2, "Integrated Gradients\n(Sundararajan et al., 2017)\nFeature-level attribution", "#E8F4F8"),
        (3.6, 0.8, 2.8, 1.2, "GAT attention weights\n(Velickovic et al., 2018)\nFlow / edge importance", "#D4EDDA"),
        (6.7, 0.8, 2.8, 1.2, "Alert output\nTop features + top flows\n(SOC triage)", "#FFF3CD"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="gray"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(3.4, 1.4), xytext=(3.3, 1.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.5, 1.4), xytext=(6.4, 1.4), arrowprops=dict(arrowstyle="->"))
    ax.text(5, 0.2, "Explainability pipeline in this project (Sources: §11; Captum: Kokhlikyan et al., 2020)", ha="center", fontsize=7, style="italic")
    fig.suptitle("Figure: Explainability methods used for SOC-oriented alerts", fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "literature_explainability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_DIR / "literature_explainability.png")


def main():
    fig_ids_taxonomy()
    fig_fedavg_flow()
    fig_dynamic_gnn_concept()
    fig_explainability_types()
    print("All literature figures generated. Run generate_literature_figure.py for the positioning matrix.")


if __name__ == "__main__":
    main()
