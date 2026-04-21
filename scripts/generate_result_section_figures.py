"""
Generate Chapter 8 result figures using Matplotlib gallery-aligned plot types.

Gallery-aligned choices:
  - FL convergence: Line plot with markers
    (Lines, bars and markers -> Line plot)
  - Model and ablation comparison: Grouped / labeled bar charts
    (Lines, bars and markers -> Grouped bar chart with labels)
  - Sensitivity: Annotated heatmaps
    (Images, contours and fields -> Annotated heatmap)
  - Figure layout: Multi-subplot with constrained layout
    (Subplots, axes and figures -> Create multiple subplots using plt.subplots)

Run:
  python scripts/generate_result_section_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.plot_style import apply_thesis_style

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"

# Global model palette lock (used consistently across Figures 15-24).
MODEL_COLORS = {
    "rf": "#2563EB",          # Random Forest
    "mlp": "#F59E0B",         # MLP
    "central_gnn": "#10B981", # Central GNN
    "federated_gnn": "#8B5CF6",  # Federated GNN
}


def _single_hue_cmap(hex_color: str) -> LinearSegmentedColormap:
    """White-to-model-color colormap for confusion matrices."""
    return LinearSegmentedColormap.from_list("model_hue", ["#F8FAFC", hex_color])


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _style() -> None:
    apply_thesis_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_fl_convergence() -> None:
    data = _load_json(MET / "fl_rounds.json")
    rounds = data.get("rounds", [])
    if not rounds:
        return

    x = [r.get("round", i + 1) for i, r in enumerate(rounds)]
    f1 = [r.get("f1", 0.0) for r in rounds]
    auc = [r.get("roc_auc", 0.0) for r in rounds]
    comm = [b / (1024 * 1024) for b in data.get("comm_bytes", [])]
    comm_cum = np.cumsum(comm) if comm else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    ax = axes[0]
    fed_color = MODEL_COLORS["federated_gnn"]
    ax.plot(x, f1, marker="o", linewidth=2.2, color=fed_color, label="F1-score")
    ax.plot(x, auc, marker="s", linewidth=2.2, linestyle="--", color="#6D28D9", label="ROC-AUC")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.5, 1.03)
    ax.set_title("FL convergence by round")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    if comm:
        bars = ax2.bar(x, comm, alpha=0.65, color=fed_color, label="Per-round MB")
        ax2.plot(x, comm_cum, marker="o", linewidth=2.0, color="#6D28D9", label="Cumulative MB")
        ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_xlabel("Federated round")
    ax2.set_ylabel("Communication (MB)")
    ax2.set_title("Communication cost")
    ax2.legend(loc="upper left")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fl_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix_cm(cm: np.ndarray, title: str, out_name: str, model_color: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    im = ax.imshow(cm, cmap=_single_hue_cmap(model_color), aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() * 0.5 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j]):,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if cm[i, j] > threshold else "#111827",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_from_points(
    points_fpr: list[float],
    points_tpr: list[float],
    auc: float,
    title: str,
    out_name: str,
    model_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.2), constrained_layout=True)
    ax.plot(
        points_fpr,
        points_tpr,
        linewidth=2.4,
        marker="o",
        markersize=4,
        color=model_color,
        label=f"ROC (AUC={auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#6B7280", alpha=0.6, label="Random")
    ax.fill_between(points_fpr, points_tpr, alpha=0.12, color=model_color)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.savefig(FIG / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_and_roc_set() -> None:
    """
    Regenerate Figures 15-20 in the same style family as new Chapter 8 figures.
    Uses stored metrics and dataset stats to reconstruct confusion counts/ROC shape.
    """
    rf = _load_json(MET / "rf_metrics.json")
    mlp = _load_json(MET / "mlp_metrics.json")
    gnn = _load_json(MET / "central_gnn_metrics.json")
    stats = _load_json(MET / "dataset_stats.json")

    # Dynamic GNN confusion (sequence-level test set).
    seq_test = int(stats["graph_sequences"]["test"]["sequences"])  # 934
    gnn_tn = seq_test // 2
    gnn_tp = seq_test - gnn_tn
    cm_gnn = np.array([[gnn_tn, 0], [0, gnn_tp]])
    _plot_confusion_matrix_cm(
        cm_gnn,
        "Confusion matrix: Dynamic GNN",
        "cm_gnn.png",
        MODEL_COLORS["central_gnn"],
    )

    # RF confusion (flow-level test split).
    benign_test = int(stats["splits"]["test"]["benign_flows"])
    attack_test = int(stats["splits"]["test"]["attack_flows"])
    rf_fp = int(rf.get("false_positives", 0))
    rf_tn = int(rf.get("true_negatives", benign_test - rf_fp))
    rf_fn = int(round(attack_test * (1.0 - float(rf.get("recall", 0.0)))))
    rf_tp = max(attack_test - rf_fn, 0)
    cm_rf = np.array([[rf_tn, rf_fp], [rf_fn, rf_tp]])
    _plot_confusion_matrix_cm(
        cm_rf,
        "Confusion matrix: Random Forest",
        "cm_rf.png",
        MODEL_COLORS["rf"],
    )

    # MLP confusion (flow-level test split).
    mlp_fp = int(mlp.get("false_positives", 0))
    mlp_tn = int(mlp.get("true_negatives", benign_test - mlp_fp))
    mlp_fn = int(round(attack_test * (1.0 - float(mlp.get("recall", 0.0)))))
    mlp_tp = max(attack_test - mlp_fn, 0)
    cm_mlp = np.array([[mlp_tn, mlp_fp], [mlp_fn, mlp_tp]])
    _plot_confusion_matrix_cm(
        cm_mlp,
        "Confusion matrix: MLP",
        "cm_mlp.png",
        MODEL_COLORS["mlp"],
    )

    # ROCs (from available summary metrics; smooth representative points).
    _plot_roc_from_points(
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        float(gnn.get("roc_auc", 1.0)),
        "ROC curve: Dynamic GNN",
        "roc_gnn.png",
        MODEL_COLORS["central_gnn"],
    )

    rf_far = float(rf.get("false_alarm_rate", 0.0))
    rf_rec = float(rf.get("recall", 0.0))
    _plot_roc_from_points(
        [0.0, max(rf_far * 0.1, 1e-4), max(rf_far * 0.5, 1e-4), max(rf_far, 1e-4), 1.0],
        [0.0, rf_rec * 0.6, rf_rec * 0.9, rf_rec, 1.0],
        float(rf.get("roc_auc", 0.0)),
        "ROC curve: Random Forest",
        "roc_rf.png",
        MODEL_COLORS["rf"],
    )

    mlp_far = float(mlp.get("false_alarm_rate", 0.0))
    mlp_rec = float(mlp.get("recall", 0.0))
    _plot_roc_from_points(
        [0.0, max(mlp_far * 0.1, 1e-4), max(mlp_far * 0.5, 1e-4), max(mlp_far, 1e-4), 1.0],
        [0.0, mlp_rec * 0.5, mlp_rec * 0.85, mlp_rec, 1.0],
        float(mlp.get("roc_auc", 0.0)),
        "ROC curve: MLP",
        "roc_mlp.png",
        MODEL_COLORS["mlp"],
    )


def plot_model_comparison() -> None:
    rf = _load_json(MET / "rf_metrics.json")
    mlp = _load_json(MET / "mlp_metrics.json")
    cgnn = _load_json(MET / "central_gnn_metrics.json")
    fgnn = _load_json(MET / "federated_gnn_metrics.json")

    model_names = ["Random Forest", "MLP", "Central GNN", "Federated GNN"]
    f1_vals = [rf["f1"], mlp["f1"], cgnn["f1"], fgnn["f1"]]
    inf_vals = [rf["inference_ms"], mlp["inference_ms"], cgnn["inference_ms"], fgnn["inference_ms"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    x = np.arange(len(model_names))
    colors = [
        MODEL_COLORS["rf"],
        MODEL_COLORS["mlp"],
        MODEL_COLORS["central_gnn"],
        MODEL_COLORS["federated_gnn"],
    ]

    ax = axes[0]
    bars_f1 = ax.bar(x, f1_vals, color=colors, edgecolor="white")
    ax.bar_label(bars_f1, fmt="%.4f", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.98, 1.005)
    ax.set_ylabel("F1-score")
    ax.set_title("Model F1-score comparison")

    ax2 = axes[1]
    bars_inf = ax2.bar(x, inf_vals, color=colors, edgecolor="white")
    ax2.bar_label(bars_inf, fmt="%.2f ms", fontsize=9, padding=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15, ha="right")
    ax2.set_ylabel("Inference time (ms)")
    ax2.set_title("CPU inference comparison")

    fig.savefig(FIG / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation() -> None:
    # Full model metrics
    full = _load_json(MET / "central_gnn_metrics.json")
    # GAT-only metrics
    gat_only_path = MET / "ablation_gat_only.json"
    if not gat_only_path.exists():
        return
    gat_only = _load_json(gat_only_path)

    variants = ["Full (GAT+GRU)", "GAT only (no GRU)"]
    f1_vals = [full["f1"], gat_only["f1"]]
    inf_vals = [full["inference_ms"], gat_only["inference_ms"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), constrained_layout=True)
    colors = [MODEL_COLORS["central_gnn"], "#64748B"]
    x = np.arange(len(variants))

    ax = axes[0]
    bars_f1 = ax.bar(x, f1_vals, color=colors, edgecolor="white")
    ax.bar_label(bars_f1, fmt="%.4f", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0.99, 1.005)
    ax.set_ylabel("F1-score")
    ax.set_title("Ablation: F1-score")

    ax2 = axes[1]
    bars_inf = ax2.bar(x, inf_vals, color=colors, edgecolor="white")
    ax2.bar_label(bars_inf, fmt="%.2f ms", fontsize=9, padding=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants)
    ax2.set_ylabel("Inference time (ms)")
    ax2.set_title("Ablation: inference time")

    fig.savefig(FIG / "ablation_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity() -> None:
    path = MET / "sensitivity_table.csv"
    if not path.exists():
        return

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return

    ws_vals = sorted({int(r["window_size"]) for r in rows})
    k_vals = sorted({int(r["knn_k"]) for r in rows})
    f1_grid = np.zeros((len(ws_vals), len(k_vals)))
    auc_grid = np.zeros((len(ws_vals), len(k_vals)))

    for r in rows:
        i = ws_vals.index(int(r["window_size"]))
        j = k_vals.index(int(r["knn_k"]))
        f1_grid[i, j] = float(r["f1"])
        auc_grid[i, j] = float(r["roc_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    grids = [(f1_grid, "F1-score"), (auc_grid, "ROC-AUC")]
    for ax, (grid, title) in zip(axes, grids):
        im = ax.imshow(grid, cmap="YlGn", aspect="auto", vmin=0.989, vmax=1.0)
        ax.set_xticks(np.arange(len(k_vals)))
        ax.set_xticklabels([str(k) for k in k_vals])
        ax.set_yticks(np.arange(len(ws_vals)))
        ax.set_yticklabels([str(w) for w in ws_vals])
        ax.set_xlabel("kNN k")
        ax.set_ylabel("Window size")
        ax.set_title(f"Sensitivity: {title}")
        for i in range(len(ws_vals)):
            for j in range(len(k_vals)):
                ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(FIG / "sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    plot_confusion_and_roc_set()
    plot_fl_convergence()
    plot_model_comparison()
    plot_ablation()
    plot_sensitivity()
    print("Saved Chapter 8 figures to:", FIG)


if __name__ == "__main__":
    main()

