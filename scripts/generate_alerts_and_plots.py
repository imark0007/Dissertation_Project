"""
Generate all missing outputs required by the spec:
1. 5 example alerts with explanations (saved to results/alerts/)
2. FL convergence plot (saved to results/figures/)
3. SOC workflow example (saved to results/soc_workflow_example.md)

Usage: python scripts/generate_alerts_and_plots.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocess import load_config
from src.data.dataset import GraphSequenceDataset
from src.models.dynamic_gnn import DynamicGNN
from src.explain.explainer import explain_sequence
from src.siem.alert_formatter import format_ecs_alert, alert_to_json

logger = logging.getLogger(__name__)


def generate_example_alerts(cfg, n_alerts=5):
    """Generate n example alerts with full explanations."""
    device = torch.device("cpu")
    model = DynamicGNN.from_config(cfg)
    ckpt = Path("results/checkpoints/dynamic_gnn_best.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    test_graphs = torch.load("data/graphs/test_graphs.pt", weights_only=False)
    seq_len = cfg.get("graph", {}).get("sequence_length", 5)
    feat_names = cfg["data"]["flow_feature_columns"]

    alerts_dir = Path("results/alerts")
    alerts_dir.mkdir(parents=True, exist_ok=True)

    alerts = []
    seen_benign, seen_attack = 0, 0
    i = 0
    while len(alerts) < n_alerts and i + seq_len <= len(test_graphs):
        seq = test_graphs[i : i + seq_len]
        true_label = seq[-1].y.item()

        # Try to get a mix of benign and attack examples
        if true_label == 0 and seen_benign >= 3:
            i += seq_len
            continue
        if true_label == 1 and seen_attack >= 3:
            i += seq_len
            continue

        bundle = explain_sequence(model, seq, device, top_k_nodes=5, top_k_features=5)
        alert = format_ecs_alert(bundle, flow_feature_names=feat_names)
        alert["_true_label"] = "benign" if true_label == 0 else "attack"
        alert["_sample_idx"] = i
        alerts.append(alert)

        if true_label == 0:
            seen_benign += 1
        else:
            seen_attack += 1
        i += seq_len

    with open(alerts_dir / "example_alerts.json", "w") as f:
        json.dump(alerts, f, indent=2, default=str)
    logger.info("Saved %d example alerts to %s", len(alerts), alerts_dir / "example_alerts.json")

    # Also save a human-readable summary
    with open(alerts_dir / "alert_summary.txt", "w") as f:
        for idx, a in enumerate(alerts, 1):
            f.write(f"=== Alert {idx} (True label: {a['_true_label']}) ===\n")
            f.write(f"Prediction: {a['ml']['prediction']}  Score: {a['ml']['score']}\n")
            f.write(f"Severity: {a['threat']['indicator']['confidence']}\n")
            f.write(f"Top features:\n")
            for feat in a.get("explanation", {}).get("top_features", []):
                f.write(f"  {feat['name']}: {feat['importance']:.4f}\n")
            f.write(f"Top nodes:\n")
            for node in a.get("explanation", {}).get("top_nodes", []):
                f.write(f"  node_{node['node_idx']}: {node['importance']:.4f}\n")
            f.write("\n")
    logger.info("Saved alert summary to %s", alerts_dir / "alert_summary.txt")
    return alerts


def generate_fl_convergence_plot():
    """Generate FL round-by-round convergence plot."""
    fl_path = Path("results/metrics/fl_rounds.json")
    if not fl_path.exists():
        logger.warning("No FL rounds data found")
        return

    with open(fl_path) as f:
        data = json.load(f)
    rounds = data.get("rounds", [])
    comm = data.get("comm_bytes", [])
    if not rounds:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    r_nums = [r.get("round", i + 1) for i, r in enumerate(rounds)]
    f1s = [r.get("f1", 0) for r in rounds]
    aucs = [r.get("roc_auc", 0) for r in rounds]

    axes[0].plot(r_nums, f1s, "g-o", markersize=5)
    axes[0].set_xlabel("FL Round")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Federated GNN — F1 per Round")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(r_nums, aucs, "b-s", markersize=5)
    axes[1].set_xlabel("FL Round")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Federated GNN — AUC per Round")
    axes[1].set_ylim(0, 1.05)

    if comm:
        cumulative = np.cumsum([c / 1e6 for c in comm])
        axes[2].bar(range(1, len(comm) + 1), [c / 1e6 for c in comm], alpha=0.6, label="Per round")
        axes[2].plot(range(1, len(cumulative) + 1), cumulative, "r-o", markersize=4, label="Cumulative")
        axes[2].set_xlabel("Round")
        axes[2].set_ylabel("MB")
        axes[2].set_title("Communication Cost")
        axes[2].legend()

    fig.tight_layout()
    out = Path("results/figures/fl_convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved FL convergence plot to %s", out)


def generate_model_comparison_bar():
    """Bar chart comparing all 4 models."""
    met_dir = Path("results/metrics")
    results = {}
    for name in ["rf", "mlp", "central_gnn", "federated_gnn"]:
        p = met_dir / f"{name}_metrics.json"
        if p.exists():
            with open(p) as fh:
                results[name] = json.load(fh)

    if len(results) < 2:
        return

    models = list(results.keys())
    metrics_names = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics_names):
        vals = [results[n].get(m, 0) for n in models]
        ax.bar(x + i * width, vals, width, label=m.replace("_", " ").upper())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([n.replace("_", " ").title() for n in models], rotation=10)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title("Model Comparison — All Metrics")
    fig.tight_layout()
    fig.savefig("results/figures/model_comparison_bar.png", dpi=150)
    plt.close(fig)
    logger.info("Saved model comparison bar chart")


def generate_soc_workflow_doc():
    """Write a short SOC workflow triage example."""
    doc = """# SOC Alert Triage Workflow Example

## Scenario
A Dynamic GNN alert fires on a 5-window sequence of IoT traffic.

## Step 1: Analyst receives SIEM alert
The SIEM dashboard shows a new alert with:
- **event.kind**: alert
- **rule.name**: IoT Dynamic GNN Detector
- **ml.score**: 0.97 (high confidence)
- **threat.indicator.confidence**: high

## Step 2: Review explanation
The alert includes an explanation bundle:
- **Top features**: `Rate` (importance: 2.14), `Srate` (1.87), `Header_Length` (1.52)
- **Top nodes**: node_12 (importance: 3.41), node_7 (importance: 2.89)

The analyst sees that the detection was driven by unusually high packet rates
and abnormal header lengths — consistent with a DDoS flood pattern.

## Step 3: Cross-reference with context
The analyst checks:
- Are the top-contributing flows from known IoT devices? (node mapping)
- Does the time window correlate with other alerts? (SIEM timeline)
- Is the traffic pattern consistent with known attack signatures? (threat intel)

## Step 4: Decision
Based on the high model score (0.97), clear feature explanations pointing to
flood-like traffic patterns, and corroboration from other SIEM events, the
analyst escalates the alert for incident response.

## Value of Explainability
Without explanations, the analyst would see only "malicious / score 0.97" and
would need to manually inspect raw flow logs. The top-feature and top-node
explanations reduce triage time by directing attention to the most relevant
traffic characteristics, supporting faster and more confident decisions.
"""
    out = Path("results/soc_workflow_example.md")
    with open(out, "w") as f:
        f.write(doc)
    logger.info("Saved SOC workflow example to %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config("config/experiment.yaml")

    logger.info("=== Generating example alerts ===")
    generate_example_alerts(cfg, n_alerts=5)

    logger.info("=== Generating FL convergence plot ===")
    generate_fl_convergence_plot()

    logger.info("=== Generating model comparison bar chart ===")
    generate_model_comparison_bar()

    logger.info("=== Generating SOC workflow document ===")
    generate_soc_workflow_doc()

    logger.info("All outputs generated.")


if __name__ == "__main__":
    main()
