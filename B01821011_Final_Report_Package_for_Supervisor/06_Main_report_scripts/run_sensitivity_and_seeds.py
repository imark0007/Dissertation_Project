"""
Run sensitivity analysis and multi-seed experiments for the dissertation.

1. Sensitivity: window_size in {30, 50, 70}, knn_k in {3, 5, 7}
2. Multi-seed: seeds 42, 123, 456 for the Central GNN

Outputs:
  results/metrics/sensitivity_table.csv
  results/metrics/multi_seed_summary.json
  B01821011_Arka_Talukder_Main_Report/figures/fig_sensitivity.png
"""
import argparse
import csv
import json
import logging
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.data.graph_builder import build_graphs_for_split
from src.data.dataset import GraphSequenceDataset, _collate_sequences
from src.models.dynamic_gnn import DynamicGNN
from src.models.trainer import train_gnn, evaluate, measure_inference_time, compute_class_weights
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

MET_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "B01821011_Arka_Talukder_Main_Report" / "figures"
FIG_DIR2 = ROOT / "results" / "figures"


def build_loaders_with_params(cfg, window_size, knn_k, seed=42):
    """Build graph dataloaders with custom window_size and knn_k."""
    feat_cols = cfg["data"]["flow_feature_columns"]
    proc_dir = Path(cfg["data"]["processed_dir"])
    graph_cfg = cfg.get("graph", {})
    seq_len = graph_cfg.get("sequence_length", 5)
    m_stride = graph_cfg.get("minority_stride", None)
    if m_stride and window_size != 50:
        m_stride = window_size // 2
    batch_size = cfg["models"]["dynamic_gnn"].get("batch_size", 16)

    loaders = {}
    for split in ["train", "validation", "test"]:
        pq = proc_dir / f"{split}.parquet"
        if not pq.exists():
            logger.warning("Missing %s", pq)
            continue
        df = pd.read_parquet(pq)
        graphs = build_graphs_for_split(
            df, feat_cols, window_size=window_size, knn_k=knn_k,
            seed=seed, minority_stride=m_stride,
        )
        ds = GraphSequenceDataset(graphs, seq_len)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=_collate_sequences,
            drop_last=(split == "train"),
        )
        logger.info("[w=%d, k=%d, seed=%d] %s: %d seqs", window_size, knn_k, seed, split, len(ds))
    return loaders


def train_and_evaluate_gnn(cfg, loaders, device, seed=42):
    """Train a fresh GNN and evaluate. Returns test metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DynamicGNN.from_config(cfg).to(device)
    gnn_cfg = cfg["models"]["dynamic_gnn"]
    train_cfg = cfg.get("training", {})

    class_weights = compute_class_weights(loaders["train"])

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_cfg.get("lr", 0.001))
    best_f1 = 0.0
    best_state = None
    no_improve = 0
    patience = train_cfg.get("early_stopping_patience", 5)

    for epoch in range(1, gnn_cfg.get("epochs", 30) + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        w = class_weights.to(device)
        for sequences, labels in loaders["train"]:
            labels = labels.to(device)
            logits = model.forward_batch(sequences, device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_loader = loaders.get("validation", loaders["train"])
        val_metrics = evaluate(model, val_loader, device)
        logger.info("  Epoch %02d loss=%.4f val_f1=%.4f", epoch, total_loss / max(n_batches, 1), val_metrics["f1"])

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, loaders["test"], device)
    test_metrics["inference_ms"] = measure_inference_time(model, loaders["test"], device)
    return test_metrics


def run_sensitivity(cfg):
    """Run sensitivity analysis: window_size x knn_k grid."""
    device = torch.device("cpu")
    window_sizes = [30, 50, 70]
    knn_ks = [3, 5, 7]
    results = []

    for ws in window_sizes:
        for k in knn_ks:
            logger.info("=== Sensitivity: window=%d, k=%d ===", ws, k)
            t0 = time.perf_counter()
            loaders = build_loaders_with_params(cfg, ws, k, seed=42)
            if "train" not in loaders or "test" not in loaders:
                logger.warning("Skipping w=%d, k=%d (missing data)", ws, k)
                continue
            metrics = train_and_evaluate_gnn(cfg, loaders, device, seed=42)
            elapsed = time.perf_counter() - t0
            row = {
                "window_size": ws,
                "knn_k": k,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
                "roc_auc": round(metrics["roc_auc"], 4),
                "inference_ms": round(metrics["inference_ms"], 2),
                "time_s": round(elapsed, 1),
            }
            results.append(row)
            logger.info("  Result: F1=%.4f AUC=%.4f inf=%.1fms (%.0fs)",
                        row["f1"], row["roc_auc"], row["inference_ms"], elapsed)

    MET_DIR.mkdir(parents=True, exist_ok=True)
    out = MET_DIR / "sensitivity_table.csv"
    if results:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        logger.info("Sensitivity table -> %s", out)

    return results


def run_multi_seed(cfg):
    """Run Central GNN with seeds 42, 123, 456 and report mean +/- std."""
    device = torch.device("cpu")
    seeds = [42, 123, 456]
    all_metrics = []

    for seed in seeds:
        logger.info("=== Multi-seed: seed=%d ===", seed)
        t0 = time.perf_counter()
        loaders = build_loaders_with_params(cfg, window_size=50, knn_k=5, seed=seed)
        if "train" not in loaders or "test" not in loaders:
            logger.warning("Skipping seed=%d", seed)
            continue
        metrics = train_and_evaluate_gnn(cfg, loaders, device, seed=seed)
        elapsed = time.perf_counter() - t0
        metrics["seed"] = seed
        all_metrics.append(metrics)
        logger.info("  seed=%d F1=%.4f AUC=%.4f (%.0fs)", seed, metrics["f1"], metrics["roc_auc"], elapsed)

    if all_metrics:
        f1s = [m["f1"] for m in all_metrics]
        aucs = [m["roc_auc"] for m in all_metrics]
        summary = {
            "seeds": seeds,
            "per_seed": all_metrics,
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
        }
        out = MET_DIR / "multi_seed_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Multi-seed summary -> %s", out)
        logger.info("  F1: %.4f +/- %.4f", summary["f1_mean"], summary["f1_std"])
        logger.info("  AUC: %.4f +/- %.4f", summary["roc_auc_mean"], summary["roc_auc_std"])
        return summary
    return None


def plot_sensitivity(results):
    """Plot sensitivity heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not results:
        return

    ws_vals = sorted(set(r["window_size"] for r in results))
    k_vals = sorted(set(r["knn_k"] for r in results))
    f1_grid = np.zeros((len(ws_vals), len(k_vals)))
    for r in results:
        i = ws_vals.index(r["window_size"])
        j = k_vals.index(r["knn_k"])
        f1_grid[i, j] = r["f1"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(f1_grid, cmap='YlGn', aspect='auto', vmin=0.99, vmax=1.0)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_yticks(range(len(ws_vals)))
    ax.set_yticklabels([str(w) for w in ws_vals])
    ax.set_xlabel('kNN k')
    ax.set_ylabel('Window Size')
    ax.set_title('Sensitivity: F1-Score vs Window Size and k', fontweight='bold', color='#1f3a5f')

    for i in range(len(ws_vals)):
        for j in range(len(k_vals)):
            ax.text(j, i, f'{f1_grid[i,j]:.4f}', ha='center', va='center', fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR2.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / 'fig_sensitivity.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR2 / 'sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("Sensitivity plot saved.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-seeds", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if not args.skip_sensitivity:
        logger.info("========== SENSITIVITY ANALYSIS ==========")
        sens_results = run_sensitivity(cfg)
        plot_sensitivity(sens_results)

    if not args.skip_seeds:
        logger.info("========== MULTI-SEED EXPERIMENTS ==========")
        run_multi_seed(cfg)

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
