"""
Run ablation experiments for the thesis (Premium Thesis Roadmap, Priority 1).

Variants:
  1. Full model (GAT + GRU) — already in main pipeline; results in central_gnn_metrics.json
  2. GAT only (no GRU) — mean pool over time instead of GRU

Usage:
  python scripts/run_ablation.py --config config/experiment.yaml

Outputs:
  results/metrics/ablation_gat_only.json   (when --variant gat_only)
  results/metrics/ablation_table.csv       (summary table for thesis §7.6)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.data.dataset import get_dataloaders
from src.models.dynamic_gnn import DynamicGNN
from src.models.trainer import train_gnn, evaluate, measure_inference_time

logger = logging.getLogger(__name__)


def run_gat_only_ablation(cfg):
    """Train and evaluate GAT-only model (use_gru=False)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt = ckpt_dir / "dynamic_gnn_best.pt"
    backup_pt = ckpt_dir / "dynamic_gnn_full_backup.pt"
    ablation_pt = ckpt_dir / "ablation_gat_only.pt"
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, backup_pt)
        logger.info("Backed up full model to %s", backup_pt)

    loaders = get_dataloaders(graphs_dir="data/graphs")
    if "train" not in loaders:
        logger.error("No train graphs found. Run: python scripts/run_all.py --config %s", cfg.get("config_path", "config/experiment.yaml"))
        return None

    # Override use_gru for ablation
    cfg_copy = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    if "models" not in cfg_copy:
        cfg_copy["models"] = {}
    if "dynamic_gnn" not in cfg_copy["models"]:
        cfg_copy["models"]["dynamic_gnn"] = {}
    cfg_copy["models"]["dynamic_gnn"]["use_gru"] = False

    model = DynamicGNN.from_config(cfg_copy).to(device)
    gnn_cfg = cfg_copy["models"]["dynamic_gnn"]
    train_cfg = cfg_copy.get("training", {})

    logger.info("Training GAT-only ablation (use_gru=False)...")
    train_gnn(
        model, loaders["train"], loaders.get("validation", loaders["train"]),
        device,
        epochs=gnn_cfg.get("epochs", 30),
        lr=gnn_cfg.get("lr", 0.001),
        patience=train_cfg.get("early_stopping_patience", 5),
        auto_class_weight=train_cfg.get("class_weight_auto", True),
    )

    # Save ablation checkpoint to separate file and restore full model if we backed it up
    if best_pt.exists():
        import shutil
        shutil.copy(best_pt, ablation_pt)
        logger.info("Saved GAT-only checkpoint to %s", ablation_pt)
        if backup_pt.exists():
            shutil.copy(backup_pt, best_pt)
            logger.info("Restored full model to %s", best_pt)

    # Load ablation checkpoint for evaluation (so we report GAT-only metrics)
    load_ckpt = ablation_pt if ablation_pt.exists() else best_pt
    if load_ckpt.exists():
        model.load_state_dict(torch.load(load_ckpt, map_location=device))

    if "test" not in loaders:
        logger.error("No test loader.")
        return None

    test_metrics = evaluate(model, loaders["test"], device)
    test_metrics["inference_ms"] = measure_inference_time(model, loaders["test"], device)
    test_metrics["variant"] = "GAT only (no GRU)"

    met_dir = Path(cfg["paths"]["metrics"])
    met_dir.mkdir(parents=True, exist_ok=True)
    out_path = met_dir / "ablation_gat_only.json"
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Ablation metrics saved to %s", out_path)
    return test_metrics


def build_ablation_table(cfg):
    """Build ablation_table.csv from central_gnn_metrics + ablation_gat_only (if present)."""
    met_dir = Path(cfg["paths"]["metrics"])
    rows = []

    # Full model (from main pipeline)
    full_path = met_dir / "central_gnn_metrics.json"
    if full_path.exists():
        with open(full_path) as f:
            m = json.load(f)
        rows.append({
            "Variant": "Full (GAT + GRU)",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "Inference (ms)": f"{m.get('inference_ms', 0):.2f}",
        })

    # GAT only
    gat_only_path = met_dir / "ablation_gat_only.json"
    if gat_only_path.exists():
        with open(gat_only_path) as f:
            m = json.load(f)
        rows.append({
            "Variant": "GAT only (no GRU)",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "Inference (ms)": f"{m.get('inference_ms', 0):.2f}",
        })

    if rows:
        import csv
        out = met_dir / "ablation_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info("Ablation table -> %s", out)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Run ablation for thesis §7.6")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--variant", choices=["gat_only", "all"], default="gat_only",
                    help="gat_only: run GAT-only; all: run and build table")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.variant in ("gat_only", "all"):
        run_gat_only_ablation(cfg)

    build_ablation_table(cfg)
    logger.info("Done. Add results to thesis §7.6 Ablation Studies.")


if __name__ == "__main__":
    main()
