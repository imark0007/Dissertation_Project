"""
Evaluate saved GAT-only ablation checkpoint and write ablation_gat_only.json + ablation_table.csv.
Use when run_ablation.py already trained and saved results/checkpoints/ablation_gat_only.pt
but did not write the metrics (e.g. process was stopped).

Usage: python scripts/eval_ablation_from_ckpt.py --config config/experiment.yaml
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
from src.models.trainer import evaluate, measure_inference_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "results" / "checkpoints" / "ablation_gat_only.pt"
    if not ckpt_path.exists():
        logger.error("No ablation checkpoint at %s. Run run_ablation.py first.", ckpt_path)
        sys.exit(1)

    loaders = get_dataloaders(graphs_dir=str(ROOT / "data" / "graphs"))
    if "test" not in loaders:
        logger.error("No test loader.")
        sys.exit(1)

    cfg_copy = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    if "models" not in cfg_copy:
        cfg_copy["models"] = {}
    if "dynamic_gnn" not in cfg_copy["models"]:
        cfg_copy["models"]["dynamic_gnn"] = {}
    cfg_copy["models"]["dynamic_gnn"]["use_gru"] = False

    model = DynamicGNN.from_config(cfg_copy).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_metrics = evaluate(model, loaders["test"], device)
    test_metrics["inference_ms"] = measure_inference_time(model, loaders["test"], device)
    test_metrics["variant"] = "GAT only (no GRU)"

    met_dir = Path(cfg["paths"]["metrics"])
    met_dir.mkdir(parents=True, exist_ok=True)
    out_path = met_dir / "ablation_gat_only.json"
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Ablation metrics saved to %s", out_path)

    # Build ablation_table.csv (full + GAT-only)
    import csv
    met_dir = Path(cfg["paths"]["metrics"])
    rows = []
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
    rows.append({
        "Variant": "GAT only (no GRU)",
        "Precision": f"{test_metrics.get('precision', 0):.4f}",
        "Recall": f"{test_metrics.get('recall', 0):.4f}",
        "F1": f"{test_metrics.get('f1', 0):.4f}",
        "ROC-AUC": f"{test_metrics.get('roc_auc', 0):.4f}",
        "Inference (ms)": f"{test_metrics.get('inference_ms', 0):.2f}",
    })
    if rows:
        out_csv = met_dir / "ablation_table.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info("Ablation table -> %s", out_csv)
    logger.info("Done. Run: python scripts/update_dissertation_table4.py")


if __name__ == "__main__":
    main()
