"""
Master script: run the full pipeline end-to-end.

  python scripts/run_all.py --config config/experiment.yaml [--nrows 50000]

Steps:
  1. Preprocess raw data -> data/processed/
  2. Build kNN graphs  -> data/graphs/
  3. Train baselines (RF + MLP) -> results/
  4. Train Dynamic GNN (central) -> results/
  5. Evaluate all models -> results/metrics/
  6. Generate comparison plots -> results/figures/
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config, run_preprocessing
from src.data.graph_builder import build_and_save_graphs
from src.data.dataset import get_dataloaders
from src.models.baselines import RandomForestBaseline, MLPBaseline, load_processed_split
from src.models.dynamic_gnn import DynamicGNN
from src.models.trainer import train_gnn, evaluate, measure_inference_time
from src.evaluation.metrics import compute_metrics, plot_roc, plot_confusion_matrix

logger = logging.getLogger(__name__)


def train_baselines(cfg, nrows=None):
    feat_cols = cfg["data"]["flow_feature_columns"]
    proc = cfg["data"]["processed_dir"]
    met_dir = Path(cfg["paths"]["metrics"])
    fig_dir = Path(cfg["paths"]["figures"])
    met_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed data for baselines...")
    X_train, y_train = load_processed_split(proc, "train", feat_cols)
    X_test, y_test = load_processed_split(proc, "test", feat_cols)
    if nrows:
        X_train, y_train = X_train[:nrows], y_train[:nrows]
        X_test, y_test = X_test[:nrows // 3], y_test[:nrows // 3]

    # Random Forest
    rf_cfg = cfg["models"]["rf"]
    logger.info("Training Random Forest...")
    rf = RandomForestBaseline(rf_cfg["n_estimators"], rf_cfg["max_depth"])
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_pred, rf_proba)

    t0 = time.perf_counter()
    for _ in range(100):
        rf.predict(X_test[:1])
    rf_metrics["inference_ms"] = (time.perf_counter() - t0) / 100 * 1000
    rf_metrics["train_time_s"] = rf_train_time
    logger.info("RF metrics: %s", rf_metrics)
    with open(met_dir / "rf_metrics.json", "w") as f:
        json.dump(rf_metrics, f, indent=2)
    plot_roc(y_test, rf_proba, fig_dir / "roc_rf.png", "Random Forest ROC")
    plot_confusion_matrix(y_test, rf_pred, fig_dir / "cm_rf.png", "Random Forest")

    # MLP
    mlp_cfg = cfg["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLPBaseline(X_train.shape[1], tuple(mlp_cfg["hidden_dims"]), mlp_cfg["dropout"]).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=mlp_cfg.get("lr", 0.001))
    Xt = torch.from_numpy(X_train).float().to(device)
    yt = torch.from_numpy(y_train).long().to(device)

    # Class weights for MLP
    counts = np.bincount(y_train, minlength=2).astype(float)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)

    logger.info("Training MLP (%d epochs)...", mlp_cfg.get("epochs", 30))
    t0 = time.perf_counter()
    bs = mlp_cfg.get("batch_size", 256)
    for epoch in range(mlp_cfg.get("epochs", 30)):
        mlp.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(mlp(Xt[idx]), yt[idx], weight=w)
            loss.backward()
            opt.step()
    mlp_train_time = time.perf_counter() - t0

    mlp.eval()
    Xte = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        logits = mlp(Xte)
        mlp_pred = logits.argmax(1).cpu().numpy()
        mlp_proba = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_proba)

    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            mlp(Xte[:1])
    mlp_metrics["inference_ms"] = (time.perf_counter() - t0) / 100 * 1000
    mlp_metrics["train_time_s"] = mlp_train_time
    logger.info("MLP metrics: %s", mlp_metrics)
    with open(met_dir / "mlp_metrics.json", "w") as f:
        json.dump(mlp_metrics, f, indent=2)
    plot_roc(y_test, mlp_proba, fig_dir / "roc_mlp.png", "MLP ROC")
    plot_confusion_matrix(y_test, mlp_pred, fig_dir / "cm_mlp.png", "MLP")


def train_central_gnn(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(graphs_dir="data/graphs")
    if "train" not in loaders:
        logger.error("No train graphs found. Run graph builder first.")
        return

    model = DynamicGNN.from_config(cfg).to(device)
    gnn_cfg = cfg["models"]["dynamic_gnn"]
    train_cfg = cfg.get("training", {})

    train_gnn(
        model, loaders["train"], loaders.get("validation", loaders["train"]),
        device,
        epochs=gnn_cfg.get("epochs", 30),
        lr=gnn_cfg.get("lr", 0.001),
        patience=train_cfg.get("early_stopping_patience", 5),
        auto_class_weight=train_cfg.get("class_weight_auto", True),
    )

    # Evaluate on test + generate plots
    if "test" in loaders:
        model.load_state_dict(torch.load("results/checkpoints/dynamic_gnn_best.pt", map_location=device))
        test_metrics = evaluate(model, loaders["test"], device)
        test_metrics["inference_ms"] = measure_inference_time(model, loaders["test"], device)
        logger.info("GNN central test metrics: %s", test_metrics)

        # Collect predictions for ROC + CM plots
        all_preds, all_labels, all_proba = [], [], []
        model.eval()
        with torch.no_grad():
            for sequences, labels in loaders["test"]:
                logits = model.forward_batch(sequences, device)
                proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_proba.extend(proba)
        y_true_gnn = np.array(all_labels)
        y_pred_gnn = np.array(all_preds)
        y_proba_gnn = np.array(all_proba)

        fig_dir = Path(cfg["paths"]["figures"])
        fig_dir.mkdir(parents=True, exist_ok=True)
        if len(np.unique(y_true_gnn)) > 1:
            plot_roc(y_true_gnn, y_proba_gnn, fig_dir / "roc_gnn.png", "Dynamic GNN ROC")
        plot_confusion_matrix(y_true_gnn, y_pred_gnn, fig_dir / "cm_gnn.png", "Dynamic GNN")

        met_dir = Path(cfg["paths"]["metrics"])
        with open(met_dir / "central_gnn_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)


def build_results_table(cfg):
    met_dir = Path(cfg["paths"]["metrics"])
    rows = []
    for name in ["rf", "mlp", "central_gnn", "federated_gnn"]:
        p = met_dir / f"{name}_metrics.json"
        if p.exists():
            with open(p) as fh:
                m = json.load(fh)
            rows.append({
                "Model": name.replace("_", " ").title(),
                "Precision": f"{m.get('precision',0):.4f}",
                "Recall": f"{m.get('recall',0):.4f}",
                "F1": f"{m.get('f1',0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc',0):.4f}",
                "FAR": f"{m.get('false_alarm_rate',0):.4f}",
                "FP": str(m.get('false_positives', 0)),
                "Inference (ms)": f"{m.get('inference_ms',0):.2f}",
            })
    if rows:
        import csv
        out = met_dir / "results_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logger.info("Results table -> %s", out)
        for r in rows:
            logger.info("  %s", r)


def main():
    ap = argparse.ArgumentParser(description="Run full pipeline")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--nrows", type=int, default=None, help="Limit rows (for fast testing)")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-graphs", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-gnn", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if not args.skip_preprocess:
        logger.info("=== Step 1: Preprocessing ===")
        run_preprocessing(args.config, args.nrows)

    if not args.skip_graphs:
        logger.info("=== Step 2: Building graphs ===")
        build_and_save_graphs(nrows=args.nrows)

    if not args.skip_baselines:
        logger.info("=== Step 3: Training baselines ===")
        train_baselines(cfg, args.nrows)

    if not args.skip_gnn:
        logger.info("=== Step 4: Training Dynamic GNN ===")
        train_central_gnn(cfg)

    logger.info("=== Step 5: Results table ===")
    build_results_table(cfg)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
