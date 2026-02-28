"""
Training and evaluation loops for the Dynamic GNN.

Features:
  - Class-weighted cross-entropy loss (auto-computed from label imbalance).
  - Early stopping on validation F1.
  - Metric logging (precision, recall, F1, ROC-AUC).
  - CPU inference timing.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models.dynamic_gnn import DynamicGNN

logger = logging.getLogger(__name__)


def compute_class_weights(loader: DataLoader) -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader."""
    counts = np.zeros(2, dtype=np.float64)
    for _, labels in loader:
        for l in labels.numpy():
            counts[l] += 1
    if counts.min() == 0:
        return torch.ones(2)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: DynamicGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    weight = class_weights.to(device) if class_weights is not None else None

    for sequences, labels in loader:
        labels = labels.to(device)
        logits = model.forward_batch(sequences, device)
        loss = F.cross_entropy(logits, labels, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: DynamicGNN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    for sequences, labels in loader:
        labels_np = labels.numpy()
        logits = model.forward_batch(sequences, device)
        proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        all_proba.extend(proba)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)

    from src.evaluation.metrics import compute_metrics
    return compute_metrics(y_true, y_pred, y_proba)


def measure_inference_time(model: DynamicGNN, loader: DataLoader, device: torch.device, n: int = 50) -> float:
    """Average inference time per sequence (ms) on given device, with warmup."""
    model.eval()
    # Warmup pass (excluded from timing)
    for sequences, _ in loader:
        seq_dev = [g.to(device) for g in sequences[0]]
        with torch.no_grad():
            model.forward(seq_dev)
        break

    times = []
    count = 0
    for sequences, _ in loader:
        for seq in sequences:
            seq_dev = [g.to(device) for g in seq]
            t0 = time.perf_counter()
            with torch.no_grad():
                model.forward(seq_dev)
            times.append((time.perf_counter() - t0) * 1000)
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    return float(np.mean(times)) if times else 0.0


def train_gnn(
    model: DynamicGNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.001,
    patience: int = 5,
    checkpoint_dir: str = "results/checkpoints",
    metrics_dir: str = "results/metrics",
    auto_class_weight: bool = True,
) -> Dict[str, List]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = compute_class_weights(train_loader) if auto_class_weight else None
    if class_weights is not None:
        logger.info("Class weights: %s", class_weights.tolist())

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    met_dir = Path(metrics_dir)
    met_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_f1": [], "val_roc_auc": []}
    best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        logger.info(
            "Epoch %02d  loss=%.4f  val_f1=%.4f  val_auc=%.4f",
            epoch, loss, val_metrics["f1"], val_metrics["roc_auc"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_dir / "dynamic_gnn_best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save history
    with open(met_dir / "gnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history
