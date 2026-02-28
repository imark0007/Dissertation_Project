"""
Baseline models: Random Forest (sklearn) and MLP (PyTorch).
Wired to load from data/processed/ parquet files.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def load_processed_split(
    processed_dir: str, split: str, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(processed_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values.astype(np.int64)
    return X, y


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 200, max_depth: int = 20, seed: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        proba = self.predict_proba(X)[:, 1]
        return _compute_metrics(y, pred, proba)


class MLPBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }
