"""
Build kNN similarity graphs from preprocessed flow data.

Each graph window contains ``window_size`` flows as nodes. Edges are
constructed via k-nearest-neighbours in feature space (cosine or euclidean).
The graph carries a binary label derived from the majority label in the window.

CLI: python -m src.data.graph_builder --config config/experiment.yaml \
        --processed-dir data/processed --output-dir data/graphs
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def flows_to_knn_graph(
    features: np.ndarray,
    majority_label: int,
    k: int = 5,
) -> Data:
    """Convert an (N, F) feature matrix into a PyG ``Data`` object with kNN edges."""
    n = features.shape[0]
    k_eff = min(k, n - 1) if n > 1 else 0

    if k_eff > 0:
        tree = cKDTree(features)
        _, indices = tree.query(features, k=k_eff + 1)
        src, dst = [], []
        for i, nbrs in enumerate(indices):
            for j in nbrs:
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([majority_label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def _build_graphs_for_split(
    df: pd.DataFrame,
    feat_cols: List[str],
    window_size: int,
    k: int,
    minority_stride: Optional[int] = None,
) -> List[Data]:
    """Slide a window over the dataframe and build one graph per window."""
    features = df[feat_cols].values.astype(np.float32)
    labels = df["binary_label"].values

    graphs: List[Data] = []
    stride = window_size

    i = 0
    while i + window_size <= len(features):
        window_feats = features[i : i + window_size]
        window_labels = labels[i : i + window_size]
        majority = int(np.bincount(window_labels).argmax())
        g = flows_to_knn_graph(window_feats, majority, k=k)
        graphs.append(g)

        is_minority = (majority == 1)
        if is_minority and minority_stride and minority_stride < window_size:
            i += minority_stride
        else:
            i += stride

    logger.info("  Built %d graphs (window=%d, k=%d)", len(graphs), window_size, k)
    return graphs


def build_and_save_graphs(
    processed_dir: str,
    output_dir: str,
    config_path: str = "config/experiment.yaml",
    nrows: Optional[int] = None,
) -> None:
    """Build graphs for all splits and save as .pt files."""
    from src.data.preprocess import load_config

    cfg = load_config(config_path)
    feat_cols = cfg["data"]["flow_feature_columns"]
    graph_cfg = cfg.get("graph", {})
    window_size = graph_cfg.get("window_size", 50)
    k = graph_cfg.get("knn_k", 5)
    minority_stride = graph_cfg.get("minority_stride", None)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test", "validation"):
        pq = Path(processed_dir) / f"{split}.parquet"
        if not pq.exists():
            logger.warning("Missing %s — skipping graph build for %s", pq, split)
            continue
        logger.info("Building graphs for %s ...", split)
        df = pd.read_parquet(pq)
        if nrows:
            df = df.head(nrows)
        graphs = _build_graphs_for_split(df, feat_cols, window_size, k, minority_stride)
        torch.save(graphs, out / f"{split}_graphs.pt")
        logger.info("Saved %s -> %s (%d graphs)", split, out / f"{split}_graphs.pt", len(graphs))


def main():
    ap = argparse.ArgumentParser(description="Build kNN graphs from processed data")
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--output-dir", default="data/graphs")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_and_save_graphs(args.processed_dir, args.output_dir, args.config, args.nrows)


if __name__ == "__main__":
    main()
