"""
Non-IID data splitting for federated learning.

Uses Dirichlet distribution to create label-skewed partitions across clients,
simulating realistic heterogeneous IoT deployments.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def dirichlet_split(
    graphs: List[Data],
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[Data]]:
    """
    Split graphs into non-IID client partitions using Dirichlet distribution.

    Lower alpha = more skewed (each client sees mostly one class).
    alpha=100 ≈ IID, alpha=0.1 = very non-IID.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([g.y.item() for g in graphs])
    classes = np.unique(labels)
    client_indices: Dict[int, List[int]] = {c: [] for c in range(num_clients)}

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        # Fix rounding so sum == len(cls_idx)
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()
        rng.shuffle(cls_idx)
        start = 0
        for cid in range(num_clients):
            end = start + proportions[cid]
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end

    client_graphs: Dict[int, List[Data]] = {}
    for cid in range(num_clients):
        indices = client_indices[cid]
        rng.shuffle(indices)
        client_graphs[cid] = [graphs[i] for i in indices]
        c0 = sum(1 for i in indices if labels[i] == 0)
        c1 = sum(1 for i in indices if labels[i] == 1)
        logger.info("Client %d: %d graphs (benign=%d, attack=%d)", cid, len(indices), c0, c1)

    return client_graphs


def split_and_save(
    graphs_path: str,
    output_dir: str,
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
) -> None:
    graphs = torch.load(graphs_path, weights_only=False)
    client_graphs = dirichlet_split(graphs, num_clients, alpha, seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cid, cg in client_graphs.items():
        torch.save(cg, out / f"client_{cid}_graphs.pt")
    logger.info("Saved %d client splits to %s", num_clients, out)
