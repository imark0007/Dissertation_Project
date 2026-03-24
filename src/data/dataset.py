"""
PyTorch Dataset and DataLoader utilities for graph sequences.

``GraphSequenceDataset`` yields (sequence_of_graphs, label) tuples suitable
for the DynamicGNN model.  ``get_dataloaders`` is a convenience function
that loads saved ``.pt`` graph files and returns ready-to-use DataLoaders.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphSequenceDataset(Dataset):
    """
    Dataset of fixed-length graph sequences.

    Each item is a tuple ``(list_of_Data, label)`` where the label is taken
    from the last graph in the sequence.
    """

    def __init__(self, graphs: List[Data], sequence_length: int = 5):
        self.graphs = graphs
        self.seq_len = sequence_length
        self.num_sequences = max(0, len(graphs) - sequence_length + 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[List[Data], int]:
        seq = self.graphs[idx : idx + self.seq_len]
        label = seq[-1].y.item()
        return seq, label


def _collate_sequences(
    batch: List[Tuple[List[Data], int]],
) -> Tuple[List[List[Data]], torch.Tensor]:
    """Custom collate: return list-of-sequences and stacked labels."""
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sequences, labels


def get_dataloaders(
    graphs_dir: str = "data/graphs",
    sequence_length: int = 5,
    batch_size: int = 16,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Load saved graph .pt files and build DataLoaders for train/test/validation.
    """
    gdir = Path(graphs_dir)
    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "test", "validation"):
        pt = gdir / f"{split}_graphs.pt"
        if not pt.exists():
            logger.warning("Graph file not found: %s", pt)
            continue
        graphs = torch.load(pt, weights_only=False)
        ds = GraphSequenceDataset(graphs, sequence_length)
        shuffle = shuffle_train if split == "train" else False
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_sequences,
        )
        logger.info("Loaded %s: %d sequences (batch_size=%d)", split, len(ds), batch_size)

    return loaders
