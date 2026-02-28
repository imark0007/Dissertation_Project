"""
Dynamic GNN: 2-layer GATConv per window, global mean pool, GRU over sequence, classifier.
Exposes attention weights for explainability.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool


class DynamicGNN(nn.Module):
    """
    Per-window kNN graph -> GATConv (with attention) -> graph embedding -> GRU -> classifier.
    """

    def __init__(
        self,
        node_dim: int = 46,
        hidden_dim: int = 64,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        gru_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * gat_heads
            self.gat_layers.append(
                GATConv(in_ch, hidden_dim, heads=gat_heads, dropout=dropout, concat=True)
            )

        self.pool_fc = nn.Sequential(
            nn.Linear(hidden_dim * gat_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.classifier = nn.Linear(gru_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._last_attention_weights: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # ---- single-graph forward ----

    def _encode_graph(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        h = self.node_embed(x)
        attn_pairs = []
        for gat in self.gat_layers:
            h, (edge_idx, alpha) = gat(h, edge_index, return_attention_weights=True)
            h = h.relu()
            h = self.dropout(h)
            if return_attention:
                attn_pairs.append((edge_idx.detach(), alpha.detach()))
        graph_emb = h.mean(dim=0, keepdim=True)
        graph_emb = self.pool_fc(graph_emb)
        if return_attention:
            self._last_attention_weights = attn_pairs
        return graph_emb

    # ---- sequence forward ----

    def forward(
        self,
        graph_sequence: List[Data],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        graph_sequence: list of PyG Data objects (one per time step).
        Returns logits (1, num_classes).
        """
        embeddings = []
        for g in graph_sequence:
            emb = self._encode_graph(g.x, g.edge_index, return_attention=return_attention)
            embeddings.append(emb)
        seq = torch.cat(embeddings, dim=0).unsqueeze(0)  # (1, T, hidden)
        _, h_n = self.gru(seq)
        logits = self.classifier(h_n.squeeze(0))
        return logits

    def forward_batch(
        self,
        sequences: List[List[Data]],
        device: torch.device,
    ) -> torch.Tensor:
        """Forward a batch of sequences; returns (B, num_classes)."""
        logits_list = []
        for seq in sequences:
            seq_on_dev = [g.to(device) for g in seq]
            logits_list.append(self.forward(seq_on_dev))
        return torch.cat(logits_list, dim=0)

    def get_attention_weights(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return attention weights from the last forward call (if return_attention=True)."""
        return self._last_attention_weights

    @classmethod
    def from_config(cls, cfg: dict) -> "DynamicGNN":
        gnn = cfg.get("models", {}).get("dynamic_gnn", {})
        return cls(
            node_dim=gnn.get("node_dim", 46),
            hidden_dim=gnn.get("hidden_dim", 64),
            num_gat_layers=gnn.get("num_gat_layers", 2),
            gat_heads=gnn.get("gat_heads", 4),
            gru_hidden=gnn.get("gru_hidden", 64),
            num_classes=gnn.get("num_classes", 2),
            dropout=gnn.get("dropout", 0.2),
        )
