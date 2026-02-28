"""
Explainability module: Captum Integrated Gradients + GAT attention weights.

Produces an ExplanationBundle dataclass per sample with:
  - Integrated Gradients attributions per node feature.
  - GAT attention weights per edge (from last forward pass).
  - Top-k contributing nodes and features.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


@dataclass
class ExplanationBundle:
    """Container for one sample's explanations."""
    prediction: int
    score: float
    ig_node_attributions: Optional[np.ndarray] = None        # (N, F) per-node IG
    attention_weights: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None  # per GAT layer
    top_nodes: List[Dict] = field(default_factory=list)      # top-k nodes by attribution
    top_features: List[Dict] = field(default_factory=list)   # top-k features globally


def _ig_wrapper(model, sequence: List[Data], device: torch.device, target: int):
    """Compute IG on the first graph's node features (most recent window)."""
    if not HAS_CAPTUM:
        return None

    last_graph = sequence[-1].to(device)
    x_input = last_graph.x.clone().requires_grad_(True)

    def forward_fn(x):
        last_graph.x = x
        logits = model.forward([last_graph], return_attention=False)
        return logits

    ig = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(x_input)
    attr = ig.attribute(x_input, baselines=baseline, target=target, n_steps=30)
    return attr.detach().cpu().numpy()


def explain_sequence(
    model,
    sequence: List[Data],
    device: torch.device,
    top_k_nodes: int = 10,
    top_k_features: int = 10,
) -> ExplanationBundle:
    """
    Run a sequence through the model, extract explanations, and return an ExplanationBundle.
    """
    model.eval()
    seq_dev = [g.to(device) for g in sequence]

    with torch.no_grad():
        logits = model.forward(seq_dev, return_attention=True)
        proba = torch.softmax(logits, dim=1)[0]
        pred = proba.argmax().item()
        score = proba[1].item()

    # GAT attention weights from last forward pass
    attn_raw = model.get_attention_weights()
    attn_list = [(ei.cpu().numpy(), a.cpu().numpy()) for ei, a in attn_raw] if attn_raw else None

    # Integrated Gradients on last window
    ig_attr = _ig_wrapper(model, sequence, device, target=pred)

    # Top nodes (by sum of absolute IG attributions)
    top_nodes = []
    if ig_attr is not None:
        node_importance = np.abs(ig_attr).sum(axis=1)
        top_idx = np.argsort(-node_importance)[:top_k_nodes]
        for i in top_idx:
            top_nodes.append({"node_idx": int(i), "importance": float(node_importance[i])})

    # Top features (by global sum of absolute IG attributions across nodes)
    top_features = []
    if ig_attr is not None:
        feat_importance = np.abs(ig_attr).sum(axis=0)
        top_fidx = np.argsort(-feat_importance)[:top_k_features]
        for i in top_fidx:
            top_features.append({"feature_idx": int(i), "importance": float(feat_importance[i])})

    return ExplanationBundle(
        prediction=pred,
        score=score,
        ig_node_attributions=ig_attr,
        attention_weights=attn_list,
        top_nodes=top_nodes,
        top_features=top_features,
    )
