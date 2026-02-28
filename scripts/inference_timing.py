"""
Measure CPU inference time for all models: RF, MLP, Dynamic GNN.
Usage: python scripts/inference_timing.py --config config/experiment.yaml
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.data.graph_builder import flows_to_knn_graph
from src.models.baselines import RandomForestBaseline, MLPBaseline
from src.models.dynamic_gnn import DynamicGNN


def main():
    cfg = load_config("config/experiment.yaml")
    n_runs = 100
    device = torch.device("cpu")

    # RF
    rf = RandomForestBaseline(100, 10)
    X = np.random.randn(100, 46).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    rf.fit(X, y)
    sample = np.random.randn(1, 46).astype(np.float32)
    for _ in range(5):
        rf.predict(sample)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        rf.predict(sample)
    rf_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"RF:          {rf_ms:.3f} ms per sample")

    # MLP
    mlp = MLPBaseline(46, (128, 64, 32), 0.0)
    mlp.eval()
    xt = torch.randn(1, 46)
    for _ in range(5):
        with torch.no_grad():
            mlp(xt)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            mlp(xt)
    mlp_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"MLP:         {mlp_ms:.3f} ms per sample")

    # Dynamic GNN
    gnn = DynamicGNN.from_config(cfg).to(device)
    gnn.eval()
    seq_len = cfg.get("graph", {}).get("sequence_length", 5)
    graphs = []
    for _ in range(seq_len):
        feats = np.random.randn(200, 46).astype(np.float32)
        graphs.append(flows_to_knn_graph(feats, 0, k=8))
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            gnn.forward(graphs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            gnn.forward(graphs)
    gnn_ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"Dynamic GNN: {gnn_ms:.3f} ms per sequence ({seq_len} windows)")

    out = Path(cfg.get("paths", {}).get("metrics", "results/metrics"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inference_timing.txt", "w") as f:
        f.write(f"RF\t{rf_ms:.3f}\nMLP\t{mlp_ms:.3f}\nDynamicGNN\t{gnn_ms:.3f}\n")
    print(f"Wrote {out / 'inference_timing.txt'}")


if __name__ == "__main__":
    main()
