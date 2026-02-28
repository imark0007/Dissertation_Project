"""
Federated Learning simulation: runs FedAvg with 3 non-IID clients in a single process.
Uses Flower's simulation API — no need for separate server/client terminals.

Usage: python scripts/run_fl_simulation.py --config config/experiment.yaml
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.data.dataset import GraphSequenceDataset, _collate_sequences
from src.models.dynamic_gnn import DynamicGNN
from src.models.trainer import evaluate, measure_inference_time
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def fedavg_simulation(
    cfg: dict,
    num_rounds: int = 10,
    local_epochs: int = 2,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fl_cfg = cfg.get("fl", {})
    num_clients = fl_cfg.get("num_clients", 3)
    gnn_cfg = cfg["models"]["dynamic_gnn"]
    seq_len = cfg.get("graph", {}).get("sequence_length", 5)
    bs = gnn_cfg.get("batch_size", 16)

    # Load client data
    client_loaders = []
    for cid in range(num_clients):
        gpath = Path("data/graphs") / f"client_{cid}_graphs.pt"
        if not gpath.exists():
            logger.error("Missing %s — run data_split first", gpath)
            return {}
        graphs = torch.load(gpath, weights_only=False)
        ds = GraphSequenceDataset(graphs, seq_len)
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=_collate_sequences)
        client_loaders.append(loader)
        logger.info("Client %d: %d sequences", cid, len(ds))

    # Load test data for evaluation
    test_graphs = torch.load("data/graphs/test_graphs.pt", weights_only=False)
    test_ds = GraphSequenceDataset(test_graphs, seq_len)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=_collate_sequences)

    # Global model
    global_model = DynamicGNN.from_config(cfg).to(device)
    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    round_metrics = []
    comm_bytes = []

    for rnd in range(1, num_rounds + 1):
        client_states = []
        client_sizes = []

        for cid, loader in enumerate(client_loaders):
            # Clone global model for this client
            local_model = DynamicGNN.from_config(cfg).to(device)
            local_model.load_state_dict({k: v.clone() for k, v in global_state.items()})
            optimizer = torch.optim.Adam(local_model.parameters(), lr=gnn_cfg.get("lr", 0.001))

            local_model.train()
            for _ in range(local_epochs):
                for sequences, labels in loader:
                    labels = labels.to(device)
                    logits = local_model.forward_batch(sequences, device)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            client_states.append(local_model.state_dict())
            client_sizes.append(len(loader.dataset))

        # FedAvg aggregation
        total_size = sum(client_sizes)
        new_state = {}
        for key in global_state:
            new_state[key] = sum(
                client_states[i][key].float() * (client_sizes[i] / total_size)
                for i in range(num_clients)
            )
        global_state = new_state

        # Evaluate global model on test
        global_model.load_state_dict(global_state)
        metrics = evaluate(global_model, test_loader, device)
        metrics["round"] = rnd
        round_metrics.append(metrics)

        # Communication cost (bytes for one round: all client params sent + global sent back)
        param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
        round_comm = param_bytes * num_clients * 2
        comm_bytes.append(round_comm)

        logger.info(
            "Round %02d  F1=%.4f  AUC=%.4f  comm=%.1f MB",
            rnd, metrics["f1"], metrics["roc_auc"], round_comm / 1e6,
        )

    # Final metrics
    global_model.load_state_dict(global_state)
    final_metrics = evaluate(global_model, test_loader, device)
    final_metrics["inference_ms"] = measure_inference_time(global_model, test_loader, device)

    # Save
    met_dir = Path(cfg["paths"]["metrics"])
    met_dir.mkdir(parents=True, exist_ok=True)

    with open(met_dir / "federated_gnn_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    with open(met_dir / "fl_rounds.json", "w") as f:
        json.dump({"rounds": round_metrics, "comm_bytes": comm_bytes}, f, indent=2)

    # Save federated model
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_model.state_dict(), ckpt_dir / "dynamic_gnn_federated.pt")

    logger.info("FL simulation complete. Final: %s", final_metrics)
    return final_metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config("config/experiment.yaml")
    fl_cfg = cfg.get("fl", {})
    fedavg_simulation(
        cfg,
        num_rounds=fl_cfg.get("num_rounds", 10),
        local_epochs=fl_cfg.get("local_epochs", 2),
    )

    # Update results table
    met_dir = Path(cfg["paths"]["metrics"])
    import csv
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
                "Inference (ms)": f"{m.get('inference_ms',0):.2f}",
            })
    if rows:
        with open(met_dir / "results_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            logger.info("  %s", r)


if __name__ == "__main__":
    main()
