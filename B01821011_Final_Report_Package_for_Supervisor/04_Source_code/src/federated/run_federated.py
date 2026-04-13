"""
Run federated learning: server or client mode.

  Server:  python -m src.federated.run_federated server
  Client:  python -m src.federated.run_federated client --cid 0
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
import flwr as fl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import load_config
from src.data.dataset import GraphSequenceDataset, _collate_sequences
from src.models.dynamic_gnn import DynamicGNN
from src.federated.client import GNNFlowerClient
from src.federated.server import run_fl_server

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["server", "client"])
    ap.add_argument("--cid", type=int, default=0)
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = load_config(args.config)
    fl_cfg = cfg.get("fl", {})

    if args.mode == "server":
        run_fl_server(
            num_rounds=fl_cfg.get("num_rounds", 10),
            min_clients=min(2, fl_cfg.get("num_clients", 3)),
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicGNN.from_config(cfg).to(device)

    client_graphs_path = Path("data/graphs") / f"client_{args.cid}_graphs.pt"
    if not client_graphs_path.exists():
        logger.error("Client graphs not found: %s. Run data_split first.", client_graphs_path)
        return

    graphs = torch.load(client_graphs_path, weights_only=False)
    seq_len = cfg.get("graph", {}).get("sequence_length", 5)
    n = len(graphs)
    split = int(0.8 * n)
    train_ds = GraphSequenceDataset(graphs[:split], seq_len)
    val_ds = GraphSequenceDataset(graphs[split:], seq_len)

    from torch.utils.data import DataLoader
    bs = cfg.get("models", {}).get("dynamic_gnn", {}).get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=_collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=_collate_sequences)

    client = GNNFlowerClient(
        model, train_loader, val_loader, device,
        local_epochs=fl_cfg.get("local_epochs", 2),
        lr=cfg.get("models", {}).get("dynamic_gnn", {}).get("lr", 0.001),
    )
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
