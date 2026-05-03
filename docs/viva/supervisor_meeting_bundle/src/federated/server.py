"""
Flower server with FedAvg, per-round metric tracking, and communication cost logging.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

logger = logging.getLogger(__name__)


class TrackedFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that tracks per-round metrics and communication bytes."""

    def __init__(self, metrics_dir: str = "results/metrics", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics: List[Dict] = []
        self.comm_bytes: List[int] = []
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        if results:
            total_n = sum(r.num_examples for _, r in results)
            avg = {}
            for _, r in results:
                for k, v in (r.metrics or {}).items():
                    avg[k] = avg.get(k, 0.0) + v * r.num_examples
            avg = {k: v / total_n for k, v in avg.items()}
            avg["round"] = server_round
            self.round_metrics.append(avg)
            logger.info("Round %d metrics: %s", server_round, avg)
        return agg

    def aggregate_fit(self, server_round, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg and agg[0]:
            size = sum(a.nbytes for a in agg[0])
            self.comm_bytes.append(size)
        return agg

    def save_metrics(self):
        with open(self.metrics_dir / "fl_rounds.json", "w") as f:
            json.dump({"rounds": self.round_metrics, "comm_bytes": self.comm_bytes}, f, indent=2)
        logger.info("Saved FL round metrics")


def run_fl_server(
    num_rounds: int = 10,
    min_clients: int = 2,
    metrics_dir: str = "results/metrics",
) -> TrackedFedAvg:
    strategy = TrackedFedAvg(
        metrics_dir=metrics_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    strategy.save_metrics()
    return strategy
