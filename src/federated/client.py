"""
Flower NumPy client that wraps DynamicGNN for federated training.
"""
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dynamic_gnn import DynamicGNN
from src.models.trainer import train_one_epoch, evaluate, compute_class_weights


def _get_params(model: torch.nn.Module) -> List[np.ndarray]:
    return [p.cpu().detach().numpy() for p in model.parameters()]


def _set_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    for p, arr in zip(model.parameters(), params):
        p.data = torch.from_numpy(arr).to(p.device)


class GNNFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: DynamicGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 2,
        lr: float = 0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.class_weights = compute_class_weights(train_loader)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return _get_params(self.model)

    def set_parameters(self, params: List[np.ndarray]) -> None:
        _set_params(self.model, params)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.train_loader, optimizer, self.device, self.class_weights)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        metrics = evaluate(self.model, self.val_loader, self.device)
        return 1.0 - metrics["f1"], len(self.val_loader.dataset), metrics
