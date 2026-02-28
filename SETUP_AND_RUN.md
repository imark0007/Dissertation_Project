# Setup and Run Guide

## Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
# PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Full Pipeline (one command)

```bash
python scripts/run_all.py --config config/experiment.yaml
```

This runs: preprocess → build graphs → train RF + MLP → train GNN → evaluate → results table.

For a quick test with limited data:

```bash
python scripts/run_all.py --config config/experiment.yaml --nrows 10000
```

## Step by Step

### 1. Preprocess

```bash
python -m src.data.preprocess --config config/experiment.yaml
```

Reads `data/raw/{train,test,validation}.csv`, normalises, creates binary labels, saves to `data/processed/`.

### 2. Build Graphs

```bash
python -m src.data.graph_builder --config config/experiment.yaml
```

Builds kNN similarity graphs (windows of 200 flows), saves to `data/graphs/`.

### 3. Train Baselines

Handled by `scripts/run_all.py` or manually:

```bash
python scripts/run_all.py --skip-graphs --skip-gnn
```

### 4. Train Dynamic GNN

```bash
python scripts/run_all.py --skip-preprocess --skip-graphs --skip-baselines
```

### 5. Federated Learning

```bash
# Terminal 1 - server
python -m src.federated.run_federated server

# Terminal 2-4 - clients
python -m src.federated.run_federated client --cid 0
python -m src.federated.run_federated client --cid 1
python -m src.federated.run_federated client --cid 2
```

Before FL, create non-IID client splits:

```python
from src.federated.data_split import split_and_save
split_and_save("data/graphs/train_graphs.pt", "data/graphs", num_clients=3, alpha=0.5)
```

### 6. API

```bash
uvicorn src.siem.api:app --reload
# POST /score with flow windows to get prediction + ECS alert
```

### 7. Tests

```bash
python tests/test_api.py
```

### 8. Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_explore_ciciot.ipynb` | Data exploration |
| `02_federated_vs_central.ipynb` | FL vs central comparison + plots |
| `03_alert_examples.ipynb` | Explainability + SIEM alert demo |

## Output

| Directory | Contents |
|-----------|----------|
| `data/processed/` | Normalised parquets + `scaler.joblib` |
| `data/graphs/` | PyG graph files per split |
| `results/checkpoints/` | `dynamic_gnn_best.pt` |
| `results/metrics/` | `rf_metrics.json`, `mlp_metrics.json`, `central_gnn_metrics.json`, `results_table.csv` |
| `results/figures/` | ROC curves, confusion matrices, comparison bar charts |
