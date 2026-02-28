# IoT Network Traffic Detection via Dynamic GNN, Federated Learning, and Explainability

MSc Dissertation — SOC-oriented IoT traffic detection using Dynamic Graph Neural Networks with Federated Learning and explainable SIEM-ready alerts.

## Dataset

**CICIoT2023** — 46 numeric flow features + label (34 attack classes + BenignTraffic).
Pre-split into `data/raw/train.csv`, `test.csv`, `validation.csv` (~1.6 GB train).

## Architecture

```
Raw flows → StandardScaler preprocessing → kNN similarity graphs (windows of 200 flows)
→ Sequences of 5 graph windows → 2-layer GATConv (4 heads) → Global pool → GRU → Classifier
```

**Baselines**: Random Forest, MLP.
**Federated**: FedAvg via Flower, 3 clients, non-IID Dirichlet split.
**Explainability**: Captum Integrated Gradients + GAT attention weights.
**SIEM**: ECS-formatted JSON alerts via FastAPI.

## Repository Structure

```
├── config/experiment.yaml          # All hyperparameters
├── data/raw/                       # CICIoT2023 CSVs
├── data/processed/                 # Normalised parquets + scaler
├── data/graphs/                    # PyG kNN graph objects
│
├── src/
│   ├── data/preprocess.py          # Clean, normalise, binary labels
│   ├── data/graph_builder.py       # kNN graph construction per window
│   ├── data/dataset.py             # GraphSequenceDataset + DataLoaders
│   ├── models/dynamic_gnn.py       # GATConv + GRU + classifier
│   ├── models/baselines.py         # RF + MLP
│   ├── models/trainer.py           # Train loop, early stopping, class weights
│   ├── federated/data_split.py     # Dirichlet non-IID splitting
│   ├── federated/client.py         # Flower NumPy client
│   ├── federated/server.py         # FedAvg + round tracking
│   ├── federated/run_federated.py  # CLI for FL server/client
│   ├── explain/explainer.py        # ExplanationBundle + IG + attention
│   ├── siem/alert_formatter.py     # ECS JSON formatter
│   ├── siem/api.py                 # FastAPI /health, /score
│   └── evaluation/metrics.py       # Precision, Recall, F1, ROC, CM plots
│
├── scripts/run_all.py              # Full pipeline: preprocess → train → evaluate
├── notebooks/
│   ├── 01_explore_ciciot.ipynb      # Data exploration
│   ├── 02_federated_vs_central.ipynb # FL vs central comparison
│   └── 03_alert_examples.ipynb      # Explainability + SIEM demo
├── tests/test_api.py               # API latency + schema tests
└── docs/                           # Dissertation + paper outlines
```

## Quick Start

```bash
# 1. Environment
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
# Install PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 2. Full pipeline (preprocess → graphs → baselines → GNN → results)
python scripts/run_all.py --config config/experiment.yaml

# 3. Quick test with limited data
python scripts/run_all.py --config config/experiment.yaml --nrows 10000

# 4. Federated learning (separate terminals)
python -m src.federated.run_federated server
python -m src.federated.run_federated client --cid 0
python -m src.federated.run_federated client --cid 1

# 5. API
uvicorn src.siem.api:app --reload

# 6. Tests
python tests/test_api.py
```

## Key Metrics

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| RF    | -         | -      | -   | -       | -              |
| MLP   | -         | -      | -   | -       | -              |
| GNN (central) | - | -     | -   | -       | -              |
| GNN (federated) | - | -   | -   | -       | -              |

*(Fill after running `scripts/run_all.py`; results saved to `results/metrics/`)*
