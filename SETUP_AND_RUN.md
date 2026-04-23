# Setup and Run Guide

Run all commands from the **project root** (`attempt 2`).

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

Reads `data/raw/{train,test,validation}.csv`, normalises, creates binary labels, saves to `data/processed/`. Optional: `--nrows 10000` for testing.

### 2. Build Graphs

```bash
python -m src.data.graph_builder --config config/experiment.yaml --processed-dir data/processed --output-dir data/graphs
```

Builds kNN similarity graphs (windows of 50 flows, k=5), saves to `data/graphs/{train,test,validation}_graphs.pt`.

### 3. Train Baselines

Handled by `scripts/run_all.py` or:

```bash
python scripts/run_all.py --config config/experiment.yaml --skip-preprocess --skip-graphs --skip-gnn
```

### 4. Train Dynamic GNN

```bash
python scripts/run_all.py --config config/experiment.yaml --skip-preprocess --skip-graphs --skip-baselines
```

### 5. Generate Figures and Example Alerts

After the pipeline (and optionally FL):

```bash
python scripts/generate_alerts_and_plots.py
```

Produces: `results/figures/` (ROC, confusion matrices, FL convergence, model comparison), `results/alerts/example_alerts.json`, `results/alerts/alert_summary.txt`, `results/soc_workflow_example.md`.

### 6. Ablation (thesis Chapter 8)

```bash
python scripts/run_ablation.py --config config/experiment.yaml
```

Trains GAT-only variant (no GRU), evaluates, writes `results/metrics/ablation_gat_only.json` and `results/metrics/ablation_table.csv`. Then update the dissertation:

```bash
python scripts/update_dissertation_table4.py
```

If `run_ablation.py` stopped before writing metrics but the checkpoint exists:

```bash
python scripts/eval_ablation_from_ckpt.py --config config/experiment.yaml
```

### 7. Sensitivity grid and multi-seed (thesis Chapter 8)

Requires `data/processed/` parquets from the main pipeline. **CPU-heavy** (retrains GNN for each grid cell and each seed):

```bash
python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
```

Writes `results/metrics/sensitivity_table.csv`, `results/metrics/multi_seed_summary.json`, and `results/figures/sensitivity.png`. Use `--skip-sensitivity` or `--skip-seeds` to run only part of the study.

### 8. Final Report (Word)

```bash
python scripts/dissertation_to_docx.py
```

Reads `Arka_Talukder_Dissertation_Final_DRAFT.md`, produces `submission/Arka_Talukder_Dissertation_Final_DRAFT.docx`. See **`docs/reports/FINAL_REPORT_GENERATION.md`** for full procedure.

### 9. Federated Learning

Before FL, create non-IID client splits (once):

```python
from src.federated.data_split import split_and_save
split_and_save("data/graphs/train_graphs.pt", "data/graphs", num_clients=3, alpha=0.5)
```

Then run server and clients in separate terminals:

```bash
# Terminal 1 - server
python -m src.federated.run_federated server

# Terminals 2–4 - clients
python -m src.federated.run_federated client --cid 0
python -m src.federated.run_federated client --cid 1
python -m src.federated.run_federated client --cid 2
```

### 10. API

```bash
uvicorn src.siem.api:app --reload
# POST /score with flow/graph data → prediction + ECS alert
```

### 11. Tests

```bash
python tests/test_api.py
```

### 12. Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_explore_ciciot.ipynb` | Data exploration |
| `02_federated_vs_central.ipynb` | FL vs central comparison + plots |
| `03_alert_examples.ipynb` | Explainability + SIEM alert demo |

## Output

| Directory | Contents |
|-----------|----------|
| `data/processed/` | Normalised parquets + `scaler.joblib` |
| `data/graphs/` | `train_graphs.pt`, `test_graphs.pt`, `validation_graphs.pt`; after FL split: `client_0_graphs.pt`, etc. |
| `results/checkpoints/` | `dynamic_gnn_best.pt`, `ablation_gat_only.pt`, `dynamic_gnn_full_backup.pt`, `dynamic_gnn_federated.pt` |
| `results/metrics/` | `rf_metrics.json`, `mlp_metrics.json`, `central_gnn_metrics.json`, `federated_gnn_metrics.json`, `results_table.csv`, `ablation_gat_only.json`, `ablation_table.csv`, `fl_rounds.json`, `gnn_training_history.json` |
| `results/figures/` | ROC curves, confusion matrices, FL convergence, model comparison bar |
| `results/alerts/` | `example_alerts.json`, `alert_summary.txt` |

**Literature Review figures (§3):**

```bash
python scripts/generate_literature_figures.py
python scripts/generate_literature_figure.py
```

These create `assets/literature_*.png` (IDS taxonomy, FedAvg flow, dynamic GNN concept, explainability, positioning matrix). Used in the dissertation with Harvard references in captions.

**Dataset statistics (for thesis §6.3):**

```bash
python scripts/dataset_statistics.py --config config/experiment.yaml
```

Writes `results/metrics/dataset_stats.json` (row counts, class balance, sequence counts). Use to verify or report exact numbers in the dissertation.

## Archive and final-report appendices

Process documentation and attendance for **Appendix A** live under **`archive/process_attendance/`** (canonical for `dissertation_to_docx.py`). Interim report backups and one-off scripts are under **`archive/interim_report/`** and **`archive/scripts_one_time/`**. See **[`archive/README.md`](archive/README.md)** for the full index and how it maps to **Chapter 13** of the dissertation.
