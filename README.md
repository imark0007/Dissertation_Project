# IoT Network Traffic Detection via Dynamic GNN, Federated Learning, and Explainability

MSc Dissertation — SOC-oriented IoT traffic detection using Dynamic Graph Neural Networks with Federated Learning and explainable SIEM-ready alerts.

## Dataset

**CICIoT2023** — 46 numeric flow features + label (34 attack classes + BenignTraffic).
Pre-split into `data/raw/train.csv`, `test.csv`, `validation.csv` (~1.6 GB train).

## Architecture

```
Raw flows → StandardScaler preprocessing → kNN similarity graphs (windows of 50 flows)
→ Sequences of 5 graph windows → 2-layer GATConv (4 heads) → Global pool → GRU → Classifier
```

**Baselines**: Random Forest, MLP.
**Federated**: FedAvg via Flower, 3 clients, non-IID Dirichlet split.
**Explainability**: Captum Integrated Gradients + GAT attention weights.
**SIEM**: ECS-formatted JSON alerts via FastAPI.

## Repository Structure

```
├── config/experiment.yaml             # All hyperparameters (seed, models, paths)
├── data/raw/                          # CICIoT2023 CSVs (train, test, validation)
├── data/processed/                    # Normalised parquets + scaler
├── data/graphs/                       # PyG graph sequences (train/test/validation + client_* for FL)
│
├── src/
│   ├── data/preprocess.py             # Clean, normalise, binary labels (CLI: python -m src.data.preprocess)
│   ├── data/graph_builder.py          # kNN graph construction per window (CLI: python -m src.data.graph_builder)
│   ├── data/dataset.py                # GraphSequenceDataset + DataLoaders
│   ├── models/dynamic_gnn.py          # GAT + GRU + classifier (use_gru ablation option)
│   ├── models/baselines.py            # RF + MLP
│   ├── models/trainer.py              # Train loop, early stopping, class weights
│   ├── federated/data_split.py        # Dirichlet non-IID splitting
│   ├── federated/client.py, server.py, run_federated.py  # Flower FedAvg
│   ├── explain/explainer.py            # ExplanationBundle + IG + attention
│   ├── siem/alert_formatter.py, api.py # ECS JSON + FastAPI
│   └── evaluation/metrics.py          # Precision, Recall, F1, ROC, CM plots
│
├── scripts/
│   ├── run_all.py                     # Full pipeline: preprocess → graphs → baselines → GNN → results table
│   ├── run_ablation.py                 # GAT-only ablation → results/metrics/ablation_*
│   ├── eval_ablation_from_ckpt.py     # Evaluate saved ablation checkpoint (writes metrics if run_ablation stopped early)
│   ├── update_dissertation_table4.py   # Update Table 4 in dissertation MD from ablation_table.csv
│   ├── generate_alerts_and_plots.py    # Example alerts, FL convergence plot, model comparison figure
│   ├── generate_figure1.py             # Pipeline diagram (assets/figure1_pipeline.png)
│   ├── dissertation_to_docx.py        # MD → Word (Arka_Talukder_Dissertation_Final.docx)
│   ├── render_appendix1_code_figures.py  # PNG line-dumps for Handbook Appendix 1 → results/figures/appendix1/
│   ├── inference_timing.py             # Measure inference time
│   ├── run_fl_simulation.py           # FL simulation (optional)
│   └── md_to_pdf.py                   # Markdown to PDF (optional)
│
├── Dissertation_Arka_Talukder.md       # Dissertation source (single source of truth)
├── Arka_Talukder_Dissertation_Final.docx  # Generated Word for submission
│
├── results/
│   ├── checkpoints/                   # dynamic_gnn_best.pt, ablation_gat_only.pt, etc.
│   ├── metrics/                       # *_metrics.json, results_table.csv, ablation_table.csv
│   ├── figures/                       # ROC, confusion matrices, FL convergence; subfolder appendix1/ = code figures for dissertation
│   └── alerts/                        # example_alerts.json, alert_summary.txt
│
├── assets/figure1_pipeline.png        # Figure 1 for dissertation
├── notebooks/                         # 01_explore_ciciot, 02_federated_vs_central, 03_alert_examples
├── tests/test_api.py                  # API latency + schema tests
├── docs/                              # See docs/README.md — reports, planning, viva, reference samples
│   ├── reports/                       # Checklists, handbook compliance, final-report procedure, structure
│   ├── planning/                      # Roadmap, outlines, publication plan, quick-start steps
│   ├── viva/                          # Viva brief + printable cheatsheet
│   └── reference/                     # school_templates/ (Moodle forms), handbook copy (optional), dissertation_samples/
├── B01821011_Final_Report_Package_for_Supervisor/  # Supervisor bundle: synced copy of dissertation, scripts, source, results snapshots
├── B01821011_Arka_Talukder_Main_Report/           # Legacy figure/script paths (some package scripts still reference); prefer root scripts/ + results/figures/
└── archive/                           # Interim report, process/attendance docs, one-time scripts (kept for records)
```

## Quick Start

```bash
# 1. Environment
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
# Install PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 2. Full pipeline (preprocess → graphs → baselines → GNN → results)
python scripts/run_all.py --config config/experiment.yaml

# 3. Quick test with limited data
python scripts/run_all.py --config config/experiment.yaml --nrows 10000

# 4. Generate figures and example alerts (after pipeline)
python scripts/generate_alerts_and_plots.py

# 5. Ablation (thesis §7.6) — GAT-only variant
python scripts/run_ablation.py --config config/experiment.yaml
python scripts/update_dissertation_table4.py   # Update Table 4 in dissertation MD

# 6. Final report Word document
python scripts/dissertation_to_docx.py        # Output: Arka_Talukder_Dissertation_Final.docx

# 7. Federated learning (separate terminals)
python -m src.federated.run_federated server
python -m src.federated.run_federated client --cid 0
python -m src.federated.run_federated client --cid 1
python -m src.federated.run_federated client --cid 2

# 8. API
uvicorn src.siem.api:app --reload

# 9. Tests
python tests/test_api.py
```

See **SETUP_AND_RUN.md** for step-by-step, **`docs/reports/FINAL_REPORT_GENERATION.md`** for the full report procedure, and **`docs/reports/PROJECT_STRUCTURE.md`** for folder and file purposes.

## Key Metrics

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| RF    | —         | —      | —   | —       | —              |
| MLP   | —         | —      | —   | —       | —              |
| GNN (central) | — | —     | —   | —       | —              |
| GNN (federated) | — | —   | —   | —       | —              |

*Fill after running `scripts/run_all.py`; results in `results/metrics/results_table.csv`. Ablation table in `results/metrics/ablation_table.csv` after `scripts/run_ablation.py`.*
