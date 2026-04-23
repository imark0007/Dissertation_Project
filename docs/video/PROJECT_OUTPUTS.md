# What is the “output” of this project? (for the video and your supervisor)

Your dissertation (Abstract and Chapters 7–8) and the [`README`](../README.md) describe a **reproducible research prototype**, not a commercial SIEM product. Use the table below as the **clear answer in the first 60–90 seconds** of the video.

| Output (tangible) | What it is | Where it lives in the repo |
|-------------------|------------|----------------------------|
| **Written dissertation** | Full MSc report: aim, literature, design, implementation, evaluation, limitations | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) |
| **Trained models + metrics** | RF, MLP, central GNN, federated GNN: precision, recall, F1, ROC-AUC, inference (ms) | [`results/metrics/results_table.csv`](../results/metrics/results_table.csv) + `*_metrics.json` in same folder |
| **Figures (evaluation evidence)** | Confusion matrices, ROC curves, FL convergence, model comparison, ablation/sensitivity (as in thesis) | [`results/figures/`](../results/figures/) |
| **Data pipeline artefacts** | Normalised parquets, PyG graph `.pt` files (after you run the pipeline) | `data/processed/`, `data/graphs/` (often gitignored) |
| **Federated run log** | Round-by-round training evidence | [`results/metrics/fl_rounds.json`](../results/metrics/fl_rounds.json), [`results/figures/fl_convergence.png`](../results/figures/fl_convergence.png) |
| **SIEM-style alerts (analyst-facing)** | ECS-like JSON with **top_features** + **top_nodes** (IG + GAT attention) | [`results/alerts/example_alerts.json`](../results/alerts/example_alerts.json), [`results/alerts/alert_summary.txt`](../results/alerts/alert_summary.txt) |
| **Live API (deployment-shaped)** | Fast **`POST /score`** → prediction + same structured alert on **CPU** | [`src/siem/api.py`](../src/siem/api.py) — run: `uvicorn src.siem.api:app` (see [`SETUP_AND_RUN.md`](../SETUP_AND_RUN.md)) |
| **Reproducibility** | One config + one driver script for the main experiment | [`config/experiment.yaml`](../config/experiment.yaml), [`scripts/run_all.py`](../scripts/run_all.py) |

**One sentence (from your abstract, adapted for speaking):**  
*The work delivers a CPU-oriented, reproducible end-to-end path from **CICIoT2023 flows** to **kNN windowed graphs** and **GAT+GRU** sequence classification, with **RF/MLP** baselines, **Flower FedAvg**, **Captum** explanations, and **FastAPI** ECS-style JSON—documented in the dissertation with fixed-split metrics and clear limitations.*

**Honest one-liner on limits (say once at the end):** fixed lab subset, no SOC user study, external deployment not claimed—strengths are **traceability** and **integrated** implementation.
