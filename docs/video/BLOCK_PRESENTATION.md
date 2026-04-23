# What to present on screen — per block

Companion to [`SPOKEN_SCRIPT.md`](SPOKEN_SCRIPT.md). Paths are relative to the **repository root**; from `docs/video/`, go up two levels (`../..`) to reach the root.

---

## Block 1 — Problem and what you built (0:00–1:00)

| Present | Notes |
|--------|--------|
| **Title slide or first screen** | Full dissertation title; your name; B01821011; UWS (optional). |
| **Optional: one pipeline figure** | Root [`README.md`](../README.md) mermaid pipeline, or one slide with the same flow. |
| **Optional: dissertation file** | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) in Explorer or first page in Word/PDF when the script says “submitted dissertation on screen”. |

**Message:** problem (edge CPU, privacy, explainable alerts) + one sentence on what you built (CIC → graphs → GNN + baselines + FL → FastAPI/JSON).

---

## Block 2 — Data and representation (1:00–2:00)

| Present | Notes |
|--------|--------|
| [`config/experiment.yaml`](../config/experiment.yaml) | Scroll to **`graph:`**: `window_size` (50), `knn_k` (5), `sequence_length` (5), `minority_stride` if you mention stratified windows. |
| **Optional 3–5 s** | [`src/data/graph_builder.py`](../src/data/graph_builder.py) — module visible; kNN in code, not a full read-through. |

**Message:** where window / k / sequence are defined; RF/MLP use the same flows (spoken); baselines live in [`src/models/baselines.py`](../src/models/baselines.py) if you open a second tab.

---

## Block 3 — Models and central results (2:00–3:10)

| Present | Notes |
|--------|--------|
| [`results/metrics/results_table.csv`](../results/metrics/results_table.csv) | Zoom: RF, MLP, central GNN — **F1**, **ROC-AUC**, **inference ms**. |
| **One figure** from [`results/figures/`](../results/figures/) | e.g. `cm_gnn.png` **or** `roc_gnn.png` (one is enough). Optional: `model_comparison.png`. |
| **Optional** | [`results/metrics/dataset_stats.json`](../results/metrics/dataset_stats.json) only if you want 5 s on class balance. |

**Message:** three model families; same evaluation protocol; limits of headline metrics (spoken, thesis).

---

## Block 4 — Federated learning (3:10–3:50)

| Present | Notes |
|--------|--------|
| [`results/figures/fl_convergence.png`](../results/figures/fl_convergence.png) | Default: one clear visual. |
| **Or** [`results/metrics/fl_rounds.json`](../results/metrics/fl_rounds.json) | If numbers are legible. |
| **Same** `results_table.csv` | Federated GNN row next to central (optional second look). |

**Optional (if time):** Flower server + one client in terminals — drop if you run long.

**Message:** Flower, FedAvg, 3 clients, non-IID; parameters not raw rows; prototype parity, not production scale.

---

## Block 5 — Explainability and live output (3:50–5:10) — main demo

| Present | Notes |
|--------|--------|
| [`results/alerts/example_alerts.json`](../results/alerts/example_alerts.json) | **First** alert: `event`, `ml`, `explanation.top_features`, `explanation.top_nodes`. |
| **Live API** | `uvicorn src.siem.api:app` (preflight). Browser: `http://127.0.0.1:8000/docs` → **POST `/score`**, or `curl` + `score_body.json` — see [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md). |
| **Mention only** | API loads [`results/checkpoints/dynamic_gnn_best.pt`](../results/checkpoints/dynamic_gnn_best.pt); do not open the binary. |

**Message:** analyst-facing ECS-style JSON; IG + attention; live same structure from `POST /score` on CPU.

---

## Block 6 — Reproduction and limits (5:10–5:50)

| Present | Notes |
|--------|--------|
| [`SETUP_AND_RUN.md`](../SETUP_AND_RUN.md) | “Full pipeline” / `run_all.py` section. |
| **Command visible** | `python scripts/run_all.py --config config/experiment.yaml` from repo root. |
| [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) | Full written report. |
| **Optional 5 s** | [`config/experiment.yaml`](../config/experiment.yaml) as single config for reproducibility. |

**Message:** how to rerun; thesis as formal output; one limitation line (subset, no SOC user study).

---

## Outro (5:50–6:00)

| Present | Notes |
|--------|--------|
| **Title or “Thank you” slide** | Name, B01821011, title, Dr Ujjan as in script. |
| **Or** clean end screen | No new content. |

---

## Suggested Alt-Tab order

1. Title / README / dissertation (Block 1)  
2. `config/experiment.yaml` (Block 2)  
3. `results_table.csv` + one PNG (Block 3)  
4. `fl_convergence.png` + optional table (Block 4)  
5. `example_alerts.json` + browser `/docs` (Block 5)  
6. `SETUP_AND_RUN.md` + `submission` docx + terminal command (Block 6)  
7. Outro slide
