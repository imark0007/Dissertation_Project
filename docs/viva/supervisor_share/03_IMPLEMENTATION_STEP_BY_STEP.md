# Implementation step by step (story you tell while scrolling code)

Each **step** is one **stage** of your dissertation pipeline. Tie each to **one** folder or script so your supervisor hears a **clear** chain.

---

## Step 1 — Freeze settings

- **File:** `config/experiment.yaml`  
- **Say:** “All **hyperparameters** and **paths** live here so my **thesis** numbers match **reruns**.”

---

## Step 2 — Load and clean data

- **File:** `src/data/preprocess.py`  
- **Say:** “I read **CSV** splits, **standardise** on **train** only, write **`data/processed/*.parquet`**.”

---

## Step 3 — Build graphs and sequences

- **Files:** `src/data/graph_builder.py`, `src/data/dataset.py`  
- **Say:** “**Separate** **benign** / **attack** pools, **kNN** **(k=5)** on **50** flows per window, **5** windows per sequence, **OR** label in time.”

---

## Step 4 — Baselines (same data, flat table)

- **File:** `src/models/baselines.py`  
- **Say:** “**RF** on **one** row per flow; same **46** features as the GNN nodes.”

---

## Step 5 — Dynamic GNN

- **File:** `src/models/dynamic_gnn.py`  
- **Say:** “**GAT** for **neighbours**, **GRU** for **temporal** sequence of **windows**.”

---

## Step 6 — Training loop

- **Files:** `src/models/trainer.py`, entry in `scripts/run_all.py`  
- **Say:** “**Early** **stopping** on **val**, **class** **weights** from config, checkpoints under **`results/checkpoints/`**.”

---

## Step 7 — Evaluate and save metrics

- **Output:** `results/metrics/*.json`, `results_table.csv`  
- **Say:** “**Precision**, **recall**, **F1**, **ROC**, **inference** **time**; **same** **test** **split** for **all** models.”

---

## Step 8 — Federated training (same GNN)

- **Files:** `src/federated/*`, `scripts` / `SETUP_AND_RUN.md` for launch  
- **Say:** “**FedAvg** across **3** clients; **only** **weights** move; **final** **test** in **`federated_gnn_metrics.json`**.”

---

## Step 9 — Explain and package alerts

- **Files:** `src/explain/explainer.py`, `src/siem/alert_formatter.py`, `src/siem/api.py`  
- **Say:** “**IG** + **attention** → **`example_alerts.json`** and **`/score`** **JSON**.”

---

## Step 10 — Robustness tables (thesis Chapter 8)

- **Scripts:** `scripts/run_ablation.py`, `scripts/run_sensitivity_and_seeds.py`  
- **Outputs:** `ablation_table.csv`, `sensitivity_table.csv`, `multi_seed_summary.json`  
- **Say:** “**GRU** **ablation** and **(window, k)** grid show the **default** **(50,5)** is **not** **accidental**.”

---

## Step 11 — Write-up

- **Files:** `Arka_Talukder_Dissertation_Final_DRAFT.md` → `submission/…Final.docx`  
- **Say:** “Every **claim** in **Results** links to a **file** in **`results/metrics/`** or **`figures/`**.”

---

Print the one-page graph from **`../PROJECT_VIVA_CHEATSHEET_PRINT.md`** if you need a **paper** prop beside the screen.
