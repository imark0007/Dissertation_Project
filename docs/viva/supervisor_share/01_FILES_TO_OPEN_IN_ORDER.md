# Files to open in order (supervisor / viva screen share)

Keep the **repo root** as your workspace. Open one tab at a time **top to bottom**. Say one sentence per file: **what it does** and **why you wrote it that way** (thesis link).

---

## A. Orientation (2 minutes)

1. **`README.md`** (repo root) — “This is the public map: pipeline, quick start, where thesis lives.”  
2. **`config/experiment.yaml`** — “Single place for **46** features, **window_size**, **knn_k**, **FL** clients, **RF** depth, **GNN** hidden sizes. I change this, not random constants in code.”  

---

## B. Data path (3 minutes)

3. **`src/data/preprocess.py`** — “StandardScaler **fit on train only**; write parquet splits. This is how I avoid **leakage** from test into the scaler.”  
4. **`src/data/graph_builder.py`** — “**Stratified pools**, **kNN** edges, **window** labels; matches my design chapter **wording**.”  
5. **`src/data/dataset.py`** — **`GraphSequenceDataset`**: **5** windows, **OR** label over time.”  

---

## C. Models (4 minutes)

6. **`src/models/dynamic_gnn.py`** — “**GAT** layers → per-window embedding → **GRU** → classifier. This is the **dynamic GNN** in my title.”  
7. **`src/models/baselines.py`** — “**Random Forest** (**200** trees, depth **20** from YAML). Same rows as the GNN’s flow features.”  
8. **`scripts/run_all.py`** — “Orchestrates preprocess → graphs → **RF/MLP/GNN** train → writes **`results/metrics/`**.”  

---

## D. Federated learning (2 minutes)

9. **`src/federated/server.py`** and **`src/federated/client.py`** — “**Flower** **FedAvg**; weights only, not raw CSV rows.”  
10. **`src/federated/data_split.py`** — “**Dirichlet** **non**-**IID** split of training graphs for **3** clients.”  

---

## E. Explainability + SIEM-shaped output (2 minutes)

11. **`src/explain/explainer.py`** — “**Captum** **Integrated Gradients** + attention hooks for **top** features / nodes.”  
12. **`src/siem/alert_formatter.py`** — “Build **ECS**-like **JSON** for **`explanation`** fields.”  
13. **`src/siem/api.py`** — “**FastAPI** **`/score`** endpoint for demo.”  

---

## F. Evidence (2 minutes)

14. **`results/metrics/results_table.csv`** — “Headline **F1**, **ROC**, **inference** **ms**.”  
15. **`results/metrics/rf_metrics.json`** and **`mlp_metrics.json`** — “**FP** / **FPR** for baselines.”  
16. **`results/metrics/fl_rounds.json`** — “Round **1** vs round **10**; comms **`comm_bytes`**.”  
17. **`results/alerts/example_alerts.json`** — “**Three** to **five** worked alerts for Chapter **8**.”  
18. **One figure**, e.g. **`results/figures/cm_gnn.png`** or **`fl_convergence.png`** — “Plot matches the **JSON** on the test split.”  

---

## G. Thesis (1 minute)

19. **`Arka_Talukder_Dissertation_Final_DRAFT.md`** (fold **to** Chapter **6** or **8**) **or** **`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`** — “Same story as the **files** above; examiner cross-checks **tables** to **`results/metrics/`**.”  

---

## What you can **collapse** in the file tree

- **`archive/`**, **`artifacts/`**, **`docs/planning/`** (unless they ask).  
- **`supervisor_package/`** on screen (say: “this is a **copy** bundle; I develop at **root** `src/`”).  
- **`.cursor/`** — do not show (local tooling folder).  

---

## Optional: prove timeline (if they ask “when did you build this?”)

- **`git log --oneline -- src/`** in terminal (shows commits touching your implementation over time).  

---

Return to hub: **[`README.md`](README.md)** in this folder.
