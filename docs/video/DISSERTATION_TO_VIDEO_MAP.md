# Dissertation (final Word) → 5–6 minute video map

**Submission file:** [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx)  
The Markdown source [`Arka_Talukder_Dissertation_Final_DRAFT.md`](../Arka_Talukder_Dissertation_Final_DRAFT.md) has the same **title, abstract, and chapter spine**; use it for **search/copy** while storyboarding. **Do not** read the full thesis aloud in six minutes—this table picks **one proof point** per area.

| Thesis section | What to say in one breath | What to show on screen (15–30 s) |
|----------------|---------------------------|----------------------------------|
| **Abstract; Ch 1 (aim, questions)** | Research question: explainable **GAT+GRU** on **kNN flow graphs**, **FL**, **SIEM-shaped** output on **CPU**; sub-questions on baselines, FedAvg, explanations | Title slide with **full title**; optional pipeline diagram from root [`README.md`](../README.md) (mermaid) |
| **Ch 2 (fit of ideas)** | SDN/IoT flows, need for **edge** + **triage**; attribute graphs; FedAvg; explainability for alerts—not depth, **motivation** | Optional: one literature figure if you have a slide; otherwise **skip** to save time |
| **Ch 4–5 Design/Implementation** | **CICIoT2023** → preprocess (scaler on train) → **windows** → **kNN** graphs → **sequences**; RF/MLP on same flows as rows | [`config/experiment.yaml`](../config/experiment.yaml) (`graph.window_size`, `knn_k`, `sequence_length`); optional peek at `src/data/graph_builder.py` |
| **Ch 6–7 Data/Methods (as in build)** | Three model **families**; same split **protocol**; metrics table | [`results/metrics/results_table.csv`](../results/metrics/results_table.csv) + **one** of [`cm_gnn.png`](../results/figures/cm_gnn.png) / `roc_gnn.png` |
| **Ch 7–8 Evaluation (FL)** | **Flower**, **FedAvg**, **3** clients, **non-IID**; privacy = **parameters**, not raw rows | [`fl_convergence.png`](../results/figures/fl_convergence.png) or [`fl_rounds.json`](../results/metrics/fl_rounds.json) |
| **Ch 5–6 Explain + API; Ch 8 XAI** | **IG** + **GAT attention** → `top_features` / `top_nodes` in **ECS-like** JSON | First object in [`example_alerts.json`](../results/alerts/example_alerts.json); then **live** `POST /score` (see [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md)) |
| **Repro; Ch 9/12 limitations** | `run_all.py` + `experiment.yaml`; dissertation as **formal** output; **scope** and **no user study** | [`SETUP_AND_RUN.md`](../SETUP_AND_RUN.md) + [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) in explorer |

**Tables your abstract name-checks (7–12):** the video is **not** a table read. If you are **over time**, name **only** `results_table.csv`; if you have **20 seconds spare**, show [`ablation_bar.png`](../results/figures/ablation_bar.png) or `ablation_table.csv` in `results/metrics/`.
