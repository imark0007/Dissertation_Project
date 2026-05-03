# Viva topics map (your checklist)

Use this page to match **what they may ask** with **what you open** in the repo and **where it is written** in the thesis.

| Topic | What it means in your project | Primary repo evidence | Thesis (typical) |
|--------|------------------------------|----------------------|------------------|
| **Code run** | You can run the pipeline and show outputs | `scripts/run_all.py`, `SETUP_AND_RUN.md`, `results/metrics/` | Ch 6–8 |
| **Reflect original** | Original aim, spec, research questions, your design bet | `DISSERTATION_PROJECT_GUIDE.md` (root), `submission/…Dissertation_Final.docx` Ch 1, 4–5 | Ch 1, 3–5 |
| **Model design** | GAT+GRU, baselines, why graphs, why FL | `src/models/dynamic_gnn.py`, `config/experiment.yaml`, `src/data/graph_builder.py` | Ch 4–5 |
| **Development** | How you built it in the IDE: modules, config, iterations | `src/` tree, `git log` (optional screen tab), `scripts/` | Ch 6 |
| **Testing** | How you check the build works | `tests/`, manual runs in `SETUP_AND_RUN.md`, metrics JSON sanity | Ch 7 |
| **Thesis** | Written argument matches code and numbers | `Arka_Talukder_Dissertation_Final_DRAFT.md` or final Word | Full report |
| **Accuracy** | Metrics, F1, FP, robustness | `results/metrics/results_table.csv`, `rf_metrics.json`, `mlp_metrics.json`, `fl_rounds.json`, ablation/sensitivity | Ch 8–9 |

**Accuracy note:** Say **“on the fixed CICIoT2023 subset in my `results_table.csv`”**, not “always 100% everywhere.” Point to **FP counts** (GNN 0 vs RF 187 vs MLP 4 on your test run) when they ask about **operational** noise.

---

## One-line “I implemented this by…” for each topic

- **Code run:** “I drive everything from **`config/experiment.yaml`** and **`scripts/run_all.py`**; outputs land in **`results/metrics/`**.”  
- **Reflect original:** “Chapter **1** states the question; Chapters **4–5** lock the design; the code follows those sections.”  
- **Model design:** “**`dynamic_gnn.py`** implements **GAT** then **GRU**; **`baselines.py`** is **RF**; **MLP** is trained in **`run_all.py`**.”  
- **Development:** “I work in **VS Code**, edit **`src/`** and **`scripts/`**, and re-run the same commands after changes.”  
- **Testing:** “I use **`tests/`** plus full pipeline runs; final numbers are the **JSON/CSV** in **`results/metrics/`**.”  
- **Thesis:** “Every table in Chapter **8** traces to a file in **`results/metrics/`** or a figure in **`results/figures/`**.”  
- **Accuracy:** “Headline **F1/ROC** are in **`results_table.csv`**; I also report **ablation**, **sensitivity**, and **multi-seed** in Chapter **8**.”

---

Next: open **[`01_FILES_TO_OPEN_IN_ORDER.md`](01_FILES_TO_OPEN_IN_ORDER.md)** during the meeting.
