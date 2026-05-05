# 15-minute viva — screen share checklist (100% aligned to your project)

This list matches **your** dissertation pipeline: **CICIoT2023**, **kNN** windows, **GAT+GRU**, **RF/MLP**, **Flower FedAvg**, **Captum** explanations, **FastAPI** JSON, evidence in **`results/metrics/`** and Chapter **8**.

**Official session rules (short):**

- **15 minutes** total. **No PowerPoint.** **Camera on.** **Recorded.**  
- After recording starts: say **full name** + **Banner ID** (**B01821011**).  
- **First ~5 minutes:** problem + what you did, **short**, no deep code.  
- **Then:** **screen share** — show **existing** code and outputs; **do not** run every script **live**.  
- **Supervisor + moderator** ask questions — answer **carefully and honestly**.

Use **either** paths from **column A** (full repo, open repo root in IDE) **or** **column B** (everything under **`docs/viva/supervisor_meeting_bundle/`**).

---

## A. Before you join the call (prep)

| Action | Why |
|--------|-----|
| Open IDE on **repo root** **or** on **`docs/viva/supervisor_meeting_bundle/`** | One workspace for the whole walk |
| Pre-open the **minimum set** (next section) in tabs | Saves time in the **~7** minute screen block |
| Close or collapse **`data/`**, **`venv/`**, **`.cursor/`**, **`archive/`** | Less clutter; **no** raw dataset needed on screen |
| Have **`VIVA_SESSION_RULES.md`** read once | Same text as **`supervisor_meeting_bundle/VIVA_SESSION_RULES.md`** |

---

## B. First ~5 minutes (talking head + camera)

**You usually do not share detailed code yet.** Speak only: SOC / IoT flow context, CPU edge, explainable alerts, **CICIoT2023**, graphs + **GAT+GRU**, baselines, **FedAvg**, **IG** + **JSON**, headline metrics from memory (**F1**, **FP** story).

If they say **“show the title”,** flash **column A:** `README.md` **or** **column B:** `PROJECT_README.md` for **10 seconds**, then return to camera.

---

## C. Screen share — **minimum set** (enough to finish fairly, ~6–8 files)

Open **in this order** when you switch to screen share. **Do not run** **`run_all.py`** unless they ask.

| Step | Main repo path (A) | Same file in bundle (B) | One sentence you say |
|------|-------------------|-------------------------|----------------------|
| 1 | `config/experiment.yaml` | `config/experiment.yaml` | “All **hparams** and **46** feature names live here; one file matches the thesis.” |
| 2 | `scripts/run_all.py` | `scripts/run_all.py` | “This script **chains** preprocess, graphs, **RF**, **MLP**, **GNN**, and writes **`results/metrics/`**.” |
| 3 | `src/models/dynamic_gnn.py` | `src/models/dynamic_gnn.py` | “**GAT** on each window, **GRU** over **5** windows, then classify. This is the **dynamic GNN** in my title.” |
| 4 | `results/metrics/results_table.csv` | `results/metrics/results_table.csv` | “Headline **P**, **R**, **F1**, **ROC**, **inference ms** for **RF**, **MLP**, central and **federated GNN**.” |
| 5 | `results/metrics/rf_metrics.json` | `results/metrics/rf_metrics.json` | “Baselines **FP** / **FPR**; I can **compare** noise vs my **GNN** on the test run.” |
| 6 | `results/figures/cm_gnn.png` | `results/figures/cm_gnn.png` | “Confusion matrix matches the **GNN** row in **`results_table.csv`** on my test split.” |
| 7 | `results/alerts/example_alerts.json` | `results/alerts/example_alerts.json` | “**IG** + **attention** fields feed **SIEM**-style **JSON** for triage, as in Chapter **8**.” |

**Optional 8** if a minute remains:

| | Main repo (A) | Bundle (B) | Say |
|--|---------------|------------|-----|
| 8 | `thesis/` … use `submission/B01821011_Arka_Talukder_Dissertation_Final.docx` **or** root `Arka_Talukder_Dissertation_Final_DRAFT.md` | `thesis/B01821011_Arka_Talukder_Dissertation_Final.docx` **or** `thesis/Arka_Talukder_Dissertation_Final_DRAFT.md` | “Chapter **8** tables match these **CSV/JSON** files.” |

That is the **smallest** set that still proves **code + numbers + explainability + thesis** line up.

---

## D. Screen share — **if they ask deeper** (open only when asked)

| Topic | Main repo (A) | Bundle (B) |
|--------|---------------|------------|
| Data leakage / scaler | `src/data/preprocess.py` | `src/data/preprocess.py` |
| Graphs, **kNN**, pools | `src/data/graph_builder.py` | `src/data/graph_builder.py` |
| Sequences, **OR** label | `src/data/dataset.py` | `src/data/dataset.py` |
| **RF** baseline | `src/models/baselines.py` | `src/models/baselines.py` |
| **Federated** | `src/federated/server.py`, `client.py`, `data_split.py` | same paths |
| Explainability code | `src/explain/explainer.py`, `src/siem/alert_formatter.py`, `src/siem/api.py` | same paths |
| **FL** curves / rounds | `results/metrics/fl_rounds.json`, `results/figures/fl_convergence.png` | same paths |
| **MLP** metrics | `results/metrics/mlp_metrics.json` | same paths |

Full **tab order** for a long demo (not required for 15 min):

- In the **full repo:** [`supervisor_share/01_FILES_TO_OPEN_IN_ORDER.md`](supervisor_share/01_FILES_TO_OPEN_IN_ORDER.md)  
- In **`supervisor_meeting_bundle` only:** `OPEN_FILES_IN_ORDER.md` in that folder

---

## E. Folders you **do not** need on screen

| Folder | Reason |
|--------|--------|
| `data/raw/`, `data/processed/` | Huge; say data is **CICIoT2023** per **`SETUP_AND_RUN.md`** |
| `venv/`, `.venv/` | Local environment only |
| `supervisor_package/` | Mirror bundle; say you develop at **root** `src/` |
| `archive/` | Records only unless they ask **Appendix A** |
| `artifacts/` | Side exports; not the main thesis path |

---

## F. Time sketch (15 minutes)

| Minutes | What |
|---------|------|
| **0–0.5** | Recording on, **name**, **Banner ID** |
| **0.5–5** | Problem + your solution, **no** deep files |
| **5–12** | Screen share: **section C** minimum set (scroll, explain, **no** full live run) |
| **12–15** | First **questions** or quick follow-up file from **section D** |

---

## G. Accuracy checks (before the viva)

- [ ] **`results_table.csv`** on disk matches what you quote (**RF**, **MLP**, **GNN** **F1** / **ROC**).  
- [ ] **`rf_metrics.json`** **FP** (**187**) and **`mlp_metrics.json`** **FP** (**4**) match your thesis if you cite them.  
- [ ] **`example_alerts.json`** opens and you can name **one** top feature line.  
- [ ] You can point to **Chapter **8**** for each minimum-set file.

---

*Project: Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning.*
