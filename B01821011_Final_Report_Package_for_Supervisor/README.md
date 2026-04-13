# Final report package — B01821011 (Arka Talukder)

**Purpose:** One folder to share with your **supervisor**: written dissertation, **evidence files** (metrics, figures, alerts), **how to reproduce** (config + guide), and a **snapshot of code**.

**Important:** This folder was copied at a point in time. The **canonical** project (latest dissertation, `scripts/run_sensitivity_and_seeds.py`, etc.) is the **repository root** (`attempt 2`). Before a meeting, refresh `01_Dissertation/` and `04_Scripts/` from the root if you have edited the thesis or moved scripts.

---

## What to show your supervisor (meeting order)

Use this as a **walkthrough** of your whole MSc work: question → design → build → evidence → honesty.

### 1. The written thesis (main story)

- **`01_Dissertation/Arka_Talukder_Dissertation_Final.docx`** (or the `.md` if they prefer source).
- **Where to point them first:** **Chapter 1 §1.6** (marking criteria ↔ chapters), **Chapter 2** (literature + gap), **Chapters 4–5** (method + design), **Chapter 6** (what you built), **Chapter 8** (results tables/figures), **Chapter 9** (what it means + limits), **Chapter 10** (reflection).
- If they ask about **specification vs literature %**: **§1.6–1.6.1** ties the thesis to the **agreed specification**; bring your **signed spec** (see `05_Appendix_documents/`).

### 2. Proof the numbers are real (not invented)

- **`02_Results/metrics/`** — JSON/CSV that back **Tables 1–4, 6–7** (e.g. `central_gnn_metrics.json`, `rf_metrics.json`, `mlp_metrics.json`, `fl_rounds.json`, `sensitivity_table.csv`, `multi_seed_summary.json`, `ablation_table.csv`, `gnn_training_history.json`).
- Open one file alongside the matching table in Chapter 8.

### 3. Proof the plots match the metrics

- **`02_Results/figures/`** — confusion matrices, ROC curves, FL convergence, model comparison, sensitivity heatmap, ablation bar, etc.
- Cross-check one figure to its metrics file if they ask.

### 4. The “SOC product” slice of the work

- **`02_Results/alerts/example_alerts.json`** (and `alert_summary.txt` if useful) — shows **explainable** SIEM-style JSON (features + nodes/flows).
- Mention **FastAPI** / `src/siem/` in the repo if they want to see runtime code.

### 5. Reproducibility (you can defend “how would I rerun this?”)

- **`03_Reproducibility/`** — `experiment.yaml`, `requirements.txt`, **`SETUP_AND_RUN.md`** (commands for preprocess → graphs → `run_all.py` → plots/alerts → ablation → **sensitivity/multi-seed**).

### 6. Implementation (if they want depth or a viva dry-run)

- **`04_Source_code/src/`** — `data/`, `models/`, `federated/`, `explain/`, `siem/`, `evaluation/`.
- **`04_Scripts/`** — `run_all.py`, `generate_alerts_and_plots.py`, `run_ablation.py`, `dissertation_to_docx.py`, etc.

**Sensitivity / multi-seed:** In the **current repo**, the driver is **`scripts/run_sensitivity_and_seeds.py`** (project root), not only under `B01821011_Arka_Talukder_Main_Report/`. The **`06_Main_report_scripts/`** copy here may be an **older snapshot** — prefer the root `scripts/` version when demonstrating.

### 7. Programme paperwork (non-technical but often required)

- **`05_Appendix_documents/`** — process documentation + attendance.
- **`ADD_SIGNED_PROJECT_SPECIFICATION_HERE.txt`** — reminder to add your **signed project specification** if it is not already there.

### 8. Checklists (optional printout)

- **Copies in this folder:** `SUBMISSION_CHECKLIST.md`, `HANDBOOK_COMPLIANCE_REPORT.md` (snapshot for the supervisor zip).
- **Canonical (kept in sync with the repo):** `docs/reports/SUBMISSION_CHECKLIST.md`, `docs/reports/HANDBOOK_COMPLIANCE_REPORT.md` at repository root.

---

## If they ask specific questions

| Question | Where to point |
|----------|----------------|
| “What did you actually build?” | Ch 5–6 + `04_Source_code/src/` |
| “Are the headline metrics reproducible?” | Ch 8 + `02_Results/metrics/` + `SETUP_AND_RUN.md` |
| “What about federation / privacy?” | Ch 4, 8, 9 + `fl_rounds.json`, `federated_gnn_metrics.json` |
| “How do you explain alerts?” | Ch 4, 6, 8 + `example_alerts.json` + `src/explain/` |
| “Robustness beyond one run?” | Ch 8 + `sensitivity_table.csv`, `multi_seed_summary.json`, ablation files |
| “What would you do with more time?” | Ch 9–10 |

**Dataset:** CICIoT2023 is **external**; the thesis and `SETUP_AND_RUN.md` describe your subset, splits, and windowing — raw CSVs may live outside this zip.

---

## Refresh this package before sending

From the repo root, re-copy at least:

- `Dissertation_Arka_Talukder.md` and `Arka_Talukder_Dissertation_Final.docx` → `01_Dissertation/`
- `scripts/*.py` → `04_Scripts/` (including **`run_sensitivity_and_seeds.py`**)
- `results/metrics/`, `results/figures/`, `results/alerts/` → `02_Results/…`
- `SETUP_AND_RUN.md` → `03_Reproducibility/`

Then regenerate the Word file if needed: `python scripts/dissertation_to_docx.py`.

---

*Updated to match repository layout (sensitivity script under `scripts/`).*
