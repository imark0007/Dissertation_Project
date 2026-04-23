# Project Structure — Folders and Purpose

This file describes every top-level folder and key file so the full dissertation work stays aligned. **Do not delete** the folders or files listed as required for submission or reproducibility.

## Root

| Item | Purpose |
|------|---------|
| `config/` | Experiment configuration (seed, data paths, model hyperparameters, FL settings) |
| `data/` | Raw CICIoT2023 CSVs, processed parquets, built graphs (required for training and evaluation) |
| `src/` | Source code: data, models, federated, explain, siem, evaluation |
| `scripts/` | Runners and utilities: run_all, run_ablation, generate_alerts_and_plots, dissertation_to_docx, etc. |
| `results/` | Checkpoints, metrics, figures, alerts (outputs of pipeline and ablation) |
| `assets/` | Figure 1 pipeline image for dissertation |
| `notebooks/` | Exploration and demos (optional for submission) |
| `tests/` | API tests |
| `docs/` | All non-code **documentation** — index in `docs/README.md` |
| `docs/project_portfolio/` | **“Start here”** for programme coordinator, supervisor, and GitHub visitors — see **`project_portfolio/README.md`** in that folder |
| `docs/viva_supervisor_materials/` | **Viva / screen share** — what to open vs. hide — see **`viva_supervisor_materials/README.md`** |
| `docs/video/` | **5–6 min demo** — script, preflight, recording pack (`README.md`, etc.) |
| `docs/reports/`, `docs/viva/`, `docs/planning/`, `docs/reference/` | Checklists, viva notes, roadmaps, school templates and samples (see `docs/README.md`) |
| `submission/` | Final **B018** Word (and optional PDF), generated Word, and **school forms** — see **`submission/README.md`** |
| `supervisor_package/` | Optional **one-folder bundle** for supervisor review (mirrors dissertation, results, code snapshot) — see **`supervisor_package/README.md`** |
| `artifacts/` | Optional packaged exports (e.g. **`main_report_figures/`**) — not required to rerun `scripts/run_all.py` |
| `archive/` | Interim report, process/attendance docs, one-time scripts — **kept for records and appendices** — see **[`archive/README.md`](../../archive/README.md)** for the authoritative index |

## Dissertation and Submission

| File | Purpose |
|------|---------|
| `Arka_Talukder_Dissertation_Final_DRAFT.md` | Dissertation source (single source of truth) |
| `submission/` | Final **Word** exports, humanized variant if used, and **school forms** — see `submission/README.md` |
| `submission/B01821011_Arka_Talukder_Dissertation_Final.docx` | **Final** hand-finished dissertation (Word) for programme / archive |
| `submission/B01821011_Final_Report.pdf` | Optional PDF export of the final report (if committed) |
| `submission/Arka_Talukder_Dissertation_Final_DRAFT.docx` | Generated Word from `dissertation_to_docx.py` / `sync_dissertation_and_docx.py` |
| `submission/Arka_Talukder_Dissertation_Final_Submission.docx` | Optional alternate filename for Turnitin-ready copy (if used) |
| `submission/forms/Arka-B01821011_MSc Cyber Security Project specification_form_2025-26.docx` | Project specification (Appendix B); referenced by `dissertation_to_docx.py` (also searched under repo root and `supervisor_package/`) |
| `docs/reports/SUBMISSION_CHECKLIST.md` | MSc submission requirements and checklist |
| `docs/reports/FINAL_REPORT_GENERATION.md` | Procedure to produce final report (ablation → Table 4 → DOCX) |
| `docs/reports/HANDBOOK_COMPLIANCE_REPORT.md` | Alignment with MSc handbook |
| `docs/reports/DISSERTATION_REVIEW_REPORT.md` | Pre-submission review notes |
| `docs/reports/PROJECT_STRUCTURE.md` (this file) | Folder and file purposes |
| `docs/reports/Project_Overview_Reference.md` | High-level project reference (optional PDF via `scripts/md_to_pdf.py`) |

## Config and Environment

| File | Purpose |
|------|---------|
| `config/experiment.yaml` | Single config for preprocessing, graph, models, FL, paths |
| `requirements.txt` | Python dependencies (PyTorch, PyG, Flower, etc.) |
| `.gitignore` | Excludes venv, cache, logs; optionally results; excludes `docs/**/MSc Project Handbook*.docx` (copyright) |

## Data and Results (do not delete for reproducibility)

- **data/raw/** — Original train/test/validation CSVs
- **data/processed/** — Normalised parquets and scaler (created by preprocess)
- **data/graphs/** — Graph sequences (created by graph_builder; required for GNN and ablation)
- **results/checkpoints/** — Trained models (dynamic_gnn_best.pt, ablation_gat_only.pt, etc.)
- **results/metrics/** — All JSON/CSV metrics and tables
- **results/figures/** — All dissertation figures (including **`appendix1/`** auto-generated code screenshots for Handbook Appendix 1)
- **results/alerts/** — Example alerts and summary

## Archive (do not delete — appendices and records)

See **[`archive/README.md`](../../archive/README.md)** for filenames, embedding behaviour, and what is **not** in `archive/` (e.g. school templates under `docs/reference/school_templates/`).

- **archive/process_attendance/** — **Canonical** process documentation + attendance (**Appendix A**); merged by `dissertation_to_docx.py` after Appendix E
- **archive/interim_report/** — Interim report artefacts and guidelines
- **archive/scripts_one_time/** — One-off drafting scripts (not the main `scripts/` pipeline)

---

*Use with `README.md` and `SETUP_AND_RUN.md` (repository root) for a full picture of the project.*
