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
| `docs/` | Dissertation outline, roadmap, publication plan, quick start (and handbook reference) |
| `archive/` | Interim report, process/attendance docs, one-time scripts — **kept for records and appendices** |

## Dissertation and Submission

| File | Purpose |
|------|---------|
| `Dissertation_Arka_Talukder.md` | Dissertation source (single source of truth) |
| `Arka_Talukder_Dissertation_Final.docx` | Generated Word for Turnitin and submission |
| `Arka-B01821011_MSc Cyber Security Project specification_form_2025-26.docx` | Project specification (Appendix B); referenced by dissertation_to_docx |
| `SUBMISSION_CHECKLIST.md` | MSc submission requirements and checklist |
| `FINAL_REPORT_GENERATION.md` | Procedure to produce final report (ablation → Table 4 → DOCX) |
| `HANDBOOK_COMPLIANCE_REPORT.md` | Alignment with MSc handbook |
| `DISSERTATION_REVIEW_REPORT.md` | Pre-submission review notes |

## Config and Environment

| File | Purpose |
|------|---------|
| `config/experiment.yaml` | Single config for preprocessing, graph, models, FL, paths |
| `requirements.txt` | Python dependencies (PyTorch, PyG, Flower, etc.) |
| `.gitignore` | Excludes venv, cache, logs; optionally results; excludes `docs/MSc Project Handbook*.docx` (copyright) |

## Data and Results (do not delete for reproducibility)

- **data/raw/** — Original train/test/validation CSVs
- **data/processed/** — Normalised parquets and scaler (created by preprocess)
- **data/graphs/** — Graph sequences (created by graph_builder; required for GNN and ablation)
- **results/checkpoints/** — Trained models (dynamic_gnn_best.pt, ablation_gat_only.pt, etc.)
- **results/metrics/** — All JSON/CSV metrics and tables
- **results/figures/** — All dissertation figures
- **results/alerts/** — Example alerts and summary

## Archive (do not delete — appendices and records)

- **archive/process_attendance/** — Process documentation and attendance log (Appendix A); embedded in final DOCX
- **archive/interim_report/** — Interim report and guidelines
- **archive/scripts_one_time/** — One-time scripts (kept for reference)

---

*Use with README.md and SETUP_AND_RUN.md for a full picture of the project.*
