# Review and Update Summary

This document summarises the review and updates made so that **every file and folder is aligned** for the full dissertation work. No important files or folders were deleted.

---

## 1. Scripts

| File | Change |
|------|--------|
| **scripts/eval_ablation_from_ckpt.py** | Removed dependency on `run_ablation` (import from sibling script). Inlined the logic to build `ablation_table.csv` so the script runs from project root without path hacks. |
| **scripts/run_all.py** | Graph step now passes `processed_dir`, `output_dir`, and `config_path` explicitly to `build_and_save_graphs` so config and paths stay consistent. |

---

## 2. Documentation

| File | Change |
|------|--------|
| **README.md** | Expanded repository structure to include all scripts (run_ablation, eval_ablation_from_ckpt, update_dissertation_table4, generate_alerts_and_plots, generate_figure1, dissertation_to_docx), dissertation and report files, results layout, archive. Added Quick Start steps for ablation, Table 4 update, and DOCX generation. Linked SETUP_AND_RUN, `docs/reports/FINAL_REPORT_GENERATION.md`, `docs/reports/PROJECT_STRUCTURE.md`. |
| **SETUP_AND_RUN.md** | Clarified “run from project root”. Step 2 (Build Graphs) now shows explicit `--processed-dir` and `--output-dir`. Added steps 5 (Generate figures and alerts), 6 (Ablation + update Table 4), 7 (Final report DOCX). Added step for `eval_ablation_from_ckpt` if ablation stopped early. FL section now includes third client. Output table extended to include ablation and alerts. |
| **archive/README.md** | Stated that archive is for records and appendices (do not delete). Clarified that **process_attendance** is the canonical location for process documentation and attendance log; `dissertation_to_docx.py` embeds from `archive/process_attendance/`. |
| **`docs/` layout** | Submission and review Markdown moved under **`docs/reports/`**; thesis/publication planning under **`docs/planning/`**; viva notes under **`docs/viva/`**; handbook + sample dissertations under **`docs/reference/`**. See **`docs/README.md`**. |

---

## 3. No Deletions

- All existing files and folders were kept.
- **archive/** (interim report, process_attendance, scripts_one_time) is documented as required for records and appendices.
- **results/** (checkpoints, metrics, figures, alerts) is documented as output of the pipeline and ablation.

---

## 4. Alignment Checks

- **Dissertation appendices:** Already reference `archive/process_attendance/` and project root for spec; no change.
- **dissertation_to_docx.py:** Already points to `archive/process_attendance/` for process and attendance docs and to project root for project spec; no change.
- **Config:** `config/experiment.yaml` is the single config; scripts use it via `load_config` (paths resolved from project root).
- **.gitignore:** Excludes venv, cache, logs, and `docs/**/MSc Project Handbook*.docx` (handbook may live under `docs/reference/`).

---

## 5. Quick Reference

| Goal | Action |
|------|--------|
| Run full pipeline | `python scripts/run_all.py --config config/experiment.yaml` |
| Generate figures and alerts | `python scripts/generate_alerts_and_plots.py` |
| Run ablation | `python scripts/run_ablation.py --config config/experiment.yaml` |
| Ablation stopped early? | `python scripts/eval_ablation_from_ckpt.py --config config/experiment.yaml` |
| Update Table 4 in dissertation | `python scripts/update_dissertation_table4.py` |
| Generate final DOCX | `python scripts/dissertation_to_docx.py` |
| Understand folders | Read **`docs/reports/PROJECT_STRUCTURE.md`** |
| Step-by-step run guide | Read **`SETUP_AND_RUN.md`** (repo root) |
| Final report procedure | Read **`docs/reports/FINAL_REPORT_GENERATION.md`** |

---

*This summary was generated after a full project review. All updates keep the project aligned for the full dissertation workflow without removing any important files or folders.*
