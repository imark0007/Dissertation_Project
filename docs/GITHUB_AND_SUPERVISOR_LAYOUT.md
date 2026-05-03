# GitHub vs supervisor: repository layout (read this before you zip or push)

This file is the **single map** for: (1) what belongs on **public GitHub**, (2) what your **supervisor** needs in a meeting or zip, and (3) what is **optional / archive** and can stay out of a **minimal** clone.

**Important:** The **canonical code** lives at the **repository root** (`src/`, `scripts/`, `config/`). **`supervisor_package/`** is a **mirror** for one-folder sharing. **Do not** physically move `src/` or `scripts/` into new folders without updating every import and script path. This document **reorganises understanding only**, not disk layout.

---

## 1. Three audiences

| Audience | Goal | Open first |
|----------|------|------------|
| **GitHub visitor / peer replicator** | Understand the project, rerun the pipeline | `README.md` → `SETUP_AND_RUN.md` → `config/experiment.yaml` → `scripts/run_all.py` |
| **Supervisor / examiner (screen share)** | Verify thesis ↔ evidence ↔ code | `docs/project_portfolio/README.md` → final Word in `submission/` → `results/metrics/` → `DISSERTATION_PROJECT_GUIDE.md` → `docs/viva_supervisor_materials/README.md` |
| **You (maintenance)** | Refresh hand-in bundle | `supervisor_package/README.md` (refresh checklist) + `scripts/sync_dissertation_and_docx.py` |

---

## 2. Top-level folders: role and who needs them

| Folder | What it is | **Needed on GitHub** (typical) | **Needed for supervisor** | Notes |
|--------|------------|--------------------------------|---------------------------|--------|
| **`src/`** | Production code: data, models, FL, explain, SIEM API | **Yes — core** | Yes (or rely on root) | **Authoritative.** Not the copy under `supervisor_package/04_Source_code/`. |
| **`scripts/`** | `run_all`, ablation, sensitivity, dissertation export, figures | **Yes — core** | Yes | Same: prefer root `scripts/`, not only `supervisor_package/04_Scripts/`. |
| **`config/`** | `experiment.yaml` (single source of truth) | **Yes — core** | Yes | |
| **`tests/`** | Pytest / API checks | **Yes** | Optional | |
| **`requirements.txt`**, **`SETUP_AND_RUN.md`** | Install and run | **Yes — core** | Yes | |
| **`results/`** | `metrics/`, `figures/`, `checkpoints/`, `alerts/` | **Yes** (thesis cites them) | **Yes** | Checkpoints (`.pt`) can be large; some teams use Git LFS or omit and say “regenerate”. This repo may track them for assessment. |
| **`submission/`** | Final dissertation Word/PDF, school forms | **Yes** (if you want assessors to find finals) | **Yes** | Coordinator paths in `submission/README.md`. |
| **`docs/`** | Reports, viva, portfolio, reference, planning | **Yes** | **Yes** | `docs/reference/` may contain school templates; handbook DOCX may be gitignored per `.gitignore`. |
| **`supervisor_package/`** | Zip-style **mirror**: dissertation copy, `02_Results`, code copy, reproducibility copy | **Optional duplicate** | **Convenient** for “one folder” email; **refresh** before sending (see its `README.md`) | Not required if supervisor uses the **full repo** with root `results/` and `submission/`. |
| **`archive/`** | Process/attendance (Appendix A sources), interim report, one-off scripts | **Yes** (audit trail / thesis embedding) | **Yes** for spec compliance | Canonical Appendix A docs: `archive/process_attendance/`. |
| **`artifacts/`** | Packaged figure bundles / extra scripts | **Optional** | Optional | Thesis pipeline is **`scripts/`** at root; artifacts are side exports. |
| **`assets/`** | Static images (e.g. literature diagrams for docs) | **Optional** | Optional | |
| **`notebooks/`** | Exploration (e.g. CIC explore) | **Optional** | Optional | Not the main reproducible path. |
| **`thesis_artifacts/`** | Draft exports / humanized tracks | **Optional** | Optional | Main thesis source: **`Arka_Talukder_Dissertation_Final_DRAFT.md`** at root. |
| **`video/`** | Root stub; real pack is **`docs/video/`** | **Optional** | Optional | Start at `docs/video/README.md`. |
| **`data/`** | Raw/processed CICIoT2023 | **Never commit** (gitignored) | N/A | Download per `README.md`; supervisor does not need your local GB of CSV. |
| **`.cursor/`**, **`.vscode/`**, **`venv/`** | Editor / local env | **No** (gitignored) | No | |

---

## 3. What the supervisor **must** see (minimum)

To defend the dissertation without relying on `supervisor_package/`:

1. **Written thesis:** `submission/B01821011_Arka_Talukder_Dissertation_Final.docx` (or PDF if you publish it).
2. **Numbers behind Chapter 8:** `results/metrics/*.json`, `*.csv`.
3. **Figures matching those numbers:** `results/figures/` (and `results/figures/appendix1/` if Appendix 1 is discussed).
4. **Example alerts:** `results/alerts/example_alerts.json`.
5. **How to rerun:** `SETUP_AND_RUN.md`, `config/experiment.yaml`, `scripts/run_all.py`.
6. **Screen-share etiquette:** `docs/viva_supervisor_materials/README.md`.

**Optional but strong:** `DISSERTATION_PROJECT_GUIDE.md`, `docs/viva/VIVA_COACH_ALL_PHASES.md`, `docs/reports/SUPERVISOR_FINAL_FEEDBACK.md`.

---

## 4. What you can **omit** from a “lean” public GitHub (policy choice)

Depends on licence and programme rules. Common choices:

- **Omit or LFS:** `results/checkpoints/*.pt` (large); say in README “run `run_all.py` to regenerate”.
- **Omit:** `archive/interim_report/` if you only need final submission (keeping it is still valid for audit).
- **Omit:** `artifacts/` if redundant with `results/figures/`.
- **Omit:** duplicate `Dissertation_Arka_Talukder_Humanized.md` if you only maintain one Markdown source (or keep for history).

**Do not omit** without thought: `archive/process_attendance/` if your Word build embeds those files; `src/`, `scripts/`, `config/`, `tests/`, `submission/README.md`, main `Arka_Talukder_Dissertation_Final_DRAFT.md`.

---

## 5. Duplication warnings (avoid confusion)

| Location | Duplicate of | Rule |
|----------|--------------|------|
| `supervisor_package/01_Dissertation/` | `submission/` + root `*.md` | Refresh before zipping; **final** name on record: see `supervisor_package/README.md`. |
| `supervisor_package/02_Results/` | `results/` | Should match after a full experiment run. |
| `supervisor_package/04_Source_code/` | `src/` | **Edit only root `src/`** for real changes; copy into package for hand-in. |
| `supervisor_package/04_Scripts/` | `scripts/` | Prefer **`scripts/run_sensitivity_and_seeds.py`** at root for live demos. |

---

## 6. Suggested “GitHub README” order for first-time cloners

1. Read **`README.md`** (pipeline and quick start).  
2. Clone → venv → `pip install -r requirements.txt` → PyG.  
3. Add dataset under **`data/raw/`** (not in git).  
4. Run **`python scripts/run_all.py --config config/experiment.yaml`**.  
5. Open **`docs/project_portfolio/README.md`** if they are staff or moderator.

---

## 7. After you change the thesis or results

1. Regenerate or sync Word: `python scripts/sync_dissertation_and_docx.py` (or `dissertation_to_docx.py`).  
2. Copy fresh artefacts into **`supervisor_package/`** using the checklist in **`supervisor_package/README.md`**.  
3. Commit and push when ready (`git status` on `submission/` and `results/metrics/`).

---

*Aligned with root `README.md`, `archive/README.md`, and `supervisor_package/README.md`. No code paths were moved in creating this document.*
