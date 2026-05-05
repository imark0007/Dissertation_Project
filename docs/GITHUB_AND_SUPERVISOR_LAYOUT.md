# GitHub vs supervisor: repository layout (read this before you zip or push)

This file is the **single map** for: (1) what belongs on **public GitHub**, (2) what your **supervisor** needs in a meeting or zip, and (3) what is **optional / archive** and can stay out of a **minimal** clone.

**Important:** The **canonical code** lives at the **repository root** (`src/`, `scripts/`, `config/`). **`supervisor_package/`** is a **mirror** for one-folder sharing. **Do not** physically move `src/` or `scripts/` into new folders without updating every import and script path. This document **reorganises understanding only**, not disk layout.

---

## 1. Three audiences

| Audience | Goal | Open first |
|----------|------|------------|
| **GitHub visitor / peer replicator** | Understand the project, rerun the pipeline | [`README.md`](../README.md) → [`SETUP_AND_RUN.md`](../SETUP_AND_RUN.md) → [`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md) → `config/experiment.yaml` → `scripts/run_all.py` |
| **Supervisor / examiner (screen share)** | Verify thesis ↔ evidence ↔ code | Use **full local workspace**: `docs/project_portfolio/README.md` (public map) → final Word in **`submission/`** (local) → `results/metrics/` → viva packs under **`docs/viva/`** (local). |
| **You (maintenance)** | Refresh hand-in bundle | **`supervisor_package/README.md`** (**local** mirror) + `scripts/sync_dissertation_and_docx.py` |

---

## 2. Top-level folders: role and who needs them

| Folder | What it is | **Needed on GitHub** (typical) | **Needed for supervisor** | Notes |
|--------|------------|--------------------------------|---------------------------|--------|
| **`src/`** | Production code: data, models, FL, explain, SIEM API | **Yes — core** | Yes (or rely on root) | **Authoritative.** Not the copy under `supervisor_package/04_Source_code/`. |
| **`scripts/`** | `run_all`, ablation, sensitivity, dissertation export, figures | **Yes — core** | Yes | Same: prefer root `scripts/`, not only `supervisor_package/04_Scripts/`. |
| **`config/`** | `experiment.yaml` (single source of truth) | **Yes — core** | Yes | |
| **`tests/`** | Pytest / API checks | **Yes** | Optional | |
| **`requirements.txt`**, **`SETUP_AND_RUN.md`** | Install and run | **Yes — core** | Yes | |
| **`results/`** | `metrics/`, `figures/`, `checkpoints/`, `alerts/` | **Yes** (metrics, figures, alerts) | **Yes** | **Public remote:** `checkpoints/*.pt` omitted (regenerate). See [`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md). |
| **`submission/`** | Final dissertation Word/PDF, school forms | **`README.md` only (typical public)** | **Yes** (local finals) | Binaries stay **local** on public GitHub; see `submission/README.md`. |
| **`docs/`** | Portfolio, video, layout maps | **Subset** (see [`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md)) | **Full tree locally** | `docs/viva/`, `docs/reports/`, etc. are **local** on public policy. |
| **`supervisor_package/`** | Zip-style **mirror**: dissertation copy, `02_Results`, code copy, reproducibility copy | **Local only** (this repo policy) | **Convenient** for “one folder” email | See [`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md). |
| **`archive/`** | Process/attendance (Appendix A sources), interim report, one-off scripts | **Local only** (this repo policy) | **Yes** for spec compliance | Appendix A paths still valid on **your laptop**. |
| **`artifacts/`** | Packaged figure bundles / extra scripts | **Local only** (this repo policy) | Optional | |
| **`assets/`** | Static images (e.g. literature diagrams for docs) | **Optional** | Optional | |
| **`notebooks/`** | Exploration (e.g. CIC explore) | **Optional** | Optional | Not the main reproducible path. |
| **`thesis_artifacts/`** | Draft exports / humanized tracks | **Local only** (this repo policy) | Optional | Thesis Markdown at repo root is **local** on public policy. |
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
6. **Screen-share etiquette:** `docs/viva_supervisor_materials/README.md` (keep **`docs/viva/`** on **local** disk for this repo’s public policy).

**Optional but strong (local workspace):** root `DISSERTATION_PROJECT_GUIDE.md`, `docs/viva/VIVA_COACH_ALL_PHASES.md`, `docs/reports/SUPERVISOR_FINAL_FEEDBACK.md`.

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

*Aligned with root `README.md`. Paths like `archive/README.md` and `supervisor_package/README.md` refer to your **local** full workspace; they are **not** on the public remote by this repo’s policy (see [`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md)).*

---

## 8. Public GitHub (“job portfolio”) vs this full map

For a **lean public remote** (employers, peers), follow **[`GITHUB_PUBLIC_PORTFOLIO.md`](GITHUB_PUBLIC_PORTFOLIO.md)**: the **canonical implementation** remains **`src/`**, **`scripts/`**, **`config/`**, **`results/metrics/`**, and **`results/figures/`**; **mirrors** (`supervisor_package/`), **viva** trees, **submission PDFs**, **checkpoints**, and **long thesis Markdown** are intended to stay **local only** on the author’s machine (see `.gitignore`).

Sections 1–7 above still describe the **full workspace** when every folder exists on disk.
