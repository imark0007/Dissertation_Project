# Supervisor meetings and viva — what to show

Use this folder as a **briefing** before you share your screen in an IDE (e.g. **Visual Studio Code** or similar). It does not duplicate your project; it points to **where** to go and what to keep closed so the session stays on **your MSc work**, not toolbar clutter.

---

## Before you share your screen (about two minutes)

1. **Close** non-essential side panels and extensions so the file tree and code are easy to read (use a **clean** editor window for the session).
2. In the file explorer, **collapse** folders you will not open (see *Do not emphasise* below).
3. Optional: use your editor’s **files exclude** or **hide pattern** so rarely used folders stay collapsed during the session; clear those settings again later if you want the full tree.

---

## What to show (examination-relevant)

These items defend **method, implementation, results, and reproducibility**.

| Area | What to open | Why |
|------|----------------|-----|
| **Written thesis** | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) (and [`submission/forms/`](../../submission/forms/) if they ask for paperwork) | **Final** Word; matches programme expectations. Stakeholder map: [`project_portfolio/README.md`](../project_portfolio/README.md). |
| **Thesis source** | [`Arka_Talukder_Dissertation_Final_DRAFT.md`](../../Arka_Talukder_Dissertation_Final_DRAFT.md) at repo root | Single source for structure, chapters, figure references. |
| **Core implementation** | [`src/`](../../src/) — especially `models/dynamic_gnn.py`, `data/`, `federated/`, `explain/`, `siem/` | Where the GNN, FL, explanations, and API live. |
| **Configuration** | [`config/experiment.yaml`](../../config/experiment.yaml) | Hyperparameters, paths, and experimental settings in one place. |
| **Reproduction** | [`SETUP_AND_RUN.md`](../../SETUP_AND_RUN.md) | Step-by-step commands for preprocess → train → plots. |
| **Main drivers** | [`scripts/run_all.py`](../../scripts/run_all.py), [`run_ablation.py`](../../scripts/run_ablation.py), [`run_sensitivity_and_seeds.py`](../../scripts/run_sensitivity_and_seeds.py), [`generate_alerts_and_plots.py`](../../scripts/generate_alerts_and_plots.py) | End-to-end pipeline and thesis experiments. |
| **Word export** | [`scripts/dissertation_to_docx.py`](../../scripts/dissertation_to_docx.py) (only if they ask how the report was generated) | Markdown → submission Word. |
| **Evidence** | [`results/metrics/`](../../results/metrics/) (JSON/CSV), [`results/figures/`](../../results/figures/) (PNG) | Numbers and plots behind Chapter 8. |
| **Tests** | [`tests/test_api.py`](../../tests/test_api.py) | Small proof of API behaviour. |
| **Supervisor bundle** | [`supervisor_package/`](../../supervisor_package/) (optional) | Pre-curated mirror of dissertation + results + code snapshot for one-folder walkthrough. |
| **Meeting checklist** | [`SUPERVISOR_FINAL_FEEDBACK.md`](../reports/SUPERVISOR_FINAL_FEEDBACK.md) | Your own prompts and evidence map (keep private if you prefer). |

**High-level narrative (for you, not necessarily on screen):** [`DISSERTATION_PROJECT_GUIDE.md`](../../DISSERTATION_PROJECT_GUIDE.md) — chapter map and Q&A-style notes.

---

## What not to emphasise (or keep closed)

You are not hiding coursework; you are **avoiding noise** and **editor-specific** material.

### Editor-only folders (not part of the degree mark)

| Path | Reason |
|------|--------|
| **`.vscode/`** or other dot-folders (if present) | Local **Visual Studio Code** or editor settings; **not** core assessment content. You may keep them out of the explorer in meetings. (The repo’s [`.gitignore`](../../.gitignore) may exclude some local IDE paths.) |
| **`.vscode/`** (if present) | Personal workspace settings; use only if you are demonstrating a deliberate `files.exclude` for a tidy tree. |

### Optional / drafting tracks (can distract from the canonical thesis)

| Path | Reason |
|------|--------|
| [`Dissertation_Arka_Talukder_Humanized.md`](../../Dissertation_Arka_Talukder_Humanized.md) | Alternate drafting track; the **examined** line is [`Arka_Talukder_Dissertation_Final_DRAFT.md`](../../Arka_Talukder_Dissertation_Final_DRAFT.md). |
| **Humanized / compare / patch Word tooling** under `scripts/` | Files such as `*humanized*`, `compare_submission_humanized.py`, `align_humanized_*.py`, `fix_humanized_*.py`, `export_humanized_*.py`, `deep_compare_final_humanized.py`, `compare_two_docx.py`, etc. — **internal QA**, not the scientific pipeline. |

### Archive and records (show only if asked)

| Path | Reason |
|------|--------|
| [`archive/interim_report/`](../../archive/interim_report/) | Historical interim submission; not the final argument unless they ask. |
| [`archive/interim_report/INTERIM_REPORT_AI_SIMILARITY_AUDIT.md`](../../archive/interim_report/INTERIM_REPORT_AI_SIMILARITY_AUDIT.md) | **Do not open in session** — title and purpose are easily misread; keep interim folder collapsed. |
| [`archive/scripts_one_time/`](../../archive/scripts_one_time/) | One-off drafting utilities, not the reproducible experiment path. |

### Packaging and planning (optional context)

| Path | Reason |
|------|--------|
| [`artifacts/`](../../artifacts/) | Extra packaged figures; canonical thesis figures live under [`results/figures/`](../../results/figures/). |
| [`docs/planning/`](../planning/) | Personal roadmap / publication notes; fine for you, rarely needed live. |
| [`notebooks/`](../../notebooks/) | Exploratory; only if you explicitly rely on them in the thesis. |

### Data

| Path | Reason |
|------|--------|
| [`data/`](../../data/) (usually **gitignored**) | Raw CICIoT2023 CSVs are large and licensed; **you do not need to scroll them** in a viva. Say where they live and that preprocessing is in `src/data/` and `scripts/run_all.py`. |

---

## Suggested on-screen order (15–25 minutes)

1. **Thesis:** `submission/*.docx` or PDF if you use one — Chapters 1, 4–6, 8–9.
2. **Code:** `src/models/dynamic_gnn.py` → `src/federated/` or `src/siem/api.py` depending on questions.
3. **Config:** `config/experiment.yaml`.
4. **Evidence:** one metrics JSON (e.g. `results/metrics/central_gnn_metrics.json`) and one matching figure in `results/figures/`.
5. **Reproduce:** `SETUP_AND_RUN.md` + one command line (`run_all.py` or `run_ablation.py`).

---

## Optional: hide selected folders in the editor during a live demo

You can add a temporary `files.exclude` entry in local editor settings so optional folders (e.g. `archive/`, `artifacts/`) stay collapsed. Remove those entries after the session if you want the full tree back.

---

## Relationship to other folders

- **[`submission/`](../../submission/README.md)** — what you hand in; primary “written output” for supervisors.
- **[`supervisor_package/`](../../supervisor_package/README.md)** — portable bundle; same story, pre-zipped layout.
- **This folder** — **your checklist only**; no extra copies of the dissertation or data.

*Keep this file updated if you add new “internal only” scripts or editor folders.*
