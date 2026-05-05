# Public GitHub profile vs full workspace (job + professionalism)

This file is the **single map** for: what **stays on the public remote** when you want a **clean, employer-facing** repository, versus what you **keep only on your laptop** (still present on disk, but **not tracked** on GitHub).

**Rule:** Nothing here deletes files from your machine. Paths listed as “local only” are removed from **Git tracking** via `git rm --cached` and blocked from future commits by **`.gitignore`**.

---

## 1. What employers and reviewers should see (on GitHub)

| Area | Path | Purpose |
|------|------|---------|
| **Entry** | [`README.md`](../README.md) | Problem, pipeline, headline results |
| **Run** | [`SETUP_AND_RUN.md`](../SETUP_AND_RUN.md), [`requirements.txt`](../requirements.txt) | Install, train, API |
| **Config** | [`config/experiment.yaml`](../config/experiment.yaml) | Reproducible hyperparameters and paths |
| **Code** | [`src/`](../src/) | GNN, baselines, FL, explainability, FastAPI |
| **Orchestration** | [`scripts/`](../scripts/) | `run_all.py`, ablation, sensitivity, plotting helpers |
| **Quality** | [`tests/`](../tests/) | e.g. API smoke tests |
| **Exploration** | [`notebooks/`](../notebooks/) | Dataset / FL exploration (optional read) |
| **Evidence** | [`results/metrics/`](../results/metrics/), [`results/figures/`](../results/figures/), [`results/alerts/`](../results/alerts/) | Numbers and plots that back the README table |
| **Images for docs** | [`assets/`](../assets/) | Diagrams used in README / narrative |
| **Demo narrative** | [`docs/video/`](../docs/video/) | Short demo script / checklist (no recording required) |
| **Integrity** | [`AUTHORSHIP.md`](../AUTHORSHIP.md), [`NOTICE.md`](../NOTICE.md), [`LICENSE`](../LICENSE) | Scope, reuse, copyright |
| **Stakeholder hub** | [`docs/project_portfolio/README.md`](project_portfolio/README.md) | How to read the **public** tree |
| **Layout notes** | [`GITHUB_AND_SUPERVISOR_LAYOUT.md`](GITHUB_AND_SUPERVISOR_LAYOUT.md), **this file** | Where things went and why |

**Not required for a job portfolio but kept in this repo:** many one-off `scripts/*` helpers for dissertation export—the core story is `run_all.py` and `src/`.

---

## 2. What stays local only (not on public GitHub)

Typical reasons: **duplicate trees**, **university / personal submission PDFs**, **copyrighted templates**, **viva prep**, **interim / process paperwork**, **large checkpoints**.

| Category | Paths (on disk; gitignored / untracked) |
|----------|----------------------------------------|
| **Hand-in mirror** | `supervisor_package/` |
| **Viva + supervisor screen-share packs** | `docs/viva/`, `docs/viva_supervisor_materials/` |
| **Internal planning & review** | `docs/planning/`, `docs/reports/` |
| **School templates, samples, handbook** | `docs/reference/` |
| **Coordinator feedback docs** | `docs/project_portfolio/feedback/` |
| **Interim + attendance + one-offs** | `archive/` |
| **Final dissertation Word/PDF + forms** | `submission/*.docx`, `submission/*.pdf`, `submission/forms/` — keep [`submission/README.md`](../submission/README.md) on GitHub |
| **Thesis Markdown / long guides at root** | `Arka_Talukder_Dissertation_Final_DRAFT.md`, `Dissertation_Arka_Talukder_Humanized.md`, `DISSERTATION_PROJECT_GUIDE.md` |
| **Draft Word exports** | `thesis_artifacts/` |
| **Packaged figure bundles** | `artifacts/` |
| **Generated school forms** | `results/generated_forms/` |
| **Large weights** | `results/checkpoints/*.pt` |

If you need a **full academic bundle** (everything for the supervisor or an offline zip), keep working in your **main workspace** or maintain a **private fork / branch**; the **public `main`** branch stays lean.

---

## 3. How to clone “job mode” vs work full-fat locally

- **Job / GitHub visitor:** clone the public repo → follow `README.md` → put CICIoT2023 under `data/raw/` (ignored) → run `python scripts/run_all.py --config config/experiment.yaml`.
- **You (full workspace):** same folder on your laptop still contains `submission/`, `docs/viva/`, `supervisor_package/`, thesis `.md`, etc.; they are simply **not pushed** after this layout.

---

## 4. Alignment with `docs/GITHUB_AND_SUPERVISOR_LAYOUT.md`

- **Canonical code** is always **`src/`**, **`scripts/`**, **`config/`** at the **repository root**.
- **`supervisor_package/`** is a **local convenience mirror** for assessment; it does **not** need to sit on a **public** GitHub for professionalism.

---

*Author: Arka Talukder (B01821011) — MSc Cyber Security, UWS.*
