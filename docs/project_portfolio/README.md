# Project portfolio — how to read this repository

**MSc Cyber Security (UWS) — B01821011 — Arka Talukder**  
Use this file as the **start page** for **GitHub visitors**, the **project coordinator**, and the **supervisor**: it tells you *what to open in order* and *where the examined artefacts live*.

> **Canonical final dissertation (Word):**  
> [`../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx)  
> **Optional PDF (if present in the repo):**  
> [`../../submission/B01821011_Final_Report.pdf`](../../submission/B01821011_Final_Report.pdf)  
> **Thesis source (Markdown, figures, tables):**  
> [`../../Arka_Talukder_Dissertation_Final_DRAFT.md`](../../Arka_Talukder_Dissertation_Final_DRAFT.md)

---

## Step 1 — One-minute summary

- **What this is:** An end-to-end **IoT flow intrusion-detection** prototype: **CICIoT2023** → *k*NN **graphs** → **GAT+GRU** → optional **Flower FedAvg** → **Captum** explanations → **FastAPI** ECS-style **JSON** alerts, evaluated on **CPU**.
- **Where the story is written:** Chapters 1–10 in the Word file above; **Chapters 11–12** references/bibliography; **Chapter 13** appendices; source text in the **Markdown** file at the repo root.
- **Where the code is:** `src/`, with `config/experiment.yaml` and `scripts/run_all.py`.

---

## Step 2 — Reproduce the experiments (readers with Python)

1. **Clone** the repository.
2. **Create a venv** and `pip install -r requirements.txt` (then install **PyTorch Geometric** for your platform — see PyG docs).
3. **Download** [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) and place `train.csv`, `validation.csv`, `test.csv` under `data/raw/`.
4. **Run** the main pipeline:  
   `python scripts/run_all.py --config config/experiment.yaml`  
5. **Details:** follow **[`../../SETUP_AND_RUN.md`](../../SETUP_AND_RUN.md)** (federated, API, ablation, sensitivity, figures).

`data/` is not committed; `results/` may contain your local metrics and figures if you have run experiments.

---

## Step 3 — For the project coordinator (programme / submission)

| Need | Where |
|------|--------|
| **Final report (Word, student-named)** | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) |
| **School / Moodle forms (front sheet, library, guideline, spec form)** | [`submission/forms/`](../../submission/forms/) |
| **Submission procedure / checklist (Markdown)** | [`../reports/SUBMISSION_CHECKLIST.md`](../reports/SUBMISSION_CHECKLIST.md) |
| **Handbook alignment notes (if used)** | [`../reports/HANDBOOK_COMPLIANCE_REPORT.md`](../reports/HANDBOOK_COMPLIANCE_REPORT.md) |

Read **[`../../submission/README.md`](../../submission/README.md)** for how submission artefacts relate to the Markdown source and sync scripts.

---

## Step 4 — For the supervisor (review + meetings + viva prep)

| Need | Where |
|------|--------|
| **What to show on screen, what to collapse** | [`../viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md) |
| **Curated one-folder package (dissertation + metrics + code snapshot)** | [`../../supervisor_package/`](../../supervisor_package/) — see its [`README`](../../supervisor_package/README.md) |
| **Meeting / feedback evidence map** | [`../reports/SUPERVISOR_FINAL_FEEDBACK.md`](../reports/SUPERVISOR_FINAL_FEEDBACK.md) |
| **Narrative + Q&amp;A (long-form)** | [`../../DISSERTATION_PROJECT_GUIDE.md`](../../DISSERTATION_PROJECT_GUIDE.md) |
| **Step-by-step: which code files to open and in what order** | [`../viva/supervisor_share/README.md`](../viva/supervisor_share/README.md) |
| **Viva brief, demo script, recording notes** | [`../viva/`](../viva/) (see **`supervisor_share/`** inside for walkthrough) |

**Suggested order in a final meeting:** final Word (link above) → `docs/reports/SUPERVISOR_FINAL_FEEDBACK.md` → `config/experiment.yaml` (repo root) → `results/metrics/` + one figure from `results/figures/` → `src/siem/api.py` if runtime is discussed.

---

## Step 5 — Repository map (where everything lives)

```
config/experiment.yaml     # One YAML for data paths, model hparams, FL, seeds
src/                        # Preprocess, GNN, FL, explain, FastAPI, metrics
scripts/                    # run_all, ablation, sensitivity, docx export, etc.
results/                    # Local outputs: metrics/, figures/, checkpoints/, alerts/
Arka_Talukder_Dissertation_Final_DRAFT.md  # Dissertation text source (at repo root)
submission/                 # Final .docx, PDF (optional), school forms
docs/project_portfolio/     # Stakeholder “start here” (this file) + feedback/
docs/viva_supervisor_materials/  # Screen-share etiquette and walkthrough
docs/video/                 # 5–6 min demo video script and checklists
docs/reports/               # Checklists, compliance, SUBMISSION, SUPERVISOR feedback
docs/viva/                  # Viva brief, cheatsheet, supervisor_share walkthrough pack
supervisor_package/         # One-folder mirror for supervisor (optional zip)
archive/                    # Interim, process/attendance sources, one-off scripts
```

---

## Step 6 — Optional: integrity check of the final Word (local)

If you have Python and `python-docx`:

```bash
python scripts/audit_b018_submission_docx.py
```

This **does not** edit the file; it prints structural checks. Update the script path in `scripts/audit_b018_submission_docx.py` if the filename changes.

---

## Step 7 — Citation and licence mindset

- **Dataset:** CICIoT2023 (Pinto *et al.*, 2023) — cite in any derivative work.  
- **Code and thesis text:** academic MSc project; **not** a commercial SOC product. See the root [`README`](../../README.md) for the standard disclaimer.

---

## Feedback / action-point file (if included)

If a feedback action sheet is in [`feedback/`](feedback/) (`.doc` from the programme), it is a **log** to align with the dissertation and meetings — it is not part of the automated pipeline.

---

*Author: Arka Talukder (B01821011) · Supervisor: Dr Raja Ujjan · UWS School of Computing, Engineering and Physical Sciences*
