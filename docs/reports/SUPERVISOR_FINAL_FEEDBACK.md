# Final supervisor meeting — what to bring and what is done

Use this **before** your last feedback session with **Dr. Raja Ujjan** (and moderator if present). The canonical technical work lives at the **repository root**; **`supervisor_package/`** is a **snapshot** for one-folder sharing (refresh from root if you edited anything after copying).

**Read first for the whole story in one document:** **`DISSERTATION_PROJECT_GUIDE.md`** at the repository root (plain-language timeline, chapter map, Q&A grid, limitations). A copy is in the supervisor package folder.

---

## 1. Bring (meeting order)

| # | Item | Location | Purpose |
|---|------|----------|---------|
| 1 | **Final Word thesis** | `submission/Arka_Talukder_Dissertation_Final_Submission.docx` (or generated `submission/Arka_Talukder_Dissertation_Final.docx`) and mirror under `supervisor_package/01_Dissertation/` | Walkthrough Ch 1 → 10; Ch 8–9 for numbers; Ch 13 for appendices |
| 2 | **Source Markdown** (optional) | `Dissertation_Arka_Talukder.md` | If they want to comment in Git / track changes in MD |
| 3 | **Metrics JSON/CSV** | `results/metrics/` or package `02_Results/metrics/` | Prove Tables 1–7 and FL rounds |
| 4 | **Figures** | `results/figures/` + `results/figures/appendix1/` | Match plots to metrics; Appendix 1 code figures |
| 5 | **Example alerts** | `results/alerts/example_alerts.json` | SIEM / explainability story |
| 6 | **Reproducibility one-pager** | `SETUP_AND_RUN.md`, `config/experiment.yaml` | Defend reruns and seeds |
| 7 | **Signed specification** | Root `Arka-B01821011_...specification_form_2025-26.docx` **and** copy in `05_Appendix_documents/` | Spec vs thesis alignment |
| 8 | **Process + attendance** | `archive/process_attendance/*.docx` and package `05_Appendix_documents/` | Programme appendix requirements |
| 9 | **School forms** (for submission, not always for supervisor) | `docs/reference/school_templates/` | Front sheet, declaration, library form |
| 10 | **This checklist + submission checklist** | `docs/reports/SUBMISSION_CHECKLIST.md`, `HANDBOOK_COMPLIANCE_REPORT.md` | Open “pending” items in Word |
| 11 | **Project explainer (optional print)** | `DISSERTATION_PROJECT_GUIDE.md` | Full narrative + viva-style Q&A — read before deep technical questions |

---

## 2. Already satisfied in the repo (high level)

- **Abstract:** trimmed to **≤200 words** (programme-style three paragraphs).
- **Main body (Ch 1–10, excl. references/appendices):** **~15,900 words** — above the old “~14k” estimate; if the handbook still asks **~18k**, expand only where thin (see checklist), not padding.
- **Ablation + Table 4:** **Chapter 8** (`run_ablation.py`, `ablation_table.csv`).
- **Sensitivity + multi-seed:** **Chapter 8** (`run_sensitivity_and_seeds.py`, tables/plots).
- **Appendix 1 (code figures):** **Appendix D** + `results/figures/appendix1/` + `scripts/render_appendix1_code_figures.py`.
- **Appendix 4 (optional):** **Appendix E** (GitHub + CICIoT2023 link; video line if you add one).
- **Train/serve *k*:** `src/siem/api.py` uses **`graph.knn_k`** from config (aligned with training).
- **Word export:** Embedded process/attendance/spec use **“Full text — …”** headings so they do not clash with **Appendix A–E** in the body.

---

## 3. Still your action (cannot be automated)

| Task | Notes |
|------|--------|
| **Insert Moodle front matter** in Word | Front sheet, signed declaration, signed library form |
| **Update TOC / List of Figures / List of Tables** in Word | Right-click → Update field; page numbers |
| **Confirm figure/table numbering** with handbook | Thesis uses **Figure 1…22** globally; renumber to **Figure 8.1** style only if the handbook requires it |
| **Turnitin + email** | Module co-ordinator; similarity guidance in checklist |
| **Viva** | Schedule with supervisor + moderator |
| **Optional:** trim or expand to **~18k** only if your **signed spec** or handbook explicitly requires it |

---

## 4. Questions supervisors often ask (quick map)

| Topic | Where in thesis / repo |
|-------|-------------------------|
| Research gap | Ch 2, §2.10 |
| Why *k*NN graphs | Ch 1, 4, 5; `graph_builder.py` |
| Federation + privacy | Ch 4, 8; `fl_rounds.json`, `federated_gnn_metrics.json` |
| Explainability | Ch 4, 6, 8; `explainer.py`, `example_alerts.json` |
| Robustness | Ch 8; sensitivity + multi-seed + ablation |
| Limitations / “100%” | Ch 9, 10 |
| Reproducibility | Ch 13 Appendix C; `SETUP_AND_RUN.md` |

---

*Regenerate Word after MD edits: `python scripts/dissertation_to_docx.py`. Approximate word counts: `python scripts/count_dissertation_words.py`.*
