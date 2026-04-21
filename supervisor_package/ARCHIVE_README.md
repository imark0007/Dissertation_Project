# Archive

Material in **`archive/`** is kept for **records**, **MSc programme appendices**, and **audit trail**. It is referenced from the **final dissertation** (**Chapter 13 — Appendices**) and from **`scripts/dissertation_to_docx.py`**, which merges selected documents into **`submission/Arka_Talukder_Dissertation_Final.docx`**.

**Do not delete** these folders if you intend to regenerate the Word export or evidence the interim / process trail for the School.

For a full map of the active codebase, see **[`README.md`](../README.md)** (root) and **[`docs/reports/PROJECT_STRUCTURE.md`](../docs/reports/PROJECT_STRUCTURE.md)**.

---

## How `archive/` connects to the final report

| Final report (thesis) | `archive/` role |
|------------------------|-----------------|
| **Appendix A** (process + attendance) | **Canonical .docx files** live under **`archive/process_attendance/`**. The dissertation lists the same filenames; Word export embeds them after Appendix E under **Full text — …** headings. |
| **Interim milestone** (historical) | **`archive/interim_report/`** holds interim report drafts, PDFs, guidelines, and backups — not part of the final technical argument, but kept for submission / moderation records. |
| **One-off tooling** | **`archive/scripts_one_time/`** holds scripts used during drafting (e.g. interim fixes, doc conversion experiments). **Not** part of the reproducible pipeline; use root **`scripts/`** for experiments that match the dissertation. |

**Not stored in `archive/`:** Moodle **front-sheet / declaration / library** template copies live under **`docs/reference/school_templates/`** (see root `README.md` → Documentation map).

**Supervisor zip mirror:** `supervisor_package/05_Appendix_documents/` may contain copies of process, attendance, and specification files for convenience — **`archive/process_attendance/`** remains the **canonical** path for the two Appendix A Word files used by `dissertation_to_docx.py`.

---

## Folder index

### `archive/process_attendance/` (Appendix A — canonical)

| File | Role |
|------|------|
| `Arka_Talukder_Process_Documentation_B01821011.docx` | Project process documentation |
| `Arka Talukder_Attendance_Jan-Feb_B01821011.docx` | Attendance log |
| `generate_process_attendance_logs.py` | Optional helper script (historical) |

**Embedding:** `dissertation_to_docx.py` reads these paths when appending **Full text — Project process documentation** and **Full text — Attendance log** after **Appendix E**.

---

### `archive/interim_report/` (interim submission — records)

| Item | Role |
|------|------|
| `Interim_Report_Arka_Talukder.md` (and backups) | Markdown interim drafts |
| `B01821011_Interim_Report_Final*.docx` / `.pdf` | Submitted interim artefacts |
| `Guidelines for the production and submission of the MSc Interim Report.docx` | School guideline copy |
| `INTERIM_REPORT_AI_SIMILARITY_AUDIT.md` | Optional audit notes |

---

### `archive/scripts_one_time/` (drafting utilities — not the main pipeline)

| Script | Typical use |
|--------|----------------|
| `generate_interim_report.py` | Interim report generation |
| `apply_interim_fixes.py`, `apply_fixes.py` | One-off text or structure fixes |
| `md_to_docx.py` | Legacy / experimental Markdown → Word |
| `refine_backup_ielts.py` | Non-thesis language refinement (historical) |

**Reproducible thesis pipeline** (what the dissertation describes): root **`scripts/run_all.py`**, **`run_ablation.py`**, **`run_sensitivity_and_seeds.py`**, **`generate_alerts_and_plots.py`**, etc. — see **`SETUP_AND_RUN.md`**.

---

## Maintenance checklist (after each major thesis edit)

1. If you replace process or attendance files, update the **same filenames** in **`archive/process_attendance/`** (or update paths in `dissertation_to_docx.py`).
2. Refresh **`supervisor_package/05_Appendix_documents/`** if you zip that folder for the supervisor.
3. Regenerate Word: `python scripts/dissertation_to_docx.py`.
4. Keep this **`archive/README.md`** in sync when you add new record-only subfolders under **`archive/`**.

---

*Last aligned with: final dissertation structure (Chapter 13 appendices, Full text embedding, GitHub `README.md`).*
