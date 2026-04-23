# Submission artefacts

This folder holds **what you hand in** (or export for Turnitin) and **school paperwork copies**, separate from the runnable codebase under `src/` and `scripts/`.

## Canonical final report (examined line)

| File | Role |
|------|------|
| **`B01821011_Arka_Talukder_Dissertation_Final.docx`** | **Final MSc dissertation (Word)** — primary hand-in / archive copy for the programme |
| **`B01821011_Final_Report.pdf`** | **Optional** PDF export of the same report (if generated and committed for convenience) |
| **`Arka_Talukder_Dissertation_Final_DRAFT.docx`** | Generated from root `Arka_Talukder_Dissertation_Final_DRAFT.md` by `python scripts/dissertation_to_docx.py` or `python scripts/sync_dissertation_and_docx.py` (may mirror earlier naming) |
| **`Arka_Talukder_Dissertation_Final_Submission.docx`** | Optional **historic** filename for a hand-finished / Turnitin copy — use the **B01821011_…** file above if that is your official final |

**Start-here guide for GitHub / coordinator / supervisor:** [`../project_portfolio/README.md`](../project_portfolio/README.md)

## Other typical files

| Item | Role |
|------|------|
| `Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx` | **Humanized** Word export (if you use the humanized drafting track) |
| `thesis_artifacts/01_Humanized_Updated.docx` | **Primary** humanized Word output (regenerated with `sync_humanized_md_and_docx.py`), if used |
| *(optional build)* | `thesis_artifacts/02_Draft_Prose_Merged.docx` only if you run merged-draft tooling — see `docs/reports/DRAFT_OLD_MERGE_POLICY.md` |
| **`forms/`** | Moodle / school templates: front sheet, library form, final-report guideline, signed **project specification** |

## Sync workflow

- After editing the main thesis Markdown at the repo root, run:

  ```bash
  python scripts/sync_dissertation_and_docx.py
  ```

  That regenerates Word when possible and refreshes `supervisor_package/01_Dissertation/` if present.

- For humanized-only regeneration (writes **`thesis_artifacts/01_Humanized_Updated.docx`** and copies under `submission/` when configured):

  ```bash
  python scripts/sync_humanized_md_and_docx.py
  ```

## Source of truth

- **Text and structure:** `Arka_Talukder_Dissertation_Final_DRAFT.md` at the **repository root** (figure paths stay `assets/...` and `results/...`).
- **Canonical process/attendance appendices** embedded by the exporter: `archive/process_attendance/` (see `archive/README.md`).

## Supervisor / viva

For **what to show on screen** (and what to keep closed) during a meeting, see **[`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md)**.

To **audit** the final Word (read-only), run: `python scripts/audit_b018_submission_docx.py` (see script for the expected filename).
