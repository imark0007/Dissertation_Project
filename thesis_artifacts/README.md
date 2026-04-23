# Principal thesis handoff files

This folder holds **synced copies** of the main thesis Markdown/Word so the repo can keep a clear handoff location alongside `submission/` and `supervisor_package/`.

| File | Role |
|------|------|
| `Arka_Talukder_Dissertation_Final_DRAFT.md` | **Copy** of root `Arka_Talukder_Dissertation_Final_DRAFT.md` (refreshed by `python scripts/sync_dissertation_and_docx.py`; edit the root file, not this copy) |
| `01_Humanized_Updated.docx` | Regenerated from `Dissertation_Arka_Talukder_Humanized.md` via `python scripts/sync_humanized_md_and_docx.py` (also copied to `submission/…_Humanized_version.docx`) |
| `02_Draft_Prose_Merged.docx` + `Dissertation_Arka_Talukder_DraftProse_Merged.md` | **Optional** merge experiment — `python scripts/build_draftprose_merged_md.py --docx` |

**Authoritative thesis source (edit):** `Arka_Talukder_Dissertation_Final_DRAFT.md` at repository root. **Humanized track (edit):** `Dissertation_Arka_Talukder_Humanized.md`.

Superseded or one-off files (e.g. old DRAFT Word) live under `archive/previous_thesis_files/`.
