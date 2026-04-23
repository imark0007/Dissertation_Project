# Merge policy: `Arka_Talukder_Dissertation_Final_DRAFT_old.docx` + current final report

## What you asked for

- **New final report** content where supervisor/programme **instructions** required updates (method, deployment, valid data pipeline, **Tables 1–13**, Ch 3.5, references, no incorrect split story, etc.).
- **Keep** the **previous humanized** wording where the new report did not replace that material—mainly the **narrative voice** in the **front of Chapter 1** and **Acknowledgements**.

## What was generated

| File | Role |
|------|------|
| `thesis_artifacts/Dissertation_Arka_Talukder_DraftProse_Merged.md` | **Source** for the merge (canonical body + DRAFT-old Ack + §1.1–1.2) |
| `thesis_artifacts/02_Draft_Prose_Merged.docx` | **Word** export; a **copy** is also written to `submission/Arka_Talukder_Dissertation_Final_DraftProse_Merged.docx` when you use `--docx` |

## What was taken from the old DRAFT (humanized)

- **Acknowledgements** (three paragraphs from `Arka_Talukder_Dissertation_Final_DRAFT_old.docx`)
- **§1.1 Chapter Overview** and **§1.2 Background and Motivation** (full humanized prose)
- **Minor edit:** moderator thanks line uses *who reviewed* (not *as she*) so the line stays accurate regardless of moderator preference.

## What stays from the **current** canonical `Dissertation_Arka_Talukder.md`

- **Abstract**, **§1.3** onwards (including marking-criteria **§1.6** in the full spec version), **Chapters 2–13**
- All **method**, **results**, **tables**, and **invalid-reference fixes** the supervisor required

**Important:** the old DRAFT had **out-of-date** methodology (e.g. 70/15/15 wording in the old Ch4) — that is **not** reintroduced. Only the **introductory** humanized prose in §1.1–1.2 and **Ack** are restored.

## How to refresh the merged files

```bash
python scripts/build_draftprose_merged_md.py --docx
```

## Other tracks (unchanged)

- **Canonical** only: `Dissertation_Arka_Talukder.md` → `submission/Arka_Talukder_Dissertation_Final.docx` (`sync_dissertation_and_docx.py --final-only`)
- **Humanized aligned to canonical** (one-line preface only): `Dissertation_Arka_Talukder_Humanized.md` + `thesis_artifacts/01_Humanized_Updated.docx` (`sync_humanized_md_and_docx.py`)
