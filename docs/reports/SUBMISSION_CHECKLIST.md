# MSc Final Report Submission Checklist

Based on MSc Project Handbook 2025-26 and distinction-level marking criteria. Refresh `B01821011_Final_Report_Package_for_Supervisor/SUBMISSION_CHECKLIST.md` from this file before sending the supervisor zip.

## Document Requirements

- **Abstract:** ≤200 words ✓ (current: **180** body words; **196** incl. Keywords; `python scripts/count_abstract_words.py`)
- **Word count (main body):** Handbook often cites **~18,000** words (excluding references and appendices). **Current estimate (Ch 1–10 only):** **~15,900** words (`python scripts/count_dissertation_words.py`). If your **signed specification** or programme explicitly requires ~18k, expand thin sections (literature critique, methodology justification, limitations) — prefer quality over padding.
- **Ablation (Table 4):** ✓ Filled from `results/metrics/ablation_table.csv`; **Chapter 8** §8.7 (`python scripts/run_ablation.py`).
- **Sensitivity + multi-seed:** ✓ **Chapter 8** §8.8–8.9; `scripts/run_sensitivity_and_seeds.py`.
- **Line spacing:** 1.5 lines (set in `scripts/dissertation_to_docx.py`)
- **Font:** 11pt+ body (Times New Roman), headings Arial 12pt (docx script)
- **Numbered figures and tables:** Thesis uses **Figure 1…** / **Table 1…** globally. **Confirm** with the printed handbook whether **Figure 8.1**-style numbering is mandatory; if yes, renumber in Markdown and regenerate Word.
- **Table of contents** + **lists of figures/tables:** Present in MD; **update fields in Word** for correct page numbers.
- **References:** Harvard style, **Chapter 11** before appendices ✓

## Front Matter (download from Moodle)

- Front sheet (MSc Project - Front sheet for final report)
- Signed Declaration of originality
- Signed Library Release form

Reference copies of forms (not substitutes for Moodle) are in **`docs/reference/school_templates/`**.

## Appendices (thesis = `Dissertation_Arka_Talukder.md` → Word)

| Thesis appendix | Handbook (typical) | Content |
|------------------|--------------------|---------|
| **A** | Appendix 3 (process) | Pointers to process documentation + attendance |
| **B** | Appendix 2 (specification) | Signed agreed specification filename |
| **C** | — (extra) | Reproducibility commands |
| **D** | Appendix 1 (code) | Figures A1-1–A1-6 + captions + interpretations; PNGs `results/figures/appendix1/` |
| **E** | Appendix 4 (optional) | GitHub + CICIoT2023 URL; video placeholder |

**Markdown + Word together:** Whenever you change `Dissertation_Arka_Talukder.md`, run `python scripts/sync_dissertation_and_docx.py` so the Word file and the copy under `B01821011_Final_Report_Package_for_Supervisor/01_Dissertation/` stay aligned (that script runs `dissertation_to_docx.py` and copies both `.md` and `.docx` into the package folder when it exists).

**Word export (`dissertation_to_docx.py`):** After Appendix E, the script merges **Full text — …** sections for process documentation, attendance, and specification (see Chapter 13 note) so appendix letters are not duplicated.

## Submission

- Submit to **Turnitin** via Aula (FINAL submission point)
- Email copy to module co-ordinator ([daune.west@uws.ac.uk](mailto:daune.west@uws.ac.uk))
- Hard copies if requested by supervisor
- Turnitin similarity: follow programme guidance (checklist historically cited &lt;10% overall; confirm current rule)

## Post-Submission

- Arrange **viva voce** with supervisor and moderator (typical window communicated by School)
- Be prepared to **demonstrate software** and answer detailed questions

## Handbook structure mapping (this dissertation uses **Chapters 1–10**)

| Handbook section | Dissertation |
|------------------|--------------|
| Introduction | **Chapter 1** |
| Background / literature | **Chapter 2** |
| Project management (if required) | **Chapter 3** |
| Research design & methodology | **Chapter 4** |
| Design / implementation | **Chapters 5–6** |
| Testing & evaluation protocol | **Chapter 7** |
| Results | **Chapter 8** |
| Conclusion, discussion, recommendations | **Chapter 9** |
| Critical self-evaluation | **Chapter 10** |
| References | **Chapter 11** |
| Bibliography (if used) | **Chapter 12** |
| Appendices | **Chapter 13** |

## Distinction-level criteria (from marking grid / §1.6)

- Clear aim, research question, scope
- Critical analysis of literature (not just description)
- Justified methodology supporting reproducibility
- Implementation linked to research questions
- Objective evidence presentation
- Critical interpretation of findings
- Honest reflection on process and limitations
- Independent thinking and clear line of argument
