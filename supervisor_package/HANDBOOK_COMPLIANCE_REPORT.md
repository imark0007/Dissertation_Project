# MSc Handbook & Thesis Writing Rules — Compliance Report

**Document checked:** Final report (Arka_Talukder_Dissertation_Final_DRAFT.md → submission/Arka_Talukder_Dissertation_Final_DRAFT.docx)  
**Reference:** MSc Project Handbook 2025-26, SUBMISSION_CHECKLIST.md, DISSERTATION_REVIEW_REPORT.md

---

## 1. Structure (Handbook Section Mapping)

The handbook expects: Introduction, Background, Description of work, Analysis/Evaluation, Conclusion, Critical Reflection.

| Handbook section        | Dissertation section                              | Status   |
|-------------------------|----------------------------------------------------|----------|
| Introduction            | **Chapter 1**                                    | ✓ Match  |
| Background              | **Chapter 2** Literature Review                    | ✓ Match  |
| Description of work     | **Chapters 4–6** (design, methodology, implementation) | ✓ Match  |
| Analysis/Evaluation     | **Chapters 7–8** (protocol, results)               | ✓ Match  |
| Conclusion              | **Chapter 9**                                    | ✓ Match  |
| Critical Reflection     | **Chapter 10**                                   | ✓ Match  |
| References              | **Chapter 11** (before appendices)               | ✓ Match  |
| Appendices              | **Chapter 13:** A–E + handbook mapping; **C** = reproducibility; **D** = code figures; **E** = optional links; Word also merges **Full text —** embedded .docx | ✓ Match  |

**Verdict:** Structure aligns with the handbook.

---

## 2. Marking Criteria Alignment (see **§1.6** and Appendix B specification)

The dissertation explicitly states alignment with the University marking scheme:

- Introduction (5%): aim, research question, scope — ✓
- Literature Review (20%): critical analysis, gap — ✓
- Research Design & Methodology (20%): justified design — ✓
- Implementation (25%): documented, replicable — ✓
- Evaluation (5%) + Results (5%): objective evidence — ✓
- Discussion: interpretation, strengths/limitations — ✓
- Conclusion (10%): summary, future work — ✓
- Critical Self-Evaluation (10%): honest reflection — ✓

**Verdict:** Marking criteria are addressed and stated in **§1.6** (and tied to the agreed specification in **Appendix B**).

---

## 3. Document Requirements

| Requirement              | Handbook / checklist       | Current status                          | Match? |
|--------------------------|----------------------------|-----------------------------------------|--------|
| Abstract                  | ≤200 words                 | **200** words (Apr 2026 count)          | ✓      |
| Word count (main body)    | ~18,000 words (if required)| **~15,900** (Ch 1–10; `scripts/count_dissertation_words.py`) | ⚠ Confirm spec / handbook |
| Line spacing              | 1.5 lines                  | Set in docx script (1.5)                 | ✓      |
| Font                      | 11pt or greater             | 11pt body (Times New Roman), 12pt headings (Arial) | ✓ |
| Numbered figures/tables   | e.g. Figure 2.1, Table 3.1| Uses “Figure 1”, “Table 1” (sequential) | ⚠ Check handbook |
| Table of contents         | With page numbers          | TOC present; page numbers “verify in Word” | ⚠ Verify in Word |
| Tables of figures/tables   | If used                    | Present (Fig 1–16 + appendix figs 17–22; Tables 1–7) | ✓      |
| References                | Harvard, before appendices  | §11 References, Harvard style            | ✓      |
| Front matter              | Front sheet, Declaration, Library form | Placeholders in docx; download from Moodle | ⚠ Complete before submit |
| Appendices                | Process, spec, code figs, repro, optional | Ch 13 + `dissertation_to_docx.py` merges **Full text —** .docx after Appendix E | ✓ / ⚠ Update TOC in Word |

**Verdict:** Most requirements met. **Optional gap:** word count **~16k** vs **~18k** if your programme mandates the higher figure; confirm **Figure 8.1**-style numbering only if handbook requires it; **TOC and front matter** need finalisation in Word before submission.

---

## 4. Thesis Writing Rules (Quality)

| Rule                         | Status |
|-----------------------------|--------|
| Clear aim and research question | ✓ Ch 1 §1.3 |
| Critical analysis of literature (not only description) | ✓ Ch 2 (e.g. gap §2.8; Table 5 in literature) |
| Justified methodology, reproducibility | ✓ Ch 4, Appendix C (seed, config, commands) |
| Implementation linked to research questions | ✓ Ch 6; results Ch 8 |
| Objective evidence (metrics, tables, figures) | ✓ Ch 7–8 |
| Critical interpretation of findings | ✓ Ch 9 |
| Honest reflection and limitations | ✓ Ch 9–10 |
| References in Harvard style | ✓ §11 |
| No plagiarism; Turnitin <10% | — Verify on submission |

**Verdict:** Thesis writing rules are satisfied.

---

## 5. Gaps and Recommended Actions

### 5.1 Word count (~18,000)

- **Current (Apr 2026):** **~15,900** words for **Chapters 1–10** (excluding references, bibliography, appendices) — see `python scripts/count_dissertation_words.py`.
- **Action:** Expand toward **~18,000** **only if** your **signed specification** or programme explicitly requires it; add substance in literature critique, design justification, or limitations/future work — avoid padding (`docs/planning/PREMIUM_THESIS_ROADMAP.md`).

### 5.2 Figure and table numbering

- **Checklist example:** “Figure 2.1, Table 3.1” (section.number).
- **Current:** “Figure 1” … “Figure 22”, “Table 1” … “Table 7” (document-wide sequence).
- **Action:** Confirm in the actual MSc Project Handbook 2526 whether numbering must be section-based (e.g. Figure 7.1, Table 7.1). If yes, renumber figures/tables in the MD and regenerate the DOCX.

### 5.3 In Word before submission

1. Replace placeholder pages with the downloaded Moodle forms (front sheet, declaration, library release).
2. Update Table of Contents and Table of Figures/Tables so page numbers are correct (Word: Update Field).
3. Confirm 1.5 line spacing and 11pt font throughout.
4. Confirm embedded **Full text —** sections (process, attendance, specification) appear after Appendix E in the exported Word file; re-merge if you moved sections manually.

---

## 6. Summary

| Area              | Matches handbook? |
|-------------------|--------------------|
| Section structure | ✓                  |
| Marking criteria  | ✓                  |
| Abstract length   | ✓                  |
| Format (spacing, font) | ✓ (in docx)  |
| References (Harvard, before appendices) | ✓ |
| Appendices structure | ✓             |
| Word count        | ⚠ ~16k vs ~18k — confirm requirement |
| Figure/table numbering | ⚠ Confirm “2.1” style |
| Front matter & TOC | ⚠ Finalise in Word |

**Overall:** The final report is aligned with the MSc handbook and thesis writing rules in structure, content, and most formatting. Remaining items are: confirm word count and figure/table numbering with the handbook, then complete front matter and TOC/page numbers in the final Word document before submission.

---

*Use this report alongside SUBMISSION_CHECKLIST.md and DISSERTATION_REVIEW_REPORT.md when preparing the final submission.*
