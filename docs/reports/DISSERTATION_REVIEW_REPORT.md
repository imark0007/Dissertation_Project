# Dissertation Review Report (Teacher/Examiner Perspective)

**Document:** Dissertation_Arka_Talukder.md  
**Date:** Pre-submission review  
**Purpose:** Identify errors and issues before final submission

---

## Critical Errors (FIXED)

### 1. Random Forest tree count mismatch
- **Location:** Section 5.4 Model Implementation
- **Issue:** Stated "100 trees" but config/experiment.yaml and Section 6.1 specify 200 trees
- **Fix applied:** Changed to 200 trees

### 2. MLP architecture mismatch
- **Location:** Section 5.4 Model Implementation
- **Issue:** Stated "two hidden layers (256 and 128 units)" but config specifies [128, 64, 32] (three layers)
- **Fix applied:** Changed to "three hidden layers (128, 64, and 32 units)"

### 3. Duplicate Ablation studies in Recommendations
- **Location:** Section 9.2
- **Issue:** Ablation studies appeared twice (inside Optimisation bullet and as separate bullet)
- **Fix applied:** Merged into single Ablation studies bullet

### 4. Incorrect API path
- **Location:** Section 5.8
- **Issue:** Referenced "src/api/" but API is in src/siem/api.py
- **Fix applied:** Changed to src/siem/api.py

---

## Submission Checklist Gaps

| Requirement | Status | Action |
|-------------|--------|--------|
| Abstract ≤200 words | ✓ 198 words | OK |
| Word count ~18,000 | ⚠ ~14,000 | Consider expanding Literature Review or Discussion if distinction target |
| Line spacing 1.5 | — | Verify in Word |
| Font 11pt+ | — | Verify in Word |
| Numbered figures (e.g. Figure 2.1) | ⚠ Uses "Figure 1", "Figure 2" | Handbook suggests "Figure 2.1" style; check if required |
| Table of contents with page numbers | — | Add in Word after final formatting |
| References before appendices | ✓ | OK |
| Front sheet, Declaration, Library form | — | Download from Moodle, embed |
| Appendices embedded | — | Embed process docs, attendance, spec in final docx |

---

## Minor Suggestions

1. **Figure order in Results:** Figures appear as 2, 3, 6, 7, 8, 9, 4, 5 (not numerical order). Grouping by content is fine; consider adding "see Figure X" cross-references if examiner expects sequential flow.

2. **Example alerts vs. alert_summary numbering:** The dissertation Examples 1–5 are ordered by type (TP, TN, TP, FP, FP). The underlying alert_summary uses Alert 1–5 by generation order. The mapping is correct but could confuse if someone cross-checks. No change needed if content matches.

3. **"prioritise" / "summarise":** UK spelling is consistent throughout. ✓

4. **Module co-ordinator email:** daune.west@uws.ac.uk is in SUBMISSION_CHECKLIST. Confirm this is current.

---

## Data Verification

- **Table 1 (Model comparison):** Matches results_table.csv ✓
- **Table 2 (FL rounds):** Values consistent with narrative ✓
- **Table 3 (GNN training):** Values consistent ✓
- **Alert examples:** Scores and features match alert_summary.txt ✓

---

## Pre-Submission Actions

1. Regenerate Word document: `python scripts/dissertation_to_docx.py`
2. Embed front sheet, Declaration, Library Release form from Moodle
3. Embed appendices (process docs, attendance, project spec)
4. Verify 1.5 line spacing and 11pt font in Word
5. Add page numbers to Table of Contents, Table of Figures, Table of Tables
6. Run Turnitin check; aim for <10% similarity
7. Email to module co-ordinator
8. Arrange viva voce within 10–14 days

---

*Report generated from pre-submission review. All critical errors have been fixed in Dissertation_Arka_Talukder.md.*
