# Final Report Generation — Procedure and Deliverables

This document describes the procedure used to produce the final dissertation report in both **Markdown (MD)** and **Word (DOCX)** form, following the Premium Thesis Roadmap and submission checklist.

---

## Deliverables

| Format | File | Description |
|--------|------|-------------|
| **MD** | `Dissertation_Arka_Talukder.md` | Single source of truth for the dissertation text (figures, tables, appendices references). |
| **DOCX** | `Arka_Talukder_Dissertation_Final.docx` | Word document generated from the MD for submission (Turnitin, Moodle, module co-ordinator). |

Both files are the **final report**; the DOCX is derived from the MD.

---

## Procedure Followed

1. **Ablation (Priority 1 evidence)**  
   - Run: `python scripts/run_ablation.py --config config/experiment.yaml`  
   - Trains GAT-only variant (no GRU), evaluates on test set, writes:  
     - `results/metrics/ablation_gat_only.json`  
     - `results/metrics/ablation_table.csv`  
   - If the script stopped before writing metrics but the checkpoint exists, run:  
     `python scripts/eval_ablation_from_ckpt.py --config config/experiment.yaml`

2. **Update Table 4 in the dissertation (MD)**  
   - Run: `python scripts/update_dissertation_table4.py`  
   - Reads `results/metrics/ablation_table.csv` (or `ablation_gat_only.json`) and replaces the “GAT only (no GRU)” row in §7.6 (Table 4) in `Dissertation_Arka_Talukder.md`.

3. **Generate Word document**  
   - Run: `python scripts/dissertation_to_docx.py`  
   - Reads `Dissertation_Arka_Talukder.md`, applies styles (1.5 line spacing, 11pt font), embeds figures and tables, and writes `Arka_Talukder_Dissertation_Final.docx`.

4. **Pre-submission (manual)**  
   - Replace front-matter placeholders with downloaded Moodle forms (front sheet, declaration, library release).  
   - Verify Table of Contents / Table of Figures / Table of Tables page numbers in Word.  
   - Submit DOCX to Turnitin and email to module co-ordinator per `SUBMISSION_CHECKLIST.md`.

---

## Quick Regeneration (after edits to MD)

If you edit only the MD and want to refresh the DOCX:

```bash
python scripts/dissertation_to_docx.py
```

If you re-run ablation and want to refresh Table 4 and then the DOCX:

```bash
python scripts/update_dissertation_table4.py
python scripts/dissertation_to_docx.py
```

---

## Ablation Results (this run)

Table 4 in §7.6 was filled with:

- **Full (GAT + GRU):** Precision 1.0000, Recall 1.0000, F1 1.0000, ROC-AUC 1.0000, Inference 22.70 ms  
- **GAT only (no GRU):** Precision 0.9923, Recall 1.0000, F1 0.9961, ROC-AUC 1.0000, Inference 16.06 ms  

The small drop in F1 for GAT-only supports the choice of the full dynamic (GAT+GRU) model.

---

*Generated as part of the final report procedure. See also: `SUBMISSION_CHECKLIST.md` (same folder), `../planning/PREMIUM_THESIS_ROADMAP.md`.*
