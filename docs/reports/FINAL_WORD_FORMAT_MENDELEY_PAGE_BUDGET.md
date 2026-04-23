# Final Word: format, reference manager, and page-% checks

This is a **process note** (not part of the dissertation body). It supports the **MSc Project Handbook 2025–26** and supervisor action points.

## 1. Body style (11pt, 1.5 spacing, heading hierarchy)

- Regenerated reports from `scripts/dissertation_to_docx.py` use:
  - **Body:** Times New Roman **11pt**, **1.5** line spacing, **justified** (Normal style).
  - **Headings:** Heading 1–4 tuned in the same script; chapter titles in Markdown `##` stay **centred** and **16pt** where the exporter applies that override.
- **After** opening the `.docx` in Word, select **Home → Styles →** apply **Normal** to any pasted text; confirm **Styles** pane shows a clear **Heading 1 → 2 → 3** ladder for the navigation pane.
- If the **sample** dissertation uses slightly different point sizes for headings, adjust **only** heading styles in Word to match the sample—do not change the meaning of the body text.

## 2. Mendeley / Zotero / EndNote (final reference list)

- **Workflow:** keep **one** library (Mendeley, Zotero, or EndNote) as the **source of truth** for Harvard (UWS) entries.
- **Options:**
  - **Zotero:** use the UWS CSL or Harvard style from the Library libguide; use **Document Preferences** in the Word plugin to pin the style before insert bibliography.
  - **Mendeley / EndNote:** use the UWS-recommended **Harvard** output; **refresh** citations after any edit.
- The Markdown **Chapter 11** list is a **failsafe**; the **examinable** list is the one **in Word** produced or checked with your reference manager (Turnitin and moderators expect consistency).

## 3. Page % vs. the 10% / 15–20% table (Section 1.5)

- Markdown **cannot** control pagination. **Check only in Word/PDF** after the last edit.
- **Practical check:**
  1. **Insert →** **Page break** so front matter, chapters, and references are stable.
  2. Note **total** pages of the main body (or the pages the module counts).
  3. For each **chapter** in the table in **§1.5**, read **(chapter last page − chapter first page + 1)** or use a **section break** per chapter and look at the status bar in Print Layout.
  4. **Approximate %** = 100 × (chapter pages ÷ main-body pages). Adjust by **editing** only where a section is off-band—**not** by padding.
- The **formal** marks may still follow the **signed Project Specification** (Appendix B); treat the **% table** in §1.5 as **alignment with supervisor guidance**, not a guarantee of the numeric mark.

## 4. Sample PDF (Nouman) and structure

- The Turnitin **PDF** in the repo is often a **cover** + metadata; it may **not** show the inner thesis order. If you need to **mirror** front-matter order or chapter titling, use the **original Word** sample (if the author shared it) or the **MSc Project Handbook** front pages (Appendix IV / V) as the authority.
