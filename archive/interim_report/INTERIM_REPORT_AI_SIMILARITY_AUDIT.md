# Interim Report AI/Similarity Audit (Manual Rewrite Guide)

This audit is for your manual rewriting workflow.  
It does **not** change structure, figures, equations, or references.

Legend:
- `[OK]` = specific enough, keep with minor edits.
- `[REWRITE NEEDED - reason]` = likely generic/boilerplate or repetitive.
- `[ADD PERSONAL DETAIL]` = insert your own real project experience.

---

## 1) Hidden text / invisible formatting check

- `Interim_Report_Arka_Talukder.md`: **No hidden characters found** (`U+200B/U+200C/U+200D/U+FEFF/U+2060` all zero).
- `B01821011_Interim_Report_Final.docx`: **No hidden runs** and **no white-colored runs** detected.
- Suspicious formatting flags: **None found**.

---

## 2) Paragraph-by-paragraph tagging (excluding References)

### Introduction

- L17 `[REWRITE NEEDED - standard template phrasing used in many interim reports]`
- L19 `[OK]`
- L21 `[OK]`
- L23 `[ADD PERSONAL DETAIL]` (add one concrete milestone + one issue you faced)

### 1. Summary Literature Review

- L47 `[REWRITE NEEDED - opening sentence is generic literature boilerplate]`
- L57 `[REWRITE NEEDED - high-frequency IoT-security wording used across many papers]`
- L59 `[OK]`
- L61 `[OK]`
- L65 `[REWRITE NEEDED - common baseline description language; vary with your own framing]`
- L67 `[REWRITE NEEDED - long generic review paragraph; break and personalise interpretation]`
- L71 `[REWRITE NEEDED - standard GNN explanation; keep concise + add your project tie-in]`
- L73 `[OK]`
- L75 `[OK]`
- L85 `[REWRITE NEEDED - FedAvg description is textbook style; shorten and personalise]`
- L93 `[OK]`
- L95 `[OK]`
- L99 `[REWRITE NEEDED - common SOC alert-fatigue wording seen in many studies]`
- L101 `[REWRITE NEEDED - standard IG/SHAP comparison language; personalise and shorten]`
- L103 `[OK]`
- L113 `[OK]`
- L128 `[REWRITE NEEDED - repeated gap claim appears in multiple sections]`
- L136 `[OK]`
- L142 `[OK]`

### 2. Research Methodology

- L152 `[REWRITE NEEDED - generic section-introduction boilerplate]`
- L156 `[REWRITE NEEDED - repeated “quantitative, experimental approach” boilerplate]`
- L158 `[OK]`
- L168 `[OK]`
- L177 `[OK]`
- L181 `[REWRITE NEEDED - graph-construction wording repeated elsewhere; reduce duplication]`
- L191 `[OK]`
- L209 `[ADD PERSONAL DETAIL]`
- L211 `[ADD PERSONAL DETAIL]`
- L215 `[REWRITE NEEDED - metric preamble is generic; keep equations, rewrite explanation]`
- L217 `[OK]` (equation)
- L221 `[OK]` (equation)
- L225 `[OK]` (equation)
- L229 `[OK]` (equation)
- L233 `[OK]` (equation)
- L237 `[REWRITE NEEDED - “Additionally” sentence is generic and long]`
- L241 `[REWRITE NEEDED - common explainability method description; shorten + personalise]`
- L245 `[OK]` (equation context)
- L249 `[ADD PERSONAL DETAIL]`
- L253 `[OK]`
- L255 `[REWRITE NEEDED - repeated claim about novelty/value appears in 1.6 and 2.8]`
- L267 `[ADD PERSONAL DETAIL]`
- L271 `[ADD PERSONAL DETAIL]`
- L280 `[OK]`
- L282 `[ADD PERSONAL DETAIL]`

### 3. Plan for Completion

- L288 `[REWRITE NEEDED - generic planning opener]`
- L292 `[OK]`
- L303 `[REWRITE NEEDED - sentence appears in similar form in previous drafts; vary phrasing]`
- L311 `[OK]`
- L319 `[OK]`
- L327 `[OK]`
- L344 `[ADD PERSONAL DETAIL]`
- L351 `[ADD PERSONAL DETAIL]`
- L355 `[OK]`

---

## 3) Similarity reduction targets (where to rephrase manually)

1. **Metric definition text around equations** (L215, L237):  
   Keep equations as-is, rewrite explanatory lines in your own words + cite method sources in-text.

2. **Boilerplate method descriptions**:
   - FedAvg summary (L85)
   - Standard GNN explanation (L71)
   - “quantitative experimental approach” wording (L156)

3. **Repeated novelty claim**:
   - Appears in L128 and L255.  
   Keep full version once (suggest: 1.6), shorten the other occurrence.

4. **Dataset/setup duplication**:
   - Similar setup language appears in L168, L181, L209, L292.  
   Keep full detail in Methodology; reduce repetition in Plan section.

---

## 4) Human voice prompts (already inserted in report)

You now have explicit prompts in:
- `2.5 Data Gathering and Analysis`
- `2.7 Explainability and SIEM Integration`
- `2.8 Academic Worth and Verifiable Goal`
- `2.9 Anticipated Critical Reflection`
- `3.3 How Results Will Be Analysed`

Fill each `[WRITE YOUR EXPERIENCE HERE: ...]` with real events from your project work.

---

## 5) Sentence-pattern variation flags (manual rewrite)

- Section 1.1 has 5+ short declarative sentences in a row.  
  Add one contrast sentence and one cause/effect sentence.
- Section 2.1 has repeated “I use / I run / I measure ...” pattern.  
  Keep some first-person voice, but vary sentence openings.
- Section 3.1 bullet list starts with repeated noun-phrase pattern.  
  Add one reflective sentence after bullets in your own words.

---

## 6) References/citation rule check

- References section left unchanged (as requested).
- In-text citation style is mostly Harvard-compatible.
- During manual rewrite, keep citation consistency:
  - Narrative: `Author et al. (Year) ...`
  - Parenthetical: `(Author et al., Year)`

