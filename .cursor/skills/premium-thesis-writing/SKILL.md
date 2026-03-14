---
name: premium-thesis-writing
description: Guides writing and editing of MSc/dissertation final reports to high-class, distinction-level standard, and the transition from thesis to journal publication. Use when drafting or revising thesis chapters, improving argument and evidence, checking structure and clarity, polishing for submission, planning publication from the same work, or when the user asks for premium thesis writing, distinction-level writing, final report quality, or thesis-to-publication.
---

# Premium Thesis Writing

Apply this skill when writing or editing the final dissertation so it meets **distinction-level** and **premium** standards. Do not change the technical content or delete sections; improve clarity, argument, and alignment with marking criteria.

---

## What “Premium Thesis” Means

1. **Distinction criteria** — Clear aim, critical literature, justified method, strong evidence, honest limitations, independent thinking.
2. **Rigour** — Not only “we did X” but “we showed X, tested why it works (e.g. ablation), and checked robustness where possible.”
3. **Reproducibility** — Config, seed, code, and commands so someone could re-run the work (e.g. Appendix C).
4. **One clear story** — Problem → gap → your approach → evidence → interpretation → limitations → future work (including publication).

---

## Structure and Argument

- **Introduction:** One crisp paragraph chain — problem → gap in literature → your contribution (3–4 bullets) → research question(s). No extra experiments; scope and limitations stated clearly.
- **Literature Review:** Critical analysis and comparison of existing work; state the **gap** your project fills. Avoid long descriptive lists; synthesise and evaluate.
- **Methodology / Design:** Justify choices (dataset, graph design, models, FL setup, metrics). Link each choice to the research questions. Support reproducibility (seed, config, commands).
- **Implementation:** Document the build so the work can be assessed and replicated. Link design decisions to research questions.
- **Results:** Present evidence objectively — tables, figures, metrics. Reference every table/figure in the text. State limitations (e.g. subset size, number of clients).
- **Discussion:** Interpret findings. For each main result (main table, ablation, FL, alerts), add 1–2 sentences: “This shows that…”; “The drop when we remove the GRU suggests…”; “Limitation: …”. Relate to literature.
- **Conclusion:** Summary of what was achieved; short “Future work” that can explicitly mention journal submission (e.g. MDPI/IEEE) with same dataset and extra validation (multi-seed, per-attack).
- **Critical Self-Evaluation:** Honest reflection — what went well, what was hard, what you would do differently. Specific, not generic.

---

## Evidence and Objectivity

- Report metrics as stated in the evaluation plan (precision, recall, F1, ROC-AUC, inference time, etc.). Do not invent numbers.
- Every claim about performance or comparison should be backed by a table, figure, or explicit metric.
- State limitations explicitly (e.g. single dataset subset, few FL clients, no user study). This strengthens the thesis.
- Ablation and (if present) sensitivity subsections should have one table each and short interpretation (why the design is justified, or how robust it is).

---

## Critical Analysis and Limitations

- **Literature:** Compare and critique related work; do not only describe. Show why your approach is justified.
- **Interpretation:** Discuss what results mean, not only what they are. Link to research questions.
- **Limitations:** Acknowledge scope, dataset, and methodological constraints. Be honest in the self-evaluation.

---

## Premium Thesis Checklist

Use this when reviewing or editing the final report:

**Aim and scope**
- [ ] Clear aim and research question(s) stated in the introduction
- [ ] Scope and limitations stated (e.g. dataset subset, 45-day project)

**Literature and method**
- [ ] Critical analysis of literature (gap and contribution clear)
- [ ] Methodology justified and linked to research questions
- [ ] Reproducibility: seed, config path, and exact commands (e.g. in Appendix C)

**Evidence**
- [ ] Results presented objectively with tables/figures
- [ ] Every table and figure referenced and interpreted in the text
- [ ] Ablation (and sensitivity if present) with table + short interpretation

**Discussion and conclusion**
- [ ] Discussion interprets findings and states limitations
- [ ] Conclusion summarises achievements and states future work (e.g. publication)
- [ ] Critical self-evaluation is specific and honest

**Submission standards**
- [ ] Abstract ≤200 words; word count in range required by handbook (~18,000 if specified)
- [ ] 1.5 line spacing, 11pt+ font, numbered figures/tables
- [ ] References in Harvard style, before appendices
- [ ] Front matter and appendices as per handbook (front sheet, declaration, process docs, attendance, project spec)

---

## Style and Tone

- Use clear, formal academic prose. Simple language is fine (e.g. IELTS Band 6–style) as long as it is precise and consistent.
- Prefer active voice where natural (“The model achieves…” rather than “It is achieved by the model…”).
- Use UK spelling consistently (e.g. “analyse”, “summarise”, “prioritise”).
- Avoid padding; add content only where it adds argument, evidence, or clarity.

---

## Project References

- **Submission and format:** See [SUBMISSION_CHECKLIST.md](../../SUBMISSION_CHECKLIST.md) for document requirements and distinction criteria.
- **Strategy and priorities:** See [docs/PREMIUM_THESIS_ROADMAP.md](../../docs/PREMIUM_THESIS_ROADMAP.md) for the four priorities (evidence, story, submission standards, self-evaluation).
- **Handbook alignment:** See [HANDBOOK_COMPLIANCE_REPORT.md](../../HANDBOOK_COMPLIANCE_REPORT.md) for structure mapping and gaps.
- **Thesis to publication:** See [docs/PUBLICATION_AND_THESIS_MASTER_PLAN.md](../../docs/PUBLICATION_AND_THESIS_MASTER_PLAN.md) for the full thesis-to-publication strategy and venue requirements.

When editing, preserve the existing section numbering and structure; improve clarity, argument strength, and alignment with this checklist and the referenced project docs.

---

## Thesis → Publication (Always Learning, Future Use)

The same dissertation content is designed to **feed directly into a journal paper**. The skill should be updated as the author moves from thesis submission to publication; the following is the bridge.

**Principle:** One premium thesis = one clear story + strong evidence. The same story, tables, and figures become the core of the paper. No re-running experiments for the first submission; reformat and target the venue.

**What carries over (thesis → paper):**
- **Introduction:** Problem → gap → contribution (condense to ~1 page for paper).
- **Related Work:** Literature review becomes “Related Work” (~0.5–1 page); same critical comparison and gap.
- **Method:** Research Design + Implementation become “Methodology” (~2 pages); same dataset, models, FL, explainability, metrics.
- **Results:** Same tables (model comparison, ablation Table 4, FL rounds, training history) and figures (ROC, confusion matrices, FL convergence, model comparison). Add multi-seed or per-attack only if already done for thesis.
- **Discussion:** Same interpretation and limitations; condense.
- **Conclusion:** Same summary + future work; paper may add “Code and data availability” (point to repo).

**What to add for publication (when the time comes):**
- Journal-specific format (MDPI/IEEE template), 150–200 word abstract, structured sections.
- If not already in thesis: multi-seed (mean ± std), per-attack-type metrics, or one extra baseline from literature. Add one sentence/table to thesis if possible so thesis and paper stay aligned.
- “Code and data availability” / reproducibility statement (repo, config, seed, commands — already in thesis Appendix C).

**Updating this skill:** When the author starts publication (e.g. MDPI Sensors, IEEE Access), add a short “Publication checklist” to this skill (venue template, abstract length, reference style) and keep the same evidence and story. The thesis is the trained base; the paper is the same work, reformatted and slightly extended.
