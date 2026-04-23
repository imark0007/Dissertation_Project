# Supervisor feedback vs final draft (recheck)

**Sources:** `Arka_Talukder_Dissertation_Final_DRAFT.md` and aligned `Dissertation_Arka_Talukder.md`; interim **Action Point** form dated **15 April 2026** (Dr. R. M. Ujjan), as summarised in **Section 3.7**.

**Binary form on disk:** `docs/reference/Arka_Talukder_B01821011-Feedback Action Point Sheet (RU).doc` (not text-readable in the repo; the dissertation body mirrors the form in **§3.7**).

| Theme | Where addressed | Status |
|--------|-----------------|--------|
| Literature: search, synthesis, debate, 15–20 core sources | Ch.2 (e.g. §2.2, §2.8, §2.10) | Covered in draft |
| Methods & deployment: own methodology, tools, baselines, FL, XAI | Ch.4–6; §4.4.1 hyperparameter table | Covered |
| Testbed / validity / resources | §7.4.1, tables in Ch.7 | Covered |
| RQs: explicit answers | §9.3 executive block | Covered |
| Societal / impact note | §9.6 | Covered |
| Abbreviations, Harvard, LOF/LOT, alignment with spec | Front matter, §1.5–1.6, §3.6–3.7 | Covered |
| **IDE, platform, evidence for supervision** | **§3.5** (restored and synced 2026-04-20) | **Fixed** — had been present in the canonical `Dissertation_Arka_Talukder.md` but was **missing** from the DRAFT, causing a **TOC vs body** mismatch; DRAFT now includes §3.5 and §3.6–§3.8 renumbered to match the TOC. |
| Self-evaluation / programme module mapping | Ch.10; garbled Ch.4–Ch.3 cross-ref sentence | **Fixed** — Ch.10 programme paragraph was corrected to match **Section 3.7** and handbook/front matter (no broken “Chapter 3.6 / 3.8” sentence). |

## Still student–supervisor (not in files)

- Final **UWS** template spacing/fonts if the School publishes one.  
- **Turnitin** &lt; 10% (per School rules and exclusions).  
- **Mendeley / Zotero / EndNote** tidy for Ch.11–12.  
- **MS Teams** or agreed channel for any last review.

## Full-draft recheck (2026-04-20, second pass)

- **Citations vs Chapter 11:** `python scripts/audit_citations_and_refs.py Arka_Talukder_Dissertation_Final_DRAFT.md` — **no missing** in-text / reference-list pairs. (The script can occasionally list *unused* for heavily cited surnames; **Velickovic** and **Kokhlikyan** *are* cited in the body — ignore false *unused* if it appears.)  
- **Chapters 1–13:** headings present in order. **Figures 1–29** caption sequence **contiguous** in Markdown. **Appendix** Figure **A1-1**–**A1-6** present.  
- **Chapter 2** word count **~5.1k+** (handbook **≥15%** band — confirm in **Word** with page %).  
- **Typos / prose** corrected in the DRAFT: `difference.p` / `altered.s` / negative inference typo; **§9.2** opening sentence; **§9.6** ENISA **Chapter 11** (not 12); **§9.7**–**9.9** (blockchain slip, **Han/Ngo** citation form, **there → their**, incomplete **gap** sentence, **9.10** summary); **§10.5**–**10.3** stray suffixes (`fail.aks`, `work.ork`, `intuition.e`, `score.e`, `scalar.r`).  
- **Tables:** some use `Table 2:` (unbold) vs `**Table 1:**` (bold) — **cosmetic**; align in Word **List of Tables** to programme style.  
- **Word-only before submission:** refresh **TOC / LOF / LOT**, Mendeley/Zotero, **Turnitin** &lt; 10% per School rules, final supervisor pass.

## Literature review feedback (2026-04-20)

**Applied in `Arka_Talukder_Dissertation_Final_DRAFT.md`:** Chapter 2 was **replaced** with the full, supervisor-aligned version from `Dissertation_Arka_Talukder.md` (search strategy **IEEE / Springer / MDPI / ScienceDirect** via Scholar; **§2.2.2.1–2.2.2.4** main-technologies **debate**; **subheadings** 2.4–2.7; **Figures 1–5** with in-text and caption **sources**; **§2.8.1–2.8.4** significance, limits, contribution, counterarguments; **~15%+** callout in **§2.1**). **ECS** in **§2.4.1** now has an in-text cite **(Elastic N.V. 2023)** with a **Chapter 11** reference entry (not duplicated in Chapter 12). **Regenerate Word** from the updated Markdown before submission.
