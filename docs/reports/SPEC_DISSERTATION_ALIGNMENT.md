# Specification vs dissertation alignment

**Generated from:** **Formal specification** (`Arka-B01821011_MSc Cyber Security Project specification_form_2025-26.docx`)

## Automated keyword coverage

| Commitment (from source doc) | In dissertation? | Notes |
|------------------------------|------------------|-------|
| Primary research question (IoT / GNN / FL / SIEM / CPU) | **Yes** | Core RQ wording from interim/spec should appear in Introduction. |
| Sub-questions / hypotheses (GNN vs baselines; federated; explanations) | **Yes** | Baselines, federation, and explainability addressed in Ch.1 / Ch.4 / Ch.8–9. |
| CICIoT2023 dataset | **Yes** | Dataset named and scoped. |
| Flower / FedAvg | **Yes** | Federated stack documented (Ch.4, 6, 8). |
| Integrated Gradients / Captum | **Yes** | Explainability pipeline. |
| FastAPI / ECS / JSON alerts | **Yes** | Deployment and SIEM-shaped output. |
| Ablation (temporal / GAT) | **Yes** | §8.7 and Table 4. |
| Sensitivity (window / kNN k) | **Yes** | §8.8 Table 6 and Figure 15. |
| Multi-seed | **Yes** | §8.9 Table 7 (seeds 42, 123, 456). |
| Ethics / public data / no human participants | **Yes** | Ch.3 §3.4; align with ethics box on signed spec when available. |
| Marking criteria / specification grid | **Yes** | §1.6 mapping; verify percentages against signed PDF. |

## Interim-specific promises vs final dissertation

- **H2: federated GNN within 2% F1 of centralised GNN:** Done — Final Ch.8: federated matches central (100% F1) — satisfies stricter bar.
- **3–5 example alerts with explanations:** Done — §8.6 lists five examples.
- **Confusion matrices, ROC, FL convergence, inference/F1 bar chart:** Done — Figures 2–9 and 4–5.
- **Contingency: drop sensitivity if out of time:** Done — Sensitivity §8.8 completed — exceeds interim contingency.

## Specification form: structured gaps / clarifications

- **Graph design vs overview text:** The specification overview assumes **device nodes** and **flow edges** where identifiers exist; the thesis implements **flow nodes** and **kNN edges** for the public CICIoT2023 release. **Status:** §1.4 now ties this explicitly to the signed form in **Appendix B** and supervisor agreement.
- **CyBOK:** **§2.9** maps the dissertation to CyBOK knowledge areas consistent with the specification (Appendix B).
- **Ethics sign-off:** The specification Section 2 states no full ethics-board approval is required. **Status:** §3.4 mirrors that sign-off (date aligned to the form).

## Manual follow-ups

1. **Marking percentages:** If the form’s marking grid differs from §1.6, update §1.6 and Appendix B.
2. **ERM ID:** If the form ever references a specific ERM case number, repeat it verbatim in §3.4.
3. **Timeline:** Interim used a *15-week* framing; the thesis uses *45 days* for the build sprint — ensure this matches what your supervisor approved on the form.

---

*Regenerate: `python scripts/compare_spec_to_dissertation.py`*