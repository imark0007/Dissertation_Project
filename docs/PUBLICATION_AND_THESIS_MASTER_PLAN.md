# Master Plan: Thesis to 90% (Distinction) + High-Class Publication (MDPI / IEEE / Q1)

**Purpose:** Use the same CICIoT2023 work, go deeper, and produce outputs that (1) lift your thesis to distinction-level (≈90%) and (2) make the work publication-ready for MDPI, IEEE, or Q1 journals.

---

## Part 1: Understanding the Venues

### MDPI (e.g. Sensors, Electronics, Applied Sciences)

| Aspect | Detail |
|--------|--------|
| **Scope** | IoT cybersecurity, intrusion detection, ML/GNN, explainability. Special Issues on "IoT Cybersecurity" are common. |
| **Requirements** | Original research, sound experiments, reproducibility (code/data when possible), 150–200 word abstract, structured sections (Introduction, Methods, Results, Discussion, Conclusion). |
| **Strengths for you** | Open access, fast review (~17 days first decision), CICIoT2023 is well accepted, your combination (GNN + FL + explainability + SIEM) fits. |
| **Typical length** | Full article: ~8–12 pages; Communication: shorter. |

### IEEE (e.g. IEEE Access, conference papers)

| Aspect | Detail |
|--------|--------|
| **Scope** | IoT security, GNN-based IDS, federated learning. Many GNN-for-IoT papers in IEEE Xplore. |
| **Requirements** | IEEE template, clear contribution, experimental validation, comparison with baselines. |
| **Strengths for you** | Strong fit for “system” papers (pipeline + FL + edge). Conferences allow 6–8 pages; IEEE Access allows longer. |

### Q1 (top quartile journals)

| Aspect | Detail |
|--------|--------|
| **What they want** | Novelty, rigorous evaluation (ablation, sensitivity, statistical tests), reproducibility, clear gap and contribution. |
| **Your gaps to fill** | Ablation studies, sensitivity analysis (window size, k, rounds), multiple seeds / confidence intervals, comparison with at least one recent GNN/FL baseline from literature. |

**Bottom line:** Your current thesis is a solid **first draft** of a journal paper. To aim for 90% thesis + publication you need: **deeper analysis on the same dataset**, **more tables/figures**, **explicit ablation and sensitivity**, and **clear “future work” that you partially already do**.

---

## Part 2: Thesis 90% (Distinction) vs Publication Readiness

| Marking criterion (thesis) | What examiners want | What also helps publication |
|----------------------------|---------------------|-----------------------------|
| **Critical analysis (Literature)** | Compare methods, identify real gap, not just describe. | Same: sharp related work and contribution statement. |
| **Justified methodology** | Why this dataset, these baselines, these hyperparameters. | Same: Methods section with rationale. |
| **Evidence-based results** | Tables, figures, numbers; limitations stated. | Same: Results + Discussion with stats. |
| **Reproducibility** | Config, code, seed, commands. | Same: MDPI/IEEE often ask for code availability. |
| **Independent thinking** | Your design choices, your interpretation. | Same: clear “we show that…” in paper. |

**Strategy:** Every “deeper” experiment you add for the thesis (ablation, sensitivity, extra table) is also a **direct input** for the publication. One set of experiments serves both.

---

## Part 3: Go Deeper on the SAME Dataset (CICIoT2023)

You do **not** need a new dataset. You need **more structure and depth** on CICIoT2023.

### 3.1 Ablation Studies (what adds what?)

| Experiment | What you do | Thesis impact | Publication impact |
|------------|-------------|---------------|--------------------|
| **Ablation 1: GAT only (no GRU)** | Train GNN with GAT but replace GRU with mean pooling over time. Compare F1/ROC-AUC to full model. | Shows “temporal part matters”. | Required in good ML papers. |
| **Ablation 2: GRU on flat features (no graph)** | Same 46 features per window, no kNN graph; sequence of window vectors → GRU → classifier. | Shows “graph part matters”. | Isolates contribution of graph. |
| **Ablation 3: kNN vs random edges** | Build graph with random edges (same degree) instead of kNN. Compare to kNN. | Justifies kNN design. | Addresses “why this graph”. |

**Output:** One new table (e.g. “Ablation on CICIoT2023 test set”) and one short subsection in thesis + one subsection in paper.

### 3.2 Sensitivity Analysis (robustness)

| Experiment | What you do | Thesis impact | Publication impact |
|------------|-------------|---------------|--------------------|
| **Window size** | Run pipeline for window_size = 30, 50, 70 (fix k=5, seq_len=5). Report F1/ROC-AUC and inference time. | Shows “choice of 50 is reasonable”. | Shows robustness. |
| **k in kNN** | Run for k = 3, 5, 7 (fix window=50). Same metrics. | Justifies k=5. | Standard sensitivity. |
| **FL rounds** | Already have round-by-round; add 5 and 15 rounds (if time) to show “10 is enough”. | Strengthens FL discussion. | Good for FL-focused reviewers. |

**Output:** One table (e.g. “Sensitivity to window size and k”) and one figure (e.g. F1 vs window size). One subsection in thesis + in paper.

### 3.3 Statistical Rigour (optional but strong)

| Experiment | What you do | Thesis impact | Publication impact |
|------------|-------------|---------------|--------------------|
| **Multiple seeds** | Run central GNN (and optionally RF/MLP) for seeds 42, 123, 456. Report mean ± std (or 95% CI) for F1, ROC-AUC. | “Results are stable.” | Q1 reviewers expect this. |
| **Per–attack-type breakdown** | If CICIoT2023 labels allow: report F1 per attack type (e.g. DDoS, Recon, Brute force) for GNN vs RF. | Deeper analysis. | Differentiates from “only binary” papers. |

**Output:** One table (mean ± std over seeds; or per-attack F1). One paragraph in thesis + in paper.

### 3.4 Comparison with “Literature” Baseline (if feasible)

| Experiment | What you do | Thesis impact | Publication impact |
|------------|-------------|---------------|--------------------|
| **Simple GNN baseline** | Implement a plain GCN or GraphSAGE (no GRU, one snapshot per sample) on same graphs. Compare to your GAT+GRU. | “Our dynamic design beats a static GNN.” | Direct comparison with common baselines. |

**Output:** One row in main results table + one sentence in text. Strong for publication.

---

## Part 4: Point-by-Point Action Plan

### Phase A: Experiments (run once, use in thesis + paper)

| Step | Action | Where it goes |
|------|--------|----------------|
| A1 | Add config options for ablation: `use_gru: true/false`, `use_graph: true/false`, `graph_type: knn/random`. | `config/experiment.yaml` + small code changes in model/dataset. |
| A2 | Implement “GAT only” (no GRU): mean pool over time steps, then classifier. Train and evaluate. | New script or flag in `run_all.py`. |
| A3 | Implement “GRU on flat” (no graph): window-level feature vector (e.g. mean of 46 features per window), sequence → GRU → classifier. | New model or flag in baselines. |
| A4 | Run sensitivity: window_size ∈ {30, 50, 70}, k ∈ {3, 5, 7}. Log metrics to JSON/CSV. | New script `scripts/run_sensitivity.py` or extend `run_all.py`. |
| A5 | Run 3 seeds (42, 123, 456) for central GNN (and optionally RF). Compute mean ± std F1, ROC-AUC. | Same pipeline, loop over seeds; aggregate in a small script. |
| A6 | If labels available: compute per–attack-type metrics (e.g. F1 per class) for GNN and RF. | Extend `src/evaluation/metrics.py` and results table. |

### Phase B: Thesis Enhancements (text + figures/tables)

| Step | Action | Section |
|------|--------|---------|
| B1 | Add **Ablation** subsection in Results: table + short commentary (which component contributes most). | §7 (e.g. 7.6 Ablation Studies). |
| B2 | Add **Sensitivity** subsection: table (window, k) + one figure (e.g. F1 vs window size). | §7 (e.g. 7.7 Sensitivity Analysis). |
| B3 | If you did multiple seeds: add one **Stability** sentence + small table (mean ± std). | §6 or §7. |
| B4 | In Discussion: interpret ablation (“temporal modelling adds X%; graph adds Y%”) and sensitivity (“performance stable for window 30–70”). | §8. |
| B5 | In Conclusion/Future work: state “This work is extended toward a journal submission (MDPI/IEEE) with ablation, sensitivity, and statistical validation.” | §9. |
| B6 | Add one or two **comparison with related work** sentences: “Our F1/ROC-AUC is comparable to [Ngo et al. / Basak et al.] on CICIoT2023; we additionally provide FL and explainability.” | §8 (Relation to Literature). |

### Phase C: Publication-Ready Paper (same content, different format)

| Step | Action | Notes |
|------|--------|--------|
| C1 | Short **paper abstract** (150–200 words): problem, approach, main result (with numbers), conclusion. | Can trim from thesis abstract. |
| C2 | **Introduction** (~1 page): motivation, gap, contributions (bullet list), paper structure. | Reuse thesis §2 + §3 gap. |
| C3 | **Related Work** (~0.5–1 page): GNN for IDS, FL for IoT, explainability; table comparing “GNN / FL / Explainability / SIEM” across papers. | Condense thesis §3. |
| C4 | **Methodology** (~2 pages): data, graph construction, model (GAT+GRU), FL, explainability, metrics. | From thesis §4–5; add equation or diagram. |
| C5 | **Results**: main table (current Table 1) + ablation table + sensitivity table + FL figure + example alert. | Direct reuse of new thesis tables/figures. |
| C6 | **Discussion**: limitations, threat to validity, comparison with literature. | From thesis §8. |
| C7 | **Conclusion**: summary + future work. | From thesis §9. |
| C8 | **References**: same as thesis, ensure journal format (MDPI/IEEE). | |
| C9 | **Data and Code Availability**: “Code and config at [GitHub]; CICIoT2023 from [cite].” | Reproducibility. |

### Phase D: Venue-Specific

| Step | Action | When |
|------|--------|------|
| D1 | Choose first target: **MDPI Sensors** (or Electronics) vs **IEEE Access** vs **conference**. | After thesis submitted. |
| D2 | Download template: MDPI Word/LaTeX or IEEE. | From journal site. |
| D3 | Map your sections into template; trim to page limit (e.g. 8–12 pages). | |
| D4 | Submit via journal system; suggest 2–3 reviewers if allowed. | |
| D5 | Respond to reviewers; revise; resubmit. | Normal process. |

---

## Part 5: What to Add in Code (Manageable)

### 5.1 Config for Ablation

In `config/experiment.yaml` (or new `config/ablation.yaml`):

```yaml
# Ablation flags (add under a new key)
ablation:
  use_gru: true      # false = mean pool over time
  use_graph: true    # false = flat sequence only (for "GRU on flat")
  graph_edges: knn   # or "random" for random edges
```

### 5.2 New Scripts (high level)

| Script | Purpose |
|--------|---------|
| `scripts/run_ablation.py` | Loop over ablation configs; train and evaluate; write `results/metrics/ablation_table.csv`. |
| `scripts/run_sensitivity.py` | Loop over window_size and k; write `results/metrics/sensitivity_table.csv` and optional plot. |
| `scripts/run_multi_seed.py` | Run central GNN (and optionally RF) for seeds 42, 123, 456; aggregate to `results/metrics/multi_seed_summary.json`. |

### 5.3 New Results to Generate

| Output | Path | Content |
|--------|------|--------|
| Ablation table | `results/metrics/ablation_table.csv` | Model variant, F1, ROC-AUC, Inference (ms). |
| Sensitivity table | `results/metrics/sensitivity_table.csv` | window_size, k, F1, ROC-AUC. |
| Multi-seed summary | `results/metrics/multi_seed_summary.json` | mean_f1, std_f1, mean_auc, std_auc. |
| Sensitivity figure | `results/figures/sensitivity_f1_vs_window.png` | F1 vs window size (optional). |

---

## Part 6: What to Add in the Thesis (Text)

### 6.1 New Subsection: Ablation Studies (e.g. §7.6)

- One short paragraph: “To assess the contribution of each component, we ran three ablations: (1) GAT without GRU (mean pooling over time), (2) GRU on flat window features (no graph), (3) kNN graph vs random edges.”
- **Table:** Ablation on CICIoT2023 test set (columns: Model variant | F1 | ROC-AUC | Inference ms).
- One paragraph interpreting: “Removing the GRU reduces F1 by X%; removing the graph reduces it by Y%. The kNN graph outperforms random edges by Z%.”

### 6.2 New Subsection: Sensitivity Analysis (e.g. §7.7)

- One short paragraph: “We varied window size (30, 50, 70) and k (3, 5, 7) while keeping other settings fixed.”
- **Table:** Sensitivity to window size and k (rows: config, F1, ROC-AUC).
- One sentence: “Performance remains stable for window sizes 30–70 and k=3–7, supporting the chosen hyperparameters.”

### 6.3 Optional: Statistical Stability

- One sentence + small table: “To check stability, we ran the central GNN with three seeds (42, 123, 456). F1 = X.XX ± X.XX, ROC-AUC = X.XX ± X.XX.”

### 6.4 In Discussion (§8)

- “Ablation shows that both the graph and the temporal (GRU) components contribute to performance…”
- “Sensitivity analysis indicates that results are robust to the chosen window size and k within the tested range.”

### 6.5 In Conclusion (§9)

- “Future work includes submission of an extended version to a Q1/MDPI/IEEE venue, including ablation, sensitivity analysis, and multi-seed validation.”

---

## Part 7: Order of Execution (Timeline)

| Priority | Task | Time (rough) | Thesis section | Paper section |
|----------|------|--------------|----------------|---------------|
| 1 | Implement and run **ablation** (GAT-only, GRU-only, kNN vs random) | 2–4 days | §7.6 | Results |
| 2 | Add **ablation table + subsection** to thesis | 0.5 day | §7.6 | — |
| 3 | Implement and run **sensitivity** (window, k) | 1–2 days | §7.7 | Results |
| 4 | Add **sensitivity table + short subsection** to thesis | 0.5 day | §7.7 | — |
| 5 | Run **multi-seed** (3 seeds) for GNN; aggregate | 1 day | §7 or §6 | Results |
| 6 | Add **stability** sentence + table to thesis | 0.5 day | §7 | — |
| 7 | Update **Discussion** with ablation/sensitivity interpretation | 0.5 day | §8 | Discussion |
| 8 | Update **Conclusion** with publication as future work | 0.25 day | §9 | — |
| 9 | (Optional) Per–attack-type metrics if labels available | 1 day | §7 | Results |
| 10 | After thesis submitted: **draft paper** (MDPI/IEEE template) | 2–3 days | — | Full paper |
| 11 | **Submit** to chosen journal | 0.5 day | — | — |

**Total extra for thesis (phases 1–8):** about 5–8 days of focused work.  
**Publication:** after thesis, 2–4 days to adapt to journal format and submit.

---

## Part 8: Checklist Summary

### For thesis (90% / distinction)

- [ ] Ablation table + short subsection (§7.6).
- [ ] Sensitivity table + short subsection (§7.7).
- [ ] (Optional) Multi-seed stability table/sentence.
- [ ] Discussion updated with ablation/sensitivity interpretation.
- [ ] Conclusion mentions extension to publication (MDPI/IEEE/Q1).
- [ ] All new figures/tables referenced and captioned.
- [ ] Word count and formatting per handbook (1.5 spacing, 11pt, etc.).

### For publication (MDPI / IEEE / Q1)

- [ ] Same dataset (CICIoT2023), deeper analysis (ablation, sensitivity, seeds).
- [ ] One clear contribution statement (e.g. “first to combine GAT+GRU, FL, and ECS-like explainable alerts on CICIoT2023”).
- [ ] Reproducibility: config, code, seed in repo; statement in paper.
- [ ] Paper structure: Abstract, Intro, Related Work, Method, Results, Discussion, Conclusion, References.
- [ ] Target venue chosen; template downloaded; submission done.

---

## Part 9: Impact on Final Report and Future Work

| What you do | Impact on thesis mark | Impact on publication |
|-------------|------------------------|------------------------|
| Ablation studies | Shows critical analysis and design justification (methodology + results). | Expected by Q1/MDPI/IEEE reviewers. |
| Sensitivity analysis | Shows robustness and justifies hyperparameters. | Strengthens Results. |
| Multi-seed / stability | Shows rigour. | Often required in good ML journals. |
| Clear “future work” + “submission to journal” | Shows ambition and planning (Conclusion). | You are literally doing that next. |
| Same dataset, deeper | No new data needed; all experiments comparable. | “Extended analysis on CICIoT2023” is a valid contribution. |

**Unique angle for publication:** “Explainable dynamic GNN with federated learning and SIEM-ready ECS alerts on CICIoT2023, with ablation and sensitivity analysis.” That combination is still rare in a single paper.

---

*Document prepared to align thesis (90% / distinction) with high-class publication (MDPI / IEEE / Q1). Execute phases in order; reuse every new table and figure in both the thesis and the paper.*
