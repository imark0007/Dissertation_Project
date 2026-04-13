<!-- Print: A4 or Letter, portrait, margins ~12–15 mm, font 9–10 pt if needed to fit two sides -->

# Viva cheat sheet — B01821011 · Arka Talukder

**Project:** Explainable dynamic GNN + federated learning + SIEM-style alerts (CPU). **Data:** CICIoT2023 subset. **Full notes:** `docs/viva/PROJECT_VIVA_MASTER_BRIEF.md`

---

## 30 seconds

Flows → **stratified kNN graphs** (50 flows/window, **k**=5) → **5-window sequences** → **GAT+GRU** classifier. **RF / MLP** on same data as **one row per flow** (46 features). **Flower + FedAvg**, 3 clients, no raw data to server. **IG + attention** → JSON alerts + **FastAPI**. Evidence: `results/metrics/`, `results/figures/`, `results/alerts/`.

---

## Graph + sequence (say this clearly)

| Step | Fact |
|------|------|
| Pools | Benign / attack **separate**; each window built from **one pool only** → graph label = that class (`graph_builder.py`). |
| Balance | Same count benign/attack graphs → **shuffle** list. |
| Sequence | **5 consecutive** graphs in that list; label = **attack if ANY** of 5 is attack (`GraphSequenceDataset`). |
| kNN | Inside window: **k**=5 neighbours on standardised 46-D Euclidean (`experiment.yaml` → `graph:`). |

---

## Models — input shape

| Model | Input |
|-------|--------|
| RF | 1 row = **1 flow**, parquet, `baselines.py` |
| MLP | Same; **2 logits**, cross-entropy |
| GNN | **5 graphs × 50 nodes × 46** features → GAT → GRU |

**Ablation:** GRU off = mean over time (`run_ablation.py`).

---

## Headline test numbers (`results_table.csv` + CM)

| Model | F1 | AUC | Inf ms | FP (test, thesis) |
|-------|-----|-----|--------|---------------------|
| RF | 0.9986 | 0.9996 | 46.09 | **187** |
| MLP | 0.9942 | 0.9984 | 0.66 | **4** |
| Central GNN | 1.0 | 1.0 | 22.70 | **0** |
| Fed GNN | 1.0 | 1.0 | 20.99 | (same split) |

**Data:** ~500k flows/split; ~**2.3%** benign at flow level (`dataset_stats.json`). **934** test **sequences** for GNN.

**Multi-seed:** 42, 123, 456 → F1/AUC **1.0**, std **0** (`multi_seed_summary.json`). **Sensitivity:** 9 configs → `sensitivity_table.csv` + `sensitivity.png`.

---

## Federated (one breath)

**FedAvg** · **3** clients · **10** rounds · **2** local epochs · Dirichlet **α=0.5** on **train graphs** · only **weights** round-trip · evidence `fl_rounds.json`, `federated_gnn_metrics.json`, `fl_convergence.png`.

---

## Explainability + demo

**IG** (Captum) + **GAT attention** → `explanation` in JSON (`explainer.py`, `alert_formatter.py`). Walk **one** line from `results/alerts/example_alerts.json` (TP + FP).

---

## Commands (project root)

```
python scripts/run_all.py --config config/experiment.yaml
python scripts/generate_alerts_and_plots.py
python scripts/run_ablation.py --config config/experiment.yaml
python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
python scripts/dataset_statistics.py --config config/experiment.yaml
```

FL: split clients then `python -m src.federated.run_federated server` + `client --cid 0..2` — see `SETUP_AND_RUN.md` / thesis Appendix C.

---

## If lost — open this file

`config/experiment.yaml` · `src/data/graph_builder.py` · `src/data/dataset.py` · `src/models/dynamic_gnn.py` · `scripts/run_all.py` · `results/metrics/results_table.csv`

---

**——— flip / page 2 ———**

## Viva Q → one-line A

| Q | A |
|---|---|
| Why F1=1? | Subset + strong tabular features + windowing; I still show baselines, ablation, sensitivity, multi-seed; **not** “solved all IoT.” |
| Leakage? | Train/val/test files; scaler on train only; sequences per split. No second external dataset. |
| Why kNN not device graph? | No IPs in this feature release — attribute similarity graph; cited in thesis. |
| What does FL prove? | Prototype parity **here**; not production at national scale. |
| IG vs SHAP? | Neural stack; IG in Captum + attention for structure; cost if every alert explained. |
| Limitations? | No SOC user study; 3 clients; lab subset; perfect metrics on **this** split only. |
| 6 months more? | Bigger/dataset #2; more clients; per-class metrics; analyst study; harder non-IID. |

---

## Do not say

“Production SIEM” · “Analysts proved faster triage” · “FL always converges this fast” · “No need to revalidate on new data”

---

## Three papers to name cold

**FedAvg:** McMahan et al. (2017) · **GAT:** Velickovic et al. (2018) · **IG:** Sundararajan et al. (2017) · **Dataset:** Pinto et al. (2023) CICIoT2023

---

## Spec / marking

**Ch 1 §1.6** — chapter ↔ marking criteria; signed spec in **Appendix B** if asked.

---

*Regenerate metrics → refresh CSV/JSON before the viva.*
