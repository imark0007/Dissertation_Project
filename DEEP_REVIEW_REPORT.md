# Deep Review Report: Dataset, Results, Scripts, and Figures for 90% Distinction

This report documents a deep check of your full dataset, every result, scripts, and figures against the dissertation and distinction-level standards. All findings are tied to your pipeline and, where relevant, to research practice.

---

## 1. Dataset Verification

### 1.1 What Was Checked

- **Raw data:** `data/raw/train.csv` (~1.6 GB), `test.csv` (~348 MB), `validation.csv` (~348 MB) — full CICIoT2023 splits as provided.
- **Processed data:** `data/processed/*.parquet` — 500,000 rows per split; scaler fitted on train only (no test/validation leakage).
- **Graph data:** `data/graphs/{train,validation,test}_graphs.pt` — built from processed splits; no cross-split mixing.
- **Script:** `scripts/dataset_statistics.py` — computes and saves `results/metrics/dataset_stats.json` for reproducible numbers.

### 1.2 Dataset Statistics (from dataset_statistics.py)

| Split      | Flows   | Benign (flow-level) | Attack (flow-level) | GNN sequences |
|-----------|---------|----------------------|----------------------|----------------|
| Train     | 500,000 | 2.32% (11,582)      | 97.68% (488,418)    | 920            |
| Validation| 500,000 | 2.34% (11,676)      | 97.66% (488,324)    | 928            |
| Test      | 500,000 | 2.35% (11,772)      | 97.65% (488,228)    | 934            |

The stratified windowing in `graph_builder.py` (separate benign/attack pools, then balance) produces roughly 50% benign / 50% attack *windows*, which addresses the severe flow-level imbalance. This is standard for imbalanced IDS data (e.g. Elkan 2001; López et al. 2013 on class imbalance in intrusion detection).

### 1.3 Data Leakage and Splits

- **Preprocessing:** `StandardScaler` is fit only on train; transform applied to train/validation/test. No leakage.
- **Graph building:** Per-split; train/validation/test graphs built independently from respective parquets. No leakage.
- **Baselines (RF, MLP):** Trained on train split only; evaluated on test. Same processed splits as GNN.
- **GNN:** Trained on train sequences; validation used for early stopping; test used only for final metrics. No test data in training.

**Verdict:** Dataset use and splits are correct and consistent with reproducible research practice.

---

## 2. Results Verification

### 2.1 Table 1 (Model comparison)

| Source | RF (P/R/F1/AUC) | MLP | Central GNN | Federated GNN |
|--------|------------------|-----|-------------|----------------|
| results_table.csv | 0.9989/0.9984/0.9986/0.9996 | 1.0/0.9885/0.9942/0.9984 | 1.0/1.0/1.0/1.0 | 1.0/1.0/1.0/1.0 |
| rf_metrics.json   | 0.9989/0.9984/0.9986/0.9996 | — | — | — |
| mlp_metrics.json  | — | 1.0/0.9885/0.9942/0.9984 | — | — |
| central_gnn_metrics.json | — | — | 1.0/1.0/1.0/1.0 | — |
| Dissertation Table 1 | 0.9989/0.9984/0.9986/0.9996 | 1.0000/0.9885/0.9942/0.9984 | 1.0/1.0/1.0/1.0 | 1.0/1.0/1.0/1.0 |

Inference (ms): RF 46.09, MLP 0.66, Central GNN 22.70, Federated GNN 20.99 — all match metrics files and dissertation.

**Verdict:** Table 1 is correct and consistent with saved metrics.

### 2.2 Table 2 (Federated learning rounds)

Round-by-round precision, recall, F1, and ROC-AUC in the dissertation match `results/metrics/fl_rounds.json` (rounds 1–10). Round 6 ROC-AUC 0.973 matches 0.9731 in JSON. Communication cost (3.07 MB per round, ~31 MB total) matches 128,002 parameters × 4 bytes × 2 (up+down) × 3 clients ≈ 3.07 MB per round.

**Verdict:** Table 2 is correct.

### 2.3 Table 3 (Central GNN training history)

**Fix applied:** The dissertation Table 3 previously showed train loss 0.412, 0.012, 0.002, … which did not match `results/metrics/gnn_training_history.json`. Table 3 was updated to the actual values: 0.484, 0.023, 0.0002, 0.0001, 0.0001, 0.0001 (Val F1 and Val ROC-AUC remain 1.0 for all epochs). This ensures the thesis is verifiable against the saved run.

**Verdict:** Table 3 now matches the metrics file.

### 2.4 Table 4 (Ablation)

Values in the dissertation match `results/metrics/ablation_table.csv`: Full (GAT+GRU) 1.0/1.0/1.0/1.0/22.70; GAT only 0.9923/1.0/0.9961/1.0/16.06.

**Verdict:** Table 4 is correct.

### 2.5 Evaluation Unit (flow vs sequence)

- **RF and MLP:** One prediction per *flow* (500k test flows).
- **GNN:** One prediction per *sequence* (934 test sequences; each sequence = 5 windows × 50 flows = 250 flows).

The dissertation now states this explicitly in §6.3: “the GNN is evaluated at sequence level (one prediction per sequence of 5 windows)” while “Random Forest and MLP baselines use the same processed flows in tabular format (one row per flow) on the same splits”. So comparison is on the same underlying test *partition*, with each model’s natural evaluation unit. This is a valid and common design (e.g. sequence-level evaluation for temporal models vs flow-level for non-temporal baselines).

---

## 3. Scripts Verification

| Script | Purpose | Checked |
|--------|---------|--------|
| `src/data/preprocess.py` | Load raw CSVs, clean, scale (train fit), binary labels, save parquets | Config columns used; scaler fit on train only; splits correct. |
| `src/data/graph_builder.py` | Build kNN graphs per window; stratified benign/attack pools; save per split | Uses config window_size, knn_k, minority_stride; output matches dataset_stats. |
| `src/data/dataset.py` | GraphSequenceDataset; sequence_length=5; get_dataloaders | Same config; train/val/test loaders; no shuffle on test. |
| `scripts/run_all.py` | Preprocess → graphs → baselines → GNN → results table | Passes config paths to graph builder; builds results_table from metrics. |
| `scripts/run_ablation.py` | GAT-only ablation; backup/restore full model; write ablation_gat_only.json + ablation_table.csv | Correct; eval on test after training. |
| `scripts/eval_ablation_from_ckpt.py` | Evaluate saved ablation checkpoint; write metrics if run_ablation stopped early | Inlined CSV build; no external import; paths correct. |
| `scripts/update_dissertation_table4.py` | Update Table 4 in MD from ablation_table.csv or ablation_gat_only.json | Reads correct paths; updates GAT-only row. |
| `scripts/generate_alerts_and_plots.py` | Example alerts, FL convergence plot, model comparison figure | Uses GNN checkpoint and test graphs; writes to results/figures and results/alerts. |
| `scripts/dataset_statistics.py` | Compute dataset_stats.json for §6.3 | Writes to results/metrics; parquets and graph .pt files read correctly. |

**Verdict:** Scripts are consistent with config, use the full dataset (no hidden nrows in normal run), and produce the outputs referenced in the dissertation.

---

## 4. Figures Assessment for 90% Distinction

### 4.1 Current Figures (9 total)

| Figure | Content | Role for distinction |
|--------|---------|----------------------|
| 1 | Research pipeline | Method clarity |
| 2, 6, 7 | Confusion matrices (GNN, RF, MLP) | Evidence for Table 1, false positives |
| 3, 8, 9 | ROC curves (GNN, RF, MLP) | Discriminative ability, AUC |
| 4 | FL convergence (F1, ROC-AUC vs round) | Sub-question 2, Table 2 |
| 5 | Model comparison (inference time, F1) | Deployment + comparison |

This covers: main results (confusion, ROC), FL behaviour, and model comparison. For an MSc thesis, this is typically **enough** for a strong (distinction-level) mark, especially with four tables (model comparison, FL rounds, training history, ablation).

### 4.2 Optional Additions (if time)

- **Dataset/class balance figure:** A simple bar chart of flow-level class distribution (e.g. train/val/test benign vs attack) or sequence counts. Not required for 90%, but strengthens §6.3 and reproducibility (e.g. Bischl et al. 2021 on reporting data splits).
- **Sensitivity analysis figure:** If you run sensitivity (window size and/or k), one figure (e.g. F1 vs window size) plus a short table would support “robustness of design choices” (§7.7) and align with common journal expectations (e.g. sensitivity as in many MDPI/IEEE IDS papers).

**Verdict:** Nine figures are **sufficient** for 90% distinction. Optional: one dataset-statistics figure and, if you run sensitivity, one sensitivity figure.

---

## 5. Recommendations with Research Justification

### 5.1 Done in This Review

1. **Table 3 aligned with saved metrics** — Training loss values in the dissertation now match `gnn_training_history.json`. This avoids examiner inconsistency and supports reproducibility (e.g. ACM/IEEE guidelines on reporting exact results).
2. **§6.3 dataset statistics** — Added explicit counts (500k flows per split, ~2.3% benign / 97.7% attack, 920/928/934 sequences) and reference to `dataset_stats.json` and `scripts/dataset_statistics.py`. This meets the expectation that “dataset and experiment statistics” are reported and verifiable (e.g. Bischl et al. 2021).
3. **Evaluation unit clarified** — Flow-level (RF, MLP) vs sequence-level (GNN) is now stated in §6.3, avoiding ambiguity and strengthening methodology.

### 5.2 Optional (Recommended if Time Permits)

4. **Sensitivity analysis (§7.7)** — Run a few experiments (e.g. window_size ∈ {30, 50, 70}, k ∈ {3, 5, 7}), record F1/ROC-AUC in `results/metrics/sensitivity_table.csv`, and add one short table + one sentence in §7.7. This is standard in ML/IDS papers (e.g. hyperparameter sensitivity in MDPI Sensors/Electronics) and would strengthen the “robustness” claim.
5. **Dataset balance figure** — One bar chart of flow-level or sequence-level class distribution, saved to `results/figures/dataset_balance.png` and referenced in §6.3. Improves transparency; not mandatory for 90%.

### 5.3 Not Required for 90%

- **Multi-seed runs:** Would strengthen a journal submission (e.g. mean ± std over seeds). For the thesis, a single seed (42) with clear reporting is acceptable.
- **Per-attack-type breakdown:** Would add depth but requires label mapping in CICIoT2023; optional for thesis.

---

## 6. Summary

| Area | Status | Action taken |
|------|--------|--------------|
| Dataset | Correct, full splits, no leakage | Added dataset_stats.json + §6.3 counts and evaluation-unit clarification. |
| Table 1 | Matches metrics | None. |
| Table 2 | Matches fl_rounds.json | None. |
| Table 3 | Mismatch | Updated to match gnn_training_history.json. |
| Table 4 | Matches ablation_table.csv | None. |
| Scripts | Consistent with config and data | Added dataset_statistics.py; documented in report. |
| Figures | Sufficient for 90% | Nine figures retained; optional: dataset figure, sensitivity figure. |

Your results and scripts are consistent with the full dataset and with the dissertation text. The only correction applied was Table 3; the rest of the changes improve transparency and reproducibility (dataset statistics, evaluation unit, and a reusable dataset-statistics script). With these in place, the thesis is in strong shape for 90% (distinction) and for future reuse in a journal paper.

---

*References (for rationale): Elkan (2001) and López et al. (2013) on class imbalance in classification; Bischl et al. (2021) on reporting experiments; ACM/IEEE reproducibility guidelines; typical MDPI/IEEE IDS paper structure.*
