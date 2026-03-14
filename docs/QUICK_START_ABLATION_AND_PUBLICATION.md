# Quick Start: First Ablation + Where It Fits

This file gives the **first concrete steps** so you can run one ablation quickly. Full strategy is in `PUBLICATION_AND_THESIS_MASTER_PLAN.md`.

---

## Step 1: One Ablation (GAT without GRU)

**Goal:** Train the same GNN but replace the GRU with mean pooling over time. Compare F1/ROC-AUC to the full model.

**Why:** Shows that "temporal modelling (GRU) adds value". This is the single most impactful ablation for a publication.

### Option A: Minimal code change (recommended first)

1. In `src/models/dynamic_gnn.py`, add an optional argument to `forward`:

   - Add `use_gru: bool = True` to `forward()` and `forward_batch()` (or read from a class attribute `self.use_gru` set in `__init__`).
   - In `forward()`, after building `seq` (shape 1, T, hidden):
     - If `use_gru=True`: keep `_, h_n = self.gru(seq)` then `logits = self.classifier(h_n.squeeze(0))`.
     - If `use_gru=False`: do `h = seq.mean(dim=1)` then `logits = self.classifier(h)`.

2. In `config/experiment.yaml`, add under `models.dynamic_gnn`:

   ```yaml
   use_gru: true   # set false for ablation "GAT only"
   ```

3. In `DynamicGNN.from_config()`, read `use_gru` and pass to `__init__`; store as `self.use_gru`.

4. Run training twice:
   - Once with `use_gru: true` (current behaviour) → you already have results.
   - Once with `use_gru: false` → save metrics to e.g. `results/metrics/ablation_gat_only.json`.

5. Add one row to your **ablation table** in the thesis:

   | Model variant   | F1    | ROC-AUC | Inference (ms) |
   |-----------------|-------|---------|----------------|
   | Full (GAT+GRU)  | 1.000 | 1.000   | 22.70          |
   | GAT only (no GRU) | (your result) | (your result) | (measure) |

### Option B: Separate script (cleaner for many ablations)

Create `scripts/run_ablation.py` that:

1. Loads config.
2. For each variant (e.g. `use_gru=True`, `use_gru=False`), builds the model, trains, evaluates.
3. Writes `results/metrics/ablation_table.csv` with columns: variant, F1, ROC-AUC, inference_ms.

Then you only run: `python scripts/run_ablation.py --config config/experiment.yaml`.

---

## Step 2: Add to Thesis

- In **Results** (e.g. new §7.6 Ablation Studies):
  - One short paragraph describing the ablation (GAT without GRU).
  - **Table:** Ablation on CICIoT2023 test set (Full vs GAT-only).
  - One sentence: “Removing the GRU reduces F1 by X% and ROC-AUC by Y%, showing that temporal modelling contributes to performance.”

- In **Discussion** (§8):
  - One sentence: “The ablation confirms that both the graph (GAT) and the temporal (GRU) components contribute; the GRU is necessary to capture evolution across windows.”

---

## Step 3: Use Same Content in the Paper

When you write the journal paper (MDPI/IEEE):

- **Results:** Reuse the same ablation table and the same sentence.
- **Discussion:** Same interpretation.

No extra experiments needed for this part of the publication.

---

## What Next (from Master Plan)

After this first ablation:

1. **Second ablation:** GRU on flat features (no graph). Requires a small separate model or a flag that builds “window mean features” instead of graphs.
2. **Sensitivity:** Run for window_size ∈ {30, 50, 70} and k ∈ {3, 5, 7}; fill sensitivity table.
3. **Multi-seed:** Run central GNN for seeds 42, 123, 456; report mean ± std.
4. Then write the full paper using the structure in `PUBLICATION_AND_THESIS_MASTER_PLAN.md` Part 7 (Publication-Ready Paper).

All of this uses the **same dataset (CICIoT2023)** and the same codebase; you only add config options and one script for ablation/sensitivity.
