# Project master brief — viva / supervisor defence

**Student:** Arka Talukder (B01821011)  
**Title:** Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning  
**Supervisor:** Dr. Raja Ujjan  

Use this file as a **revision sheet**: read sections aloud, trace claims to **files** in the repo, and practise short answers (60–90 seconds). Examiners care that **you** can join the parts together—not that you memorise adjectives.

**Printable 1–2 page squeeze:** `docs/viva/PROJECT_VIVA_CHEATSHEET_PRINT.md` (take to viva; keep the master brief for depth).

---

## 1. Thirty-second summary (elevator)

IoT flow records from **CICIoT2023** are turned into **kNN graphs per time window**, then **sequences of windows** are classified by a **dynamic GNN (GAT + GRU)**. **Random Forest** and **MLP** run on the **same flows** as flat **46-feature rows** (no graph). **Federated learning** (Flower, FedAvg, three clients, non-IID split) trains the same GNN without centralising raw flows. **Integrated Gradients** and **GAT attention** feed into **ECS-like JSON alerts** served from a **FastAPI** app on **CPU**. Headline test metrics for the central and federated GNN on your fixed split are in `results/metrics/results_table.csv`; you still explain **limits** (subset, lab data, no analyst user study).

---

## 2. Research aim and questions (answer verbatim, then expand if asked)

**Main question:** How can an explainable dynamic GNN, trained with federated learning, detect attacks in Software-Defined IoT **flow** data and generate SIEM-style alerts that support SOC triage on **CPU** edge devices?

**Sub-questions:**

1. Does the dynamic graph model beat **RF** and **MLP** on your evaluation setup?  
2. Does **FedAvg** keep performance **without** sharing raw data?  
3. Are **explanations** (top features / top flows) usable for triage—**qualitatively** in this project?

**One-line answers you can give:**

1. On the **held-out test split** in your results table, central GNN reaches **F1 = 1.0, ROC-AUC = 1.0**; RF **F1 ≈ 0.9986**; MLP **F1 ≈ 0.9942**—with **0** GNN false positives vs **187** RF and **4** MLP on that split (`results_table.csv` + confusion matrices).  
2. Federated GNN matches central on those headline metrics in your logged run; raw data stays on clients.  
3. You attach **IG + attention** to five **worked example alerts** (`results/alerts/example_alerts.json`); usefulness is **argued** in the thesis, not proven by a user study.

---

## 3. Dataset and splits (expect “where is leakage?”)

- **Source:** CICIoT2023 (Pinto et al., 2023) — public IoT flow captures with **46 numeric features** per flow in the version you used; **no device IPs** in that feature set, so graphs are **feature-similarity kNN**, not device topology.  
- **Subset:** **500,000 rows per split** (train / validation / test), heavily **attack-skewed** at flow level (~**2.3%** benign, ~**97.7%** attack) — see `results/metrics/dataset_stats.json`.  
- **Splits:** Fixed files under `data/raw/` → preprocessing → `data/processed/*.parquet`. **Scaler fit on train only**, applied to val/test (`src/data/preprocess.py`).  
- **Leakage line:** Test flows never enter training or hyperparameter tuning; sequences are built **per split**. If pressed: you did not tune on test; validation used for early stopping / sanity.

---

## 4. Graph construction (expect “majority vote vs pool?”)

**Implemented behaviour** (`src/data/graph_builder.py`):

- Benign and attack flows are **separated into two pools**.  
- Windows of **`window_size`** flows (default **50**) are drawn **from one pool only**.  
- The **graph-level label** is the **pool’s class** (every node in that window shares the same binary label).  
- Minority class uses a **smaller stride** (`minority_stride`, default **25**) to get more benign windows.  
- Equal numbers of benign and attack **graphs** are taken, then the list is **shuffled**.

**Sequences** (`src/data/dataset.py` — `GraphSequenceDataset`):

- A training item is **five consecutive graphs** in that **shuffled** list (`sequence_length` = **5**).  
- **Sequence label = attack if any of the five windows is attack**; otherwise benign. That is an **OR** over windows, not “majority over mixed labels inside one window.”

**kNN inside each window:** each flow linked to **`knn_k`** nearest neighbours in Euclidean distance on standardised features (default **k = 5**). All in `config/experiment.yaml` under `graph:`.

---

## 5. Models — what each one actually sees

| Model | Input unit | Notes |
|--------|----------------|------|
| **Random Forest** | **One row = one flow** | 46 features + label from parquet; `RandomForestBaseline` in `src/models/baselines.py`; 200 trees, depth 20 from config. |
| **MLP** | **One row = one flow** | Same matrix as RF; two-class logits + cross-entropy in `scripts/run_all.py`. |
| **Dynamic GNN** | **Sequence of 5 graph snapshots** | Each snapshot = kNN graph over 50 flows; GAT → graph embedding → **GRU** → classifier (`src/models/dynamic_gnn.py`). |

**Ablation:** `use_gru: false` → **mean pool over time** instead of GRU (`run_ablation.py`); metrics in `results/metrics/ablation_gat_only.json` / `ablation_table.csv`.

---

## 6. Federated learning (expect “non-IID how?”)

- **Framework:** Flower (`src/federated/`).  
- **Algorithm:** FedAvg — server averages client weights.  
- **Setup from config:** **`num_clients: 3`**, **`num_rounds: 10`**, **`local_epochs: 2`**, **`alpha: 0.5`** for Dirichlet partitioning over **graph** labels on the training set (`src/federated/data_split.py` — run `split_and_save` on `train_graphs.pt` before clients).  
- **Privacy story:** updates are **parameters**, not raw PCAP/CSV rows.  
- **Evidence:** round metrics `results/metrics/fl_rounds.json`; final `federated_gnn_metrics.json`; convergence figure `results/figures/fl_convergence.png`.  
- **Communication:** thesis gives **order-of-MB** totals derived from parameter count (~128k floats); say “approximate” not “measured on wire with TLS overhead.”

---

## 7. Explainability and alerts

- **Integrated Gradients:** Captum; attributions on **node features** (last window often used for cost — see `src/explain/explainer.py`).  
- **Attention:** from **GAT** layers — edge/neighbour weights mapped to “top nodes” / flows in the alert JSON.  
- **Output:** `src/siem/alert_formatter.py` → ECS-like fields + `explanation.top_features` / `top_nodes`.  
- **Examples:** `results/alerts/example_alerts.json` — know **one true positive** and **one false positive** well enough to walk through **which features** fired.

---

## 8. Numbers you should be able to point to (from `results_table.csv`)

These are the **aggregated** headline lines in your table (central test):

| Model | F1 | ROC-AUC | Inference (ms) |
|--------|-----|---------|------------------|
| RF | 0.9986 | 0.9996 | 46.09 |
| MLP | 0.9942 | 0.9984 | 0.66 |
| Central GNN | 1.0000 | 1.0000 | 22.70 |
| Federated GNN | 1.0000 | 1.0000 | 20.99 |

**Multi-seed:** `multi_seed_summary.json` — seeds **42, 123, 456**; headline F1/AUC **1.0** with **0** std on that test setup; per-seed **inference_ms** varies (CPU timing noise).

**Sensitivity:** nine (**window**, **k**) pairs in `results/metrics/sensitivity_table.csv` + heatmap `results/figures/sensitivity.png`.

If a number is challenged: **open the JSON/CSV** in the meeting folder—you are not defending digits from memory alone.

---

## 9. Reproduction — commands you should know cold

From project root (paths as in `SETUP_AND_RUN.md`):

```text
python scripts/run_all.py --config config/experiment.yaml
```

Quick dry run (smaller data): add `--nrows 10000` where supported.

```text
python scripts/generate_alerts_and_plots.py
python scripts/run_ablation.py --config config/experiment.yaml
python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
python scripts/dataset_statistics.py --config config/experiment.yaml
```

Federated: **split** training graphs to clients, then server + three `client` processes — exact invocations in **Appendix C** of the dissertation and in `SETUP_AND_RUN.md`.

---

## 10. Repository map (where to look when stumped)

| Topic | Location |
|--------|-----------|
| Config single source of truth | `config/experiment.yaml` |
| Preprocess + scaler | `src/data/preprocess.py` |
| Graphs | `src/data/graph_builder.py` |
| Sequences / loaders | `src/data/dataset.py` |
| GNN | `src/models/dynamic_gnn.py` |
| Training loop | `src/models/trainer.py` |
| Baselines | `src/models/baselines.py` |
| Full central pipeline | `scripts/run_all.py` |
| FL entry | `src/federated/run_federated.py`, `client.py`, `server.py` |
| Explain | `src/explain/explainer.py` |
| API | `src/siem/api.py`, `alert_formatter.py` |
| Written thesis source | `Dissertation_Arka_Talukder.md` → Word via `scripts/dissertation_to_docx.py` |

---

## 11. Viva-style questions — short honest answers

**“Why 100% F1 — is the task trivial?”**  
Subset + strong features + careful windowing can make separation easy. You still report **baselines**, **ablation**, **sensitivity**, **multi-seed**, and you **do not** claim “solved IoT security globally.”

**“Is there data leakage?”**  
Train/val/test are file-level splits; scaler from train; test not used for tuning. Sequences built **within** each split. If they push harder: acknowledge **no formal independent external test set** beyond this subset.

**“Why kNN graphs instead of real topology?”**  
Public release you used lacks **IPs** as node identities in the feature list; kNN on flow attributes is a **documented design choice** aligned with recent attribute-graph IDS work (you cite Ngo et al., Basak et al. in the thesis).

**“What does federated prove here?”**  
A **prototype** that FedAvg can reach **parity** with central training on this split and architecture—not proof at national scale or under adversarial clients.

**“Why Integrated Gradients and not SHAP?”**  
Neural pipeline; IG is standard, implemented in Captum, and you also use **attention** for structure-aware cues. Trade-off: **latency** if run on every alert.

**“What would you do with six more months?”**  
Larger slice of CICIoT2023 or a second dataset; more clients and stragglers; per-attack breakdown; **SOC user study**; harder non-IID; optional GPU vs CPU trade study.

**“Biggest implementation pain?”**  
Class imbalance at flow level vs balanced graph training; wiring PyG batches + Flower + Captum without silent shape bugs.

---

## 12. What *not* to claim

- Not a production SIEM product.  
- Not a statistically powered study of analyst behaviour.  
- Not proof that FL always converges this fast on **your** future data.  
- Perfect ROC on a **fixed** lab split is **encouraging**, not a certificate of deployment readiness.

---

## 13. Marking / specification (if they ask “did you follow the spec?”)

Chapter **1.6** maps **chapters** to the **criteria in your agreed specification** (and tells you to align percentages with the **signed** appendix if anything drifted). Know where **Literature**, **Design**, **Implementation**, **Evaluation**, **Results**, **Discussion/Conclusions**, and **Self-evaluation** live in **your** chapter numbering.

---

## 14. Day-before checklist

- [ ] Open `results_table.csv` and **one** confusion matrix PNG side by side.  
- [ ] Read **one** `example_alerts.json` entry out loud.  
- [ ] Re-skim **Chapter 2 gap** and **Chapter 9 limitations** (same story, no contradiction).  
- [ ] Know **three** references by heart: **FedAvg** (McMahan et al.), **GAT** (Velickovic et al.), **Integrated Gradients** (Sundararajan et al.) — dataset paper **Pinto et al.** for CICIoT2023.  
- [ ] Sleep. Caffeine is not a substitute for having opened the repo once that week.

---

*This brief is tied to the repository layout and `config/experiment.yaml` as of when you last regenerated metrics. If you change config or rerun experiments, refresh the JSON/CSV figures before the viva.*
