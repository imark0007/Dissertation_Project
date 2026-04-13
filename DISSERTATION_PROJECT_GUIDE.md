# Your dissertation project — complete guide & supervisor prep

**Who this is for:** You (Arka), so you can **re-learn the whole story in one sitting**, **walk a supervisor through it**, and **answer questions** without guessing. **Anyone** with basic ML/security vocabulary can follow it if you read it top to bottom once.

**How to use it**

1. Read **§2–4** before a supervisor meeting (what you built + how the thesis is organised + technical path).  
2. Skim **§5** (chapter map) with `Dissertation_Arka_Talukder.md` open.  
3. Memorise the **decision list in §6** and **limitations in §11** — examiners often probe there.  
4. Use **§10** as a flashcard grid: cover the answer column and quiz yourself.

**Related files:** Short GitHub overview → [`README.md`](README.md). Commands → [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md). Archive / appendices → [`archive/README.md`](archive/README.md). Final meeting checklist → [`docs/reports/SUPERVISOR_FINAL_FEEDBACK.md`](docs/reports/SUPERVISOR_FINAL_FEEDBACK.md).

---

## Table of contents

1. [What you built (elevator pitch)](#1-what-you-built-elevator-pitch)  
2. [Research questions you answer](#2-research-questions-you-answer)  
3. [What you did, in order (project timeline)](#3-what-you-did-in-order-project-timeline)  
4. [Dissertation chapter map (what each chapter is for)](#4-dissertation-chapter-map-what-each-chapter-is-for)  
5. [Technical path: from CSV to SIEM alert](#5-technical-path-from-csv-to-siem-alert)  
6. [Design decisions you must be able to justify](#6-design-decisions-you-must-be-able-to-justify)  
7. [Results you can state (and how to defend them)](#7-results-you-can-state-and-how-to-defend-them)  
8. [Where everything lives in the repo](#8-where-everything-lives-in-the-repo)  
9. [Reproduction order (commands)](#9-reproduction-order-commands)  
10. [Supervisor & viva Q&A (question → answer)](#10-supervisor--viva-qa-question--answer)  
11. [Honest limitations (say these out loud)](#11-honest-limitations-say-these-out-loud)  
12. [Glossary](#12-glossary)

---

## 1. What you built (elevator pitch)

You built a **working research prototype** that:

1. Takes **IoT flow records** from the public **CICIoT2023** dataset (46 numeric features per flow, no raw IPs in the release you used).  
2. Turns consecutive flows into **windows**, then into ***k*-nearest-neighbour graphs** in feature space (each flow = node, similar flows = edges).  
3. Sequences several windows in time and runs a **dynamic graph neural network**: **GAT** (attention over neighbours) + **GRU** (over time) → **benign vs attack** prediction.  
4. Compares against **Random Forest** and **MLP** on the **same** train/validation/test splits.  
5. Trains the same GNN with **federated learning** (**Flower**, **FedAvg**, three clients, **non-IID** split) so raw data does not leave “sites.”  
6. Produces **explainable**, **SIEM-shaped JSON alerts** (ECS-like) using **Captum Integrated Gradients** + **GAT attention**, exposed through **FastAPI** on **CPU**.  
7. Documents everything in a **full MSc dissertation** (`Dissertation_Arka_Talukder.md` → Word) with **ablation**, **sensitivity grid**, and **multi-seed** checks.

**One sentence for your supervisor:** *“I combined graph-based IoT intrusion detection, federated training, and explainable SOC-style alerts in one reproducible pipeline on CICIoT2023, with strong baselines and robustness experiments.”*

---

## 2. Research questions you answer

These match **Chapter 1** and are answered in **Chapters 8–9**.

| Question | Short answer | Evidence |
|----------|--------------|----------|
| **Main:** Can an explainable dynamic GNN with FL produce useful SIEM alerts on CPU for SD-IoT flows? | Yes **for this prototype and subset**: pipeline works end-to-end; metrics and alerts in Ch 8; limits in Ch 9–10. | Ch 5–6 (design/build), Ch 8 (tables/figures), Ch 9 (interpretation). |
| **Sub 1:** Does the dynamic GNN beat simple models? | On your split: **higher F1/AUC** than RF/MLP; **fewer false positives** than RF on the test set. | Table 1, Ch 8, confusion matrices. |
| **Sub 2:** Does FL match central training? | **Yes** on your setup (same headline metrics). | FL convergence figure, `fl_rounds.json`, Ch 8. |
| **Sub 3:** Are explanations useful for triage? | **Qualitative**: top features/nodes align with attack/benign patterns; no formal SOC user study. | Example alerts JSON, Ch 6 & 8, Ch 9 limits. |

---

## 3. What you did, in order (project timeline)

Think of this as **the story of the repo**, not calendar dates. When you explain the project, you can walk this list.

| Phase | What you did | Why it matters | Where it shows |
|-------|----------------|----------------|------------------|
| **A. Scoping** | Chose CICIoT2023, binary classification, subset for MSc time; accepted **no device IPs** → **feature *k*NN** graphs instead of device–device topology. | Defines what “graph” means in *your* thesis. | Ch 1, 4; `graph_builder.py`. |
| **B. Literature** | Surveyed IoT IDS, SIEM/SOC, GNNs, FL, XAI; argued a **gap** (combined prototype + SIEM-shaped output). | Motivates originality at MSc level. | Ch 2; Table 5. |
| **C. Project management** | Timeline, risks, ethics (public data), CyBOK mapping, interim feedback loop. | Programme requirements. | Ch 3; `archive/interim_report/`. |
| **D. Methodology** | Defined splits, metrics (precision, recall, F1, ROC-AUC, FPR), evaluation protocol, federation simulation design. | Examiners see you did “real” science. | Ch 4, 7. |
| **E. System design** | Pipeline figure, graph definition, model architecture, alert schema, API surface. | Links research questions to code. | Ch 5; `assets/figure1_pipeline.png`. |
| **F. Implementation** | Preprocess → graphs → datasets → RF/MLP → GNN trainer → FL client/server → explainer → alert formatter → FastAPI. | This is the bulk of `src/` + `scripts/run_all.py`. | Ch 6; `src/`. |
| **G. Central training & baselines** | Trained RF, MLP, GNN; saved metrics and plots. | Fair comparison story. | Ch 8; `results/metrics/`. |
| **H. Federated learning** | `split_and_save` for clients; Flower server + 3 clients; logged rounds. | Privacy narrative. | Ch 8; `src/federated/`; `fl_rounds.json`. |
| **I. Explainability & SIEM** | Integrated Gradients + attention; ECS-like JSON; example alerts file. | SOC angle. | Ch 6, 8; `results/alerts/`. |
| **J. Robustness** | **Ablation:** GAT-only (no GRU). **Sensitivity:** window size × *k*. **Multi-seed:** 42, 123, 456. | Shows you did not rely on one lucky config. | Ch 8 §8.7–8.9; `run_ablation.py`, `run_sensitivity_and_seeds.py`. |
| **K. Dissertation writing** | Full UWS-style chapters; abstract (3 paragraphs, ≤200 words); appendices A–E; handbook mapping; code figure appendix. | Submission artefact. | `Dissertation_Arka_Talukder.md`. |
| **L. Submission engineering** | MD → Word script; embedded process/attendance/spec; school templates folder; GitHub README; supervisor package. | Makes submission reproducible and professional. | `dissertation_to_docx.py`; `docs/reference/school_templates/`. |

---

## 4. Dissertation chapter map (what each chapter is for)

Use this when your supervisor says *“walk me through the document.”*

| Ch | Title | Say this in one breath |
|----|--------|-------------------------|
| **1** | Introduction | Problem: IoT + SOC load; why graphs + FL + XAI; **your** research aim and sub-questions; scope (subset, 45-day, no device graph). |
| **2** | Literature Review | What others did on IDS, GNN, FL, XAI; **gap** you fill; Table 5 positioning; CyBOK §2.9. |
| **3** | Project Management | How you ran the project, risks, ethics, interim alignment. |
| **4** | Research design | Dataset description, splits, **stratified windowing** (why), metrics, FL protocol, threat to validity. |
| **5** | Design | Architecture, components, alert schema, figures (pipeline, concepts). |
| **6** | Implementation | Modules, key classes, API, config-driven behaviour, where FL and XAI hook in. |
| **7** | Testing & evaluation | What “success” means; hardware; how you measure inference and federation cost. |
| **8** | Results | **Facts only:** tables, figures, ablation, sensitivity, multi-seed; points forward to Ch 9 for meaning. |
| **9** | Conclusion & discussion | Answers RQs, strengths/limits, implications, future work; includes **four-paragraph programme-style** block in §9.2. |
| **10** | Critical self-evaluation | First-person reflection: what you’d redo, “100%” honesty, learning outcomes. |
| **11** | References | Harvard list. |
| **12** | Bibliography | Reserved / minimal. |
| **13** | Appendices | A process, B spec, C commands, D code figures, E GitHub/dataset/video; handbook 1–4 mapping table. |

---

## 5. Technical path: from CSV to SIEM alert

Tell this story **slowly** if asked *“how does data move?”*

1. **Raw CSVs** (`data/raw/train.csv`, etc.) — flows with 46 features + label.  
2. **Preprocess** (`src/data/preprocess.py`) — cleaning, **StandardScaler**, **binary** labels (benign vs attack), parquet splits.  
3. **Graph build** (`src/data/graph_builder.py`) — for each split, **separate benign/attack pools** → sliding windows → **`flows_to_knn_graph`** (Euclidean *k*NN, bidirectional edges) → list of PyG `Data` objects saved as `.pt`.  
4. **Sequences** (`src/data/dataset.py`) — group graphs into length-**5** sequences for the GRU (from `config/experiment.yaml`: `sequence_length`).  
5. **Baselines** (`src/models/baselines.py`) — RF on flattened windows / tabular; MLP same idea.  
6. **Dynamic GNN** (`src/models/dynamic_gnn.py`) — per window: node linear → 2× GAT → mean pool nodes → GRU over time → logits; `use_gru=False` for ablation.  
7. **Federated** (`src/federated/`) — same model; Flower averages weights; clients hold different graph subsets.  
8. **Explain** (`src/explain/explainer.py`) — forward with attention; IG on **last window’s** `x`; top features/nodes.  
9. **Alert** (`src/siem/alert_formatter.py`) — bundle → ECS-like nested JSON.  
10. **Serve** (`src/siem/api.py`) — load config + checkpoint; `POST /score` builds graphs with **`graph.knn_k`** from config; returns prediction + alert + timing.

---

## 6. Design decisions you must be able to justify

| Decision | If asked “why?” | Your answer (short) |
|----------|-----------------|---------------------|
| ***k*NN in feature space** | Why not a device graph? | Public release used here has **no device IDs** for a classic topology graph; *k*NN is a standard stand-in used in recent papers; agreed with supervisor. |
| **Stratified windows** | Why not random windows from the whole CSV? | Raw data is **heavily attack-heavy**; pooling by class avoids trivial “always attack” classifiers and matches graph-level labels cleanly. |
| **Window 50, *k* 5, sequence 5** | Why these numbers? | Chosen for MSc scope; **sensitivity grid** (30/50/70 × *k* 3/5/7) shows behaviour; default sits in a good cell on your subset. |
| **FedAvg + 3 clients** | Why not real geo-distributed hardware? | **Feasible simulation** within time; still demonstrates non-IID + privacy narrative. |
| **IG on last window only** | Why not full sequence IG? | **Cost and API simplicity**; Captum wrapper is defined that way in code; state limitation if challenged. |
| **ECS-like JSON** | Why not CEF or vendor format? | Programme asks for SIEM-oriented structured output; ECS is a recognisable pattern; not full Elastic integration. |
| **CPU inference** | Why not GPU? | **Edge / SOC realism** — thesis stresses commodity hardware. |

---

## 7. Results you can state (and how to defend them)

**From your committed `results/metrics/results_table.csv` (subset; do not over-generalise):**

- **Central GNN:** F1 = 1.0, ROC-AUC = 1.0, inference ~**22.7 ms**/sequence.  
- **Federated GNN:** same headline metrics in your log; inference ~**21 ms** (may differ slightly by timing).  
- **RF:** F1 ~**0.9986**, ROC-AUC ~**0.9996**; **many more false positives** on the test set than GNN (see Ch 8 / confusion matrices).  
- **MLP:** F1 ~**0.9942**, ROC-AUC ~**0.9984**.

**If they push on “100%”:**

- You used a **fixed subset** and **careful splits**; test has **934** sequences in the thesis narrative — perfect scores are **possible** but **not** “IoT security is solved.”  
- You added **ablation** (GRU matters a bit), **sensitivity** (not every cell is identical), **multi-seed** (stable on reported metrics).  
- You document **limits** in **Ch 9–10** honestly.

---

## 8. Where everything lives in the repo

| You want… | Open… |
|-----------|--------|
| Thesis text | `Dissertation_Arka_Talukder.md` |
| Generated Word | `Arka_Talukder_Dissertation_Final.docx` (regenerate with `python scripts/dissertation_to_docx.py`) |
| All hyperparameters | `config/experiment.yaml` |
| One-command pipeline | `scripts/run_all.py` |
| FL + sensitivity + seeds | `scripts/run_sensitivity_and_seeds.py`, `src/federated/run_federated.py` |
| Ablation | `scripts/run_ablation.py`, `results/metrics/ablation_table.csv` |
| Plots & alerts | `scripts/generate_alerts_and_plots.py`, `results/figures/`, `results/alerts/` |
| Code figure PNGs (Appendix D) | `scripts/render_appendix1_code_figures.py` → `results/figures/appendix1/` |
| Process / attendance originals | `archive/process_attendance/` |
| School Moodle forms (copies) | `docs/reference/school_templates/` |
| Supervisor zip bundle | `B01821011_Final_Report_Package_for_Supervisor/` |
| Checklists | `docs/reports/SUBMISSION_CHECKLIST.md`, `HANDBOOK_COMPLIANCE_REPORT.md` |

---

## 9. Reproduction order (commands)

Run from repo root (after `venv` + `pip install -r requirements.txt` + PyG).

```text
1. Place CICIoT2023 CSVs in data/raw/ (train, test, validation).
2. python scripts/run_all.py --config config/experiment.yaml
3. python scripts/generate_alerts_and_plots.py
4. python scripts/run_ablation.py --config config/experiment.yaml
5. python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
6. python scripts/render_appendix1_code_figures.py
7. python scripts/dissertation_to_docx.py
```

Federated training needs **`split_and_save`** once — exact snippet in **`SETUP_AND_RUN.md`**.

---

## 10. Supervisor & viva Q&A (question → answer)

| Question | What to say | Where to look |
|----------|-------------|----------------|
| What is the **scientific contribution**? | An **integrated** MSc-scale system: dynamic GNN + FL + XAI + SIEM-shaped alerts on IoT **flow** data, with reproducible scripts and robustness checks—not a new dataset. | Ch 1–2, 9 |
| Why **graphs**? | To encode **relational structure** between similar flows; improves separation vs flat models on **your** split, especially **false alarm** story. | Ch 5, 8 |
| Why **federated learning**? | IoT data is **sensitive** / siloed; FedAvg trains a shared model **without** centralising raw flows; you show parity on your simulation. | Ch 2, 4, 8 |
| Why **Integrated Gradients**? | Model-agnostic attributions for **feature-level** explanations; pairs with **attention** for analyst-facing cues. | Ch 4, 6; `explainer.py` |
| How do you handle **class imbalance**? | **Stratified** windowing from benign vs attack pools; minority stride for more benign windows. | Ch 4; `build_graphs_for_split` |
| Is there **data leakage**? | Train/val/test at **sequence** level after your pipeline design; you checked overlap narrative in reflection. | Ch 7, 10 |
| What is **non-IID** here? | Dirichlet-style **label skew** across three clients (`alpha` in config). | `data_split.py`, Ch 4 |
| What would you do **with 6 more months**? | Real traffic, more clients, user study with analysts, per-attack breakdown, stronger heterogeneity tests. | Ch 9 §9.8 |
| How do I **trust** the code? | Open repo, fixed seed in config, saved metrics JSON, plots generated from those metrics, `SETUP_AND_RUN.md`. | Appendix C, E.1 |
| What is **not** implemented? | Full production SIEM integration, large-scale deployment, formal human factors evaluation. | Ch 9–10 |

---

## 11. Honest limitations (say these out loud)

- **Subset** of CICIoT2023, **lab** conditions, **45-day** MSc scope.  
- **No** formal **SOC user study** — explanation quality is argued from examples + literature, not user trials.  
- **Small** federated topology (**3** clients, **10** rounds).  
- **100%** headline metrics are **subset-specific**; sensitivity shows some (**window**, *k*) cells are slightly weaker.  
- **API** and **training** both use **`graph.knn_k`** from config now — good for consistency; still not “proof” on enterprise traffic.

Saying these clearly **raises** trust; hiding them does the opposite.

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| **CICIoT2023** | Public IoT attack dataset (UNB/CIC); you use flow-level features. |
| ***k*NN graph** | Connect each flow to *k* nearest neighbours in **feature** space. |
| **PyG** | PyTorch Geometric — graph tensors and `GATConv`. |
| **FedAvg** | Federated averaging of model weights across clients each round. |
| **Flower** | Python framework used to run the FL server/clients. |
| **Integrated Gradients** | Attribution method: integral of gradients from baseline input to actual input. |
| **ECS-like JSON** | Nested JSON shaped similarly to Elastic Common Schema for alerts. |
| **Non-IID** | Client data distributions differ (here, label skew). |

---

**You are done when:** you can tell §5 in your own words without slides, answer any row in §10 in under a minute, and point to a file on disk for each claim in §7.

*This guide is descriptive of the repository state as of the final dissertation draft; if you change `config/experiment.yaml` or rerun experiments, refresh numbers from `results/metrics/` before citing them.*
