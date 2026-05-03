# MSc viva coach — all phases in one document

**Student:** Arka Talukder (B01821011)  
**Thesis title:** Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning.

**Not this project (ignore old viva text):** UGRansome, XGBoost/DT/LR as main table, SHAP+Gini as your core method, McNemar, 5-fold CV, hand-written byte rules, “132 hours” from generic prompts.

**Ground truth:** `Arka_Talukder_Dissertation_Final_DRAFT.md`, `submission/B01821011_Arka_Talukder_Dissertation_Final.docx` (same story). **Metrics:** `results/metrics/*.csv` and `*.json`. If Word and JSON disagree, trust JSON/CSV.

**B1 English, spoken style. Short sentences. No em dashes.**

**Examiner weak spots to own:** F1 1.0 is on a **fixed lab subset**; no formal SOC user study; FL round 1 has ROC-AUC 0.557 in `fl_rounds.json` (then improves). RF uses scikit-learn **Gini** default (no `criterion=` in `src/models/baselines.py`).

---

## PHASE 1 — Rebuild your knowledge (what you did and why)

### 1.1 The big picture (3 to 4 sentences)

IoT flow data has to be checked for attacks, often on **CPU** at the **edge**, not only in a big data centre. Security teams need more than a single score, they need **explanations** and **JSON** a SIEM can read. This dissertation builds one **end-to-end** path: **CICIoT2023** → **kNN** graphs in windows → **GAT+GRU** model → optional **Federated** training with **FedAvg** → **Integrated Gradients** and **GAT** attention in **FastAPI** output. The difference from many papers is you **integrate** graph learning, **FL**, and **SIEM**-shaped **alerts** in one **reproducible** package with one **`config/experiment.yaml`**, not only one more accuracy result.

### 1.2 Why CICIoT2023 (one dataset, not two)

**CICIoT2023** (Pinto et al., 2023) gives **public** **train**, **val**, and **test** with **46** **numeric** **flow** **features** from IoT traffic. I picked it for **repro** and so I can **compare** with recent IoT **IDS** work. This thesis does **not** use a second dataset like UGR, the **story** is one **consistent** protocol, not a cross-dataset class-balance **study**. If they ask “why not **NSL-KDD** or **CIC-IDS2017**?” say those are **old** and not focused on the **IoT** **flow** feature setup you use. CICIoT2023 matches your **time** and **comparability** **goal**.

### 1.3 No hand-written “rule” baseline; stratified windows and the OR in time

This project does **not** use fixed IF-rules like “bytes greater than X” in your **results** tables. The design that matters is: **benign** and **attack** **pools** are **separate**; each **graph** **window** is built from **one** pool, so the window label is **not** a messy mix. For a **sequence** of **five** **windows**, the label is **attack** if **any** of the five is attack (**OR** in **time**). That is **not** the same as mixing **benign** and **attack** **flows** inside one window. I use **kNN** in **46-D** because the public **feature** list has **no** **device** **IP** field for a full **device**-level graph. The **95th** **percentile** in **old** viva text is **not** your **baseline**; you use **stratified** **windowing** and **`minority_stride: 25`** in **`experiment.yaml`** to get **enough** **benign** **graph** **samples** when raw flows are about **2.3%** **benign**.

### 1.4 Why Random Forest, MLP, GAT+GRU, and federated GNN

- **Random Forest:** strong **row**-level **baseline** on the **same** **46** **features** (`n_estimators=200`, `max_depth=20` in config).  
- **MLP:** **neural** **baseline** on the **flat** table, fair to compare with a **net**-based GNN path.  
- **Central** **GAT+GRU:** each **flow** in a **window** is a **node**, **k=5** **neighbours**, **GAT** then **GRU** over **5** **windows** (sequence).  
- **Federated** **GNN:** same **arch**, **Flower** **FedAvg**, **3** **clients**, **10** **rounds**, **non**-**IID** **(alpha** 0.5**)**.

**“Lightweight”** here means **CPU**-**first** **inference** **logging** in **ms**, not a claim that the **GNN** is the **lightest** model on every hardware. I did **not** use **XGBoost**, **DT**, **LR** as the **core** **table** **comparators**. I did **not** use a **CNN** on **raw** **packets** because the **MSc** uses **CIC** **pre**-**extracted** **flow** **features**.

### 1.5 Why Integrated Gradients and GAT attention (not SHAP on the GNN)

**IG** in **Captum** walks from a **baseline** **vector** to the **input** and **adds** how each **feature** **shifted** the **score** **(top**-**k** in **alerts**). **GAT** **attention** shows which **neighbour** **links** the model **favoured**. **SHAP** is **cited** in the **thesis** as **context** and for **tree**-style **work**, but your **GNN** **explanation** path is **IG** + **attention** in code. There is **no** “IAT rank 12 vs Gini 1” **results** file in this repo, **do** **not** say that in the viva.

### 1.6 Why this evaluation (fixed split, not McNemar, not 5-fold in results)

I use **public** **fixed** **splits** and **one**-**shot** test **reporting**. I report **F1**, **ROC**-**AUC**, **FPR**, and **confusion**-style **FP** **counts**; **not** only **accuracy**, because at **flow** level about **98%** are **attack** and **“always** **attack**” can look “accurate” and still **fail** on **benign** **FPR**. **Stability** uses **3** **seeds** (42, 123, 456) for the **GNN** in `multi_seed_summary.json`, **ablation** (GAT+GRU **vs** GAT-only), **9**-cell **sensitivity** on **(window,** **k)**, and **per**-**round** **FL** **metrics** in `fl_rounds.json`. I **do** **not** have **McNemar** or **5**-**fold** **CV** in **`results/metrics`**.

### 1.7 Deployment numbers you can say (from data, not “hours saved” in JSON)

- **GNN** **inference** about **22.7** **ms** **per** **sequence** (5 windows), **Federated** GNN about **20.99** **ms** (`results_table.csv`). **RF** about **46.09** **ms** **per** **flow** **row**; **MLP** about **0.66** **ms** **per** **row** (same file). **Compare** **units** **carefully** when they ask.  
- **False** **positives** on **our** test **run**: **GNN** **0** **,** **RF** **187** **,** **MLP** **4** **(thesis** **Chapter** **8** **and** `rf_metrics.json` / `mlp_metrics.json`). **FPR** in **json**: RF **~0.048** **,** MLP **~0.001** **. **  
- **Federated** **comms** **order** of **~3.07** **MB** per **round**, **~31** **MB** **total** over **10** **rounds** (thesis **Chapter** **8**; `comm_bytes` line **3072048** per round in `fl_rounds.json`). **Do** **not** **claim** **“132** **hours**” **from** this **JSON** path unless that **exact** **line** is in your **Word** file.

### 1.8 Five key findings (say out loud, 2 to 3 sentences each)

1. **Strong** **tabular** **baselines:** RF **0.9986** F1 and MLP **0.9942** F1 show **flow** **features** **alone** are **very** **separable** on this **split**. The **GNN** **F1=1.0** is **in** the **same** **headline** **range**, but **see** point **2** on **error** type.  
2. **False** **positive** **gap:** On **our** test **,** **GNN** has **0** **FP** **,** **RF** **187** **,** **MLP** **4** **. If** that **pattern** **held** in the **field**, it **reduces** **triage** **noise** for **benign** **rows** **wrongly** **tagged** **as** **attack** **(no** **measured** **analyst** **hours** in **your** `results` **JSON** **)**.  
3. **Federated** **GNN** can **match** **central** on **F1/ROC=1.0** in **`results_table.csv`** so **data** can **stay** on **three** **clients** while **the** **global** model **reaches** **parity** on **this** test **(sub**-**Q2** in the thesis**)**. **Early** **rounds** are **weaker** **(round** **1** **F1** **~0.983** **,** **ROC** **~0.557** in **`fl_rounds.json`**) **,** I **state** that **in** the **viva** **,** I **do** **not** **hide** it.  
4. **Explain** **+** **serve:** **IG** and **GAT** **attention** go into **ECS**-**like** **JSON** via **FastAPI** for **a** **SIEM**-**shaped** **hand**-**off** **(qualitative** **triage** **,** not a **formal** **user** **study** **)**.  
5. **Ablation** and **sensitivity:** **GAT**-**only** **(no** **GRU**)** is **F1** **0.9961** in **`ablation_table.csv`**. **Default** **(window** 50, **k=5**)** is **in** a **high**-**F1** **region** in **`sensitivity_table.csv`**; some **(50,3)**-style **cells** **drop** **.

### 1.9 Numbers cheat sheet (print this table)

| Item | Value |
|------|--------|
| **Title** | Explainable **Dynamic** **GNN** **SIEM** for **SD** IoT with **Edge** **AI** and **Federated** **Learning** |
| **Train / val / test flows** | **500,000** each split (`dataset_stats.json`) |
| **Test** flow mix (example) | **~2.35%** benign, **~97.65%** attack (same JSON) |
| **Test** **graph** **sequences** | **934** (`dataset_stats.json` → graph_sequences.test.sequences) |
| **Default** **graph** **config** | window **50**, **knn_k=5**, **sequence_length=5**, **minority_stride=25** |
| **RF** (results_table) | P **0.9989**, R **0.9984**, F1 **0.9986**, ROC **0.9996**, **46.09** **ms** |
| **MLP** | P **1.0**, R **0.9885**, F1 **0.9942**, ROC **0.9984**, **0.66** **ms** |
| **Central** **GNN** | P/R/F1/ROC all **1.0**, **22.70** **ms** **per** **sequence** |
| **Federated** **GNN** | P/R/F1/ROC all **1.0**, **20.99** **ms** |
| **RF** FP / TN / FPR (`rf_metrics.json`) | **187** / **3676** / **~0.0484** |
| **MLP** FP / TN / FPR (`mlp_metrics.json`) | **4** / **3859** / **~0.0010** |
| **GNN** test CM (thesis) | **0** **FP** **,** **0** **FN** **(Figures** **16**-**19** **text** **)** |
| **Ablation** ( `ablation_table.csv` ) | **Full** **(GAT+GRU)**: F1 **1.0** **;** **GAT** **only**: F1 **0.9961**, P **0.9923** |
| **Multi-seed** **GNN** | Seeds **42**,**123**,**456** **;** **F1** **mean** **1.0** **,** **std** **0** **(`multi_seed_summary.json`)** |
| **FL** **(Table** **8** / `fl_rounds.json`) | **10** **rounds** **,** **3** **clients** **;** **R1** F1 **~0.983** **,** **ROC** **~0.557** **;** **R7**-**10** **F1=ROC=1.0** **(rounded** in **thesis** **table** **)** |
| **FL** **per**-**round** **bytes** | **3072048** **(≈2.93** **MiB** **per** line**)** × **10** **rounds** in **`comm_bytes`**; thesis says **~3.07** **MB** **/round** and **~31** **MB** **total** **(narrative** **order**-**of**-**magnitude** **)** |
| **Model** **params** **(thesis** **Ch** **8** **)** | **~128,002** **floats** **for** **comms** **order**-**of**-**magnitude** |
| **FL** **config** | **num_clients:3**, **num_rounds:10**, **local_epochs:2**, **alpha:0.5** |
| **Python** **stack** | See **`requirements.txt`**: **torch** **≥2.0** **,** **torch**-**geometric** **≥2.3** **,** **scikit**-**learn** **≥1.2** **,** **flwr** **≥1.5** **,** **captum** **≥0.6** **,** **fastapi** **≥0.1** **,** **pyyaml** **≥6** **,** etc. **(exact** **versions** in **your** **env** if **asked** **)** **.** |

**Not in this** `results` **folder** **(do** **not** **claim** **unless** you **add** it**):** **McNemar** **,** **5**-**fold** **CV** **means** **,** **UGR** **,** second-dataset **SHAP** table **,** IAT-**Gini** **rank** story **.**


---

## PHASE 2 — Question bank (model answers, B1)

Format: **Q (likelihood):** … then **A:** 3 to 6 short sentences, with your numbers where useful.

### A. Framing and motivation (5)

**Q (HIGH):** Why this topic?  
**A:** SOCs get too many events and need tools that can run on **CPU** at the edge, not only in a data centre. I built one pipeline that goes from data to a **GNN**, optional **Federated** training, and **JSON** **alerts** with explanations. I care about false alarm load and time, not only a headline score. That matches my research question in Chapter 1.

**Q (HIGH):** What real-world problem is this?  
**A:** People must triage many IoT flow records and decide if something is a real attack. A black-box score is not enough, you need a clear story. My work is a **prototype** that still reports **FPR and FP** as well as F1, using **CICIoT2023** on a public split.

**Q (MEDIUM):** What is new here?  
**A:** I connect **kNN graphs**, **GAT+GRU**, **Flower FedAvg**, **Captum** Integrated Gradients, **GAT** attention, and **FastAPI** in one place with one **`config/experiment.yaml`**. The thesis Table 1 style gap is that most papers do not show all of these in one end-to-end build.

**Q (MEDIUM):** Who would use it in practice?  
**A:** A **research** or **R&D** team could reuse the code path as a **starting point** for a pilot. It is not a product-ready SIEM. The value is full pipeline and honest limits in Chapter 9.

**Q (LOW):** One-sentence pitch?  
**A:** I built an explainable **dynamic GNN** for IoT **flows** with **Federated** training and **SIEM**-shaped **JSON** on **CICIoT2023**, with **strong** baselines and ablation on CPU.

### B. Literature and gap (5)

**Q (HIGH):** What gap?  
**A:** Work often has **GNNs**, or **FL**, or **XAI**, or **JSON** hand-off, but not all four in one **reproducible** MSc build on the same subset. I made that **integration** the contribution, not a single SOTA claim on the whole internet.

**Q (MEDIUM):** Is fifteen to twenty core sources enough?  
**A:** The programme asked for that breadth, Section 2.10 extends beyond the core story. I use peer venues for **claims**, and tools in Chapter 12 for build facts.

**Q (MEDIUM):** How is this different from “another CIC paper”?  
**A:** I add **temporal** graph sequences, **FedAvg** with three non-IID clients, and **end-to-end** **ECS**-like **alerts** with **latency** and **communication** numbers in Chapter 8.

**Q (MEDIUM):** Why not just read surveys?  
**A:** I had to **implement** the graph, FL, and explainable stack, else I cannot defend **k**, **GRU** ablation, or **round** curves with evidence.

**Q (LOW):** Few 2025 papers?  
**A:** I locked citations for submission time; future work can add newer **IoT** **IDS** **surveys** as they stabilise.

### C. One dataset and preprocessing (8)

**Q (HIGH):** Why only CICIoT2023?  
**A:** I needed **one** public IoT **flow** set with **fixed** train, validation, and test so others can re-run. A second set was out of my MSc time box, but I name a second public set as **future** work in discussion.

**Q (HIGH):** How are splits made?  
**A:** I use the **public** file split, **500,000** **flows** per part in `dataset_stats.json`. I do not merge test into training.

**Q (HIGH):** Data leakage?  
**A:** I fit the **StandardScaler** on **train** only, then apply to validation and test. Sequences for each part are built **inside** that part. Test is not used to tune the split choice.

**Q (MEDIUM):** 46 features, no IP?  
**A:** The **config** list has **no** **IP** column. Graph edges are **kNN** in 46-D space, not a true device **topology** map, as I say in the design chapter.

**Q (MEDIUM):** What is stratified windowing?  
**A:** I keep **benign** and **attack** streams in **separate** pools. Each small graph **window** comes from one pool, so the window label is clean. I balance graph counts, then I shuffle. Raw flows are still about **2.3%** **benign** on the file you see in the stats JSON.

**Q (MEDIUM):** Why OR over five windows for the sequence label?  
**A:** I mark the sequence as attack if **any** of the five **window**-graphs is attack. A short **burst** of bad windows should still count as a hit, so I do not average away the signal.

**Q (MEDIUM):** Class weighting?  
**A:** The training block uses `class_weight_auto: true` in `experiment.yaml` for neural training so the model does not just predict the majority class at the raw level.

**Q (LOW):** Categorical encodings?  
**A:** I use the **numeric** **flow** **columns** defined in the YAML, one row per flow for baselines.

### D. Not rule baseline (5)

**Q (HIGH):** Why kNN in feature space, not a true device graph?  
**A:** I do not have a reliable **IP**-level field in the features I use. I follow attribute-graph work cited in the thesis and I state the limit in discussion.

**Q (MEDIUM):** Is Euclidean kNN too simple?  
**A:** It is a **clear** and **cited** way to get edges when topology is missing. The **GAT** then **learns** which neighbours matter, not a fixed hand rule on bytes.

**Q (MEDIUM):** Is the OR in time “too easy”?  
**A:** I report it openly. I also run **sensitivity** on window size and k in `sensitivity_table.csv` to show the design is not random.

**Q (MEDIUM):** Imbalanced at flow level but balanced for graphs?  
**A:** I use a **smaller** **stride** for the minority class and **equal** counts of **benign** and **attack** **graphs** before I build the **GNN** **sequences** as the thesis says.

**Q (LOW):** Do you have hand-cut IF-byte rules?  
**A:** **No** for the main `results` story. I removed that story from the old viva template. I use model scores and **IG** **/ attention** in JSON.

### E. RF, MLP, GAT+GRU, federated (8)

**Q (HIGH):** What does lightweight mean?  
**A:** I target **CPU** first and log **inference** **in** **ms**. GNN is about **22.7** **ms** **per** **sequence** in `results_table.csv` on the machine used for the thesis run. I do not claim a GPU farm is required for the main thread.

**Q (HIGH):** What does Random Forest see?  
**A:** **One** **row** = **one** **flow** with 46 **features**. It is strong at tabular patterns but it does not see **kNN** **edges** or **order** of windows.

**Q (HIGH):** What does the GNN see?  
**A:** I build **5** **windows**; each has **50** **flows** as nodes and **k=5** **neighbours**. I run **GAT** per window, then **GRU** on the 5 **embeddings**, then a classifier.

**Q (MEDIUM):** Why not XGBoost?  
**A:** I picked **RF** and **MLP** as the **agreed** **baselines** for **one**-row **inputs**. The story is to test if **graph**+**time** helps beyond a strong table model.

**Q (MEDIUM):** Why not a CNN on raw packets?  
**A:** I use **CIC** **pre-made** **flow** **stats**, not full **pcap** in this work, so I can finish the FL and JSON path in the MSc window.

**Q (MEDIUM):** Explain Random Forest in 60 seconds.  
**A:** I train **200** **trees** with `max_depth=20` from config, default **Gini** **impurity** in sklearn, each tree on a **bootstrap** **sample** and **feature** **subset** at each **split**; the **final** **vote** is a **soft** or **hard** **aggregate** in **sklearn**.

**Q (MEDIUM):** GAT vs GRU in one line each.  
**A:** **GAT** **weights** **neighbour** **links** in each **layer**. **GRU** **remembers** a **short** **sequence** of **window** **vectors**; ablation without GRU is in `ablation_table.csv` (F1 **0.9961**).

**Q (LOW):** Federated vs central on test?  
**A:** The **federated** **GNN** line in **`results_table.csv`** shows F1 and ROC **1.0** with central on that **test**, matching the parity sub-goal.

### F. Evaluation (8)

**Q (HIGH):** Why F1 and ROC, not only accuracy?  
**A:** The flow mix is about **98%** **attack**. A trivial **“always** **attack**” **rule** can get high **accuracy** and still miss **FPR** on **benign** traffic, so I report F1, ROC, and FPR/FP as Chapter 7 states.

**Q (MEDIUM):** Why not k-fold CV?  
**A:** I use the **public** **fixed** **split** and I add ablation, sensitivity, and **multi-seed** **(42**, **123**, **456**). I do **not** list McNemar in `results` **(not** in `results/metrics` **)**.

**Q (MEDIUM):** Why three seeds?  
**A:** A **MSc** **stability** check. **`multi_seed_summary.json`** has **F1** **mean** **1.0** and **std** **0** on the logged run.

**Q (MEDIUM):** “Always attack” and 97.7% class share?  
**A:** Yes, so I use **class**-**weighted** **training** and **graph**-level **balance** for the GNN, and I still read P, R, F1, AUC, not a single **accuracy** line on raw rows.

**Q (MEDIUM):** Are you overfitting?  
**A:** I use validation for training control. The subset **is** very separable, so the scores are high. I do **not** say this solves all IoT sites.

**Q (MEDIUM):** Is test F1 1.0 “suspicious” next to val?  
**A:** I show **convergence** in training history. I still name **static** data and **lab** scope as a **threat** in discussion.

**Q (MEDIUM):** What is FPR?  
**A:** **FP** **/ (** **FP+TN** **)** for **benign** **treated** as **negative** if you follow the same label rule as the thesis. In JSON, RF is about **0.048** FPR, MLP about **0.001**, with **187** and **4** **FPs** for RF and MLP.

**Q (LOW):** Why no external test set?  
**A:** **Time** and **scope**; I state it as a clear **next** **step** in the conclusion / future work.

### G. Results (8)

**Q (HIGH):** RF 0.9986 F1 vs GNN 1.0, what is the real difference?  
**A:** On **our** test log, the **GNN** has **0** **FP** while **RF** has **187** **FP** and MLP has **4** **(json** and thesis**)**. If that pattern held, it would **cut** a lot of **triage** **noise** for **benign** items **wrongly** **flagged**.

**Q (MEDIUM):** Ablation: why does GRU matter?  
**A:** GAT only with mean time pool gets F1 **0.9961** in `ablation_table.csv` **vs** **1.0** full model; precision is a bit lower on that run.

**Q (MEDIUM):** FL round 1: F1 0.983, ROC 0.557?  
**A:** `fl_rounds.json` shows the **first** **global** **model** is **weaker** on the dev curve used there; from **round** **7** the log is **1.0** F1 and **1.0** ROC. Round 6 has a small ROC wobble. I do not hide the early **rounds**.

**Q (MEDIUM):** GNN 22.7 ms vs RF 46.09 ms, can you directly compare?  
**A:** **Careful**: RF time is per **flow** **row**; GNN is per **sequence** of **five** **windows** of **fifty** **flows** each. Not one-to-one for cost per **single** **flow** at the GNN.

**Q (MEDIUM):** MLP 0.66 ms vs GNN, why is MLP not “best” for production?  
**A:** MLP is fast per **row** and only **4** **FPs** in `mlp_metrics.json`, but the **GNN** shows **0** **FPs** in our run. Real deploy would add ops cost for graph **batching** and the whole stack, not the raw ms only.

**Q (MEDIUM):** Sensitivity: when does F1 drop?  
**A:** In `sensitivity_table.csv`, (50,3) and some other cells drop below **1.0**; default (50,5) stays at **1.0** in the grid, which matches my config.

**Q (LOW):** MLP recall 0.9885?  
**A:** MLP has **higher** **precision** **(1.0** in table) but **lower** **recall** on **attack** than GNN; check exact trade-off in the thesis confusion for FN counts.

**Q (LOW):** Fed communication?  
**A:** `comm_bytes` in `fl_rounds.json` is **3072048** **bytes** per **round** line in the file. Thesis states order **~3.07** **MB** per **round** and **~31** **MB** **total** over **10** **rounds** with three clients. Say “about” and point to the figure, not a wire line with TLS.

### H. Explainability (6)

**Q (HIGH):** What is Integrated Gradients?  
**A:** I start from a **baseline** **vector**, move **in** a **line** to the **input**, and **add** the **signed** effect on the **logit** **for** each **feature** in **small** **steps** in Captum. I take **top-k** for the alert.

**Q (HIGH):** Why not SHAP on the GNN?  
**A:** I use **PyTorch** + **Captum** for **this** build. **SHAP** is a **known** method for **trees**; for the **GNN** I **use** **IG** **+** **attention** in code.

**Q (MEDIUM):** What does attention give an analyst?  
**A:** It **highlights** which **neighbour** **links** the **GAT** used most, as **“top** **nodes** / **top** **flows**” in **JSON** when mapped back.

**Q (MEDIUM):** Cost of 50 integration steps?  
**A:** It **adds** **forward** **passes**; I note we can run **explanations** on a **subset** of alerts in practice.

**Q (MEDIUM):** Can IG mislead?  
**A:** It is a **local** view for **one** **input**; it is **not** a **ground**-**truth** **attack** **story** by itself.

**Q (LOW):** Where are the examples?  
**A:** `results/alerts/example_alerts.json` and Section 8.6, including **variance**-related features on some **FP** **examples**.

### I. Workload / operations (5)

**Q (HIGH):** You claim specific analyst hours saved?  
**A:** I **do** **not** **measure** **hours** in a **live** **SOC** here. I give **FP** **counts** and **latency** as **factors** that **drive** **load**, and I cite the **alert**-**load** **theme** in the literature in Chapter 2.

**Q (MEDIUM):** “Three minutes per alert” in your work?  
**A:** I **do** **not** **fix** a **triage** **time** in the **main** `results` **metrics** **files** I used. I **only** use **a** time **if** the **thesis** **body** has it for a **sensitivity** line you actually wrote.

**Q (MEDIUM):** 30 minutes per alert, does your FPR point still hold?  
**A:** If triage is **long**, then **false** **positives** still **waste** more **work**; I keep the order-of-magnitude point from 187 vs 0 on our test, not a labour-time model.

**Q (MEDIUM):** Is this a real SOC measurement?  
**A:** It is a **table**-top **build** and **file**-based test. I say so in limits.

**Q (LOW):** Is FastAPI a SIEM?  
**A:** It is a **local** **HTTP** path for a **demo** **;** a **true** **SIEM** has **parsers** and **governance** **. **

### J. Limitations (6)

**Q (HIGH):** Biggest weakness?  
**A:** I train on a **static** public **file**; **F1=1.0** may not **hold** for **new** **sites** or **drift** without **retrain** and **new** **val**.

**Q (MEDIUM):** Concept drift?  
**A:** I do not **retrain** on **live** data here; the thesis names **retrain** and **monitoring** in future work.

**Q (MEDIUM):** Adversarial evasion?  
**A:** I do not run an **adversary** **study**; the model is a **classifier** test, not a full **game**-**theoretic** proof.

**Q (MEDIUM):** “Easy” dataset warnings in 2024–2025 literature?  
**A:** I accept **public** **sets** can be **too** **clean**; I add **sensitivity** and **strong** **baselines** to show the score is not a **hand**-**tuned** **test** **cheat**.

**Q (MEDIUM):** No GPU, is that a limit?  
**A:** I **chose** **edge**-**minded** **CPU** **inference** as a design **goal**; a GPU ablation is future work if needed.

**Q (LOW):** Only three clients?  
**A:** I needed a **feasible** **FL** **graph** in the **MSc** time box; I say **scale**-**out** in future work.

### K. Reproducibility, ethics, contribution (4)

**Q (HIGH):** What did you contribute?  
**A:** I built **kNN** **graphs**, the **GAT+GRU** model, the **FL** run, the **IG** **/ attention** code, the **API**, and the **ablation** / **sensitivity** / **seed** **scripts**. CIC only published the data and **labels**; the pipeline is mine.

**Q (MEDIUM):** Replicate?  
**A:** Use **`config/experiment.yaml`**, the **`results`** you freeze, and **`python scripts/run_all.py --config config/experiment.yaml`**. Appendix in the thesis has more detail for FL client launch.

**Q (MEDIUM):** Ethics approval?  
**A:** I use only **public** data with **no** **human** **subjects**; I follow the **signed** **ethics** text in the **spec** / process docs referenced in the thesis.

**Q (LOW):** GitHub?  
**A:** State **only** what **your** **final** **thesis** **says** in **Appendix** **D** or **module** **hand**-**in** **(private** or **link** as **per** **rules**). Do not invent a public URL.

### L. Future work (4)

**Q (MEDIUM):** Next step?  
**A:** Broader CIC **slice** or a **second** **dataset**, **per**-**attack** class **table**, and if possible a **user** **study** with **analysts** on the **alerts** **JSON** **. **

**Q (MEDIUM):** PhD?  
**A:** Deeper **topology** if **IPs** **return**, harder **FL** and maybe **formal** **privacy** **(DP)** **. **

**Q (MEDIUM):** Would you deploy to production?  
**A:** I would **pilot** with **tight** **watch** and **retrain** before full **roll**-**out** **. **

**Q (LOW):** Bigger k-fold?  
**A:** I could add **k**-**fold** **if** the **set** is **larger** and **if** a **second** set **arrives** **. **

### M. Hardest gotchas (10)

**Q (HIGH):** F1 = 1.0. Do you trust it?  
**A:** I trust it for **our** **fixed** test **split** and **ablation** **/ sensitivity** **show** the **set** is **separable** **. **I** **do** **not** **trust** it as **a** **global** **number** for **all** **IoT** in the world without **re**-**val** **. **

**Q (HIGH):** Brand new dataset tomorrow, same RF, same score?  
**A:** I would **retrain** and **check** a **dev** set from the new site. I do **not** **promise** 0.9986 **or** 1.0 **. **

**Q (MEDIUM):** Gini and entropy, what does your RF use?  
**A:** I did not set `criterion=` in `baselines.py`, so it is the default Gini impurity in scikit-learn RandomForest (unless a future edit changes it). The GNN does not use Gini; it is end-to-end gradient learning.

**Q (MEDIUM):** Shapley value, formal definition?  
**A:** I did not use Shapley in the core run. A short formal line is in Lundberg 2017: a fair average of marginal gains over all feature orderings in a value game, but my GNN explanations are IG and attention, not SHAP.

**Q (MEDIUM):** You say no one model is always best. What do you deploy?  
**A:** I would **not** **choose** from F1 only. I would **weigh** **FP**, **time**, **and** **ops** **cost**; on **our** test **GNN** had **0** **FP** **,** but **I** need **a** real **pilot** before I **bet** a **line** on it **. **

**Q (MEDIUM):** You did not tune hyperparams much: lazy?  
**A:** I fixed a **sensitivity** **grid** in Chapter 8 and **MSc** time **;** the **sensitivity** **table** and **ablation** show I **probed** key **(window**, **k**)** and **GRU** **,** not **endless** **sweep** **. **

**Q (MEDIUM):** MLP and scaling?  
**A:** I use **scaler** for **all** **features**; I do **not** **claim** that **Z**-**scale** is the **bottleneck**; the **GNN** **/ MLP** **use** the **prepared** **matrix** in code **. **

**Q (MEDIUM):** Why Federated if central is 1.0?  
**A:** The work is to show data can stay on three clients while the model still reaches parity on this test, for governance, not because central training fails.

**Q (MEDIUM):** If we redo the project from scratch, first change?  
**A:** I would **add** a **true** **held**-**out** set from **a** different **day** or **a** small **new** public **set** to **test** **general** **. **

**Q (MEDIUM):** “Literature is thin, only 19 references in an old viva” — your thesis?  
**A:** I answer from **this** work: Section 2.10 and Chapter 11 **;** the **old** 19-**ref** viva was **a** different **template** on **a** different **project** **,** not **this** **GNN** **thesis** **(check** your **word** count** for** the **true** number**)**. **

---

## PHASE 3 — Five-minute script (~650 words) + five opening Q&A

**[Slide 1 – Title]**  
Good morning. I am Arka Talukder. My title is **Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning**.

**[Slide 2 – Problem]**  
Software-defined networks and many IoT sites send a lot of flow data to a SOC. Teams need fast triage, often on **CPU** at the **edge**, not only in a data centre. They also get tired from bad alerts, so a score without context is not enough. So the problem is: how to detect intrusions, train in a data-safe way, and return clear alert **JSON** that an analyst can read.

**[Slide 3 – Gap]**  
Many papers have strong **GNN** ideas, or **Federated** learning, or some **XAI**, or a nice API, but they are often separate. I tried to show one **reproducible** path that **joins** all of these in one pipeline on a **public** benchmark.

**[Slide 4 – Data]**  
I use the **CICIoT2023** IoT flow set with **public** **train**, **val**, and **test** files. Each part has about **500,000** **rows** in my stats file. The raw class mix is very **attack**-**heavy** at the flow level, so I do **stratified** **windowing** and I treat **kNN** in **46** **D** because I do not have a full device **IP** graph in the public **feature** list I used.

**[Slide 5 – Pipeline]**  
I **standardise** on **train** and apply to the other **splits** so there is no scaler **leakage** from **test** **. **I** build **kNN** **graphs** in **50**-flow **windows** and **5**-window **sequences** with **GAT+GRU** **. **I** also train a **strong** **Random** **Forest** and an **MLP** on the **flat** **table** of **one** **row** per **flow** **,** to ask if **time** and **neighbours** really **add** **. **

**[Slide 6 – Federated]**  
With **Flower** and **FedAvg** I use **three** **clients** and **10** **rounds** on a **non**-**IID** **Dirichlet** **split** on **label** **ratios** **(alpha** 0.5 in **config**)**. I log **per**-**round** **F1** and **ROC** **and** **the** **bytes** **moved** **,** to show the **data** can **stay** local while the **model** still **reaches** **the** same **end**-**line** on **our** test **,** in **our** run **. **

**[Slide 7 – Explain and serve]**  
For explainability I use **Captum** **Integrated** **Gradients** and **GAT** **attention** **. They **put** top **feature** **names** and top **neighbour** **flow** **hints** into **ECS**-**like** **JSON** from a small **FastAPI** **/score** **path** **,** for **a** **SIEM**-**minded** **shape** **,** not a full **product** **.

**[Slide 8 – Headline results]**  
On **`results_table.csv`**, **Random** **Forest** is **0.9986** F1, **MLP** is **0.9942** F1, **and** the **central** and **Federated** **GNNs** are **1.0** F1 and **1.0** **ROC**-**AUC** in **our** file **. **In** the **per**-**model** **json** **,** the **GNN** has **0** **false** **positives** on the **test** we **use** **,** while **Random** **Forest** has **187** **and** the **MLP** has **4** **(false** **positives**)**. **GNN** **inference** is about **22.7** **ms** **per** **sequence** in **our** run **,** and **Federated** is about **21** **ms** in **the** **table** **. **FL** comms are about **3** **MB** **per** **round** in **our** log **,** or **~31** **MB** **in** **total** in **this** run **(Chapter** 8**)**. **

**[Slide 9 – Checks]**  
I also ran **GAT**-**only** **ablation** **(no** **GRU**), **nine** **(window,** **k**)** **sensitivity** **cells**, and **multi**-**seed** **GNN** **stability** **(42**, **123**, **456**)**. **

**[Slide 10 – Limits and close]**  
The **test** is **a** **fixed** public **lab** file **,** and **F1=1.0** is **not** a **promise** for **all** real **edge** **sites** without **a** new **re**-**val** **. **I** have **no** **formal** **analyst** **user** **study** **,** and **Federated** **round** one **in** the **log** is **weaker** before it stabilises (Chapter 8).

So in short, I did build an **end**-**to**-**end** path from **CIC** **to** **explained** **alerts** **,** and I **gave** **evidence** on **FPR** **,** **latency** **,** and **Federated** **cost** **. **Next** I would add **a** **wider** **data** **slice** or a **user** test with **a** real **triage** **queue** **. **Thank** **you** **. **

---

### Five likely opening questions (after the talk)

**Q1: Why graphs if Random Forest is already so strong?**  
**A:** I wanted to test if **neighbour** **links** in **kNN** and **order** in **time** add **value** on top of a **row**-level **classifier** **. **In** this **file** the **GNN** has **0** false **positives** while **Random** **Forest** has **187** **,** and **I** have **GAT**-**only** ablation in **`ablation_table.csv`** to **show** the **time** part **(GRU**)**. **I** do **not** **say** **GNNs** are **always** **better** on **all** data **,** I **test** the **claim** in **our** build **. **

**Q2: What does “dynamic” GNN mean in your project?**  
**A:** I **use** a **time** **sequence** of **graph** **snapshots** **(five** **windows**)**. **GAT** looks at each **window**; **GRU** links **across** **time** **. **That** is the **“dynamic”** in my **thesis** **,** not a **separate** **new** name **,** the **ablation** **strips** the **RNN** to **test** that **,** in **`ablation_table.csv`** **. **

**Q3: How do you know there is no data leakage?**  
**A:** I **fit** the **scaler** on **train** **rows** only **,** and **I** build **test** **sequences** from **test** only **,** the **splits** follow **separate** **CIC** **files** **,** the **thesis** **says** **so** in **the** method **,** and **I** can **point** to **`preprocess.py`** **in** the **code** if **you** want **,** the **key** is **the** test **is** not **in** the **scaler** **. **

**Q4: Why Federated learning if we could train everything in one place?**  
**A:** The **research** **sub**-**goal** is **to** show **a** way to **move** only **model** **weights** when **raw** **flows** **cannot** **pool** in **one** **site** **(policy** or **safety**)**. **I** also **log** the **comms** **(about** **3** **MB** **per** **round** in our JSON line**)**. **I** do **not** **claim** **full** **privacy** **like** a **formal** **DP** **proof** **,** the **work** is **a** **prototype** **,** the **thesis** **says** **that** in **the** **limits** **,** the **federated** **GNN** **still** **hits** **1.0** **F1** on **our** test **,** the **`results_table.csv`** line **. **

**Q5: Your F1 is 1.0. Is the task too easy?**  
**A:** I **agree** you **can** not **treat** this **as** a **solved** **global** **IoT** **task** **. **The** **set** in **CIC** in **our** file **is** very **separable** **,** the **GNN** **and** the **sensitivity** table **(window** **,** **k**)** and **seeds** **are** in **the** **thesis** **,** the **GNN** **is** not **a** **magic** **black** **box** on **a** new **line** with **new** **mix** of **benign** **,** the **threat** is **in** the **“static** **lab** **data**”** point** in **Chapter** **9** **. **I** can **name** the **re**-**val** as **a** first **next** **move** if **I** have **a** new **set** **. **

---

*Practise: read Phase 3 aloud, aim for 5.5 to 6.5 minutes at slow B1 speed. If you edit metrics, re-run scripts and update this file.*

---

## Related files

- **`VIVA_COACH_FULL_PROMPT.md`** — original three-phase prompt (instructions + outline).  
- **`VIVA_COACH_PHASE2_AND_3_RUN.md`** — duplicate of Phase 2 and 3 only (kept for history).  
- **`PROJECT_VIVA_MASTER_BRIEF.md`**, **`PROJECT_VIVA_CHEATSHEET_PRINT.md`** — short one-page squeeze.

**Single all-in-one document:** this file, **`VIVA_COACH_ALL_PHASES.md`**.
