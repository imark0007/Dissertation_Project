# 5–6 minute supervisor demo — spoken script (one page)

**Title:** *Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning*  
**Student:** Arka Talukder (B01821011)  
**Read at a calm pace; pause on transitions. Total target: ~5:20–5:50.**

Use with the block timings in the master plan. Adjust names/paths to match your recording setup.

---

### Block 1 — Problem and what you built (0:00–1:00)

Hello. This project is a **research prototype** for **intrusion detection** on **IoT network flows**, aimed at **edge and SOC** settings.

Today’s software-defined IoT networks generate huge volumes of flow data. We need models that can run on **CPUs** at the edge, that do **not require pooling raw traffic** in one place when we want to train collaboratively, and that give analysts more than a black-box score: they need **explainable, SIEM-style alerts** that can feed triage and automation.

**What I built** is a full pipeline: public **CICIoT2023** flow data, turned into **time-windowed graphs**, classified by a **dynamic graph neural network** with **baselines** and **federated learning**, and exposed through a **FastAPI** service that returns **ECS-like JSON** with **feature- and node-level explanations**.

---

### Block 2 — Data and representation (1:00–2:00)

I preprocess CSV flows, normalise features with a scaler **fit on training only**, and build **windows** of flows—by default **fifty** flows per window. Inside each window I connect flows with **k-nearest-neighbour** edges in feature space, so each window is a small **graph**. The public feature set I use does **not** use device IP addresses as node identities, so this is **attribute similarity**, not a physical topology.

For the main model, I turn those windows into **sequences** of graphs and classify **sequences**: the **random forest** and **MLP** see the **same** underlying flows as **flat rows** of forty-six features, so the comparison is fair.

**Configuration** for window size, *k*, and sequence length lives in **`config/experiment.yaml`** as the single source of truth for the experiments in the thesis.

---

### Block 3 — Models and central results (2:00–3:10)

I evaluate **three** model families: **Random Forest** and **MLP** on flow rows, and a **dynamic GNN** with **GAT**—graph attention—layers and a **GRU** over the sequence of window embeddings, plus a **binary** classifier.

On the **held-out test split** I report in the dissertation, the metrics are in **`results/metrics/results_table.csv`**, with **ROC curves** and **confusion matrices** under **`results/figures`**. I’ll show **F1, ROC-AUC, and inference time in milliseconds** for the GNN and baselines. The thesis discusses **class balance** and the **lab subset**—so I’m not claiming “solved IoT security everywhere”; I’m showing a **reproducible** evaluation on this split.

---

### Block 4 — Federated learning (3:10–3:50)

For **privacy-sensitive** training, I use **Flower** with **FedAvg**: **three** clients with a **non-IID** split of the **training** graphs, **parameter** updates to the server, not raw **CSV** rows. I show **convergence** from the logged rounds and the **federated** line in the same results table. This proves a **prototype**: federated training can **match** central training on this setup—it is not a claim about national-scale deployment or every adversary model.

---

### Block 5 — Explainability and live output (3:50–5:10)

The **practical output** of the work is **analyst-facing**: **Integrated Gradients** and **GAT attention** are turned into **ranked features** and **top nodes** inside an **Elastic Common Schema**-style **JSON** **alert**, stored as examples in **`results/alerts/example_alerts.json`**.

I will walk one alert: **severity**, **ml.prediction** and **score**, and **explanation.top_features** and **top_nodes**—**why** the model focused where it did.

I then call the **running API**—**POST `/score`**—so you see a **live** **prediction** and the same **structured alert** in the **HTTP** response, suitable to wire to a **SIEM** or **SOAR** in a future deployment. The service loads the **trained checkpoint** and runs on **CPU**.

---

### Block 6 — Reproduction and limits (5:10–5:50)

**Reproducibility** is: clone the repo, follow **`SETUP_AND_RUN.md`**, and run **`python scripts/run_all.py --config config/experiment.yaml`** for preprocess through training and the **metrics** **table**. The **written dissertation** in **`submission/`** is the full research **report**—**literature, design, implementation, evaluation, and limitations**.

**Limitations in one line:** this is **CICIoT2023** in a **fixed** **subset**, **no** real **SOC** **user** **study**—strengths are **reproducibility** and a **clear** **end-to-end** **path** from **flows** to **explained** **alerts**.

---

### Outro (5:50–6:00)

Thank you, Dr. Ujjan. I’m happy to take questions on the design, the federated setup, or the evaluation.

---

*Companion files: `VIDEO_DEMO_PREFLIGHT_CHECKLIST.md`, `VIDEO_DEMO_RECORDING_GUIDE.md` in this folder.*
