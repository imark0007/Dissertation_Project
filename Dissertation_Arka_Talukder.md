# Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning

**Arka Talukder | B01821011**  
**MSc Cyber Security (Full-time)**  
**University of the West of Scotland**  
**School of Computing, Engineering and Physical Sciences**  
**Supervisor: Dr. Raja Ujjan**

---

## Front Matter

The **submitted Word/PDF** must include, **before** this technical body (insert from Moodle / School templates): **(1)** completed **Dissertation Front Sheet**; **(2)** signed **Declaration of originality**; **(3)** signed **Library release form**. Those pages are not reproduced here. The technical body below follows the programme dissertation guideline (abstract structure, acknowledgements, chapters, references, appendices).

---

## 1. Abstract

Internet of Things (IoT) and software-defined flow telemetry increase the volume of security-relevant data that Security Operations Centres must triage. Analysts need accurate detection on commodity CPUs, respect for data locality, and SIEM-friendly justified alerts—motivating explainable, federated, graph-based intrusion detection at the edge.

Much prior work treats graph models, federated learning, or explainable AI for intrusion detection in isolation; few student-scale prototypes combine all three with SIEM-shaped JSON on a public benchmark. On CICIoT2023, strong Random Forest and MLP baselines already perform well, so the dissertation asks where temporal graphs and FedAvg under non-IID splits add value when only public flow features exist (no device-level topology).

A dynamic GNN (GAT + GRU over *k*NN flow graphs) is trained with Flower FedAvg (three clients); Captum attributions and attention cues feed ECS-like alerts from FastAPI. Chapter 8 reports 100% test F1 and ROC-AUC for centralised and federated GNN on the fixed subset, fewer false alarms than Random Forest on the same split, 22.7 ms CPU inference per sequence, 31 MB federated messaging over ten rounds, plus ablation, sensitivity, and multi-seed checks. Limitations and future work are in Chapter 9.

**Keywords:** IoT security, dynamic graph neural network, federated learning, SIEM, explainable AI, edge AI, SOC, CICIoT2023

---

## Acknowledgements

I thank my **supervisor, Dr. Raja Ujjan**, for technical guidance, feedback on design and evaluation, and support throughout the project. I thank my **moderator, Muhsin Hassanu**, for reviewing interim work and helping sharpen the final report. I am grateful to **School and programme staff** for module materials and submission guidance, and to the **MSc Project co-ordinator** for administrative communication around milestones and ethics. Finally, I thank **friends and family** for patience during intensive writing and experiment runs.

---

## Table of Contents

**Acknowledgements**

**Chapter 1** – Introduction  
**Chapter 2** – Literature Review  
**Chapter 3** – Project Management  
**Chapter 4** – Research Design and Methodology  
**Chapter 5** – Design  
**Chapter 6** – Implementation and System Development  
**Chapter 7** – Testing and Evaluation  
**Chapter 8** – Results Presentation  
**Chapter 9** – Conclusion, Discussion, and Recommendations  
**Chapter 10** – Critical Self-Evaluation  
**Chapter 11** – References  
**Chapter 12** – Bibliography  
**Chapter 13** – Appendices (A: process; B: specification; C: reproducibility; D: handbook App. 1 code figures; E: handbook App. 4 optional; mapping table + Word export note inside)

### Table of Figures

| Figure | Title | Page |
|--------|-------|------|
| 1 | Research pipeline: from raw IoT flow data to explainable SIEM alerts | — |
| 2 | Confusion matrix for Dynamic GNN on test set | — |
| 3 | ROC curve for Dynamic GNN on test set | — |
| 4 | Federated learning convergence (F1 and ROC-AUC vs. round) | — |
| 5 | Model comparison: inference time and F1-score | — |
| 6 | Confusion matrix for Random Forest on test set | — |
| 7 | Confusion matrix for MLP on test set | — |
| 8 | ROC curve for Random Forest on test set | — |
| 9 | ROC curve for MLP on test set | — |
| 10 | Positioning of related work (GNN, FL, explainability, SIEM) | — |
| 11 | Taxonomy of IDS approaches relevant to IoT and this project | — |
| 12 | Conceptual flow of a dynamic GNN (GAT + GRU) for temporal graph classification | — |
| 13 | Federated learning (FedAvg) flow — no raw data leaves clients | — |
| 14 | Explainability methods used for SOC-oriented alerts | — |
| 15 | Sensitivity of central GNN performance to window size and kNN *k* | — |
| 16 | Ablation comparison: full model vs. GAT-only (F1 and inference time) | — |
| 17 | Appendix 1 — Figure A1-1: `DynamicGNN` class (GAT + GRU / ablation path) | — |
| 18 | Appendix 1 — Figure A1-2: `flows_to_knn_graph` — building one *k*NN window graph | — |
| 19 | Appendix 1 — Figure A1-3: Stratified windowing — `build_graphs_for_split` | — |
| 20 | Appendix 1 — Figure A1-4: `explain_sequence` — Integrated Gradients and top-*k* ranking | — |
| 21 | Appendix 1 — Figure A1-5: Flower federated client entry — `run_federated.py` | — |
| 22 | Appendix 1 — Figure A1-6: FastAPI `/score` — inference, explanation, ECS-style alert | — |

### Table of Tables

| Table | Title | Page |
|-------|-------|------|
| 1 | Model comparison on CICIoT2023 test set | — |
| 2 | Federated learning round-by-round metrics | — |
| 3 | Central GNN training history (loss and validation metrics) | — |
| 4 | Ablation on CICIoT2023 test set (centralised GNN variants) | — |
| 5 | Comparison of selected related work (Literature Review) | — |
| 6 | Sensitivity analysis: window size and kNN *k* (central GNN test set) | — |
| 7 | Multi-seed stability (central GNN, seeds 42, 123, 456) | — |

---

## Chapter 1 – Introduction

### 1.1 Chapter overview

This chapter establishes the **problem** (IoT and SOC alert load), **motivation** for graph-based, federated, explainable detection on **CPU edge** devices, and the **research aim** with sub-questions and **scope**. It explains how the rest of the report maps onto the **marking criteria in the agreed Project Specification** (see **§1.6** and **Appendix B**).

### 1.2 Background and Motivation

The Internet of Things (IoT) has grown quickly in recent years. Smart sensors, cameras, and industrial controllers are now common in homes, offices, and critical infrastructure. But many of these devices were not built with security in mind. They often have weak passwords, old software, and limited computing power. This makes them easy targets for attackers. When IoT devices are hacked, they can be used for botnets, data theft, or to attack other parts of the network. So, monitoring IoT traffic and finding attacks has become very important for security teams.

Software-Defined Networking (SDN) and Software-Defined IoT let you control and view network flows from one place. Flow data (statistics about traffic between endpoints) can be collected at switches and routers. This data is often enough for intrusion detection without needing full packet capture. So storage and privacy concerns are lower. The challenge is to build systems that can analyse this flow data well, explain their decisions, and run on simple edge devices.

Security Operations Centres (SOCs) watch over networks, look into alerts, and respond to incidents. They use SIEM systems to collect and analyse logs and flow data. A common problem is that these systems create too many alerts. Many of them are false positives. Analysts then spend a lot of time deciding which alerts are real and which to ignore. When an alert does not explain why it was triggered, it is harder for analysts to triage it quickly. So, there is a need for detection systems that are both accurate and explainable. SOC staff need to understand and trust the alerts they receive.

Traditional machine learning models like Random Forest or simple neural networks can work on flow data. But network traffic has a natural structure: flows relate to one another (similar statistics, shared endpoints where visible, or temporal proximity). Graph-based models can encode this by treating **flows** (or, when identifiers exist, devices) as **nodes** and linking related flows with **edges**—in this project, *k*NN edges in feature space because device IPs are not in the public release. Dynamic graph neural networks (GNNs) can learn from how these graphs change over time. This may capture attack patterns that simpler models miss. At the same time, IoT data is often spread across different sites or organisations. Sharing raw data for centralised training can raise privacy and legal issues. Federated learning lets you train a model across many clients without moving the raw data to one server. This is useful for IoT and edge environments.

Finally, many IoT and edge systems run on devices with limited hardware. Not every organisation can afford powerful GPUs at the edge. So, a system that can run on CPU-based edge devices and still provide useful detection and explanations would be more practical for real-world SOC use.

### 1.3 Research Aim and Questions

This project aims to design and build a small prototype system that can detect attacks in IoT network traffic. The system will use a dynamic graph neural network and federated learning, and it should provide simple explanations for alerts so that SOC analysts can understand and use them easily.

The main research question is:

*How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?*

To support this, the following sub-questions are addressed:

1. Does a dynamic graph model perform better than simple models like Random Forest and MLP?
2. Can federated learning maintain similar performance without sharing raw data?
3. Can the model generate useful explanations for SOC alert triage?

Answering these will help show whether the approach works well and is useful for SOC workflows. The project aims to make IoT security monitoring better. By putting graph-based detection, federated learning, and explainability in one prototype, it shows that this can be done within an MSc project. The project uses CICIoT2023, a well-known IoT security dataset. This lets the results be compared with other work in the literature.

### 1.4 Scope and Limitations

The project uses the CICIoT2023 dataset (Pinto et al., 2023) and works with a manageable subset so that experiments can be done in the available time. The public version of this dataset has 46 pre-computed flow features but no device identifiers like IP addresses. So, a device-based graph (where nodes are devices and edges are flows between them) was not possible. The **agreed project specification** (Appendix B) describes that ideal device–flow graph in its overview; **this dissertation implements flow-level nodes with kNN feature-similarity edges** because that is what the public release supports, following agreement with the supervisor. Consecutive flows are grouped into fixed-size windows. Each flow becomes a node with its 46 features. Edges connect each flow to its k-nearest neighbours in feature space. Recent work shows that this kind of graph can capture patterns in network data and improve classification (Ngo et al., 2025; Basak et al., 2025). The main model is a dynamic GNN with Graph Attention and GRU. It is compared with Random Forest and MLP. Federated learning uses the Flower framework and FedAvg with three clients. Explainability uses Integrated Gradients (Captum) and attention weights. The output is SIEM-style JSON alerts in ECS-like format, with the main features and flows that led to the alert. The system runs as a FastAPI endpoint. CPU inference time is measured to check it works for edge deployment.

The work is limited to a 45-day MSc project. So the prototype is kept small: a subset of CICIoT2023, a few federated clients, and CPU-only inference. The focus is on showing that the idea works and that the alerts and explanations are useful for SOC triage. It is not a production-ready product. If the dataset is too large, a smaller subset can be used. If federated learning is too slow, rounds or clients can be reduced. If explainability is too slow, it can be applied only to selected alerts. The project repository has the full code, config files, and scripts to reproduce all experiments.

### 1.5 Dissertation Structure

The rest of the dissertation follows a **UWS-style chapter layout** (similar to the School’s exemplar reports). **Chapter 2** reviews literature. **Chapter 3** covers project management and risks. **Chapter 4** sets out research design and methodology. **Chapter 5** presents system and graph **design** (including the pipeline figure). **Chapter 6** documents **implementation**. **Chapter 7** describes **testing and evaluation** protocol and metrics. **Chapter 8** presents **results** objectively. **Chapter 9** combines **discussion**, **conclusions**, and **recommendations** to meet the combined interpretation and closing weighting in the assessment grid. **Chapter 10** is critical self-evaluation. **Chapters 11–12** are references and bibliography. **Chapter 13** lists appendices (process, specification, reproducibility, **Handbook Appendix 1** code figures with captions and interpretations, and optional repository or demonstration).

### 1.6 Alignment with the marking criteria in the Project Specification

The dissertation is organised against the **marking criteria and weightings recorded in the agreed Project Specification** (**Appendix B**). That is the same grid you committed to in formally submitted progress work: **Introduction 5%**; **Context / Literature Review 20%**; **Research Design 20%**; **Implementation (Practical Work) 25%**; **Evaluation 5%**; **Presentation of Results 5%**; **Conclusions and Recommendations 10%**; **Critical Self-Evaluation 10%** (total **100%**).

The chapter mapping is:

| Criterion (weight) | Where it is addressed in this dissertation |
|--------------------|---------------------------------------------|
| Introduction (5%) | **Chapter 1** — aim, questions, scope, structure |
| Literature (20%) | **Chapter 2** — critical review, comparison table, figures, gap |
| Research design (20%) | **Chapters 4–5** — methodology, dataset, graph and system design |
| Implementation (25%) | **Chapter 6** — build, modules, tooling, reproducibility |
| Evaluation (5%) | **Chapter 7** — protocol, metrics, setup |
| Results presentation (5%) | **Chapter 8** — tables and figures, minimal interpretation |
| Conclusions & recommendations (10%) | **Chapter 9** — answers to questions, implications, limits, future work |
| Critical self-evaluation (10%) | **Chapter 10** — reflection on process and learning |

**Approximate balance:** In the current Markdown draft, **Chapter 2** is **about 19%** of the main-body word count — close to the **20%** literature weighting in the specification grid above.

#### 1.6.1 If your signed specification PDF shows different percentages

Some forms or drafts may still show **15%** for literature (or other variants). **What matters for you is the signed specification in Appendix B.** If that signed copy does **not** match the table above, update **this section** and the **Appendix B** file so the thesis and the embedded specification match what was formally approved.

### 1.7 Chapter summary

Chapter 1 has framed the **SOC/IoT** problem, stated the **research questions**, explained **scope and limits**, and mapped chapters to the **marking criteria in the Project Specification** (**§1.6**).

---

## Chapter 2 – Literature Review

### 2.1 Chapter overview

This chapter critically reviews **IoT intrusion detection**, **SIEM/SOC** needs, **graph and dynamic GNN** approaches, **federated learning**, and **explainable AI** in security. It ends with an explicit **gap** and this project’s **contribution**.

### 2.2 Themes and structure

The review is organised by theme. Each subsection links to the research questions. Critical analysis is given where it helps justify design choices for the prototype.

### 2.3 IoT Security and the Need for Detection

IoT devices are now common in many sectors, but they are often built with cost and convenience in mind rather than security. Researchers have pointed out that many devices use default credentials, have unpatched software, and lack strong encryption (Kolias et al., 2017). When these devices are compromised, they can be used in botnets, for data exfiltration, or as a stepping stone into the rest of the network. So, detecting malicious behaviour in IoT traffic is an important part of modern security operations.

The scale of the IoT threat is well known. Kolias et al. (2017) looked at the Mirai botnet. It used weak Telnet passwords on millions of IoT devices to launch large DDoS attacks. Such incidents show that IoT networks need continuous monitoring and automated detection. Manual inspection is not possible at scale. IoT devices vary a lot (cameras, thermostats, industrial controllers). So attack surfaces vary widely. Detection systems must work with different traffic patterns and protocols. SDN and Software-Defined IoT add another layer. Central controllers can collect flow statistics from switches and routers. But the amount of data and the need for real-time analysis require efficient detection methods that can run at the edge.

Intrusion detection for IoT can be done in different ways. Signature-based methods look for known attack patterns. Anomaly-based methods learn what is normal and flag the rest. Machine learning has been used a lot for attack detection on flow or packet data. The CICIoT2023 dataset used in this project was made to support such research. It includes attacks like DDoS, reconnaissance, and brute force from real IoT devices in a controlled environment (Pinto et al., 2023). The dataset has 46 pre-computed flow features from packet captures. This reduces the need for raw packet inspection. It also matches the kind of data that SIEM systems usually use. Pinto et al. (2023) describe how the data was collected. Traffic was captured from real IoT devices (smart plugs, cameras, doorbells) under controlled attack scenarios. Flow features were extracted using standard tools. The features include protocol indicators (TCP, UDP, ICMP, HTTP, DNS), packet and byte statistics, TCP flag counts, and statistics like mean, variance, and standard deviation. Using a public dataset lets results be compared with other work. But the setting is still lab-based, not a live network. Yang et al. (2025) recently showed that graph neural networks can improve intrusion detection for industrial IoT. They showed that modelling network structure can improve accuracy compared to flat feature models. The CICIoT2023 dataset has a class imbalance (about 97.8% attack, 2.2% benign at the flow level). This makes training harder. This project deals with it through stratified windowing and balanced graph construction.

Many studies treat each flow or packet on its own. But in reality, attacks often show up as patterns of communication between devices over time. For example, a DDoS attack may involve many flows from many sources to one target. A reconnaissance scan may show sequential probing of ports. Flat feature vectors lose this structure. Wang et al. (2025) did a scoping review. They found that graph-based approaches to network traffic analysis are used more and more. Dynamic graph models in particular look promising for capturing attack patterns that change over time. Zhong et al. (2024) surveyed graph neural networks for intrusion detection. They found a clear trend towards combining GNNs with temporal models. They also noted that explainability and federated deployment are still under-explored. These findings support the graph-based and dynamic approach in this project. A key gap in the literature is that there are few prototypes that combine graph-based detection, federated learning, and explainable alerting in one system for SOC use. Most papers focus on only one of these. Figure 11 summarises the main categories of intrusion detection approaches relevant to IoT and this project.

**Figure 11: Taxonomy of IDS approaches relevant to IoT and this project**

![Figure 11: IDS taxonomy](assets/literature_ids_taxonomy.png)

*Sources: Kolias et al. (2017); Pinto et al. (2023); Wang et al. (2025); Zhong et al. (2024). Full references in Chapter 11.*

### 2.4 SIEM, SOC Workflows, and Alert Quality

SOC teams use SIEM and related tools to collect logs and flow data and to generate alerts. A well-known problem is alert fatigue. Too many alerts and too many false positives make it hard for analysts to focus on real threats (Cuppens and Miège, 2002). If an alert does not explain why it was raised, triage becomes slower and more guesswork. Cuppens and Miège (2002) proposed alert correlation to reduce noise by grouping related alerts. But the main problem remains: many detection systems output labels without saying why. This is still relevant today.

The Elastic Common Schema (ECS) gives a standard structure for security events. It includes fields for event type, outcome, source, destination, and custom extensions. Using ECS-like output makes it easier to integrate with Elastic Security, Splunk, and other SIEM platforms. This project uses an ECS-like structure for the alert JSON. So the output can be used in existing SOC tools with little change.

Modern SIEM platforms like Splunk, Elastic Security, and Microsoft Sentinel support custom rules and machine learning models. But the use of explainable AI in alert workflows is still developing. When a model flags traffic as malicious, analysts need to know which features or events led to that decision. This helps them prioritise and avoid wasting time on false positives. Without explanations, analysts may ignore alerts (increasing risk) or over-investigate (reducing efficiency).

There is growing interest in making security tools more explainable. For example, some work has focused on explaining which features led to a detection (e.g. Lundberg and Lee, 2017, on SHAP). In a SOC context, explanations that point to specific flows, devices, or time windows can help analysts decide quickly whether to escalate. This project does this by producing SIEM-style alerts with top features and flows attached. So the output is not only a label but something an analyst can act on. The ECS-like JSON format works with common SIEM tools.

### 2.5 Graph Neural Networks and Dynamic Graphs

Graph neural networks (GNNs) work on data that has a graph structure: nodes and edges. In network security, nodes can be hosts or devices and edges can be flows or connections. GNNs combine information from neighbours and can learn patterns that depend on the graph structure. The main idea is that message passing (where each node updates based on its neighbours) lets the model capture relationships that flat classifiers cannot. Graph Attention Networks (GATs) go further. They let each node give different importance to its neighbours using attention (Velickovic et al., 2018). This helps the model focus on the most relevant connections. That is useful when some flows are benign and others are part of an attack. Han et al. (2025) proposed DIMK-GCN for intrusion detection. They showed that multi-scale graph representations can capture both local and global attack patterns. Ngo et al. (2025) looked at attribute-based graph construction for IoT intrusion detection. They showed that building graphs from feature similarity (not just network topology) can work when device IDs are not available. This supports the kNN-based graph construction in this project. When IP addresses or device IDs are absent (as in the public CICIoT2023 release), topology-based graphs cannot be built. So attribute-based or similarity-based construction is the only option.

Networks change over time. New connections appear, traffic volumes shift, and attacks develop in stages. Dynamic or temporal graph models try to capture this. One common approach is to combine a GNN with a recurrent module (e.g. GRU or LSTM). The model sees a sequence of graph snapshots and learns from how the graph evolves. Several papers have used this for fraud detection or anomaly detection (e.g. Zheng et al., 2019). Lusa et al. (2025) proposed TE-G-SAGE for network intrusion detection. It combines temporal edge features with GraphSAGE. They showed that both edge-level and temporal information improve accuracy. Basak et al. (2025) developed X-GANet for network intrusion detection. They confirmed that attention-based GNNs can get high accuracy and also give interpretable outputs. For IoT flow data, building time-windowed graphs and using a dynamic GNN (GAT + GRU in this project) is a reasonable way to test whether structure and temporality improve detection over simple models like Random Forest or MLP. The literature supports the idea that graph structure can add value. But the actual gain depends on the data and task. That is why this project compares the dynamic GNN to baselines. GNNs also cost more to run than tabular models. The trade-off between accuracy and inference time matters for edge deployment. This project evaluates it. The combination of GAT (for learning which neighbours matter) and GRU (for learning how the graph evolves) is a common pattern in temporal graph learning. This project applies it to IoT flow data with kNN graph construction. This has not been explored much in prior work. Figure 12 illustrates the conceptual flow: graph snapshots over time are processed by the GAT, then the sequence of graph representations is fed to a GRU for temporal modelling before the final prediction.

**Figure 12: Conceptual flow of a dynamic GNN (GAT + GRU) for temporal graph classification**

![Figure 12: Dynamic GNN concept](assets/literature_dynamic_gnn_concept.png)

*Sources: Velickovic et al. (2018); Zheng et al. (2019); Lusa et al. (2025); Basak et al. (2025). Full references in Chapter 11.*

### 2.6 Federated Learning and Privacy

Federated learning trains a model across many clients without putting raw data in one place. Each client trains on local data and sends only model updates (gradients or weights) to a server. The server combines them and sends back an updated model. FedAvg (McMahan et al., 2017) is a standard algorithm. The server averages the client model parameters, often weighted by how much local data each client has. This reduces privacy risk compared to sending raw IoT traffic to a central site. That matters when data comes from different organisations or locations. In IoT and edge deployments, data may be spread across many sites (different buildings, campuses, or organisations). Rules like GDPR may not allow centralising sensitive network data. Federated learning also reduces the bandwidth needed for training. Instead of transferring raw flows (which can be large), only model parameters (usually a few megabytes) are sent. So federated learning works well for bandwidth-limited edge networks. Figure 13 shows the FedAvg flow: clients train locally and send model updates to the server; the server aggregates and broadcasts the global model back.

**Figure 13: Federated learning (FedAvg) flow — no raw data leaves clients**

![Figure 13: FedAvg flow](assets/literature_fedavg_flow.png)

*Sources: McMahan et al. (2017); Lazzarini et al. (2023). Full references in Chapter 11.*

Federated learning has been used in IoT and fog-edge settings (e.g. Qu et al., 2020). Lazzarini et al. (2023) looked at federated learning for IoT intrusion detection. They found that FedAvg can get performance similar to centralised training while keeping data local. Albanbay et al. (2025) did a performance study on federated intrusion detection in IoT. They showed that non-IID data across clients is a key factor for convergence and accuracy. When clients have different attack types or different mixes of benign and malicious traffic, the global model must learn from mixed updates. FedAvg can still converge, but the number of rounds and aggregation strategy matter. Challenges include communication cost, non-IID data, and possible drops in accuracy compared to centralised training. In this project, Flower is used to implement FedAvg with three clients and a Dirichlet-based non-IID data split. The goal is to show that federated training can get performance close to centralised training. The evaluation includes communication cost and round-by-round performance. Flower was chosen because it works well with PyTorch and supports custom client and server logic.

### 2.7 Explainability in ML-Based Security

Explainability can be added to a system in different ways. Post-hoc methods explain a trained model. They show which inputs or features mattered most for a prediction. Integrated Gradients is a gradient-based method. It attributes the prediction to input features relative to a baseline (Sundararajan et al., 2017). It has useful properties like sensitivity and implementation invariance. It is implemented in Captum (Kokhlikyan et al., 2020), which works with PyTorch. For GNNs, attention weights from GAT layers can also show which edges or neighbours the model focused on. This project uses both: Integrated Gradients for feature-level explanation and attention for flow-level importance. So the alert can include both which features (e.g. packet rate, flag counts) and which flows contributed most to the prediction. Figure 14 summarises the explainability pipeline used in this project for SOC-oriented alerts.

**Figure 14: Explainability methods used for SOC-oriented alerts**

![Figure 14: Explainability pipeline](assets/literature_explainability.png)

*Sources: Sundararajan et al. (2017); Velickovic et al. (2018); Kokhlikyan et al. (2020). Full references in Chapter 11.*

Alabbadi and Bajaber (2025) showed that explainable AI (XAI) can be used for intrusion detection over IoT data streams. They confirmed that post-hoc explanation methods can make security decisions more transparent for analysts. Lundberg and Lee (2017) introduced SHAP for model interpretation. SHAP can be applied to any model. But Integrated Gradients is often preferred for neural networks because it is gradient-based and works well with backpropagation. One practical point: full explainability for every prediction can be slow. Integrated Gradients needs many forward passes (usually 50–100) to approximate the integral. For a GNN processing sequences of graphs, this can add a lot of latency. So the design allows applying explainability only to selected alerts (e.g. high-confidence positives) if needed. The literature does not always address this trade-off. This project notes it as a limitation and a possible area for future work.

Table 5 and Figure 10 summarise how selected related work maps onto the four pillars of this project (GNN/graph-based detection, federated learning, explainability, and SIEM-style alert output). All sources are given in the References (Chapter 11). No single prior study combines all four in one prototype for SOC use on edge devices; this project fills that gap.

**Table 5: Comparison of selected related work (sources: Chapter 11)**

| Study | GNN/Graph | Federated learning | Explainability | SIEM/Alert | Dataset/context |
|-------|-----------|--------------------|----------------|------------|------------------|
| Pinto et al. (2023) | — | — | — | — | CICIoT2023 dataset |
| Velickovic et al. (2018) | Yes (GAT) | — | — | — | General GNN |
| McMahan et al. (2017) | — | Yes (FedAvg) | — | — | General FL |
| Sundararajan et al. (2017) | — | — | Yes (IG) | — | Attribution method |
| Han et al. (2025) | Yes (DIMK-GCN) | — | — | — | IDS |
| Ngo et al. (2025) | Yes (attribute-based) | — | — | — | IoT IDS |
| Basak et al. (2025) | Yes (X-GANet) | — | Yes | — | NID |
| Lusa et al. (2025) | Yes (TE-G-SAGE) | — | Yes | — | NID |
| Lazzarini et al. (2023) | — | Yes | — | — | IoT IDS |
| Albanbay et al. (2025) | — | Yes | — | — | IoT IDS |
| Alabbadi and Bajaber (2025) | — | — | Yes (XAI) | — | IoT streams |
| Yang et al. (2025) | Yes | — | — | — | Industrial IoT |
| **This project** | **Yes (GAT+GRU)** | **Yes (Flower)** | **Yes (IG+attention)** | **Yes (ECS-like)** | **CICIoT2023** |

*Source: Author compilation from the papers cited in Chapter 11; table is descriptive only.*

*Note: IG = Integrated Gradients. NID = network intrusion detection. Full references in Chapter 11.*

**Figure 10: Positioning of related work — GNN, federated learning, explainability, and SIEM-style alerts**

![Figure 10: Positioning of related work](assets/literature_positioning.png)

*Sources: Pinto et al. (2023); Velickovic et al. (2018); McMahan et al. (2017); Sundararajan et al. (2017); Han et al. (2025); Ngo et al. (2025); Basak et al. (2025); Lusa et al. (2025); Lazzarini et al. (2023); Albanbay et al. (2025); Alabbadi and Bajaber (2025); Yang et al. (2025). This figure synthesises **Chapter 2** and supports the gap stated in **§2.8**.*

### 2.8 Gap and Contribution

Existing work covers IoT intrusion detection, GNNs, federated learning, and explainability separately. There is less work that combines an explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype for SOC use on CPU-based edge devices. Yang et al. (2025) and Han et al. (2025) focus on GNN architectures for detection. But they do not add federated learning or SIEM output. Lazzarini et al. (2023) and Albanbay et al. (2025) look at federated learning for IoT intrusion detection. But they do not combine it with dynamic graphs or explainable alerts. Alabbadi and Bajaber (2025) and Basak et al. (2025) look at explainability. But they do it in centralised settings. This project tries to fill that gap with a small but complete system. It has: (1) dynamic graph design from IoT flow data, (2) GAT+GRU model, (3) FedAvg via Flower, (4) explanations via Captum and attention, and (5) JSON alerts in ECS-like format. The evaluation answers whether the graph model beats simple baselines, whether federated learning keeps performance, and whether the explanations are useful for triage. The scope is limited to a 45-day MSc project and a subset of CICIoT2023. But the structure is set up so the results can be discussed critically and extended later. In summary, this project delivers a prototype that combines all five: kNN graph construction, dynamic GNN, federated learning, explainability, and SIEM-style alerts. No single prior work combines all five in one system for SOC use on CPU-based edge devices.

### 2.9 Mapping to CyBOK (Cyber Security Body of Knowledge)

The **agreed project specification** (Appendix B) records this work against **CyBOK** knowledge areas. The dissertation aligns with the following topics in particular: **Attacks & Defences → Security Operations & Incident Management** (SIEM-style alerting, SOC triage, and intrusion detection outputs); **Infrastructure Security → Network Security** (flow-based analysis and IoT traffic classification); **Systems Security → Distributed Systems Security** (federated learning without centralising raw client data); and **Software and Platform Security → Software Security** (engineering of the prototype pipeline, APIs, and reproducible scripts). Together, these areas situate the project within the School’s CyBOK mapping and match the specification’s indicative coverage.

### 2.10 Chapter summary

The literature supports using **dynamic GNNs**, **federated learning**, and **XAI** for intrusion detection, but rarely **all three** together with **SIEM-oriented** outputs for edge SOC workflows. This gap motivates the prototype evaluated in later chapters.

---

## Chapter 3 – Project Management

### 3.1 Chapter overview

This chapter records **planning, milestones, risks, and ethics** in the style expected of UWS MSc project reports—without duplicating the technical methodology in Chapters 4–7.

### 3.2 Project plan and timeline (45-day MSc)

Work was staged in six phases: (1) freeze requirements and literature; (2) fix the CICIoT2023 subset and preprocessing; (3) implement graph construction and central GNN training; (4) run baselines and Flower-based federated training; (5) add explainability, alert JSON, and FastAPI; (6) run ablation, sensitivity, multi-seed experiments, figures, and final writing. Each phase had concrete artefacts (metrics files, checkpoints, plots) to reduce ambiguity about “done.”

### 3.3 Risk assessment

| Risk | Mitigation |
|------|------------|
| Training too slow on CPU | Fixed subset; configurable rounds/clients; early stopping |
| Severe class imbalance | Stratified windowing; class weights; balanced graph-level labels |
| Federated instability | Three clients; reproducible split; checkpoints each round |
| Explainability runtime | Top-*k* attributions; optional subset of alerts |
| Scope creep (extra models) | Baselines limited to RF and MLP; one main GNN architecture |

### 3.4 Ethics and data

Only **public** CICIoT2023 data is used; there are no human participants or proprietary organisational datasets. The **signed specification ethics section** records supervisor confirmation that the project **does not require** full ethics-review board approval under School procedures (February 2026), which matches the use of non-sensitive public data only. Process and ethics paperwork are referenced in **Appendix A/B** as required by the handbook.

### 3.5 Interim report feedback incorporated

The MSc Project Handbook requires that feedback on the **interim report** (written and verbal from supervisor and moderator) be taken into account in the final submission. After interim submission, feedback reinforced: (1) keeping the **end-to-end** story visible (data → model → federation → explainability → SOC-shaped output), not only literature; (2) tightening the **evaluation plan** before reporting numbers; (3) being explicit about **limits** of a lab subset and very high test metrics; and (4) reserving effort for **final analysis and write-up** rather than scope creep.

Actions taken in the final dissertation include: completing the full evaluation and results chapters (**7–8**) with tables and figures promised at interim; adding **ablation**, **sensitivity**, and **multi-seed** checks so robustness is evidenced (going beyond the interim contingency of dropping sensitivity if time ran short); strengthening **critical reflection** (**Chapter 10**) on methodology, federation, and the “100% metrics” question; and documenting **reproducibility** (**Chapter 13**). Any remaining suggestions from the interim review are acknowledged as **limitations** and **future work** in **Chapter 9**.

### 3.6 Chapter summary

Transparent **time-boxing** and **risk controls** kept the project feasible while still delivering graph, federated, explainable, and deployable components.

---

## Chapter 4 – Research Design and Methodology

### 4.1 Chapter overview

This chapter explains **how** the research questions are answered: research stance, **dataset**, **model choices**, **federated setup**, **explainability**, and **evaluation plan**. **Graph and pipeline design** are expanded in Chapter 5; **build details** appear in Chapter 6.

### 4.2 Research Approach

The project is practical. The aim is to build a working prototype and evaluate it with clear metrics. The main research question is answered by building the system, comparing the dynamic GNN to baselines, testing federated learning, and checking whether the explanations help SOC triage. Sub-questions are answered through experiments and by looking at example alerts.

The methodology follows a design-science approach. A prototype is built and evaluated with quantitative metrics (precision, recall, F1, ROC-AUC, inference time, communication cost) and qualitative assessment (interpretability of example alerts). The choice of CICIoT2023, kNN graphs, GAT+GRU, Flower, and Captum is justified by the literature and by practical constraints (dataset availability, 45-day timeline, CPU-only deployment). The evaluation plan is defined before experiments are run. So results can be interpreted objectively.

### 4.3 Dataset and Subset

The CICIoT2023 dataset (Pinto et al., 2023) is used. It has network flow data from IoT devices under various attacks (DDoS, DoS, reconnaissance, brute force, spoofing). The full dataset is large (millions of flows). So a manageable subset is selected to fit the 45-day timeline. The subset includes both benign and attack traffic and a spread of attack types. The same subset is used for all experiments so results are comparable. Data is split into fixed train, validation, and test sets (e.g. 70% train, 15% validation, 15% test, with no overlap). This avoids data leakage and allows fair comparison between centralised and federated training.

The 46 flow-level features in CICIoT2023 include packet and byte counts, duration, protocol indicators (TCP, UDP, ICMP), TCP flag counts (SYN, ACK, RST, PSH, FIN), and statistics (mean, variance, standard deviation). These features are standardised before training and graph construction. The binary label indicates benign vs. attack. This project focuses on binary classification for simplicity. The subset ensures both classes are represented. The test set is held out from the start. So no information from the test set influences model design or hyperparameter choice. The config file (config/experiment.yaml) specifies the 46 flow feature columns. These include flow_duration, Header_Length, Protocol Type, Rate, Srate, Drate, TCP flag counts, protocol indicators, and statistics.

### 4.4 Models

Three types of models are used so that the value of the graph and temporal structure can be assessed.

- **Random Forest:** A baseline on **tabular flow-level data**: each training/test instance is **one processed flow** (the same 46 standardised features as in the graph pipeline), with no graph structure. Implemented via scikit-learn on the train/test parquet splits (`src/models/baselines.py`).
- **MLP (Multi-Layer Perceptron):** The same **flow-level** representation as Random Forest—a feed-forward network on 46-dimensional rows without graph or temporal modelling (`MLPBaseline` in `src/models/baselines.py`).
- **Dynamic GNN (GAT + GRU):** The main model. Graph Attention layers process each graph snapshot; a GRU (or similar RNN) then processes the sequence of graph representations over time. The output is used for attack vs. benign (or multi-class) prediction. This model uses both structure and temporality. The GAT allows the model to weight neighbours differently, and the GRU captures how the graph representation changes across the sequence.

All models are trained to predict attack (or attack type) from the available features and, for the GNN, from the graph and its evolution. Hyperparameters (e.g. learning rate, number of layers, hidden size) are chosen with the validation set; the test set is used only for final reporting. The same train/validation/test splits are used for all models to ensure fair comparison.

### 4.5 Federated Learning Setup

Federated learning is implemented with the Flower framework. FedAvg is used: each client trains the model on its local partition of the data and sends updated parameters to the server; the server averages them and broadcasts the global model back. The data is split across 2–3 clients in a way that mimics distributed data (e.g. by device or by time period). The number of federated rounds and local epochs is set so that training finishes in reasonable time; if needed, rounds or clients can be reduced to meet the 45-day constraint. The same model architecture is used in federated and centralised training so that performance can be compared fairly. Communication cost is approximated (e.g. bytes per round for model parameters) and reported.

The Dirichlet distribution (alpha=0.5) is used to partition the training data across clients. Each client samples from a Dirichlet distribution over class proportions, so that one client might have 80% attack and 20% benign, another 60% attack and 40% benign, and so on. This creates non-IID conditions that are more realistic than random IID splits. The alpha parameter controls the degree of heterogeneity: smaller alpha means more heterogeneous client distributions. Alpha=0.5 was chosen to create noticeable but not extreme heterogeneity.

### 4.6 Explainability

Two forms of explanation are used:

- **Integrated Gradients (Captum):** Applied to the chosen model (e.g. the dynamic GNN or the MLP) to attribute the prediction to input features. The top contributing features are extracted and included in the alert.
- **Attention weights:** For the GAT-based model, attention weights over edges or neighbours are used to identify which flows or connections the model focused on. These can be summarised as “top flows” or “top edges” in the alert.

Explanations are produced for selected predictions (e.g. positive alerts). The top-k (k=5) features and flows are included in each alert to keep the explanation concise. If computing explanations for every prediction is too slow, they are applied only to a subset (e.g. high-confidence alerts) and this is noted as a limitation.

### 4.7 Evaluation Plan

Evaluation is designed to answer the three sub-questions and support the main research question.

- **Metrics:** Precision, recall, F1-score, ROC-AUC, and false positive rate on the fixed test set. A confusion matrix is reported.
- **Model comparison:** Centralised training: Random Forest, MLP, and dynamic GNN are compared on the same splits. Federated training: the same GNN (or MLP) is trained with FedAvg and compared to its centralised version.
- **Time and cost:** Time-window analysis (e.g. performance across different window sizes or positions), federated round-by-round performance, approximate communication cost (bytes), and CPU inference time per sample or per batch.
- **Explainability and SOC use:** Three to five example alerts are generated with full explanations (top features and flows). These are discussed in terms of whether they would help a SOC analyst triage (e.g. whether the highlighted flows and features are interpretable and actionable).

Risks are handled as follows: if the dataset is too large, a smaller subset is used; if federated learning is too slow, rounds or clients are reduced; if explainability is too slow, it is applied only to selected alerts. All choices are documented so that the work is reproducible and the limitations are clear. The evaluation plan explicitly links each metric and comparison to a research question: precision, recall, F1, ROC-AUC, and false positive rate address sub-question 1 (model comparison) and sub-question 2 (federated vs. centralised); round-by-round metrics and communication cost address sub-question 2; example alerts with explanations address sub-question 3; CPU inference time addresses the edge deployment aspect of the main question.

### 4.8 Chapter summary

Chapter 4 has defined the **research stance**, **data**, **models**, **federation**, **XAI ingredients**, and **evaluation plan**. **Chapter 5** gives structural graph and pipeline design; **Chapter 6** gives implementation detail.

---

## Chapter 5 – Design

### 5.1 Chapter overview

This chapter presents **what** is built at a design level—**graphs**, **temporal sequences**, and **alert/deployment shape**—before the implementation detail in Chapter 6.

### 5.2 Graph design (flows to kNN snapshots)

The graph is built from flow data to capture structural relationships between network flows within each observation window.

- **Nodes:** Each node represents one flow record. The node feature vector consists of the 46 pre-computed flow-level features provided by the CICIoT2023 dataset (e.g. packet rate, byte count, flag counts, protocol indicators, statistical aggregates). Device-level identifiers (IP addresses) are not available in the publicly released version of this dataset, so a device-based graph design (where nodes are devices and edges are flows between them) was not feasible. Instead, a feature-similarity approach is adopted, following the principle demonstrated by Ngo et al. (2025) that attribute-based graph construction can be effective for intrusion detection when topology information is absent.
- **Edges:** For each window, a k-nearest-neighbour (kNN) graph is constructed in feature space. Each flow is connected to its *k* most similar flows (by Euclidean distance on the 46 features), producing undirected edges. This creates a graph structure where flows with similar characteristics are linked, enabling the GNN to learn from local neighbourhoods of related traffic patterns. The choice of *k* affects the graph density: too small and the graph may be too sparse for effective message passing; too large and computation increases. *k* = 5 was chosen as a balance (see sensitivity analysis in Chapter 8).
- **Windows and sequences:** Flows are grouped into fixed-size windows (e.g. 50 flows per window). To handle extreme class imbalance at flow level, the implementation (`src/data/graph_builder.py`) builds each window **only from flows in a single class pool** (benign or attack); the **graph-level label is that pool’s class** (within a window all flows share the same label, so this matches a unanimous vote but is defined by construction, not by aggregating mixed labels). Benign and attack windows are balanced, then **shuffled** into one list. Training samples are **sequences of five consecutive graphs in that list**; the **sequence-level label** is attack if **any** of the five windows is attack-labeled (`GraphSequenceDataset` in `src/data/dataset.py`). The GRU reads the five graph embeddings in order. Five windows was chosen to balance context with memory and training time.

This design is kept simple so that it can be implemented and tested within the project scope. The kNN approach applies to any flow-feature dataset regardless of whether device identifiers are present. More complex designs (e.g. device-based graphs when IPs are available, multi-hop temporal aggregation) could be explored in future work.

### 5.3 Pipeline, alerts, and deployment (conceptual)

Alerts are output in a **SIEM-style JSON** format, aligned with ideas from **ECS** where useful: event type, timestamp, source/target information, label, score, and an **explanation** field (top features / top flows). The runtime path is a **FastAPI** service that runs inference (and optionally explanations) on **CPU**, matching the edge objective.

**Figure 1: Research pipeline — from raw IoT flow data to explainable SIEM alerts**

![Figure 1: Research pipeline](assets/figure1_pipeline.png)

*Source: Author’s own diagram; reflects the implementation and data path described in Chapters 5–6 and **Appendix C**.*

### 5.4 Chapter summary

The design encodes **relational** (*k*NN), **temporal** (window sequences), and **SOC-facing** outputs in one architecture.

---

## Chapter 6 – Implementation and System Development

### 6.1 Chapter overview

This chapter describes **how** the prototype was built—modules, libraries, training loops, federated orchestration, explainers, and the API.

Data loading and graph construction, model implementation, federated learning, explainability with Captum and attention, alert generation, and the FastAPI deployment are all implemented in Python. The code is organised in the `src/` package with sub-modules for data processing (`src/data/`), models (`src/models/`), federated learning (`src/federated/`), explainability (`src/explain/`), SIEM output (`src/siem/`), and evaluation (`src/evaluation/`). A master pipeline script (`scripts/run_all.py`) orchestrates the full workflow from preprocessing to results.

The implementation prioritises modularity and reproducibility. Each component (data loading, graph construction, model training, federated learning, explainability, alert formatting) can be run independently or as part of the full pipeline. Configuration is externalised in YAML files so that experiments can be reproduced without code changes. The codebase follows Python conventions (e.g. type hints where helpful, docstrings for public functions) to support maintenance and extension.

### 6.2 Environment and Tools

The system is implemented in Python 3.10. PyTorch 2.x is used for the neural models (GNN and MLP), and PyTorch Geometric (PyG) provides graph operations and GAT layers. Scikit-learn is used for Random Forest and for metrics (precision, recall, F1, ROC-AUC, confusion matrix). The Flower framework is used for federated learning (client and server logic). Captum is used for Integrated Gradients. FastAPI is used for the REST API. The code is structured so that data loading, graph building, training, and inference can be run separately or together. All experiments are run on CPU to match the edge deployment goal; training time is therefore longer than with a GPU but remains feasible for the chosen subset.

The project structure follows a modular design. The `src/data/` module handles CSV loading, label encoding, and train/validation/test splitting. The `src/models/` module defines the Dynamic GNN (GAT + GRU), MLP, and interfaces for scikit-learn Random Forest. The `src/federated/` module implements Flower client and server logic, including data partitioning via Dirichlet distribution for non-IID simulation. The `src/explain/` module wraps Captum's Integrated Gradients and extracts attention weights from GAT layers. The `src/siem/` module formats alerts in ECS-like JSON. Configuration is managed via YAML files (`config/experiment.yaml`) so that hyperparameters can be changed without modifying code. The configuration specifies window size (50 flows), kNN k (5), sequence length (5 windows), federated rounds (10), local epochs (2), and model hyperparameters (hidden dimensions, dropout, learning rate). All experiments use seed 42 for reproducibility.

### 6.3 Data Loading and Preprocessing

The CICIoT2023 subset is loaded from CSV (or the format provided by the dataset). Required fields include identifiers for source and destination (e.g. IP or device ID where available), flow-level features (e.g. packet count, byte count, duration, protocol), timestamp, and label (benign vs. attack or attack type). Missing values are handled (e.g. dropped or filled with median), and labels are encoded numerically (0 for benign, 1 for attack). The data is split into train, validation, and test sets with a fixed seed so that splits are reproducible. For federated learning, the training set is partitioned across clients using a Dirichlet distribution (alpha=0.5) over class labels, so each client receives a different proportion of benign and attack samples—simulating non-IID conditions where different sites observe different threat profiles. The same test set is used for all models to ensure fair comparison.

Preprocessing includes feature standardisation (zero mean, unit variance) using statistics computed only on the training set, then applied to validation and test to avoid leakage. For the GNN, flows are windowed and converted to graphs as in **§6.4**. For Random Forest and MLP, **the same processed splits** are used in tabular form with **exactly one row per flow** (46 features + binary label), loaded by `load_processed_split` in `src/models/baselines.py`—not window aggregates. Graph-sequence construction uses `src/data/dataset.py`; configuration is in `config/experiment.yaml`.

### 6.4 Graph Construction

For each window, a kNN similarity graph is built from the flow records. All flows in the window become nodes, each carrying its 46-feature vector. A k-nearest-neighbour search (with k = 5 and Euclidean distance) is performed on the feature matrix, and bidirectional edges are created between each flow and its k nearest neighbours. The choice of k = 5 balances connectivity (enough edges for message passing) with sparsity (avoiding overly dense graphs that increase computation). Features are standardised (zero mean, unit variance) before distance computation to prevent scale-dominated distances. The kNN graph is undirected: if flow A is among the k nearest neighbours of flow B, then B is also among the k nearest neighbours of A, and a bidirectional edge is created. This symmetry ensures that message passing can flow in both directions along each edge.

Stratified windowing, single-class pools, balancing, shuffle, and kNN edge construction follow **§5.2** and `build_graphs_for_split` / `flows_to_knn_graph` in `src/data/graph_builder.py`. Random Forest and MLP consume **per-flow** rows from the processed parquet files as above—not graph windows.

### 6.5 Model Implementation

**Random Forest:** Implemented with scikit-learn on **one row per processed flow** (46 features + label). The implementation uses 200 trees and max_depth=20; class weights address flow-level imbalance in the training parquet.

**MLP:** Feed-forward network with three hidden layers (128, 64, 32), ReLU, dropout 0.2; **two output logits** for benign vs. attack, trained with **cross-entropy** and Adam (learning rate 1e-3), matching `MLPBaseline` and the training loop in `scripts/run_all.py`.

**Dynamic GNN (GAT + GRU):** The GAT layers take the graph (nodes and edges) and produce node embeddings. The GAT uses 4 attention heads and a hidden dimension of 64. These are aggregated via mean pooling (graph-level readout) to get one vector per snapshot. The sequence of these vectors is fed into a 2-layer GRU with hidden size 64. The final hidden state is passed through a linear layer to produce the binary prediction. The implementation uses PyTorch and PyG for the GAT. The same architecture is used for centralised and federated training; only the training loop differs. Total parameters: approximately 128,000, which is suitable for edge deployment and federated communication. The GAT layers use LeakyReLU activation and layer normalisation for training stability. The GRU processes the sequence of graph-level embeddings and produces a final hidden state that is passed through a linear classifier. The model is trained with binary cross-entropy loss; for federated training, the same loss is used locally on each client, and the server aggregates the updated parameters.

### 6.6 Federated Learning (Flower)

The Flower server is configured to run FedAvg: it waits for client updates, averages the model parameters (optionally weighted by local dataset size), and sends the updated model back. Each client loads its local partition of the data, trains the model for a fixed number of local epochs per round, and sends the updated parameters to the server. The process repeats for a set number of rounds (10 in this project). Model parameters are serialised and sent over the network; the size of these messages is used to approximate communication cost in bytes. The global model is evaluated on the central test set after each round (or at the end) to track performance. No raw data is sent between clients and server, only model weights or gradients.

The Flower framework was chosen for its PyTorch compatibility and its support for custom client and server strategies. The client logic (`src/federated/client.py`) loads the local data partition, initialises the model with the global weights received from the server, trains for a fixed number of local epochs per round (set in `config/experiment.yaml`, e.g. 2), and returns the updated parameters. The server logic (`src/federated/server.py`) aggregates the client updates using FedAvg (simple average of parameters) and broadcasts the new global model. The number of clients (3) and rounds (10) were chosen to balance convergence speed with simulation realism; in a real deployment, more clients and rounds might be used. The Dirichlet-based data split ensures that each client has a different class distribution, which tests whether FedAvg can handle non-IID data—a known challenge in federated learning. The Flower server runs in a separate process; clients connect, receive the global model, train locally, and send updates. The server aggregates updates using FedAvg (equal weighting in this implementation) and broadcasts the new global model. Model checkpoints are saved after each round for analysis and for resuming interrupted runs. The federated simulation can be run locally with multiple client processes or distributed across machines for a more realistic deployment test.

### 6.7 Explainability

**Integrated Gradients:** For a given input (e.g. a graph or a feature vector) and the model output, Captum’s Integrated Gradients is called with a suitable baseline (e.g. zero vector or mean feature vector). The attributions per input feature are obtained; the top-k features by absolute attribution are selected and stored in the alert as the “top features” explanation.

**Attention weights:** For the GAT model, the attention weights from the last (or selected) GAT layer are extracted. They indicate how much each edge or neighbour contributed. These are mapped back to flow identifiers (e.g. source–destination pairs) and the top flows are written into the alert. If the model has multiple layers, a simple strategy (e.g. average or use the last layer) is applied and documented.

Explanations are attached to the alert JSON. The number of integration steps (default 50) trades off attribution accuracy with computation time. The `src/explain/explainer.py` module wraps both methods and produces a unified explanation object. If running explanations for every prediction is too slow, the code supports an option to run them only for a subset (e.g. when the model confidence is above a threshold).

### 6.8 Alert Generation and SIEM-Style Output

When the model predicts an attack (or a specific attack type), an alert object is built. It includes: event type (e.g. “alert” or “detection”), timestamp, source and destination (e.g. IP or device ID), predicted label, confidence score, and an explanation object. The explanation object contains the list of top features (from Integrated Gradients) and/or top flows (from attention). The structure follows ECS-like conventions where possible (e.g. event.category, event.outcome, and custom fields for explanation). The alert is returned as JSON. This format is suitable for ingestion into a SIEM or for display in a SOC dashboard. The `src/siem/alert_formatter.py` module implements the ECS-like structure with fields such as `event.category`, `event.outcome`, and a custom `explanation` object holding `top_features` and `top_nodes` or `top_flows`. Severity levels (low, medium, high) are derived from the confidence score.

### 6.9 FastAPI Deployment and CPU Inference

A FastAPI application is set up with an endpoint that accepts input (e.g. a single flow, a batch of flows, or a pre-built graph). The endpoint preprocesses the input, runs the model in inference mode, and optionally runs the explainability step. The response is the alert JSON. CPU inference time is measured (e.g. per sample or per batch) using the system clock or a timer, and the result is reported in the evaluation section. No GPU is required; the design targets edge devices with CPU only. The API can be run locally or in a container for demonstration.

The FastAPI app (`src/siem/api.py`) loads the trained model from checkpoint at startup. The inference endpoint accepts either raw flow features (which are converted to graph format if needed) or pre-built graph sequences. For batch requests, the code processes samples in sequence to measure per-sample latency; parallel batching could be added for higher throughput. The measured inference times (e.g. 22.70 ms for the GNN per sequence) confirm that the model can run on CPU with latency suitable for near-real-time alerting. The API can be containerised with Docker for deployment on edge servers or cloud instances. The `scripts/run_all.py` script orchestrates preprocessing, graph construction, baseline training, central GNN training, and metric/plot outputs for the centralised path. Federated training is run via the Flower entry points (see **Appendix C**). The `scripts/generate_alerts_and_plots.py` script produces example alerts, the FL convergence plot, and additional figures. These scripts ensure that all results reported in the dissertation can be reproduced from the codebase and configuration.

### 6.10 Chapter summary

Implementation delivers a **modular** codebase with scripted **reproducibility**, covering data through to **SIEM-style** JSON and **CPU** inference.

---

## Chapter 7 – Testing and Evaluation

### 7.1 Chapter overview

This chapter specifies **how** experiments were run and **what** was measured—**metrics**, **splits**, **statistics**, and **comparison logic**—so that Chapter 8 can present results objectively.

### 7.2 Evaluation scope

Evaluation is designed to answer the three sub-questions and to support the main research question. This section describes the experimental setup, metrics, and how the results are produced and compared. The evaluation plan was defined before experiments were run. So the choice of metrics and comparison design was not influenced by the results.

### 7.3 Experimental Setup

All experiments use the same fixed train, validation, and test split from the CICIoT2023 subset. The random seed is fixed (e.g. 42) so that results are reproducible. Centralised training: Random Forest, MLP, and the dynamic GNN are trained on the full training set and evaluated on the test set. Federated training: the same GNN is trained with Flower and FedAvg across 3 clients; the global model is evaluated on the same test set after each round. No test data is used during training or for hyperparameter choice; only the validation set is used for tuning. Time windows for graph construction are set to a fixed length (e.g. 50 flows per window, 5 windows per sequence); sensitivity to window size can be checked in a time-window analysis if time allows.

The federated data split uses a Dirichlet distribution (alpha=0.5) to simulate non-IID conditions: each client receives a different proportion of attack types and benign traffic, mimicking scenarios where different sites observe different threat profiles. This is more challenging than IID splits and tests whether FedAvg can handle heterogeneity. The central test set is held out and used only for evaluation; it is never seen by any client during training. The experiment configuration specifies 200 trees and max_depth 20 for Random Forest, hidden dimensions [128, 64, 32] and dropout 0.2 for the MLP, and 2 GAT layers with 4 heads, GRU hidden size 64, and dropout 0.2 for the Dynamic GNN. Early stopping with patience 5 is used for the neural models to prevent overfitting. Class weights are computed automatically from the training set to address any remaining imbalance. All models are trained to convergence (or early stopping) before final evaluation on the test set.

### 7.4 Metrics

Classification performance is measured with: precision, recall, F1-score (macro or weighted if multi-class), ROC-AUC, and false positive rate. A confusion matrix is reported for the test set. These metrics are derived from the confusion matrix, where TP (true positives), TN (true negatives), FP (false positives), and FN (false negatives) are the counts of correct and incorrect predictions for the attack class (positive) and benign class (negative). The formal definitions are:

**Precision** (fraction of predicted attacks that are correct):
*Precision = TP / (TP + FP)*

**Recall** (fraction of actual attacks that are detected; also called sensitivity or true positive rate):
*Recall = TP / (TP + FN)*

**F1-score** (harmonic mean of precision and recall):
*F1 = 2 × (Precision × Recall) / (Precision + Recall)*

**False Positive Rate** (fraction of benign traffic incorrectly flagged as attack):
*FPR = FP / (FP + TN)*

**ROC-AUC** (area under the Receiver Operating Characteristic curve): The ROC curve plots true positive rate (TPR = Recall) against false positive rate (FPR) at varying classification thresholds. AUC is the area under this curve; a value of 1.0 indicates perfect ranking (all positives above all negatives), and 0.5 indicates random guessing.

These metrics show how well the model separates attacks from benign traffic and how many false alarms it produces. Precision is important for SOC use because false positives add to alert fatigue. Recall matters for not missing real attacks. F1-score balances both. ROC-AUC shows how well the model ranks positive instances higher than negative ones across thresholds. For federated learning, the same metrics are computed after each round so convergence can be seen. Communication cost is approximated from the size in bytes of the model parameters sent per round. CPU inference time is measured on the FastAPI endpoint and reported in milliseconds per sample. Inference time matters for edge deployment. If the model takes too long, it may not be suitable for real-time alerting.

### 7.5 Dataset and Experiment Statistics

The CICIoT2023 subset used in this project was selected to include a representative mix of benign and attack traffic. Each split (train, validation, test) contains 500,000 flows; at flow level the class distribution is approximately 2.3% benign and 97.7% attack (see `results/metrics/dataset_stats.json`, generated by `scripts/dataset_statistics.py`). The stratified windowing strategy produces balanced graph-level data (approximately 50% benign, 50% attack windows) despite this flow-level imbalance. After windowing, the GNN dataset comprises 920 training sequences, 928 validation sequences, and 934 test sequences (each sequence = five **consecutive** graph snapshots in the shuffled graph list; each snapshot = 50 flows; sequence label = attack if any snapshot is attack-labeled—see `GraphSequenceDataset`). The train/validation/test split ensures no overlap; the test set is used only for final reporting. The Random Forest and MLP baselines use the same processed flows in tabular format (one row per flow) on the same splits, so comparison is on the same underlying data; the GNN is evaluated at sequence level (one prediction per sequence of 5 windows). The federated split assigns each client a different proportion of classes via Dirichlet(alpha=0.5), simulating non-IID conditions where different sites observe different threat profiles.

### 7.6 Comparison Design

To answer sub-question 1 (dynamic graph vs. simple models), the dynamic GNN is compared to Random Forest and MLP on the same test set. If the GNN achieves higher F1 or AUC while keeping a reasonable false positive rate, that supports the use of graph and temporal structure. To answer sub-question 2 (federated vs. centralised), the federated model’s final metrics are compared to the centralised model’s. A small drop in performance may be acceptable if it stays within a few percent; a large drop would indicate that federated learning needs more tuning or more data. To answer sub-question 3 (usefulness of explanations), three to five example alerts are generated with full explanations and discussed in terms of whether the top features and flows would help a SOC analyst triage. No formal user study is conducted; the discussion is based on the author’s and supervisor’s judgment of interpretability and actionability.

### 7.7 Chapter summary

Chapter 7 fixed the **setup**, **metrics**, **data scales**, and **comparison protocol** used for all models and federated runs. Chapter 8 reports the **numerical and graphical** outcomes only.

---

## Chapter 8 – Results Presentation

### 8.1 Chapter overview

This chapter presents **results** mainly as **tables, figures, and short factual description**. Deeper interpretation (what the numbers imply for design, federation, and robustness) is reserved for **Chapter 9**. Core training and evaluation use `scripts/run_all.py` and `scripts/generate_alerts_and_plots.py` on the fixed splits; sensitivity and multi-seed experiments use `scripts/run_sensitivity_and_seeds.py` (outputs in `results/metrics/sensitivity_table.csv`, `results/metrics/multi_seed_summary.json`, and `results/figures/sensitivity.png`). All artefacts are stored under `results/metrics/`, `results/figures/`, and `results/alerts/`.

### 8.2 Centralised Model Comparison (Sub-Question 1)

The dynamic GNN (GAT + GRU), Random Forest, and MLP were evaluated on the same test set. Table 1 summarises precision, recall, F1-score, ROC-AUC, and false positive rate. On this split, the central and federated GNN report **100.0%** F1 and **100.0%** ROC-AUC; Random Forest reports **99.86%** F1 and **99.96%** ROC-AUC; MLP reports **99.42%** F1 and **99.84%** ROC-AUC. The false alarm rate for the GNN is **0.0%**; for Random Forest **4.84%** (187 false positives); for MLP **0.10%** (4 false positives). Figure 2 shows the confusion matrix for the Dynamic GNN. Figure 3 shows the ROC curve for the GNN. Figures 6 and 7 show confusion matrices for Random Forest and MLP. Figures 8 and 9 show ROC curves for RF and MLP. Figure 5 compares inference time and F1 across models.

**Table 1: Model comparison on CICIoT2023 test set**

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| Random Forest | 0.9989 | 0.9984 | 0.9986 | 0.9996 | 46.09 |
| MLP | 1.0000 | 0.9885 | 0.9942 | 0.9984 | 0.66 |
| Central GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| Federated GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20.99 |

*Source: Author-derived metrics on the fixed CICIoT2023 subset (Pinto et al., 2023); `results/metrics/results_table.csv`; reproducibility in Chapter 13.*

**Figure 2: Confusion matrix for Dynamic GNN on test set**

![Figure 2: Confusion matrix for Dynamic GNN](results/figures/cm_gnn.png)

*Source: Author’s own plot from test-set predictions; data CICIoT2023 (Pinto et al., 2023); `scripts/run_all.py`.*

**Figure 3: ROC curve for Dynamic GNN on test set**

![Figure 3: ROC curve for Dynamic GNN](results/figures/roc_gnn.png)

*Source: Author’s own plot from test-set scores; data CICIoT2023 (Pinto et al., 2023); `scripts/run_all.py`.*

**Figure 6: Confusion matrix for Random Forest on test set**

![Figure 6: Confusion matrix for Random Forest](results/figures/cm_rf.png)

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

**Figure 7: Confusion matrix for MLP on test set**

![Figure 7: Confusion matrix for MLP](results/figures/cm_mlp.png)

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

**Figure 8: ROC curve for Random Forest on test set**

![Figure 8: ROC curve for Random Forest](results/figures/roc_rf.png)

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

**Figure 9: ROC curve for MLP on test set**

![Figure 9: ROC curve for MLP](results/figures/roc_mlp.png)

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

The confusion matrices (Figures 2, 6, 7) report TP, TN, FP, and FN counts per model. Figure 2 shows **zero** FP and **zero** FN for the GNN on this test set. Figures 6 and 7 show **187** FP for Random Forest and **4** FP for MLP (benign predicted as attack). The ROC curves (Figures 3, 8, 9) plot TPR vs. FPR across thresholds; reported ROC-AUC values are in Table 1. Implications for SOC triage and model choice are discussed in **Chapter 9**.

### 8.3 Federated Learning (Sub-Question 2)

Federated training was run for **10 rounds** with **3 clients** and non-IID splits (Dirichlet **alpha = 0.5**). The global model was evaluated on the **same central test set** after each round. **Table 2** lists precision, recall, F1, and ROC-AUC per round; **Figure 4** plots F1 and ROC-AUC vs. round. Reported values include: round 1 F1 **0.983**, ROC-AUC **0.557**; ROC-AUC reaches **1.000** from round 2; F1 reaches **1.000** from round 7; round 6 ROC-AUC **0.973**. The final round reports F1 **1.000** and ROC-AUC **1.000**, matching the centralised GNN on this test set.

Communication cost is approximated from float32 parameter size: **128,002** parameters, order of **~1.0 MB** per client upload/download per round; **~3.07 MB** per round aggregate messaging is reported, **~31 MB** total over 10 rounds with three clients. Interpretation (privacy vs. centralised parity, feasibility) is in **Chapter 9**.

**Table 2: Federated learning round-by-round metrics**

| Round | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| 1 | 0.967 | 1.000 | 0.983 | 0.557 |
| 2 | 0.967 | 1.000 | 0.983 | 1.000 |
| 3 | 0.967 | 1.000 | 0.983 | 1.000 |
| 4 | 0.994 | 1.000 | 0.997 | 1.000 |
| 5 | 0.997 | 1.000 | 0.998 | 1.000 |
| 6 | 0.990 | 1.000 | 0.995 | 0.973 |
| 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 |

*Source: Author-derived round metrics; Flower FedAvg run; `results/metrics/` (see Chapter 13).*

**Figure 4: Federated learning convergence (F1 and ROC-AUC vs. round)**

![Figure 4: FL convergence](results/figures/fl_convergence.png)

*Source: Author’s own plot; `scripts/generate_alerts_and_plots.py` / federated training logs; data CICIoT2023 (Pinto et al., 2023).*

### 8.4 Central GNN Training Convergence

The centralised Dynamic GNN was trained for **6 epochs** (with early-stopping configured in the training script). **Table 3** lists train loss and validation F1 / ROC-AUC per epoch (`results/metrics/gnn_training_history.json`). Validation F1 and ROC-AUC are **1.000** from epoch 1 onward on this run; train loss falls from **0.484** to **0.0001** by epoch 4 and stays there through epoch 6.

**Table 3: Central GNN training history (loss and validation metrics)**

| Epoch | Train Loss | Val F1 | Val ROC-AUC |
|-------|------------|--------|-------------|
| 1 | 0.484 | 1.000 | 1.000 |
| 2 | 0.023 | 1.000 | 1.000 |
| 3 | 0.0002 | 1.000 | 1.000 |
| 4 | 0.0001 | 1.000 | 1.000 |
| 5 | 0.0001 | 1.000 | 1.000 |
| 6 | 0.0001 | 1.000 | 1.000 |

*Source: `results/metrics/gnn_training_history.json`; central GNN training run (`scripts/run_all.py`).*

### 8.5 Time-Window and CPU Inference (Sub-Question 2 and Deployment)

CPU inference times (reported in Table 1): Random Forest **46.09 ms** per sample; MLP **0.66 ms**; central GNN **22.70 ms**; federated GNN **20.99 ms** (GNN times are per **sequence** of 5 graph windows). **Figure 5** compares F1 and inference time across models.

**Figure 5: Model comparison — inference time and F1-score**

![Figure 5: Model comparison](results/figures/model_comparison.png)

*Source: Author’s own plot; metrics from Table 1 and inference timings in `results/metrics/`; `scripts/generate_alerts_and_plots.py`.*

### 8.6 Example Alerts with Explanations (Sub-Question 3)

Five example alerts were generated with full explanations (top features from Integrated Gradients and top flows from attention weights), in the same order as `results/alerts/example_alerts.json`. Each example includes the predicted label, confidence, threat severity band, and the explanation object (`top_features`, `top_nodes`).

**Example 1 (True negative):** Predicted benign; score **0.163** (low severity). Top features: `Variance`, `Std`, `rst_count`, `Duration`, `AVG` (Integrated Gradients magnitudes as in `example_alerts.json`).

**Example 2 (True positive):** Predicted malicious; score **0.997**. Top features: `psh_flag_number`, `ICMP`, `rst_flag_number`.

**Example 3 (True positive):** Predicted malicious; score **0.996**. Top features: `rst_flag_number`, `ICMP`, `Protocol Type`.

**Example 4 (False positive):** Benign labelled as malicious; score **0.711** (medium). Top features: `Variance`, `rst_count`.

**Example 5 (False positive):** Benign labelled as malicious; score **0.945**. Top features: `Variance`, `Std`, `rst_count`.

Each record follows the ECS-like shape: event metadata, rule name, threat indicator, ML prediction and score, and `explanation` (`top_features`, `top_nodes`). Whether these fields are sufficient for SOC triage is discussed in **Chapter 9**.

### 8.7 Ablation Studies (Priority 1: Evidence)

To show that both the graph and the temporal parts of the model add value, one ablation was run: the same GAT-based model but with the GRU replaced by mean pooling over time (so the model sees each window’s graph embedding but does not model the sequence with an RNN). This variant is called “GAT only (no GRU)”. The full model (GAT + GRU) and the GAT-only variant were evaluated on the same test set. Table 4 summarises the results. To reproduce the ablation, run: `python scripts/run_ablation.py --config config/experiment.yaml`; results are saved to `results/metrics/ablation_gat_only.json` and `results/metrics/ablation_table.csv`.

**Table 4: Ablation on CICIoT2023 test set (centralised GNN variants)**

| Variant | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|---------|-----------|--------|-----|---------|----------------|
| Full (GAT + GRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| GAT only (no GRU) | 0.9923 | 1.0000 | 0.9961 | 1.0000 | 16.06 |

*Source: `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`; test split CICIoT2023 (Pinto et al., 2023).*

**Numeric summary:** full (GAT + GRU) — precision **1.0000**, F1 **1.0000**, ROC-AUC **1.0000**, inference **22.70 ms**; GAT-only (mean pool over windows) — precision **0.9923**, F1 **0.9961**, ROC-AUC **1.0000**, inference **16.06 ms**.

**Figure 16: Ablation comparison — full GAT+GRU vs. GAT-only (F1 and inference time)**

![Figure 16: Ablation bar chart](results/figures/ablation_bar.png)

*Source: Author’s own plot; `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`.*

Interpretation of the temporal (GRU) vs. pooling trade-off is in **§9.3** and **§9.5**.

### 8.8 Sensitivity Analysis (Robustness of Design Choices)

Sensitivity analysis checks whether the main results hold when key hyperparameters change. Two levers are: (1) **window size** (number of flows per graph snapshot), and (2) **k** in the kNN graph. The main experiments use window size **50** and **k = 5**. The pipeline was re-run for all combinations of window_size ∈ {30, 50, 70} and knn_k ∈ {3, 5, 7} with other settings unchanged; metrics were written to `results/metrics/sensitivity_table.csv` (script: `scripts/run_sensitivity_and_seeds.py`).

**Table 6: Sensitivity analysis — central GNN on test set (nine configurations)**

| Window size | *k* | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------------|-----|-----------|--------|-----|---------|----------------|
| 30 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12.20 |
| 30 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 17.42 |
| 30 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11.89 |
| 50 | 3 | 0.9923 | 1.0000 | 0.9961 | 0.9985 | 15.60 |
| 50 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 17.00 |
| 50 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 17.00 |
| 70 | 3 | 0.9939 | 1.0000 | 0.9969 | 0.9947 | 19.91 |
| 70 | 5 | 0.9939 | 1.0000 | 0.9969 | 0.9895 | 15.49 |
| 70 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28.07 |

*Source: `results/metrics/sensitivity_table.csv`; `scripts/run_sensitivity_and_seeds.py`; test split CICIoT2023 (Pinto et al., 2023).*

Grid interpretation is in **§9.3**.

**Figure 15: Sensitivity of test F1 and ROC-AUC to window size and kNN *k***

![Figure 15: Sensitivity heatmap](results/figures/sensitivity.png)

*Source: Author’s own plot; `results/metrics/sensitivity_table.csv`; `scripts/run_sensitivity_and_seeds.py`.*

### 8.9 Multi-Seed Stability

To check that the central GNN result is not an accident of one random seed, the same training and evaluation procedure was repeated with seeds **42**, **123**, and **456** (other config unchanged). Summary statistics are in `results/metrics/multi_seed_summary.json`.

**Table 7: Multi-seed summary (central GNN, test set)**

| Seed | Precision | Recall | F1 | ROC-AUC | False positives (test) |
|------|-----------|--------|-----|---------|-------------------------|
| 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 123 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 456 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| Std | 0 | 0 | 0 | 0 | — |

*Source: `results/metrics/multi_seed_summary.json`; `scripts/run_sensitivity_and_seeds.py`; test split CICIoT2023 (Pinto et al., 2023).*

**Table 7** reports **mean F1 = ROC-AUC = 1.0** and **zero** standard deviation on this test split for all three seeds. Per-seed inference times in the raw logs vary with CPU timing; the headline **~22.7 ms** deployment figure is from the primary **seed 42** run (**Table 1**). What this implies for generalisation is discussed in **Chapter 9**.

### 8.10 Chapter summary

Chapter 8 reported **metrics**, **plots**, and **robustness tables**; **Chapter 9** interprets them against the research questions and limitations.

---

## Chapter 9 – Conclusion, Discussion, and Recommendations

### 9.1 Chapter overview

This chapter answers the research questions, discusses **strengths and limits**, relates findings to the **literature**, summarises **achievements**, and lists **future work**—equivalent to the exemplar dissertation’s separation between “Results” and “Conclusion” but combined here to match the programme’s percentage allocation for conclusions **and** interpretive discussion. **§9.2** gives a **four-paragraph programme-style conclusion** (field → contribution → limits → outlook); **§9.3–§9.9** retain the detailed discussion structure used throughout the draft.

### 9.2 Structured conclusion (programme format)

This dissertation addressed **SOC-oriented**, **CPU-edge** intrusion detection for **Software-Defined IoT** flow telemetry, where analysts need **accurate** models, **privacy-preserving** training where raw flows cannot be pooled, and **explainable** outputs that fit **SIEM-style** workflows. The work situates that need in **CICIoT2023** and in prior literature that often treats **GNNs**, **federated learning**, and **XAI** separately rather than as one deployable path.

The **main outcome** is an end-to-end prototype: *k*NN feature-similarity graphs over windowed flows, a **dynamic GNN** (GAT + GRU), **Flower FedAvg** with three clients, **Captum** attributions and attention cues folded into **ECS-like JSON** via **FastAPI**. On the **fixed subset and splits** documented in **Chapters 7–8**, the central and federated GNN matched **strong RF and MLP baselines** on headline accuracy while reporting **fewer false alarms** on the test split, **sub-23 ms** CPU inference per sequence, **modest** federated communication over ten rounds, and supporting **ablation**, **sensitivity**, and **multi-seed** evidence.

**Limitations** are substantial and explicit: a **45-day** MSc scope, a **subset** of a **lab** dataset, **no device-level graph** where the public release lacks identifiers, **small** federated topology, **no** formal SOC **user study**, and headline **100%** metrics that must be read as **subset-specific** rather than universal—mitigated by split hygiene checks, baseline parity, and robustness tables, not dismissed.

**Outlook:** scaling data and clients, live-traffic validation, analyst-facing evaluation, tighter SIEM integration, and publication-oriented breakdowns (per attack type, stronger heterogeneity) are the natural next steps; they are expanded in **§9.8**. Together, the chapters show that the **research questions** are **answered** for a **documented prototype**, with honest limits and a **clear** path beyond the thesis.

### 9.3 Answering the Research Questions

**Main question:** The project set out to show how an explainable dynamic graph neural network, trained with federated learning, can detect attacks in Software-Defined IoT flow data and generate SIEM alerts for SOC use on CPU-based edge devices. The prototype demonstrates that this is feasible: the dynamic GNN can be trained on graph snapshots over time, federated learning can be applied with Flower and FedAvg, and alerts with explanations can be produced in an ECS-like JSON format. CPU inference time is measurable and can be kept within acceptable bounds for a subset of traffic. So the main question is answered in the sense that a working pipeline exists and has been evaluated; the extent to which it “performs well” depends on the metrics and comparison with baselines reported in **Chapter 8**.

**Sub-question 1 (dynamic graph vs. simple models):** The dynamic GNN did better than Random Forest and MLP. It got 100% F1 and ROC-AUC compared to 99.86% and 99.42% respectively. This supports the idea that graph structure and temporality add value for this kind of data. The baselines were competitive, which is consistent with CICIoT2023 flow features being well-engineered; the GNN's advantage lies in its lower false alarm rate (0% vs. 4.84% for RF and 0.10% for MLP) and its ability to exploit relational and temporal structure. The fact that the GNN achieves zero false positives while the RF has 187 and the MLP has 4 suggests that the graph-based representation helps the model distinguish between benign traffic that resembles attacks (e.g. high variance, unusual flag patterns) and genuine attacks. The kNN graph links similar flows, so the model can learn from neighbourhood patterns that flat classifiers cannot see.

**Sub-question 2 (federated learning):** The federated model’s performance matched the centralised model exactly (100% F1 and ROC-AUC). So federated learning is a viable option when raw data cannot be shared. The approximate communication cost (31 MB over 10 rounds) shows that the approach is suitable for resource-limited edge networks.

**Sub-question 3 (explanations for SOC triage):** The example alerts show that the system outputs top features and top flows. The highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number for attacks; Variance, Std for benign) are interpretable and match the type of traffic. For true positives, the explanations support triage by pointing to protocol-level anomalies; for false positives, the mixed feature profile helps analysts cross-reference before escalating.

**Ablation (Section 8.7):** The full model (GAT + GRU) reaches F1 = 1.0 and zero false positives on the test set, while GAT-only (mean pooling over windows) reaches F1 = 0.9961 with precision 0.9923 — a small but clear gap. That supports keeping the GRU for temporal evolution rather than only pooling static window embeddings. The bar chart (Figure 16) also shows the latency trade-off (GAT-only is faster).

**Sensitivity (Section 8.8):** Across the nine (window, *k*) settings, F1 ranges from **0.9961** to **1.0000** and ROC-AUC from **0.9895** to **1.0000**. Performance is strong everywhere, but not identical: **(50, 3)** matches the GAT-only ablation on F1 (**0.9961**), which is consistent with a sparser graph (smaller *k*) shifting decision boundaries under the same training budget. The chosen default **(50, 5)** sits with **F1 = ROC-AUC = 1.0** and inference **~17 ms** in this grid. Several other cells also reach **F1 = 1.0** (e.g. all three settings at window 30; (50, 7); (70, 7)). Pairs **(70, 3)** and **(70, 5)** show the largest ROC-AUC dips (**~0.99–0.995**) while recall stays **1.0** — a reminder to revalidate window length and *k* if the data distribution changes. Overall, **(50, 5)** is well supported: full F1/AUC together with reasonable latency; Figure 15 makes the pattern visible at a glance.

**Multi-seed (Section 8.9):** Three seeds gave identical headline test metrics (F1 and AUC 1.0, zero false positives). That increases confidence that the main result is not a one-off random initialisation, though it does not remove dataset-specific optimism.

### 9.4 Strengths and Limitations

**Strengths:** The project delivers an end-to-end prototype. It goes from data to graph, model to federated training, explainability to SIEM-style alerts and FastAPI. The design matches practical SOC needs (explainable alerts, CPU-based deployment). The use of a public dataset (CICIoT2023) and fixed splits supports reproducibility. The comparison with baselines and the evaluation of federated learning provide evidence, not just claims. Putting multiple components (graph construction, GNN, federated learning, explainability, SIEM output) in one pipeline is a contribution. Most prior work looks at these in isolation. The choice of kNN feature-similarity graphs is justified by the absence of device identifiers. It is supported by Ngo et al. (2025) and Basak et al. (2025). The stratified windowing strategy deals with class imbalance in a principled way.

**Limitations:** The scope is limited to a subset of CICIoT2023 and the **45-day** MSc project timeline, so the results may not generalise to all IoT environments or attack types. The subset size was chosen to fit that constraint; a larger subset might reveal different trade-offs between models. The kNN graph construction assumes that feature similarity is a meaningful proxy for relational structure; when device identifiers are available, topology-based graphs might perform better. The federated simulation uses 3 clients and 10 rounds; real deployments with more clients, more heterogeneous data, or bandwidth constraints might require different configurations. The explainability analysis is based on five example alerts and author/supervisor judgment; a formal user study with SOC analysts would strengthen the claim that the explanations support triage. The number of federated clients is small (2–3), and the data split across clients may not reflect real-world non-IID settings. Explainability is applied to a subset of alerts if full run-time explanation is too slow. There is no formal user study with SOC analysts; the assessment of explanation usefulness is based on the author’s and supervisor’s judgment. The system is a prototype, not a production SIEM integration. The CICIoT2023 lab environment may not capture the noise, diversity, and evasion tactics present in real-world IoT deployments. Hyperparameter choices (e.g. k=5, window size, sequence length) were validated on the validation set but not exhaustively tuned; different configurations might yield different trade-offs. The 100% F1 and ROC-AUC on the test set, while encouraging, may reflect the relative ease of the chosen subset and should not be over-interpreted as evidence of universal performance.

### 9.5 Practical Implications

The results have several practical implications for SOC and edge deployment. First, the Dynamic GNN's zero false positive rate on this test set reduces alert fatigue compared to Random Forest (187 false positives) and MLP (4 false positives). Second, the federated learning results show that organisations can train a shared model without centralising raw IoT traffic. This addresses privacy and regulatory concerns. Third, the CPU inference time (under 23 ms per sample in the primary run) means the model can run on edge devices without GPUs. Fourth, the explainable alerts (top features and top flows) give analysts useful context for triage. Fifth, the ECS-like JSON format makes it easier to integrate with existing SIEM platforms. The main caveat is that these results are from a lab dataset and subset. Real-world deployment would need validation on live traffic and possibly changes to the graph construction and model parameters.

The ablation (Section 8.7) confirms that the temporal GRU improves over mean-pooling alone on this split. Sensitivity analysis (Section 8.8) shows that performance remains strong across a grid of window sizes and *k* values, with the chosen (50, 5) settings well supported. Multi-seed runs (Section 8.9) suggest stable optimisation for the reported metrics. Together, these checks mirror what a reviewer would ask for in a publication: not only headline accuracy, but **why** the architecture is shaped as it is and **how fragile** the numbers are to reasonable hyperparameter shifts.

### 9.6 Relation to Literature

The findings can be related back to the literature. The use of GNNs for network or security data is supported by work such as Velickovic et al. (2018). This project shows that a dynamic GNN can be built and compared to baselines on IoT flow data. The GNN's better false positive rate (0% vs. 4.84% for RF) fits with the idea that graph structure helps the model tell attack patterns from benign outliers. Han et al. (2025) and Ngo et al. (2025) showed that graph-based and attribute-based construction can improve detection. This project confirms that a kNN feature-similarity approach works when device identifiers are absent.

Federated learning with FedAvg (McMahan et al., 2017) was shown to work with Flower. It achieved performance matching centralised training. This is consistent with Lazzarini et al. (2023) and Albanbay et al. (2025). They found that FedAvg can keep accuracy for IoT intrusion detection. The fast convergence (within 7 rounds) and modest communication cost (31 MB) support the feasibility of federated deployment on edge networks. Explainability via Integrated Gradients (Sundararajan et al., 2017) and attention (Velickovic et al., 2018) was added to the alert output. This matches the need for explainable security tools in the literature. The example alerts show that the highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number) are interpretable and consistent with known attack signatures. Alabbadi and Bajaber (2025) suggested this. The gap identified in the Literature Review (combining explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype) has been addressed within the stated scope and limitations.

### 9.7 Summary of the Project

This dissertation set out to design and build a small prototype system for detecting attacks in IoT network traffic. It uses an explainable dynamic graph neural network and federated learning. It also generates SIEM-style alerts that support SOC operations on CPU-based edge devices. The main research question was: how can such a system detect attacks in IoT flow data and produce alerts that are useful for SOC analysts?

Empirically, the pipeline met its stated targets on the fixed subset: the dynamic GNN matched or exceeded baselines on headline metrics (**Chapter 8**), federated training matched centralised GNN performance, and SIEM-style alerts with explanations were produced. Robustness artefacts (ablation, sensitivity grid, multi-seed summaries) live under `results/metrics/` and are produced by `scripts/run_ablation.py` and `scripts/run_sensitivity_and_seeds.py` as documented in **Chapter 13**.

### 9.8 Recommendations for Future Work

- **Larger scale:** Use a larger subset of CICIoT2023 or other IoT datasets, more federated clients, and more attack types to strengthen the evidence. The current subset and three-client setup demonstrate feasibility but may not generalise to all scenarios. Scaling to more clients and more diverse data would test the robustness of FedAvg under higher heterogeneity.
- **Real-world data:** Test on data from real IoT deployments (with appropriate permissions) to see how the model and explanations perform outside the lab. Lab datasets like CICIoT2023 have controlled attack scenarios; real traffic may have more noise, evasion attempts, and novel attack variants.
- **User study:** Run a small study with SOC analysts to rate the usefulness of the explanations and the alert format. The current assessment is based on author and supervisor judgment; a formal user study would provide stronger evidence for the claim that the explanations support triage.
- **Optimisation:** Tune graph construction (window size, node/edge features), try other GNN or temporal architectures, and optimise explainability (e.g. only for high-confidence alerts) to balance accuracy and speed.
- **Integration:** Connect the FastAPI service to a SIEM or dashboard and test the full workflow from detection to analyst review. Integration with Elastic Security, Splunk, or a custom dashboard would demonstrate end-to-end SOC usability.
- **Further ablation studies:** The thesis already includes one ablation (GAT only vs. full model, Section 8.7). Future work could add ablations for kNN vs. other graph construction, or Integrated Gradients vs. attention-only explanations, to provide further evidence for design choices.

- **Journal publication:** This work is structured so that it can be extended into a journal submission (e.g. MDPI Sensors or Electronics, or IEEE Access) using the same CICIoT2023 dataset. The thesis now includes ablation (Section 8.7), a full sensitivity grid (Section 8.8), and multi-seed validation (Section 8.9). Per-attack-type breakdowns and a formal user study would further strengthen a publication. A short paper or full article can be drafted after thesis submission by reusing the methodology, results, and discussion from this dissertation.

### 9.9 Chapter summary

The prototype **answers** the stated questions within documented **limits**, **relates** the outcomes to prior work, and sets out a **credible path** for scaling, evaluation, and publication.

---

## Chapter 10 – Critical Self-Evaluation

### 10.1 Chapter overview

This chapter reflects on the research process in a first-person, honest way: what went well, what I would change with hindsight, and what the experience taught me about security research at MSc level. The handbook asks for roughly 10% of the report for critical reflection; I use it to connect **technical choices** to **personal learning**, not to repeat Chapter 8.

### 10.2 Planning, scope, and risk

The project was scoped to fit a **45-day** MSc timeline. Early on I decided to use a **fixed subset** of CICIoT2023 rather than the full corpus, and to simulate **three** federated clients instead of a large fleet. Those decisions were partly pragmatic — disk, RAM, and training time — but they were also methodological: a smaller, controlled subset makes it easier to **re-run** experiments and to **debug** the pipeline when something breaks. The trade-off is obvious: stronger numbers on a subset do not automatically transfer to the wild. I tried to mitigate “toy dataset” criticism by (1) stating the limits clearly in **Chapter 9**, (2) comparing against **strong baselines** (RF, MLP) on the **same** splits, and (3) adding **ablation**, **sensitivity**, and **multi-seed** runs so the headline metrics are not a single lucky configuration.

What I would improve with more time: negotiate **earlier** with my supervisor a precise cap on subset size and document the **exact** filtering steps in one place (I relied on scripts and config, which is good for reproducibility, but a one-page “data contract” in the dissertation would help readers who do not open the repo).

### 10.3 Literature and alignment with the questions

The literature review deliberately mixed **IoT intrusion detection**, **SIEM / SOC workflows**, **GNNs**, **federated learning**, and **explainable AI**. The gap I argued for was not “nobody used GNNs” — many papers do — but that **combining** explainable dynamic graphs, federated training, and **SIEM-shaped** outputs in one **CPU-oriented** prototype is still relatively rare in student-scale work. Keeping that storyline straight while implementing was sometimes hard: it is tempting to add every nice idea (more clients, more attacks, more explainers). I had to keep deleting side ideas to protect the core questions.

A clear weakness remains: **no formal user study** with practising analysts. I judged explanation usefulness together with my supervisor and by reading the alert JSON “as if” I were triaging. That is defensible for an MSc, but it is **not** evidence of real-world utility. If I did the project again, I would reserve one week for **three semi-structured interviews** with volunteers (even fellow students role-playing SOC) and a tiny Likert questionnaire — not publication-grade, but stronger than intuition alone.

### 10.4 Implementation: what was harder than it looked

Building the pipeline end-to-end meant **gluing** PyTorch, PyTorch Geometric, Flower, Captum, and FastAPI. Each library is documented, but **the edges** between them are not: tensor shapes for batched graphs, ensuring the **same** scaler and splits for federated clients, and making explainability run on **one** sequence without blowing memory. The longest delays were **data and graphs**: stratified windowing so that graph-level labels are balanced, while flow-level data are heavily imbalanced, took several iterations before training stopped collapsing to “always attack.”

Federated learning was conceptually simple (FedAvg) but **operationally** fiddly: starting the server, then clients, ensuring partitions are **non-IID** but still trainable, and logging per-round metrics in a form I could plot later. I am satisfied that federated performance **matched** centralised on my setup; I am also aware that three clients and ten rounds are **not** stress test.

On the positive side, I structured the code into **modules** (data, graph, models, federated, explainability, API) so I could swap pieces — for example, running ablation by changing the temporal head without touching the GAT stack. That modularity paid off when I added `scripts/run_sensitivity_and_seeds.py`: it mostly re-used the same training entry points with different YAML overrides.

### 10.5 Results, honesty, and the “100%” question

When test F1 and ROC-AUC hit **1.0**, my first reaction was relief; my second was suspicion. Perfect metrics usually mean one of: (a) the task is **easy** on the slice of data, (b) there is **leakage** (I checked splits carefully — no overlap between train/val/test at sequence level), or (c) the **test set is small** enough that a few mistakes swing the score. Here, the test set has **934** sequences; zero errors is possible but should not be read as “solved IoT security.” I tried to be explicit about that in **Chapter 9**. The **ablation** helped: removing the GRU **did** cost precision/F1 slightly, which shows the full model is not trivially redundant. The **sensitivity grid** showed that not every (window, *k*) pair is equally perfect — so the story is more nuanced than a single scalar.

If I sound cautious, it is intentional. I would rather a reader trusts me for **transparency** than for bragging.

### 10.6 What I learned (skills and mindset)

Technically, I learned to **think in graphs** (nodes as flows, *k*NN edges as similarity) and in **time** (sequences of snapshots). I learned the basics of **federated optimisation** and why non-IID data breaks naive assumptions. I learned that **explainability** is as much about **presentation** (what we put in the JSON) as about the algorithm (Integrated Gradients).

Personally, I learned to **time-box**: the sensitivity and multi-seed runs were scheduled late; they completed, but I would start **robustness experiments earlier** next time so writing-up is not waiting on overnight jobs. I also learned to **document while coding** — commit messages, `config/experiment.yaml`, and CSV outputs saved me when I forgot which run produced which figure.

### 10.7 Time management: what I would reorder

Data loading and graph construction **underestimated** at the start. Federated setup **overestimated** in difficulty once the pattern was clear — but the first week of Flower debugging felt slow. Explainability + FastAPI were faster **after** the model stabilised. With hindsight I would allocate: **week 1** — data contract + baseline; **week 2** — GNN + central training; **week 3** — federated + API; **week 4** — explainability + plots; **buffer** — ablation and sensitivity. I used the buffer mainly for robustness runs; in another timeline I might have sacrificed one fancy plot for a **user interview**.

Overall, the project met my own bar: **one coherent system**, **evidence-backed claims**, and **clear limits**. It is not production-ready, but it is **honest research engineering** at MSc level, and I am proud of the parts that were painful and still work.

### 10.8 Chapter summary

Critical reflection highlights **scoped realism**, **modularity**, and **honest limits**—and concrete habits I would repeat on a future project.

---

## Chapter 11 – References

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847. Available at: https://doi.org/10.3390/s25030847

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78. Available at: https://doi.org/10.3390/jsan14040078

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002. Available at: https://doi.org/10.3390/app15095002

Cuppens, F. and Miège, A. (2002) 'Alert correlation in a cooperative intrusion detection framework', in *Proceedings of the 2002 IEEE Symposium on Security and Privacy*, pp. 202-215. Available at: https://doi.org/10.1109/secpri.2002.1004372

Han, Z., Zhang, C., Yang, G., Yang, P., Ren, J. and Liu, L. (2025) 'DIMK-GCN: a dynamic interactive multi-channel graph convolutional network model for intrusion detection', *Electronics*, 14(7), p. 1391. Available at: https://doi.org/10.3390/electronics14071391

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Reynolds, J., Melnikov, A., Lunova, N. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint arXiv:2009.07896*. Available at: https://arxiv.org/abs/2009.07896

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80-84. Available at: https://doi.org/10.1109/mc.2017.201

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509-530. Available at: https://doi.org/10.3390/ai4030028

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774. Available at: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

Lusa, R., Pintar, D. and Vranic, M. (2025) 'TE-G-SAGE: explainable edge-aware graph neural networks for network intrusion detection', *Modelling*, 6(4), p. 165. Available at: https://doi.org/10.3390/modelling6040165

McMahan, H.B., Moore, E., Ramage, D., Hampson, S. and Agüera y Arcas, B. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, pp. 1273-1282. Available at: https://proceedings.mlr.press/v54/mcmahan17a.html

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection - a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499. Available at: https://doi.org/10.3390/info16060499

Pinto, C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941. Available at: https://doi.org/10.3390/s23135941

Qu, Y., Gao, L., Luan, T.H., Xiang, Y., Yu, S., Li, B. and Zheng, G. (2020) 'Decentralized Privacy Using Blockchain-Enabled Federated Learning in Fog Computing', *IEEE Internet of Things Journal*, 7(6), pp. 5171-5183. Available at: https://doi.org/10.1109/JIOT.2020.2977383

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319-3328. Available at: http://proceedings.mlr.press/v70/sundararajan17a.html

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *International Conference on Learning Representations (ICLR)*. Available at: https://openreview.net/forum?id=rJXMpikCZ

Wang, R., Zhao, J., Zhang, H., He, L., Li, H. and Huang, M. (2025) 'Network traffic analysis based on graph neural networks: a scoping review', *Big Data and Cognitive Computing*, 9(11), p. 270. Available at: https://doi.org/10.3390/bdcc9110270

Yang, S., Pan, W., Li, M., Yin, M., Ren, H., Chang, Y., Liu, Y., Zhang, S. and Lou, F. (2025) 'Industrial Internet of Things intrusion detection system based on graph neural network', *Symmetry*, 17(7), p. 997. Available at: https://doi.org/10.3390/sym17070997

Zheng, L., Li, Z., Li, J., Li, Z. and Gao, J. (2019) 'AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN', in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 4419-4425. Available at: https://doi.org/10.24963/ijcai.2019/614

Zhong, M., Lin, M., Zhang, C. and Xu, Z. (2024) 'A survey on graph neural networks for intrusion detection systems: methods, trends and challenges', *Computers and Security*, 141, p. 103821. Available at: https://doi.org/10.1016/j.cose.2024.103821

---

## Chapter 12 – Bibliography

**Chapter 11** lists all cited sources in Harvard style. **Chapter 12** is reserved for any examiner-requested supplementary bibliography and is otherwise unused.

---

## Chapter 13 – Appendices

*For submission: embed or attach the following documents in the final Word/PDF. They are required by the MSc Project Handbook and the School’s dissertation guideline.*

**Handbook numbering vs. this document.** Programme materials often use **Appendix 1–4**. This thesis uses **Appendix A–E** in the technical body, then **separate embedded headings** in the automated Word export so letters are not duplicated.

| Handbook appendix | Role | Where it appears in this dissertation |
|-------------------|------|----------------------------------------|
| **Appendix 1** — code screenshots (figures + captions + interpretation) | **Appendix D** below; figures **A1-1**–**A1-6**; PNGs under `results/figures/appendix1/` (`scripts/render_appendix1_code_figures.py`) |
| **Appendix 2** — signed project specification | **Appendix B** (file pointer); **Full text — Agreed project specification (embedded)** at end of Word export |
| **Appendix 3** — process documentation (and attendance where required) | **Appendix A** (file pointers); **Full text — … (embedded)** sections at end of Word export |
| **Appendix 4** (optional) — GitHub / dataset / demo video | **Appendix E** |

**Appendix C** is **not** a handbook “Appendix 1–4” slot: it is **reproducibility** (environment and commands) so examiners can re-run experiments from **Appendix E.1**’s repository.

**Word export note.** `scripts/dissertation_to_docx.py` appends the process documentation, attendance log, and specification **after** the Markdown body (after **Appendix E**) under the headings **Full text — …** so those headings do **not** clash with **Appendix A–E** already printed above. If your School template requires a single combined PDF order (e.g. process before code figures), reorder sections once in Word.

### Appendix A: Project Process Documents

In the **automated Word export**, the full .docx files named below are also inserted **after Appendix E** under **Full text — …** headings (see Chapter 13 opening note) so assessors receive both the manifest and the originals.

- **Project Process Documentation** — Arka_Talukder_Process_Documentation_B01821011.docx  
  *Canonical location:* **`archive/process_attendance/`** (see **`archive/README.md`**). *Supervisor zip mirror:* `B01821011_Final_Report_Package_for_Supervisor/05_Appendix_documents/`. The Word exporter embeds from the archive paths.

- **Attendance Log** — Arka Talukder_Attendance_Jan-Feb_B01821011.docx  
  *Canonical location:* **`archive/process_attendance/`** (see **`archive/README.md`**). *Supervisor zip mirror:* `B01821011_Final_Report_Package_for_Supervisor/05_Appendix_documents/`.

### Appendix B: Project Specification

The signed specification .docx is also embedded **after Appendix E** in the Word export (**Full text — Agreed project specification**).

- **Agreed Project Specification** — Arka-B01821011_MSc Cyber Security Project specification_form_2025-26.docx  
  *Location: project root*

### Appendix C: Reproducibility

This appendix gives **commands and paths** to reproduce the quantitative results and figures in **Chapters 7–8** from the codebase. It is **separate** from the handbook’s optional **Appendix 4** (repository / video), which is provided in **Appendix E**. Clone or download the project from **Appendix E.1**, obtain **CICIoT2023** per **Appendix E.2**, then run the steps below.

To reproduce the experiments and results reported in this dissertation:

**Environment:**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Full pipeline (preprocess → graphs → baselines → GNN → evaluation):**
```
python scripts/run_all.py --config config/experiment.yaml
```

**Federated learning** (run server in one terminal, clients in others):
```
python -m src.federated.run_federated server
python -m src.federated.run_federated client --cid 0
python -m src.federated.run_federated client --cid 1
python -m src.federated.run_federated client --cid 2
```
*Note: Run `split_and_save` from `src.federated.data_split` first to create non-IID client splits.*

**Generate figures and example alerts:**
```
python scripts/generate_alerts_and_plots.py
```

**Sensitivity grid (window size × kNN *k*) and multi-seed central GNN:**
```
python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
```
This writes `results/metrics/sensitivity_table.csv`, `results/metrics/multi_seed_summary.json`, and `results/figures/sensitivity.png` (in addition to per-seed metric files if configured).

All main experiments use seed 42 (set in `config/experiment.yaml`) unless a script overrides it for the multi-seed study. Outputs are written to `results/metrics/`, `results/figures/`, and `results/alerts/`. See `SETUP_AND_RUN.md` in the project repository for full step-by-step instructions.

### Appendix D: Handbook Appendix 1 — Screenshots of project code

The handbook requires **Appendix 1** code views to be presented as **figures**, each with a **caption** and an explanation of **how to interpret** the code in the context of this dissertation. Below, **Figure A1-1**–**Figure A1-6** satisfy that requirement.

The bitmaps are **generated automatically** from the submission codebase so line numbers stay aligned with the files on disk. Regenerate them after editing source with:

`python scripts/render_appendix1_code_figures.py`

Output directory: `results/figures/appendix1/`. For the Word build, figures **17–22** in the **Table of Figures** correspond to **A1-1**–**A1-6**. End-to-end batch orchestration (preprocess → graphs → baselines → GNN → plots) lives in `scripts/run_all.py` and is documented in **Appendix C**; it is not duplicated as a separate figure here to avoid overlap with **Figure A1-1** and the pipeline description in **Chapter 6**.

![Figure A1-1 — Dynamic graph classifier: `DynamicGNN` (GAT layers, pooling, GRU vs. mean-pool ablation, logits). Source: `src/models/dynamic_gnn.py`, lines 12–97.](results/figures/appendix1/fig_a1_01_dynamic_gnn.png)

**Caption (formal):** Figure A1-1 — Core implementation of the dynamic GNN (`DynamicGNN`): node embedding, two `GATConv` layers with multi-head attention and dropout, per-window graph embedding, sequence encoding with `GRU` (or mean-pool when `use_gru` is false for ablation), and two-class logits. Attention weights may be retained for explainability (`return_attention_weights`).

**How to interpret:** This class is the **main learnable model** described in **Chapters 5–6**. Each time step is one PyTorch Geometric `Data` object (nodes = flows in a window, edges = *k*NN). `_encode_graph` applies GAT message passing and reduces node states to one vector per window; `forward` stacks windows in time order and either runs the **GRU** (full model) or **mean-pools** over time (ablation in **§8.7**). The `from_config` constructor ties hyperparameters to `config/experiment.yaml`, which supports the sensitivity study in **§8.8**.

![Figure A1-2 — Constructing a single-window *k*NN graph from flow feature rows. Source: `src/data/graph_builder.py`, lines 25–58.](results/figures/appendix1/fig_a1_02_graph_builder_knn_graph.png)

**Caption (formal):** Figure A1-2 — `flows_to_knn_graph`: for one window of `N` flows with `F` features, fits `sklearn.neighbors.NearestNeighbors` (Euclidean), adds bidirectional edges among each node and its `k` actual neighbours (capped when `N` is small), and returns a `torch_geometric.data.Data` object with `x`, `edge_index`, and graph-level label `y`.

**How to interpret:** This function shows **how “graph structure” is defined** when device IPs are unavailable (**Chapter 1**): similarity in the **46-dimensional** flow feature space replaces physical topology. Bidirectional edges make the graph undirected for GAT. The label on the graph is supplied by the caller (binary benign vs. attack) and matches the stratified pool in **Figure A1-3**, not a per-flow vote—important for reading the evaluation chapters.

![Figure A1-3 — Stratified windowing so both classes appear in training graphs. Source: `src/data/graph_builder.py`, lines 89–129.](results/figures/appendix1/fig_a1_03_graph_builder_stratified.png)

**Caption (formal):** Figure A1-3 — `build_graphs_for_split`: splits flows into benign and attack pools, builds sliding (or strided) windows from each pool with `_build_windows_from_pool`, balances counts between classes, shuffles, and logs totals—addressing severe class imbalance in raw CICIoT2023.

**How to interpret:** This block explains **why training does not collapse** to “always predict attack” despite a very high attack ratio in the raw CSV (**Chapter 4**): windows are drawn **within** each class, so each graph’s supervision matches the pool. The `minority_stride` argument increases overlap for the benign pool when it is smaller. The dissertation’s **window size** and **k** used in experiments come from `config/experiment.yaml` and feed the sensitivity grid in **§8.8**.

![Figure A1-4 — Post-hoc explanations: forward pass with attention, Integrated Gradients, top nodes/features. Source: `src/explain/explainer.py`, lines 53–102.](results/figures/appendix1/fig_a1_04_explainer_integrated_gradients.png)

**Caption (formal):** Figure A1-4 — `explain_sequence`: runs the model with attention enabled, wraps Captum **Integrated Gradients** on the **last window’s** node features (`_ig_wrapper`), aggregates absolute attributions to rank top nodes and top feature indices, and returns an `ExplanationBundle` for JSON alert formatting.

**How to interpret:** This is the bridge between **Chapter 6** (implementation) and **Chapter 8** (example alerts): SOC-facing text is not arbitrary—it is derived from **IG magnitudes** and **GAT edge attention** from the same forward pass analysts would get at inference. IG is computed on the **most recent** graph in the sequence (design choice documented in code comments). If Captum is missing, the bundle degrades gracefully (`HAS_CAPTUM` guard in the same module).

![Figure A1-5 — Federated learning CLI: Flower server vs. client, local data, `GNNFlowerClient`. Source: `src/federated/run_federated.py`, lines 28–71.](results/figures/appendix1/fig_a1_05_federated_flower_client.png)

**Caption (formal):** Figure A1-5 — `main` in `run_federated.py`: loads YAML config; **server** mode calls `run_fl_server` with round count and quorum; **client** mode loads `client_{cid}_graphs.pt`, builds train/validation sequence loaders, constructs `GNNFlowerClient`, and connects to `127.0.0.1:8080` via `fl.client.start_numpy_client`.

**How to interpret:** This file is the **student-facing entry point** for **Chapter 8** federated results: each client trains only on its partition (**non-IID** split from `src.federated.data_split`, referenced in **Appendix C**). The address and round counts come from `config/experiment.yaml` under `fl`. The pattern matches the FedAvg narrative in **Chapter 2**—local epochs, then aggregation on the server (server implementation in `src/federated/server.py`, not shown).

![Figure A1-6 — HTTP API: load checkpoint, `/score` builds graphs, runs `explain_sequence`, returns ECS-style alert JSON. Source: `src/siem/api.py`, lines 32–89.](results/figures/appendix1/fig_a1_06_fastapi_score_alert.png)

**Caption (formal):** Figure A1-6 — FastAPI application: startup loads `experiment.yaml` and `dynamic_gnn_best.pt`; `POST /score` accepts a list of flow windows, reads **`graph.knn_k`** from the loaded config for `flows_to_knn_graph`, runs `explain_sequence` with top-5 nodes and features, measures wall-clock milliseconds, and returns `format_ecs_alert` output alongside prediction and score.

**How to interpret:** This is the **edge deployment surface** argued in **Chapters 1 and 5**: a single HTTP call turns raw feature windows into **SIEM-shaped JSON** suitable for triage. Inference-time *k*NN **k** therefore **matches** training graph construction from **`config/experiment.yaml`**, avoiding train/serve skew. Latency reported in **Chapter 8** is consistent with this synchronous CPU path.

### Appendix E: Handbook Appendix 4 (optional) — repository, dataset, demonstration

The programme allows an optional fourth appendix: a **GitHub** link to project code (and any data you host there), or a **video** of the system with a **OneDrive** or **YouTube** link. This appendix records what applies to **this** submission; URLs match the **actual** remote and the **official** dataset distributor at the time of writing.

#### E.1 Project source code (GitHub)

The prototype code for this dissertation is available in a **public** Git repository (this is the `origin` remote of the working tree used to produce the reported results):

- **Repository:** https://github.com/imark0007/Dissertation_Project  

It contains `src/`, `scripts/`, `config/`, `requirements.txt`, `README.md`, and `SETUP_AND_RUN.md`. Clone the default branch and follow **Appendix C** to re-run training and evaluation.

#### E.2 Dataset (official source; not bundled in GitHub)

The experiments use **CICIoT2023** (Pinto *et al.*, 2023). The full corpus is **not** committed to the GitHub repository (size and licence). Download it from the Canadian Institute for Cybersecurity (CIC), University of New Brunswick:

- **Dataset page:** https://www.unb.ca/cic/datasets/iotdataset-2023.html  

The dataset paper is cited in **Chapter 11** (DOI: https://doi.org/10.3390/s23135941). Preprocessing in this project expects the user to place downloaded files according to `SETUP_AND_RUN.md`.

#### E.3 Optional demonstration video (OneDrive / YouTube)

**No** narrated walkthrough video was submitted with this dissertation; assessment relies on the **GitHub** repository (**E.1**), **Appendix C**, and the figures and tables in the main text. **If you record a video later**, replace the following line with a single public **OneDrive** or **YouTube** URL and update the Word/PDF:

- **Demonstration video:** *Not included.*

---
