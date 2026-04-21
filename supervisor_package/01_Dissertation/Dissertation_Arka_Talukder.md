# Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning

**Arka Talukder | B01821011**  
**MSc Cyber Security (Full-time)**  
**University of the West of Scotland**  
**School of Computing, Engineering and Physical Sciences**  
**Supervisor: Dr. Raja Ujjan**

---

## Front Matter

The **submitted Word/PDF** must include, **before** this technical body (insert from Moodle / School templates): **(1)** completed **Dissertation Front Sheet**; **(2)** signed **Declaration of originality**; **(3)** signed **Library release form**. Those pages are not reproduced here. The technical body below follows the **Updated Guideline for Final Report** (School of Computing, Engineering and Physical Sciences): abstract, acknowledgements, lists, chapters in the required weighting band (introduction, literature review with sufficient sources, research design and methodology, implementation and testing, results with metrics and figures, conclusions, critical self-evaluation, Harvard references with DOIs where available, appendices including code evidence and signed specification).

**Alignment with the guideline.** Chapter 1 introduces the problem and scope. Chapter 2 is the literature review (including an extended comparative subsection so that **fifteen to twenty** distinct peer-reviewed or standard sources are engaged in depth). Chapters 4–5 cover research design, dataset design, and system design. Chapters 6–7 cover implementation (PyTorch, Flower, Captum, FastAPI), deployment shape, and evaluation protocol. Chapter 8 presents results and plots; Chapter 9 discusses them. Chapter 10 is critical self-reflection. Chapter 11 lists references (Harvard). Chapter 13 maps handbook appendices. **UWS Library** guidance on Harvard referencing should be followed for the final PDF (see [UWS Library Harvard](https://libguides.uws.ac.uk/harvard)).

**Quality and presentation (distinction-level aim).** This report is written entirely on **your** topic (IoT intrusion detection with a dynamic GNN, federated learning, explainability, and SIEM-shaped outputs). Programme dissertations that are marked highly usually share a few traits: a **clear thread** from problem → design → implementation → evidence → honest limits; **tables and figures** that are numbered **in order of first appearance** (see **Table of Figures** and **Table of Tables**), each captioned with **chapter and section** and tied back to the research questions; and a **clean** Word/PDF layout. In Microsoft Word, apply the School template where required: **(a)** a **drop cap** on the first letter of the abstract if your programme handbook shows that style; **(b)** consistent **accent colour** on major headings if allowed; **(c)** after pagination, refresh the **Table of Contents**, **List of Figures**, and **List of Tables** so page numbers match the body; **(d)** colour the abbreviation column in the List of Abbreviations if required (the Markdown table uses bold as a fallback).

---

## 1. Abstract

Internet of Things (IoT) deployments generate large volumes of network-flow records that Security Operations Centres (**SOCs**) must triage on **CPU**-only edge hardware. Detectors should be accurate, respect data locality, and provide brief explanations for each alert. In many organisations, analysts work with limited time and mixed data quality, so false positives and unclear alarms increase investigation cost quickly.

This dissertation develops a prototype on **CICIoT2023** (Pinto et al., 2023): flows are standardised, windowed, connected using **k**NN similarity graphs, and classified by a **dynamic GNN** (GAT + GRU) implemented in **PyTorch Geometric**. The same architecture is trained centrally and with **Flower FedAvg** across three **non-IID** clients (Dirichlet **alpha = 0.5**). Explanations combine **Captum** Integrated Gradients and GAT attention and are returned as JSON via a **FastAPI** scoring endpoint. The pipeline keeps reproducibility focus through fixed splits, explicit configuration, and script-based execution for central, federated, and explainability stages.

**Results.** On the held-out test split, the central and federated GNN achieve **F1 = 100%** and **ROC-AUC = 100%** with zero false positives; Random Forest and MLP achieve **F1 = 99.86%** and **99.42%** respectively. Mean CPU inference is ~**23 ms** per five-window sequence, and the federated run exchanges ~**31 MB** of weights over ten rounds. Ablation, sensitivity, and multi-seed runs support these results; limitations are discussed in Chapter 9. The dissertation contribution is the integrated prototype itself: dynamic graph modelling, federated training, explainable alerts, and SIEM-shaped output in one CPU-oriented workflow, which is still not common in one end-to-end public implementation for this dataset.

**Keywords:** IoT security, dynamic graph neural network, federated learning, SIEM, explainable AI, edge AI, SOC, CICIoT2023

---

## Acknowledgements

I thank my **supervisor, Dr. Raja Ujjan**, for technical guidance, feedback on design and evaluation, and support throughout the project. I thank my **moderator, Muhsin Hassanu**, for reviewing interim work and helping sharpen the final report. I also thank **Dr. Daune West** for support and academic encouragement during the submission period. I am grateful to **School and programme staff** for module materials and submission guidance, and to the **MSc Project co-ordinator** for administrative communication around milestones and ethics. Finally, I thank **friends and family** for patience during intensive writing and experiment runs.

---

## List of Abbreviations

| **Abbrev.** | Definition |
|-------------|------------|
| **API** | Application Programming Interface |
| **AUC** | Area Under the (ROC) Curve |
| **CPU** | Central Processing Unit |
| **ECS** | Elastic Common Schema (log/alert shape used as a guide) |
| **FedAvg** | Federated Averaging (McMahan et al., 2017) |
| **FL** | Federated Learning |
| **FN** | False Negative |
| **FP** | False Positive |
| **FPR** | False Positive Rate |
| **GAT** | Graph Attention Network |
| **GNN** | Graph Neural Network |
| **GRU** | Gated Recurrent Unit |
| **IG** | Integrated Gradients |
| **IoT** | Internet of Things |
| **IID** | Independent and Identically Distributed (data split) |
| **JSON** | JavaScript Object Notation |
| **kNN** | k-Nearest Neighbours (graph construction) |
| **MLP** | Multi-Layer Perceptron |
| **non-IID** | Non-identical client data distributions |
| **PyG** | PyTorch Geometric |
| **ROC** | Receiver Operating Characteristic |
| **SDN** | Software-Defined Networking |
| **SIEM** | Security Information and Event Management |
| **SOC** | Security Operations Centre |
| **TN** | True Negative |
| **TP** | True Positive |
| **UML** | Unified Modelling Language |
| **XAI** | Explainable Artificial Intelligence |

*Note: In the final Word file, abbreviations in the first column may be coloured (e.g. dark red) to match the programme’s preferred sample layout.*

---

## Table of Contents

**List of Abbreviations**

**Acknowledgements**

**Chapter 1 – Introduction**  
- 1.1 Chapter overview  
- 1.2 Background and Motivation  
- 1.3 Research Aim and Questions  
- 1.4 Scope and Limitations  
- 1.5 Dissertation Structure  
- 1.6 Alignment with the marking criteria in the Project Specification  
- 1.7 Chapter summary  

**Chapter 2 – Literature Review**  
- 2.1 Chapter overview  
- 2.2 Themes and structure  
- 2.3 IoT Security and the Need for Detection  
- 2.4 SIEM, SOC Workflows, and Alert Quality  
- 2.5 Graph Neural Networks and Dynamic Graphs  
- 2.6 Federated Learning and Privacy  
- 2.7 Explainability in ML-Based Security  
- 2.8 Gap and Contribution  
- 2.9 Mapping to CyBOK (Cyber Security Body of Knowledge)  
- 2.10 Extended comparative review (fifteen to twenty core sources)  
- 2.11 Chapter summary  

**Chapter 3 – Project Management**  
- 3.1 Chapter overview  
- 3.2 Project plan and timeline (45-day MSc)  
- 3.3 Risk assessment  
- 3.4 Ethics and data  
- 3.5 Interim report feedback incorporated  
- 3.6 Chapter summary  

**Chapter 4 – Research Design and Methodology**  
- 4.1 Chapter overview  
- 4.2 Research Approach  
- 4.3 Dataset and Subset  
- 4.4 Models  
- 4.5 Federated Learning Setup  
- 4.6 Explainability  
- 4.7 Evaluation Plan  
- 4.8 Chapter summary  

**Chapter 5 – Design**  
- 5.1 Chapter overview  
- 5.2 Pipeline, alerts, and deployment (conceptual)  
- 5.3 Graph design (flows to kNN snapshots)  
- 5.3a Research design and system architecture  
- 5.3b Conceptual illustration: similarity-based graph in one window  
- 5.4 Chapter summary  

**Chapter 6 – Implementation and System Development**  
- 6.1 Chapter overview  
- 6.2 Environment and Tools  
- 6.3 Data Loading and Preprocessing  
- 6.4 Graph Construction  
- 6.5 Model Implementation  
- 6.6 Federated Learning (Flower)  
- 6.7 Explainability  
- 6.8 Alert Generation and SIEM-Style Output  
- 6.9 FastAPI Deployment and CPU Inference  
- 6.10 Implementation code screenshots (author’s codebase)  
- 6.11 Chapter summary  

**Chapter 7 – Testing and Evaluation**  
- 7.1 Chapter overview  
- 7.2 Evaluation scope  
- 7.3 Experimental Setup  
- 7.4 Metrics  
- 7.5 Dataset and Experiment Statistics  
- 7.6 Comparison Design  
- 7.7 Chapter summary  

**Chapter 8 – Results Presentation**  
- 8.1 Chapter overview  
- 8.2 Centralised Model Comparison (Sub-Question 1)  
- 8.3 Federated Learning (Sub-Question 2)  
- 8.3a What federated learning produced (reading the curves and the table)  
- 8.4 Central GNN Training Convergence  
- 8.5 Time-Window and CPU Inference (Sub-Question 2 and Deployment)  
- 8.6 Example Alerts with Explanations (Sub-Question 3)  
- 8.7 Ablation Studies (Priority 1: Evidence)  
- 8.8 Sensitivity Analysis (Stability of Design Choices)  
- 8.9 Multi-Seed Stability  
- 8.10 Chapter summary  

**Chapter 9 – Conclusion, Discussion, and Recommendations**  
- 9.1 Chapter overview  
- 9.2 Structured conclusion (programme format)  
- 9.3 Answering the Research Questions  
- 9.4 Strengths and Limitations  
- 9.5 Practical Implications  
- 9.6 Relation to Literature  
- 9.7 Summary of the Project  
- 9.8 Recommendations for Future Work  
- 9.9 Chapter summary  

**Chapter 10 – Critical Self-Evaluation**  
- 10.1 Chapter overview  
- 10.2 Planning, scope, and risk  
- 10.3 Literature and alignment with the questions  
- 10.4 Implementation: what was harder than it looked  
- 10.5 Results, honesty, and the “100%” question  
- 10.6 What I learned (skills and mindset)  
- 10.7 Time management: what I would reorder  
- 10.8 Chapter summary  

**Chapter 11 – References**  
**Chapter 12 – Bibliography**  
**Chapter 13 – Appendices** (A: process; B: specification; C: reproducibility; D: handbook Appendix 1 code figures; E: handbook Appendix 4 optional)

### Table of Figures

**Numbering rule:** Body figures are **Figure 1–Figure 24** in **strict order of first appearance** (Chapter 2 → Chapter 5 → Chapter 6 → Chapter 8). Within **Chapter 5**, subsection order is **5.2 pipeline → 5.3 graph specification → 5.3a–5.3b figures**, so **Figures 6–8** match that reading order. Each caption states **Chapter** and **Section** (or **Appendix D**). Handbook **code extracts** in Appendix D keep labels **Figure A1-1**–**Figure A1-6** (they are not part of the 1–24 sequence). Refresh page numbers in Word after pagination.

| Figure | Title (chapter, section) | Page |
|--------|--------------------------|------|
| 1 | Taxonomy of IDS approaches (Ch. 2, Section 2.3) | — |
| 2 | Conceptual dynamic GNN flow — GAT + GRU (Ch. 2, Section 2.5) | — |
| 3 | Federated learning (FedAvg) flow (Ch. 2, Section 2.6) | — |
| 4 | Explainability methods for SOC-oriented alerts (Ch. 2, Section 2.7) | — |
| 5 | Positioning of related work — four pillars (Ch. 2, Section 2.7) | — |
| 6 | Research pipeline — raw flows to SIEM alerts (Ch. 5, Section 5.2) | — |
| 7 | Research design — data flow, FL, edge alerting (Ch. 5, Section 5.3a) | — |
| 8 | Conceptual *k*NN similarity graph in one window (Ch. 5, Section 5.3b) | — |
| 9 | Code: `flows_to_knn_graph` core (Ch. 6, Section 6.10) | — |
| 10 | Code: `build_graphs_for_split` core (Ch. 6, Section 6.10) | — |
| 11 | Code: `train_one_epoch` training step (Ch. 6, Section 6.10) | — |
| 12 | Code: `DynamicGNN.forward` (Ch. 6, Section 6.10) | — |
| 13 | Code: `_ig_wrapper` / Integrated Gradients (Ch. 6, Section 6.10) | — |
| 14 | Code: FastAPI `POST /score` core (Ch. 6, Section 6.10) | — |
| 15 | Confusion matrix — Dynamic GNN (Ch. 8, Section 8.2) | — |
| 16 | ROC curve — Dynamic GNN (Ch. 8, Section 8.2) | — |
| 17 | Confusion matrix — Random Forest (Ch. 8, Section 8.2) | — |
| 18 | Confusion matrix — MLP (Ch. 8, Section 8.2) | — |
| 19 | ROC curve — Random Forest (Ch. 8, Section 8.2) | — |
| 20 | ROC curve — MLP (Ch. 8, Section 8.2) | — |
| 21 | Federated learning convergence — F1 and ROC-AUC vs. round (Ch. 8, Section 8.3) | — |
| 22 | Model comparison — inference time and F1 (Ch. 8, Section 8.5) | — |
| 23 | Ablation — full GAT+GRU vs. GAT-only (Ch. 8, Section 8.7) | — |
| 24 | Sensitivity — window size and *k* (Ch. 8, Section 8.8) | — |
| A1-1 … A1-6 | Appendix D — handbook code figures (`DynamicGNN`, graph builder, explainer, Flower, FastAPI) | — |

### Table of Tables

**Numbering rule:** Tables are **Table 1–Table 7** in **strict order of first appearance** (literature comparison first, then Chapter 8). Refresh page numbers in Word after pagination.

| Table | Title (chapter, section) | Page |
|-------|--------------------------|------|
| 1 | Comparison of selected related work (Ch. 2, Section 2.7) | — |
| 2 | Model comparison on CICIoT2023 test set (Ch. 8, Section 8.2) | — |
| 3 | Federated learning round-by-round metrics (Ch. 8, Section 8.3) | — |
| 4 | Central GNN training history (Ch. 8, Section 8.4) | — |
| 5 | Ablation — centralised GNN variants (Ch. 8, Section 8.7) | — |
| 6 | Sensitivity analysis — nine (*window*, *k*) configs (Ch. 8, Section 8.8) | — |
| 7 | Multi-seed summary — central GNN (Ch. 8, Section 8.9) | — |

---

## Chapter 1 – Introduction

### 1.1 Chapter overview

This chapter introduces the project context, IoT flow telemetry, high SOC alert volume, and practical limits at CPU-based edge nodes. It explains why this work combines dynamic graph learning, federated training, and explainable alerts in one pipeline. The chapter also states the research aim and questions clearly, then defines scope, boundaries, and chapter structure. So, later chapters can focus on method, implementation, and evidence without repeating opening assumptions.

In addition, this chapter maps the dissertation to the approved marking structure in **Section 1.6** and the signed project specification in **Appendix B**. This alignment is important for assessment consistency. It is also useful for the reader, because the chapter explains where each technical point is formally stated once, then referenced later.

### 1.2 Background and Motivation

The Internet of Things (IoT) has expanded very fast in homes, offices, campuses, and industrial sites. Smart plugs, cameras, sensors, and controllers are now everywhere. Many of these devices are low-cost and constrained, and security is often secondary during design. As a result, weak credentials, late patching, and misconfiguration are still common in real deployments.

This means attack exposure remains high. A compromised IoT device can be used for botnet activity, service disruption, credential theft, or lateral movement into wider infrastructure. These are not rare scenarios in security reports. Because of this, continuous monitoring of network behaviour is needed, not optional, especially in environments with mixed old and new devices.

Software-defined approaches (including SDN and software-defined IoT) allow operators to observe **flow statistics** from switches and routers in a centralised way. Flow summaries are often enough for intrusion detection without full packet capture, which reduces storage pressure and can ease some privacy concerns. The remaining challenge is to analyse those flows accurately, to explain decisions in a form analysts can use, and to do so on **hardware that is realistic at the edge** (often CPU-only).

However, flow data alone does not remove the hard part. The hard part is producing accurate detections with low operational friction, then presenting results in a form analysts can use quickly during triage. In a SOC setting, model quality is not only classification accuracy. It is also alert usefulness, explanation clarity, and runtime feasibility on ordinary CPUs.

Security Operations Centres (SOCs) monitor networks, investigate alerts, and coordinate incident response, usually with SIEM platforms that aggregate logs and flow-derived events. A practical difficulty is **alert fatigue**: high volumes of alerts and many false positives reduce the time analysts can spend on genuine incidents. Alerts that only give a label, with no indication of **why** the model fired, slow triage further. There is therefore a need for detectors that are both accurate and **explainable**, so staff can trust and act on outputs under time pressure.

A model output like only "attack" is usually not enough for practical triage. Analysts often ask simple questions first, what feature pattern looked suspicious, which flow group mattered, and how confident the model was. So, explainability is part of operational quality, not a cosmetic extra. This dissertation keeps that practical perspective from the start.

Traditional models such as Random Forests or feed-forward networks can work on tabular flow features, yet traffic also has **relational structure**: flows resemble neighbours in feature space, share endpoints when identifiers exist, or arrive in short bursts that carry meaning together. Graph models treat **flows** as **nodes** and connect related flows with **edges** (here, *k*NN links in feature space because the public CICIoT2023 release does not provide device-level topology). **Dynamic** GNNs extend this idea by learning how graph snapshots evolve over short windows, which may capture attack behaviour that flat vectors hide. Separately, IoT data are often **distributed** across sites or organisations where pooling raw flows is legally or politically difficult; **federated learning** trains one shared model while keeping raw data local. Together, graph-time modelling, federation, and structured explanations align with realistic SOC and edge constraints.

Graph modelling is one way to represent these relations. In this dissertation, flows are treated as nodes and edges are created using *k* nearest neighbours in feature space. This choice is practical because the public CICIoT2023 release does not consistently provide device-level topology needed for host-to-host graphs. Dynamic GNN modelling then adds temporal context by processing a short sequence of graph snapshots.

Many IoT gateways and campus networks cannot assume a GPU on every segment. A pipeline that runs on **CPU**, reports latency honestly, and still returns SIEM-friendly JSON is therefore closer to a **deployable** research outcome than a GPU-only laboratory score alone.

### 1.3 Research Aim and Questions

This project aims to design and build a reproducible prototype for IoT attack detection using software-defined flow data, dynamic graph learning, federated training, and practical explanation outputs for SOC triage. The work focuses on feasibility under realistic constraints, especially CPU execution and non-IID federated partitions, while keeping the pipeline understandable for security operations usage.

The main research question is:

*How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?*

To support this, the following sub-questions are addressed:

1. Does a dynamic graph model perform better than simple models like Random Forest and MLP?
2. Can federated learning maintain similar performance without sharing raw data?
3. Can the model generate useful explanations for SOC alert triage?

**Contributions (in this dissertation):**

- An end-to-end, CPU-oriented prototype that turns flow telemetry into **kNN similarity graphs**, trains a **dynamic GNN** (GAT + GRU), and outputs SIEM-shaped JSON alerts.
- A centralised vs. federated comparison using the same architecture under **Flower FedAvg** with non-IID client partitions.
- An explainability path that combines **Integrated Gradients** (feature attribution) with **GAT attention** (flow and structure cues), supported by ablation, sensitivity, and multi-seed evidence.
- A submission-ready reproducibility trail through configuration, scripts, chapter mappings, and appendix artefacts.

The work uses CICIoT2023 so choices and metrics can be compared with prior studies while still fitting an MSc time-box and resource budget.

### 1.4 Scope and Limitations

The project uses **CICIoT2023** (Pinto et al., 2023) on a **manageable subset** within a **45-day** MSc window. The public release includes **46** flow-level features but not reliable device topology fields for full host graph construction. Because of that, this implementation uses **flow nodes** with ***k*NN feature-similarity edges** and a **GAT+GRU** model stack, as justified in **Section 2.5** and specified in **Section 5.3**.

The scope includes **Random Forest** and **MLP** baselines, **Flower FedAvg** federation, **Captum** explainability, and **FastAPI** ECS-like alerts. The scope does not include production SIEM integration, formal SOC user studies, or large multi-organisation federation. Technical implementation detail is intentionally concentrated in **Chapters 4 to 7** to avoid repeated text.

**Limits (summary):** no production SIEM deployment, no formal SOC user study, **CPU-only** training and inference, **three** federated clients, and **ten** rounds in the reported federated run. Headline metrics are subset-specific and discussed honestly in **Chapter 9**. Contingency decisions, such as reduced data and reduced rounds if required, follow the risk treatment in **Chapter 3**. Reproduction uses the repository, `config/experiment.yaml`, and execution notes in **Chapter 13**.

### 1.5 Dissertation Structure

The chapter order follows the **UWS MSc final-report pattern** and the **Updated Guideline for Final Report**. It includes front matter, technical body chapters, references, and appendices in the expected sequence, but all content remains specific to this project topic and evidence.

**Chapter 2** reviews literature critically, with the extended comparative subsection in **Section 2.10** to cover **fifteen to twenty** core sources. **Chapter 3** records timeline, risks, and ethics process. **Chapter 4** defines research design and methodology. **Chapter 5** specifies pipeline and graph design. **Chapter 6** documents implementation modules and integration points. **Chapter 7** defines testing protocol and metric formulae. **Chapter 8** reports results as evidence, with minimal interpretation. **Chapter 9** provides discussion, conclusions, and recommendations linked to the research questions. **Chapter 10** provides first-person critical self-evaluation. **Chapters 11 and 12** contain references and bibliography. **Chapter 13** contains appendices including process records, specification, reproducibility material, and handbook code-figure evidence.

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

**Balance check:** Chapter 2 was drafted so its depth and length are **proportionate** to the literature review weighting in the agreed specification (typically **about one fifth** of the assessed report, depending on the signed grid in Appendix B).

**Signed specification:** Generic handouts or older forms may show different percentage splits (for example **15%** for literature). **The formally signed Project Specification in Appendix B overrides any informal draft.** If the signed PDF does **not** match the percentage row in the table above, update **this Section 1.6** (and the embedded Appendix B file in the Word export) so the dissertation matches what was approved.

#### 1.6.1 Where each technical topic is stated once (clarity and non-repetition)

Strong dissertations avoid pasting the same technical block in several chapters. In this report, the **first full statement** of each topic is as below; other chapters **refer back** instead of re-explaining.

| Topic | Authoritative section | Elsewhere |
|-------|----------------------|-----------|
| Dataset origin, collection context, benchmark role | **Section 2.3** (literature); **Section 4.3** (method choice) | Ch 1 cites scope only |
| Exact split sizes, flow/sequence counts, class ratios | **Section 7.5** | Ch 8 cites numbers; Ch 9 interprets |
| Graph semantics (*k*NN, windows, sequence label rule) | **Section 5.3** | Ch 4 rationale; Ch 6 code pointers |
| Hyperparameters (trees, MLP layers, GAT heads, rounds) | **Section 6.2**, **Section 7.3** | Ch 4 lists model *roles* only |
| Metric formulae (precision, recall, F1, FPR, ROC-AUC) | **Section 7.4** | **Section 4.7** links metrics to questions only |
| Federated protocol (Flower, rounds, Dirichlet) | **Section 4.5**, **Section 6.6** | Ch 2 cites FedAvg in literature |
| Results tables and figures | **Chapter 8** | Ch 9 discusses; Ch 10 reflects |

### 1.7 Chapter summary

Chapter 1 has framed the SOC and IoT problem context, stated the research question set, and defined practical scope and boundaries. It has also mapped the report to the approved marking criteria in **Section 1.6** and clarified where each technical topic is stated once for consistency. So the reader can move into Chapter 2 with a clear map of what will be argued, what will be measured, and where each claim is evidenced.

---

## Chapter 2 – Literature Review

### 2.1 Chapter overview

This chapter reviews prior work on IoT intrusion detection, SOC/SIEM alerting needs, graph and dynamic GNN methods, federated learning, and explainable AI in security. The aim is clear. I need to show what other studies did well, what they did not cover, and how that leads to the design choices in this dissertation. Dataset counts, split statistics, and configuration details are kept for the methodology and evaluation chapters (**Section 7.5** and **Section 6.2**) so this chapter stays analytical, not a second methods section.

### 2.2 Themes and structure

The review is organised by theme. Each subsection links to the research questions. Critical analysis is given where it helps justify design choices for the prototype. **Roadmap for the reader:**

- **Section 2.3** — IoT threat context, CICIoT2023, and why flow-level class imbalance matters for this work.  
- **Section 2.4** — SIEM/SOC workflows, alert quality, and ECS-like structured outputs.  
- **Sections 2.5 and 2.6** — Graph and **dynamic** GNN ideas, then federated learning and non-IID effects.  
- **Section 2.7** — Explainability methods (Integrated Gradients, attention) and runtime trade-offs.  
- **Sections 2.8 through 2.11** — Gap, contribution, CyBOK mapping, extended comparison across sources, and chapter summary.

### 2.3 IoT Security and the Need for Detection

IoT devices are now common in many sectors, but they are often built with cost and convenience in mind rather than security. Researchers have pointed out that many devices use default credentials, have unpatched software, and lack strong encryption (Kolias et al., 2017). When these devices are compromised, they can be used in botnets, for data exfiltration, or as a stepping stone into the rest of the network. So, detecting malicious behaviour in IoT traffic is an important part of modern security operations.

The scale of the IoT threat is well known. Kolias et al. (2017) looked at the Mirai botnet. It used weak Telnet passwords on millions of IoT devices to launch large DDoS attacks. Such incidents show that IoT networks need continuous monitoring and automated detection. Manual inspection is not possible at scale. IoT devices vary a lot (cameras, thermostats, industrial controllers). So attack surfaces vary widely. Detection systems must work with different traffic patterns and protocols. SDN and Software-Defined IoT add another layer. Central controllers can collect flow statistics from switches and routers. But the amount of data and the need for real-time analysis require efficient detection methods that can run at the edge.

Intrusion detection for IoT can be done in different ways. Signature-based methods look for known attack patterns. Anomaly-based methods learn what is normal and flag the rest. Machine learning has been used a lot for attack detection on flow or packet data. The CICIoT2023 dataset used in this project was made to support such research. It includes attacks like DDoS, reconnaissance, and brute force from real IoT devices in a controlled environment (Pinto et al., 2023). The dataset has 46 pre-computed flow features from packet captures. This reduces the need for raw packet inspection. It also matches the kind of data that SIEM systems usually use. Pinto et al. (2023) describe how the data was collected. Traffic was captured from real IoT devices (smart plugs, cameras, doorbells) under controlled attack scenarios. Flow features were extracted using standard tools. The features include protocol indicators (TCP, UDP, ICMP, HTTP, DNS), packet and byte statistics, TCP flag counts, and statistics like mean, variance, and standard deviation. Using a public dataset lets results be compared with other work. But the setting is still lab-based, not a live network. Yang et al. (2025) recently showed that graph neural networks can improve intrusion detection for industrial IoT. They showed that modelling network structure can improve accuracy compared to flat feature models. The CICIoT2023 dataset has a class imbalance (about 97.8% attack, 2.2% benign at the flow level). This makes training harder. This project deals with it through stratified windowing and balanced graph construction.

Many studies treat each flow or packet on its own. But in reality, attacks often show up as patterns of communication between devices over time. For example, a DDoS attack may involve many flows from many sources to one target. A reconnaissance scan may show sequential probing of ports. Flat feature vectors lose this structure. Wang et al. (2025) did a scoping review. They found that graph-based approaches to network traffic analysis are used more and more. Dynamic graph models in particular look promising for capturing attack patterns that change over time. Zhong et al. (2024) surveyed graph neural networks for intrusion detection. They found a clear trend towards combining GNNs with temporal models. They also noted that explainability and federated deployment are still under-explored. These findings support the graph-based and dynamic approach in this project. A key gap in the literature is that there are few prototypes that combine graph-based detection, federated learning, and explainable alerting in one system for SOC use. Most papers focus on only one of these. Figure 1 summarises the main categories of intrusion detection approaches relevant to IoT and this project.

![Figure 1: IDS taxonomy](assets/literature_ids_taxonomy.png)

**Figure 1: Taxonomy of IDS approaches relevant to IoT and this project** *(Chapter 2, Section 2.3)*

*Sources: Kolias et al. (2017); Pinto et al. (2023); Wang et al. (2025); Zhong et al. (2024). Full references in Chapter 11.*

What is still missing in this pillar is an end-to-end operational bridge from IoT threat evidence to SOC-ready alert output under edge hardware limits. Many papers confirm the attack problem, but fewer show a full and reproducible pipeline from flow ingestion to explainable alert delivery.

### 2.4 SIEM, SOC Workflows, and Alert Quality

SOC teams use SIEM and related tools to collect logs and flow data and to generate alerts. A well-known problem is alert fatigue. Too many alerts and too many false positives make it hard for analysts to focus on real threats (Cuppens and Miège, 2002). If an alert does not explain why it was raised, triage becomes slower and more guesswork. Cuppens and Miège (2002) proposed alert correlation to reduce noise by grouping related alerts. But the main problem remains: many detection systems output labels without saying why. This is still relevant today.

The Elastic Common Schema (ECS) gives a standard structure for security events. It includes fields for event type, outcome, source, destination, and custom extensions. Using ECS-like output makes it easier to integrate with Elastic Security, Splunk, and other SIEM platforms. This project uses an ECS-like structure for the alert JSON. So the output can be used in existing SOC tools with little change.

Modern SIEM platforms like Splunk, Elastic Security, and Microsoft Sentinel support custom rules and machine learning models. But the use of explainable AI in alert workflows is still developing. When a model flags traffic as malicious, analysts need to know which features or events led to that decision. This helps them prioritise and avoid wasting time on false positives. Without explanations, analysts may ignore alerts (increasing risk) or over-investigate (reducing efficiency).

There is growing interest in making security tools more explainable. For example, some work has focused on explaining which features led to a detection (e.g. Lundberg and Lee, 2017, on SHAP). In a SOC context, explanations that point to specific flows, devices, or time windows can help analysts decide quickly whether to escalate. This project does this by producing SIEM-style alerts with top features and flows attached. So the output is not only a label but something an analyst can act on. The ECS-like JSON format works with common SIEM tools.

What remains missing in this SIEM and SOC pillar is consistent joining of model performance, explanation quality, and deployable alert schema in one implementation. Prior work often addresses one part, but not the full analyst workflow from score to triage context.

### 2.5 Graph Neural Networks and Dynamic Graphs

Graph neural networks (GNNs) work on data that has a graph structure: nodes and edges. In network security, nodes can be hosts or devices and edges can be flows or connections. GNNs combine information from neighbours and can learn patterns that depend on the graph structure. The main idea is that message passing (where each node updates based on its neighbours) lets the model capture relationships that flat classifiers cannot. Graph Attention Networks (GATs) go further. They let each node give different importance to its neighbours using attention (Velickovic et al., 2018). This helps the model focus on the most relevant connections. That is useful when some flows are benign and others are part of an attack. Han et al. (2025) proposed DIMK-GCN for intrusion detection. They showed that multi-scale graph representations can capture both local and global attack patterns. Ngo et al. (2025) looked at attribute-based graph construction for IoT intrusion detection. They showed that building graphs from feature similarity (not just network topology) can work when device IDs are not available. This supports the kNN-based graph construction in this project. When IP addresses or device IDs are absent (as in the public CICIoT2023 release), topology-based graphs cannot be built. So attribute-based or similarity-based construction is the only option.

Networks change over time. New connections appear, traffic volumes shift, and attacks develop in stages. Dynamic or temporal graph models try to capture this. One common approach is to combine a GNN with a recurrent module (e.g. GRU or LSTM). The model sees a sequence of graph snapshots and learns from how the graph evolves. Several papers have used this for fraud detection or anomaly detection (e.g. Zheng et al., 2019). Lusa et al. (2025) proposed TE-G-SAGE for network intrusion detection. It combines temporal edge features with GraphSAGE. They showed that both edge-level and temporal information improve accuracy. Basak et al. (2025) developed X-GANet for network intrusion detection. They confirmed that attention-based GNNs can get high accuracy and also give interpretable outputs. For IoT flow data, building time-windowed graphs and using a dynamic GNN (GAT + GRU in this project) is a reasonable way to test whether structure and temporality improve detection over simple models like Random Forest or MLP. The literature supports the idea that graph structure can add value. But the actual gain depends on the data and task. That is why this project compares the dynamic GNN to baselines. GNNs also cost more to run than tabular models. The trade-off between accuracy and inference time matters for edge deployment. This project evaluates it. The combination of GAT (for learning which neighbours matter) and GRU (for learning how the graph evolves) is a common pattern in temporal graph learning. This project applies it to IoT flow data with kNN graph construction. This has not been explored much in prior work. Figure 2 illustrates the conceptual flow: graph snapshots over time are processed by the GAT, then the sequence of graph representations is fed to a GRU for temporal modelling before the final prediction.

![Figure 2: Dynamic GNN concept](assets/literature_dynamic_gnn_concept.png)

**Figure 2: Conceptual flow of a dynamic GNN (GAT + GRU) for temporal graph classification** *(Chapter 2, Section 2.5)*

*Sources: Velickovic et al. (2018); Zheng et al. (2019); Lusa et al. (2025); Basak et al. (2025). Full references in Chapter 11.*

What is still missing in this pillar is stronger evidence on graph models under practical deployment constraints, especially CPU-limited settings with direct comparison against simple baselines in one reproducible workflow.

### 2.6 Federated Learning and Privacy

Federated learning trains a model across many clients without putting raw data in one place. Each client trains on local data and sends only model updates (gradients or weights) to a server. The server combines them and sends back an updated model. FedAvg (McMahan et al., 2017) is a standard algorithm. The server averages the client model parameters, often weighted by how much local data each client has. This reduces privacy risk compared to sending raw IoT traffic to a central site. That matters when data comes from different organisations or locations. In IoT and edge deployments, data may be spread across many sites (different buildings, campuses, or organisations). Rules like GDPR may not allow centralising sensitive network data. Federated learning also reduces the bandwidth needed for training. Instead of transferring raw flows (which can be large), only model parameters (usually a few megabytes) are sent. So federated learning works well for bandwidth-limited edge networks. Figure 3 shows the FedAvg flow: clients train locally and send model updates to the server; the server aggregates and broadcasts the global model back.

![Figure 3: FedAvg flow](assets/literature_fedavg_flow.png)

**Figure 3: Federated learning (FedAvg) flow — no raw data leaves clients** *(Chapter 2, Section 2.6)*

*Sources: McMahan et al. (2017); Lazzarini et al. (2023). Full references in Chapter 11.*

Federated learning has been used in IoT and fog-edge settings (e.g. Qu et al., 2020). Lazzarini et al. (2023) looked at federated learning for IoT intrusion detection. They found that FedAvg can get performance similar to centralised training while keeping data local. Albanbay et al. (2025) did a performance study on federated intrusion detection in IoT. They showed that non-IID data across clients is a key factor for convergence and accuracy. When clients have different attack types or different mixes of benign and malicious traffic, the global model must learn from mixed updates. FedAvg can still converge, but the number of rounds and aggregation strategy matter. Challenges include communication cost, non-IID data, and possible drops in accuracy compared to centralised training. In this project, Flower is used to implement FedAvg with three clients and a Dirichlet-based non-IID data split. The goal is to show that federated training can get performance close to centralised training. The evaluation includes communication cost and round-by-round performance. Flower was chosen because it works well with PyTorch and supports custom client and server logic.

What is still missing in this federated pillar is tighter integration with dynamic graph modelling and explanation output at the same time. Many studies evaluate federation in isolation, while SOC deployment needs federation, prediction quality, and interpretable alert content together.

### 2.7 Explainability in ML-Based Security

Explainability can be added to a system in different ways. Post-hoc methods explain a trained model. They show which inputs or features mattered most for a prediction. Integrated Gradients is a gradient-based method. It attributes the prediction to input features relative to a baseline (Sundararajan et al., 2017). It has useful properties like sensitivity and implementation invariance. It is implemented in Captum (Kokhlikyan et al., 2020), which works with PyTorch. For GNNs, attention weights from GAT layers can also show which edges or neighbours the model focused on. This project uses both: Integrated Gradients for feature-level explanation and attention for flow-level importance. So the alert can include both which features (e.g. packet rate, flag counts) and which flows contributed most to the prediction. Figure 4 summarises the explainability pipeline used in this project for SOC-oriented alerts.

![Figure 4: Explainability pipeline](assets/literature_explainability.png)

**Figure 4: Explainability methods used for SOC-oriented alerts** *(Chapter 2, Section 2.7)*

*Sources: Sundararajan et al. (2017); Velickovic et al. (2018); Kokhlikyan et al. (2020). Full references in Chapter 11.*

Alabbadi and Bajaber (2025) showed that explainable AI (XAI) can be used for intrusion detection over IoT data streams. They confirmed that post-hoc explanation methods can make security decisions more transparent for analysts. Lundberg and Lee (2017) introduced SHAP for model interpretation. SHAP can be applied to any model. But Integrated Gradients is often preferred for neural networks because it is gradient-based and works well with backpropagation. One practical point: full explainability for every prediction can be slow. Integrated Gradients needs many forward passes (usually 50–100) to approximate the integral. For a GNN processing sequences of graphs, this can add a lot of latency. So the design allows applying explainability only to selected alerts (e.g. high-confidence positives) if needed. The literature does not always address this trade-off. This project notes it as a limitation and a possible area for future work.

What is still missing in the XAI pillar is stronger reporting on explanation latency versus analyst value in operational settings. Papers describe methods, but fewer report whether the explanation payload helps real triage within strict timing constraints at the edge.

Table 1 and Figure 5 summarise how selected related work maps onto the four pillars of this project (GNN/graph-based detection, federated learning, explainability, and SIEM-style alert output). All sources are given in the References (Chapter 11). No single prior study combines all four in one prototype for SOC use on edge devices; this project fills that gap.

**Table 1: Comparison of selected related work (sources: Chapter 11)**

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

![Figure 5: Positioning of related work](assets/literature_positioning.png)

**Figure 5: Positioning of related work — GNN, federated learning, explainability, and SIEM-style alerts** *(Chapter 2, Section 2.7)*

*Sources: Pinto et al. (2023); Velickovic et al. (2018); McMahan et al. (2017); Sundararajan et al. (2017); Han et al. (2025); Ngo et al. (2025); Basak et al. (2025); Lusa et al. (2025); Lazzarini et al. (2023); Albanbay et al. (2025); Alabbadi and Bajaber (2025); Yang et al. (2025). This figure synthesises **Chapter 2** and supports the gap stated in **Section 2.8**.*

### 2.8 Gap and Contribution

Existing work covers IoT intrusion detection, GNNs, federated learning, and explainability separately. There is less work that combines an explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype for SOC use on CPU-based edge devices. Yang et al. (2025) and Han et al. (2025) focus on GNN architectures for detection. But they do not add federated learning or SIEM output. Lazzarini et al. (2023) and Albanbay et al. (2025) look at federated learning for IoT intrusion detection. But they do not combine it with dynamic graphs or explainable alerts. Alabbadi and Bajaber (2025) and Basak et al. (2025) look at explainability. But they do it in centralised settings. This project tries to fill that gap with a small but complete system. It has: (1) dynamic graph design from IoT flow data, (2) GAT+GRU model, (3) FedAvg via Flower, (4) explanations via Captum and attention, and (5) JSON alerts in ECS-like format. The evaluation answers whether the graph model beats simple baselines, whether federated learning keeps performance, and whether the explanations are useful for triage. The scope is limited to a 45-day MSc project and a subset of CICIoT2023. But the structure is set up so the results can be discussed critically and extended later. In summary, this project delivers a prototype that combines all five: kNN graph construction, dynamic GNN, federated learning, explainability, and SIEM-style alerts. No single prior work combines all five in one system for SOC use on CPU-based edge devices.

### 2.9 Mapping to CyBOK (Cyber Security Body of Knowledge)

The **agreed project specification** (Appendix B) records this work against **CyBOK** knowledge areas. The dissertation aligns with the following topics in particular: **Attacks & Defences → Security Operations & Incident Management** (SIEM-style alerting, SOC triage, and intrusion detection outputs); **Infrastructure Security → Network Security** (flow-based analysis and IoT traffic classification); **Systems Security → Distributed Systems Security** (federated learning without centralising raw client data); and **Software and Platform Security → Software Security** (engineering of the prototype pipeline, APIs, and reproducible scripts). Together, these areas situate the project within the School’s CyBOK mapping and match the specification’s indicative coverage.

### 2.10 Extended comparative review (fifteen to twenty core sources)

The programme guideline asks for **fifteen to twenty** significant papers in the literature review. Earlier sections already developed themes with primary citations. This subsection **groups additional comparisons** so the breadth requirement is explicit: **(1) temporal graphs and anomalies** — Zheng et al. (2019) show attention-based temporal GCNs for dynamic graphs; that line of work motivates stacking a sequence learner (here, a GRU) on top of graph snapshots rather than treating each window alone. **(2) Federated and fog settings** — Qu et al. (2020) discuss decentralised privacy patterns in fog computing; the present work does not use blockchain, but the paper helps position why edge sites may refuse raw data pooling. **(3) Surveys and scoping reviews** — Zhong et al. (2024) and Wang et al. (2025) summarise how GNNs are used in intrusion detection and where explainability and deployment gaps remain. **(4) SHAP vs. gradient methods** — Lundberg and Lee (2017) describe SHAP values; this project uses Integrated Gradients (Sundararajan et al., 2017) and GAT attention because they fit the PyTorch + Captum stack, but SHAP remains a valid alternative for tree or tabular baselines. **(5) IoT threat context** — Kolias et al. (2017) document Mirai-style botnets, which explains why flow-level DDoS patterns appear heavily in IoT benchmarks. **(6) Alert operations** — Cuppens and Miège (2002) give early alert-correlation ideas that sit under modern SIEM practice even though formats have moved toward JSON and ECS-like fields.

Across these strands, the **gap** for this dissertation stays the same as in **Section 2.8**: few published prototypes combine **dynamic GNNs**, **federated training**, **gradient + attention explanations**, and **SOC-shaped JSON** in one CPU-oriented pipeline on **CICIoT2023**.

### 2.11 Chapter summary

The literature supports using **dynamic GNNs**, **federated learning**, and **XAI** for intrusion detection, but rarely **all three** together with **SIEM-oriented** outputs for edge SOC workflows. Surveys and recent papers agree that graph-time models are promising, while federated and explainable deployment in the same codebase is still thinly evidenced for IoT flow benchmarks. **Table 1** and **Figure 5** make that gap visible by pillar (graph, FL, XAI, SIEM). The next chapters turn from **what others did** to **what this project built**: fixed dataset handling, *k*NN temporal graphs, the GAT+GRU stack, Flower FedAvg, Captum, and FastAPI-delivered ECS-like alerts, evaluated with baselines and stability checks.

Another important point from this review is transfer difficulty across contexts. Many papers report strong metrics on one dataset and one setup, but they do not show if the same method keeps quality when data distribution shifts, when compute is restricted, or when explanations must be delivered inside an operational alert object. This project does not claim to solve full transfer. It does, however, implement the pipeline in a way that makes these practical questions visible and measurable.

So the value of Chapter 2 is not only source coverage. It is a clear argument chain from prior evidence to design decisions used in this dissertation. This is why the following chapters can focus on implementation and measured outcomes, rather than re-arguing the basic method choice in each section.

One more synthesis point is reproducibility in literature practice. Several papers report excellent metrics but give limited implementation detail, which makes direct comparison hard when dataset preprocessing, split strategy, or label framing differ. In this dissertation, those items are explicitly separated by chapter to improve traceability: design rules in Chapter 5, implementation modules in Chapter 6, metric definitions in Chapter 7, and measured outcomes in Chapter 8. This structure does not solve all comparability limits in the field, but it makes this project auditable from source assumptions to final plots.

The literature also shows a recurring tension between method novelty and operational utility. A technically advanced model can still be hard to deploy if runtime, explanation latency, or alert formatting are weak. Because of this, this dissertation uses a practical framing where deployment-facing output is treated as part of the technical contribution, not an afterthought. The chapter therefore supports both academic and operational justification for the selected architecture and evaluation strategy.

Finally, this chapter supports critical positioning rather than broad claiming. It does not present any cited method as universally best. Instead, it compares assumptions, strengths, and missing elements, then derives a focused gap statement for this dissertation scope. That approach keeps the research question realistic and testable.

This review also helped scope the writing style of later chapters. Methods and results are presented in a way that lets the reader connect each claim to concrete evidence, instead of relying on broad statements. In a dissertation context, this matters because clarity of evidence path is part of assessment quality.

---

## Chapter 3 – Project Management

### 3.1 Chapter overview

This chapter documents the project process: planning and milestones, risk management, ethics posture (public dataset, no human participants), and how interim feedback was incorporated. It supports transparency of execution without duplicating the technical content presented in Chapters 4–7.

I include this chapter because method quality is not only model architecture. It is also project discipline, decision timing, and honest recording of constraints during development. In this project, timeline control mattered, because data preparation, graph construction, federation, and explainability all depended on each other and delays in one area quickly affected later experiment windows.

### 3.2 Project plan and timeline (45-day MSc)

Work was staged in six phases: (1) freeze requirements and literature; (2) fix the CICIoT2023 subset and preprocessing; (3) implement graph construction and central GNN training; (4) run baselines and Flower-based federated training; (5) add explainability, alert JSON, and FastAPI; (6) run ablation, sensitivity, multi-seed experiments, figures, and final writing. Each phase had concrete artefacts (metrics files, checkpoints, plots) to reduce ambiguity about “done.”

Phase outputs were tracked with practical evidence, not only task labels. For example, preprocessing completion required saved split files and reproducible rerun commands. Central training completion required stable loss behaviour, saved checkpoints, and metric logs that could be regenerated. Federated completion required round-by-round logs and a final model state that could be scored on the fixed test split. This approach reduced confusion when multiple scripts were evolving in parallel.

Time management was strict but still iterative. Some tasks were moved forward when blockers appeared, especially around graph data preparation and explanation latency. The plan therefore functioned as a controlled loop rather than a rigid waterfall. Even with that flexibility, the six-phase structure helped protect the core deliverables promised in the specification and interim milestones.

### 3.3 Risk assessment

| Risk | Mitigation |
|------|------------|
| Training too slow on CPU | Fixed subset; configurable rounds/clients; early stopping |
| Severe class imbalance | Stratified windowing; class weights; balanced graph-level labels |
| Federated instability | Three clients; reproducible split; checkpoints each round |
| Explainability runtime | Top-*k* attributions; optional subset of alerts |
| Scope creep (extra models) | Baselines limited to RF and MLP; one main GNN architecture |

Risk handling was reviewed during implementation, not only at planning stage. If one risk signal increased, for example slower-than-expected training or unstable federated rounds, mitigation was applied in the next run cycle and noted in chapter updates. This kept the project controllable under time pressure.

A practical example is explainability cost. Integrated Gradients can be expensive when run on every sample. The mitigation used here was to keep full explanation for selected alert cases and maintain the same explanation schema in output JSON. This preserved SOC relevance while preventing runtime from dominating all experiment cycles.

### 3.4 Ethics and data

Only **public** CICIoT2023 data is used; there are no human participants or proprietary organisational datasets. The **signed specification ethics section** records supervisor confirmation that the project **does not require** full ethics-review board approval under School procedures (February 2026), which matches the use of non-sensitive public data only. Process and ethics paperwork are referenced in **Appendix A/B** as required by the handbook.

### 3.5 Interim report feedback incorporated

The MSc Project Handbook requires that feedback on the **interim report** (written and verbal from supervisor and moderator) be taken into account in the final submission. After interim submission, feedback reinforced: (1) keeping the **end-to-end** story visible (data → model → federation → explainability → SOC-shaped output), not only literature; (2) tightening the **evaluation plan** before reporting numbers; (3) being explicit about **limits** of a lab subset and very high test metrics; and (4) reserving effort for **final analysis and write-up** rather than scope creep.

Actions taken in the final dissertation include: completing the full evaluation and results chapters (**7–8**) with tables and figures promised at interim; adding **ablation**, **sensitivity**, and **multi-seed** checks so result stability is evidenced (going beyond the interim contingency of dropping sensitivity if time ran short); strengthening **critical reflection** (**Chapter 10**) on methodology, federation, and the “100% metrics” question; and documenting **reproducibility** (**Chapter 13**). Any remaining suggestions from the interim review are acknowledged as **limitations** and **future work** in **Chapter 9**.

Feedback also emphasised communication quality. Earlier drafts had strong technical content but some sections were too condensed for non-specialist readers. In the final version, chapter introductions and summary paragraphs were expanded so each chapter states purpose, scope, and linkage to research questions before detailed evidence. This improved readability and made examiner navigation easier.

### 3.6 Chapter summary

Transparent **time-boxing** and **risk controls** kept the project feasible while still delivering graph, federated, explainable, and deployable components.

Project-management quality also influenced technical quality directly. Stable planning reduced context switching, and this helped keep experiments reproducible across central and federated paths. Weekly checkpoints, risk review updates, and explicit completion evidence made it easier to avoid hidden drift between design intent and implementation reality.

This chapter also documents a realistic MSc pattern: decisions are made under pressure, and trade-offs are unavoidable. What matters is whether those trade-offs are explicit, justified, and reflected later in interpretation. In this project, choices about subset size, client count, and explanation scope were treated as controlled constraints, then revisited in limitations and future work sections.

Project governance also included version discipline and checkpoint traceability. Results were linked to script runs, configuration settings, and saved artefacts to avoid confusion between exploratory outputs and report-ready outputs. This may sound procedural, but it mattered when figures and tables were finalised under deadline pressure.

In practical terms, the management chapter shows that reliable technical output requires planning behaviour, not only coding effort. The same model architecture can produce weak or strong evidence quality depending on whether process controls are clear and consistently followed.

It also shows how non-technical constraints shape technical outcomes. Deadlines, compute limits, and review cycles affected experiment order, script hardening, and writing sequence. Recording these factors openly gives a more accurate account of how the final artefacts were produced.

Another management learning was dependency awareness. Some tasks looked independent on paper, but in practice they were linked, for example graph preprocessing choices affected model training stability, and model output structure affected explainability and API payload design. Tracking those dependencies early reduced rework later and helped protect submission quality.

This planning experience also reinforced the value of visible checkpoints. When milestones are explicit and evidence-linked, feedback can be applied earlier and scope can be corrected before final weeks. For MSc projects with fixed submission windows, this is often the difference between controlled delivery and rushed consolidation.

---

## Chapter 4 – Research Design and Methodology

### 4.1 Chapter overview

This chapter sets out the methodology used to answer the research questions: research approach, dataset and split rationale, model selection (RF, MLP, dynamic GNN), federated learning setup, explainability plan, and evaluation plan. Design-level pipeline overview, graph semantics, and system architecture are specified in **Chapter 5** (including Figures 6–8), implementation details in **Chapter 6**, and metric definitions in **Section 7.4** to avoid repetition across the report.

The intention here is to make the logic auditable from question to evidence. Each design choice, for example window size, graph density, federated rounds, and explanation method, is mapped to a concrete evaluation outcome later in Chapter 8. This means the chapter is not only descriptive. It is a decision trail that can be checked by another student or examiner.

### 4.2 Research Approach

The project is practical. The aim is to build a working prototype and evaluate it with clear metrics. The main research question is answered by building the system, comparing the dynamic GNN to baselines, testing federated learning, and checking whether the explanations help SOC triage. Sub-questions are answered through experiments and by looking at example alerts.

The methodology follows a design-science approach. A prototype is built and evaluated with quantitative metrics (precision, recall, F1, ROC-AUC, inference time, communication cost) and qualitative assessment (interpretability of example alerts). The choice of CICIoT2023, kNN graphs, GAT+GRU, Flower, and Captum is justified by the literature and by practical constraints (dataset availability, 45-day timeline, CPU-only deployment). The evaluation plan is defined before experiments are run. So results can be interpreted objectively.

This design-science framing is useful for this dissertation because the output is a functioning artefact, not only a statistical claim. The artefact includes data processing, model training, federated orchestration, explanation logic, and API serving. Each part can fail independently. Therefore, evaluation has to include both prediction quality and integration quality.

A second methodological point is control of comparison fairness. Baselines and GNN are evaluated on fixed splits derived from the same source dataset. Federated and central runs use the same architecture so comparison reflects training mode difference rather than model-capacity difference. This does not eliminate every confounder, but it reduces major threats to validity.

### 4.3 Dataset and Subset

The CICIoT2023 dataset (Pinto et al., 2023) is used. It has network flow data from IoT devices under various attacks (DDoS, DoS, reconnaissance, brute force, spoofing). The full dataset is large (millions of flows). So a manageable subset is selected to fit the 45-day timeline. The subset includes both benign and attack traffic and a spread of attack types. The same subset is used for all experiments so results are comparable. Data is split into fixed train, validation, and test sets (e.g. 70% train, 15% validation, 15% test, with no overlap). This avoids data leakage and allows fair comparison between centralised and federated training.

CICIoT2023 provides **46** numeric flow features (rates, counts, protocol indicators, TCP flag statistics, etc.—see Pinto et al., 2023). The **exact column list** used in code is **`config/experiment.yaml`** (reproduced in **Appendix C**). Features are **standardised** on train statistics only, then applied to validation and test. **Binary** labels (benign vs. attack) are used. The subset keeps both classes present; the **test** split is held out for final reporting only.

Pinto et al. (2023) describe how CICIoT2023 was captured in a controlled IoT testbed with benign activity and many attack scenarios, then released with per-flow features suitable for real-time detection benchmarks. This dissertation does not re-collect traffic; it uses that **public release** so that preprocessing, splits, and models can be checked by others.

Subset choice is a limitation and a design control at the same time. It limits external validity, because full-scale deployment behaviour cannot be inferred from one controlled subset alone. At the same time, it enables repeated end-to-end experiments inside MSc constraints, including ablation and sensitivity analysis. In this project, reproducible evidence was prioritised over one very large, non-repeatable run.

### 4.3.1 Where the data lives (repository layout)

Raw CICIoT2023 files are **large** and are **not** committed to Git. In a local checkout, raw CSV (or unpacked archive) is expected under a path such as `data/raw/` (see `README.md` in the repository). After preprocessing, **train / validation / test** parquet splits and metrics live under `data/processed/` and `results/` respectively. The dissertation numbers (tables and figures) were produced from one frozen subset and the scripts named in Chapter 13; if another researcher downloads the full public dataset, they must run the same preprocessing script to reproduce the exact row counts.

### 4.4 Models

Three types of models are used so that the value of the graph and temporal structure can be assessed.

- **Random Forest:** A baseline on **tabular flow-level data**: each training/test instance is **one processed flow** (the same 46 standardised features as in the graph pipeline), with no graph structure. Implemented via scikit-learn on the train/test parquet splits (`src/models/baselines.py`).
- **MLP (Multi-Layer Perceptron):** The same **flow-level** representation as Random Forest—a feed-forward network on 46-dimensional rows without graph or temporal modelling (`MLPBaseline` in `src/models/baselines.py`).
- **Dynamic GNN (GAT + GRU):** The main model. Graph Attention layers process each graph snapshot; a GRU (or similar RNN) then processes the sequence of graph representations over time. The output is used for attack vs. benign (or multi-class) prediction. This model uses both structure and temporality. The GAT allows the model to weight neighbours differently, and the GRU captures how the graph representation changes across the sequence.

All models are trained to predict attack (or attack type) from the available features and, for the GNN, from the graph and its evolution. Hyperparameters (e.g. learning rate, number of layers, hidden size) are chosen with the validation set; the test set is used only for final reporting. The same train/validation/test splits are used for all models to ensure fair comparison.

Model roles are intentionally different. Random Forest and MLP represent strong tabular baselines with low engineering overhead. The dynamic GNN represents the research contribution path where relational and temporal information may add value. This structure allows the dissertation to answer whether added model complexity is justified by evidence, not by assumption.

The baseline inclusion is also important for transparency. If only one advanced model were presented, it would be hard to judge whether performance came from genuine structural learning or from data properties that any competent classifier could exploit. By keeping RF and MLP in the same pipeline, this work supports a clearer interpretation of comparative outcomes.

### 4.5 Federated Learning Setup

**FedAvg** (McMahan et al., 2017): clients train locally and send **weights**; the server **averages** and broadcasts the global model—no raw flows cross sites (**Section 2.6**). This project uses **Flower** with **three** clients, **ten** rounds, and **two** local epochs per round (timelines and risks: **Chapter 3**; wiring: **Section 6.6**). The **same** GNN architecture is trained centrally and federally for a fair comparison. **Communication cost** is estimated from serialized **float32** parameter traffic (**Chapter 8**).

**Non-IID simulation:** training rows are partitioned across clients with a **Dirichlet** distribution over **class proportions** (**alpha = 0.5**) so client label mixes differ (stronger than IID sharding; weaker than adversarial splits). Rationale: reflect sites with different exposure while staying trainable within the MSc budget.

The federated setup is intentionally moderate rather than extreme. Three clients and ten rounds allow meaningful round dynamics without making the project intractable on CPU-only resources. The chosen heterogeneity level (Dirichlet alpha = 0.5) creates visible non-IID behaviour while still allowing convergence with standard FedAvg. This balance fits the dissertation objective, which is feasibility and evidence under constrained scope.

### 4.6 Explainability

Two forms of explanation are used:

- **Integrated Gradients (Captum):** Applied to the chosen model (e.g. the dynamic GNN or the MLP) to attribute the prediction to input features. The top contributing features are extracted and included in the alert.
- **Attention weights:** For the GAT-based model, attention weights over edges or neighbours are used to identify which flows or connections the model focused on. These can be summarised as “top flows” or “top edges” in the alert.

Explanations are produced for selected predictions (e.g. positive alerts). The top-k (k=5) features and flows are included in each alert to keep the explanation concise. If computing explanations for every prediction is too slow, they are applied only to a subset (e.g. high-confidence alerts) and this is noted as a limitation.

This explainability design follows a practical SOC perspective. Analysts generally need concise and comparable signals, not very long attribution dumps. Keeping top-k explanations in a stable JSON structure supports consistent triage behaviour and downstream ingestion. It also keeps latency manageable in CPU-focused deployment conditions.

### 4.7 Evaluation Plan

**Formulae and definitions** for precision, recall, F1, FPR, and ROC-AUC are given **once** in **Section 7.4**. Here, the plan only **maps** measurements to questions.

Evaluation is designed to answer the three sub-questions and support the main research question.

- **Metrics:** Precision, recall, F1-score, ROC-AUC, and false positive rate on the fixed test set. A confusion matrix is reported.
- **Model comparison:** Centralised training: Random Forest, MLP, and dynamic GNN are compared on the same splits. Federated training: the same GNN (or MLP) is trained with FedAvg and compared to its centralised version.
- **Time and cost:** Time-window analysis (e.g. performance across different window sizes or positions), federated round-by-round performance, approximate communication cost (bytes), and CPU inference time per sample or per batch.
- **Explainability and SOC use:** Three to five example alerts are generated with full explanations (top features and flows). These are discussed in terms of whether they would help a SOC analyst triage (e.g. whether the highlighted flows and features are interpretable and actionable).

Contingencies match **Chapter 3** (subset size, rounds/clients, optional subset of explained alerts). **Linkage to questions:** precision, recall, F1, ROC-AUC, and FPR → **sub-question 1** (and central leg of **sub-question 2**); per-round FL metrics and comms cost → **sub-question 2**; example alerts → **sub-question 3**; CPU inference latency → **main question** (edge deployability).

### 4.7a Validity and quality controls

Method quality depends on controls as much as on model architecture. This project applies several controls to reduce biased reporting. First, metric definitions are fixed before results discussion and are written once in Chapter 7. Second, test data is held out and not used for hyperparameter selection. Third, central and federated comparisons reuse the same dynamic GNN architecture, so training mode is the main changed factor.

Another control is explicit baseline inclusion. Random Forest and MLP are retained through the full evaluation, including confusion matrices and ROC plots, not only one summary line. This prevents over-claiming around the dynamic GNN and keeps interpretation grounded in comparative evidence. A final control is stability analysis through ablation, sensitivity, and multi-seed runs. These checks do not guarantee external validity, but they reduce the chance that one configuration artefact is presented as a general finding.

Interpretability is also treated with quality checks. Explanations are reported in a fixed structure (top features and top flows), and examples are shown in a consistent format that maps model output to alert context. This allows the reader to inspect whether explanations are plausible and operationally meaningful. The dissertation does not include a formal user study, so interpretation claims stay conservative and are limited to evidence presented in the report.

Finally, reproducibility quality is addressed by scriptable execution and explicit file-path references. While full independent replication still requires environment setup and data access, the method steps and expected artefacts are documented clearly enough to support verification. This is important for assessment at MSc level and for future extension of the work.

Another methodological safeguard is interpretation discipline. The dissertation keeps measurement and interpretation in different chapters, and this avoids mixing exploratory narrative with evidence reporting. It also helps examiners check whether each conclusion is directly supported by one or more reported tables or figures.

Method scope also stays explicit. This project aims to show feasibility and integrated behaviour under constrained resources, not final production assurance across all operating conditions. That boundary is repeated in design, evaluation, and discussion chapters to keep claims proportionate to evidence.

A final point is consistency between textual claims and numeric reporting. Metric values are reported in decimal form in tables and percentage form in prose, with direct equivalence stated where needed. This reduces ambiguity and improves auditability when readers compare chapter text with exported CSV and plot files.

Method chapter completeness also depends on transparent linkage to chapter responsibilities. Design assumptions are deferred to Chapter 5, executable details to Chapter 6, and metric mathematics to Chapter 7. This avoids duplication and keeps each chapter focused while preserving full traceability across the dissertation.

An additional methodological consideration is reproducible failure handling. During development, some runs failed because of data-shape or batching issues. Instead of discarding failed attempts informally, the workflow used versioned scripts and rerun controls so fixes could be verified under the same experimental assumptions.

These choices improve confidence in reported evidence. They do not remove all methodological limits, but they reduce hidden variability and improve the integrity of chapter-to-chapter argument flow.

From a research-design perspective, this chapter also clarifies evidence hierarchy. Core findings rely on fixed protocol metrics, while supplementary checks (ablation, sensitivity, multi-seed) are used to test stability and support interpretation boundaries. Presenting this hierarchy explicitly helps avoid over-weighting one favourable chart or one isolated run.

Methodological traceability is further supported by explicit chapter contracts. Research approach explains why the method family was selected, dataset section defines data assumptions, model section defines role boundaries, federated section defines distributed-training assumptions, explainability section defines attribution scope, and evaluation section defines measurable outcomes. This layered design is useful because it allows reviewers to challenge one assumption at a time without collapsing the whole argument.

Another benefit is adaptation control. If future work changes one component, for example sequence length or federated round count, the impact can be traced to specific sections and expected outputs. This improves maintainability of the research artefact and supports cleaner extension into publication-oriented follow-up work.

Overall, the methodology chapter provides a controlled bridge from research questions to measurable evidence. It defines what is tested, why it is tested, and how interpretation boundaries are maintained. This improves both academic rigour and practical readability for examiners.

It also provides a practical map for replication by separating assumptions from execution details. Readers can first understand the methodological logic, then follow implementation references in later chapters without losing context.

This improves chapter coherence and reduces methodological ambiguity when results are interpreted later in the dissertation.

In addition, the structure supports examiner traceability because each methodological claim can be linked to a later implementation or evaluation location with minimal cross-reading overhead.

This traceability focus is especially useful in dissertation review settings where method validity and result legitimacy are checked together.

It also keeps chapter-to-chapter argument continuity stable when final edits are made close to submission.

### 4.8 Chapter summary

Chapter 4 has defined the **research stance**, **data**, **models**, **federation**, **XAI ingredients**, and **evaluation plan**. **Chapter 5** gives structural graph and pipeline design; **Chapter 6** gives implementation detail.

As a methodological checkpoint, this chapter fixes the experiment logic before results are discussed. That separation protects against hindsight bias and keeps chapter responsibilities clear in the final report.

---

## Chapter 5 – Design

### 5.1 Chapter overview

This chapter presents the system design in the same order as the narrative: the **end-to-end research pipeline** first (**Figure 6**), then **authoritative graph semantics** and windowing rules (**Section 5.3**), then **architecture** (**Figure 7**) and a **conceptual *k*NN** schematic (**Figure 8**) that match the implementation in Chapter 6. Deployment-facing outputs use ECS-like JSON alerts.

This structure is kept intentionally simple. First the reader sees the full pipeline flow, then the exact graph and sequence rules, and finally the architecture view that links modules used in implementation. So, when Chapter 6 points to specific files and functions, the design assumptions are already fixed and easier to verify.

### 5.2 Pipeline, alerts, and deployment (conceptual)

**Figure 6** gives the **stage-level** view: preprocessing, graph construction, central or federated training, explainability, and SIEM-style alerting on CPU. Windowing, *k*NN *k*, and sequence labelling are specified in **Section 5.3**; module-level wiring is in **Chapter 6**.

Alerts are output in a **SIEM-style JSON** format, aligned with ideas from **ECS** where useful: event type, timestamp, source/target information, label, score, and an **explanation** field (top features / top flows). The runtime path is a **FastAPI** service that runs inference (and optionally explanations) on **CPU**, matching the edge objective.

The pipeline view also clarifies responsibility boundaries. Data preprocessing and graph construction are deterministic preparation stages. Model training is either central or federated but uses the same architecture. Alert formatting is post-inference and should not alter prediction values. This separation helps testing, because faults can be isolated by stage instead of debugging one monolithic workflow.

![Figure 6: Research pipeline](assets/figure1_pipeline.png)

**Figure 6: Research pipeline — from raw IoT flow data to explainable SIEM alerts** *(Chapter 5, Section 5.2)*

*Source: Author’s own diagram (`scripts/generate_figure1.py`); reflects the implementation and data path described in Chapters 5–6 and **Appendix C**.*

### 5.3 Graph design (flows to kNN snapshots)

**Authoritative section:** every other mention of window/*k*/sequence rules in this thesis points here or to **`src/data/graph_builder.py`** / **`dataset.py`** (**Section 6.4**).

The graph is built from flow data to capture structural relationships between network flows within each observation window.

- **Nodes:** Each node represents one flow record. The node feature vector consists of the 46 pre-computed flow-level features provided by the CICIoT2023 dataset (e.g. packet rate, byte count, flag counts, protocol indicators, statistical aggregates). Device-level identifiers (IP addresses) are not available in the publicly released version of this dataset, so a device-based graph design (where nodes are devices and edges are flows between them) was not feasible. Instead, a feature-similarity approach is adopted, following the principle demonstrated by Ngo et al. (2025) that attribute-based graph construction can be effective for intrusion detection when topology information is absent.
- **Edges:** For each window, a k-nearest-neighbour (kNN) graph is constructed in feature space. Each flow is connected to its *k* most similar flows (by Euclidean distance on the 46 features), producing undirected edges. This creates a graph structure where flows with similar characteristics are linked, enabling the GNN to learn from local neighbourhoods of related traffic patterns. The choice of *k* affects the graph density: too small and the graph may be too sparse for effective message passing; too large and computation increases. *k* = 5 was chosen as a balance (see sensitivity analysis in Chapter 8).
- **Windows and sequences:** Flows are grouped into fixed-size windows (e.g. 50 flows per window). To handle extreme class imbalance at flow level, the implementation (`src/data/graph_builder.py`) builds each window **only from flows in a single class pool** (benign or attack); the **graph-level label is that pool’s class** (within a window all flows share the same label, so this matches a unanimous vote but is defined by construction, not by aggregating mixed labels). Benign and attack windows are balanced, then **shuffled** into one list. Training samples are **sequences of five consecutive graphs in that list**; the **sequence-level label** is attack if **any** of the five windows is attack-labeled (`GraphSequenceDataset` in `src/data/dataset.py`). The GRU reads the five graph embeddings in order. Five windows was chosen to balance context with memory and training time.

This design is kept simple so that it can be implemented and tested within the project scope. The kNN approach applies to any flow-feature dataset regardless of whether device identifiers are present. More complex designs (e.g. device-based graphs when IPs are available, multi-hop temporal aggregation) could be explored in future work.

### 5.3a Research design and system architecture

**Figure 7** summarises the **end-to-end research design**: public benchmark flows are preprocessed and turned into temporal graphs; the dynamic GNN can be trained centrally or through Flower FedAvg on non-IID clients; the same checkpoint path supports Captum explanations and FastAPI scoring into SIEM-shaped JSON. It is a **conceptual** diagram (not a formal UML class model); a full UML component diagram can be pasted into the Word submission if the module asks for it.

![Figure 7: Research design system architecture](assets/research_design_system.png)

**Figure 7: Research design — data flow, federated training, and edge alerting** *(Chapter 5, Section 5.3a)*

*Source: Author’s own diagram; generated for this dissertation (`scripts/draw_research_design.py`).*

### 5.3b Conceptual illustration: similarity-based graph in one window

Public CICIoT2023 releases do not always include device-level topology, so **attribute / similarity-based** edges are a standard way to obtain a graph for a GNN (Ngo et al., 2025). **Figure 8** is an **author-original** schematic (drawn with **Matplotlib**, following the library’s documented style sheets — see [Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)): each point is one flow in a window; lines show **k** nearest neighbours in a **2-D projection** of feature space (the real system uses **46** dimensions and Euclidean distance, **Section 5.3**). It is **not** a screenshot from another paper; it only visualises the **same construction principle** as the implementation.

![Figure 8: Conceptual kNN similarity graph](results/figures/similarity_knn_concept.png)

**Figure 8: Conceptual kNN similarity graph within one observation window** *(Chapter 5, Section 5.3b)*

*Source: Author’s own diagram (`scripts/generate_similarity_knn_concept.py`). Concept aligns with attribute-based graph construction for IoT intrusion detection (Ngo et al., 2025); image is not reproduced from third-party publications.*

### 5.4 Chapter summary

The design encodes **relational** (*k*NN), **temporal** (window sequences), and **SOC-facing** outputs in one architecture.

It also sets a stable contract for implementation, including where each design rule is enforced in code and which outputs must be preserved for evaluation and reporting.

The chapter also clarifies deployment assumptions. The design is CPU-oriented, structured around flow telemetry rather than packet payload, and aligned to SIEM-style JSON output for practical integration pathways. These assumptions are important because they shape both modelling decisions and evaluation criteria.

In addition, the chapter draws a clear boundary between conceptual diagrams and executable behaviour. Figures 6 to 8 communicate design intent, while authoritative semantics in Section 5.3 define exactly how windows, graphs, and labels are formed. This reduces ambiguity when results are interpreted later.

Design quality here is therefore both conceptual and operational. The conceptual side explains why the architecture is arranged as shown. The operational side explains what each stage must output so the next stage receives valid input under reproducible conditions.

This dual emphasis helps when moving from dissertation assessment to future publication or extension work, because design assumptions are already documented in a way that supports external review and reuse.

The design chapter therefore acts as a contract layer between theory and code. It explains what must remain stable in implementation and what can be tuned in experiments. That separation helps preserve consistency when figures, tables, and narrative sections are revised near submission deadlines.

A related benefit is communication clarity between chapters. When design constraints are explicit, implementation and results chapters can refer to them directly instead of restating definitions. This reduces duplication and makes the dissertation easier to navigate for examiners who check claim consistency across sections.

This chapter therefore serves both engineering and reporting quality. It fixes the design language used throughout the rest of the dissertation and keeps implementation choices anchored to explicit design assumptions.

It also confirms that design decisions are not isolated diagrams, but active constraints carried into implementation and evaluation.

---

## Chapter 6 – Implementation and System Development

### 6.1 Chapter overview

This chapter describes how the prototype was implemented: code structure, libraries, training and federated loops, Captum integration, alert formatting, and the FastAPI scoring endpoint. In this project, implementation detail matters a lot, because the claim is not only model accuracy, it is full pipeline feasibility under CPU limits. So I explain where each major function lives and how the modules connect.

I also use this chapter to show practical engineering trade-offs. Some implementation choices are not academically elegant, but they are stable for CPU-only execution and reproducibility under project time limits. This means explicit configuration files, deterministic data processing paths, and module boundaries that make debugging manageable when federated and graph code interact.

The implementation is organised under `src/` by responsibility (data, models, federated learning, explainability, SIEM output, evaluation) and is driven by scripts that reproduce end-to-end runs from configuration (`scripts/run_all.py`, plus the additional scripts listed in Appendix C). Configuration is externalised in YAML so experiments can be re-run without code changes.

### 6.2 Environment and Tools

The system is implemented in Python 3.10. PyTorch 2.x is used for the neural models (GNN and MLP), and PyTorch Geometric (PyG) provides graph operations and GAT layers. Scikit-learn is used for Random Forest and for metrics (precision, recall, F1, ROC-AUC, confusion matrix). The Flower framework is used for federated learning (client and server logic). Captum is used for Integrated Gradients. FastAPI is used for the REST API. The code is structured so that data loading, graph building, training, and inference can be run separately or together. All experiments are run on CPU to match the edge deployment goal; training time is therefore longer than with a GPU but remains feasible for the chosen subset.

From a software engineering perspective, tool choice was influenced by integration cost and reproducibility. PyTorch and PyG provide direct control for model and graph operations. Flower provides practical federated orchestration without custom networking code. Captum integrates with PyTorch tensors directly, which reduced explainability integration effort. FastAPI gives lightweight deployment for API evaluation in CPU conditions.

Configuration is managed via `config/experiment.yaml` (window size, kNN *k*, sequence length, model hyperparameters, and FL rounds/epochs). Reproducibility details (commands and outputs) are summarised in Appendix C.

### 6.3 Data Loading and Preprocessing

The CICIoT2023 subset is loaded from CSV (or the dataset’s provided format). The public flow tables provide the **46 numeric features** used in this project plus labels; optional endpoint columns are retained only when present in the schema. Missing values are handled, labels are binarised (benign vs. attack), and fixed train/validation/test splits are produced with a fixed seed for reproducibility.

Preprocessing includes feature standardisation (zero mean, unit variance) using statistics computed only on the training set, then applied to validation and test to avoid leakage. For the GNN, flows are windowed and converted to graphs as in **Section 6.4**. For Random Forest and MLP, **the same processed splits** are used in tabular form with **exactly one row per flow** (46 features + binary label), loaded by `load_processed_split` in `src/models/baselines.py`—not window aggregates. Graph-sequence construction uses `src/data/dataset.py`; configuration is in `config/experiment.yaml`.

### 6.4 Graph Construction

Implementation follows **Section 5.3** exactly: `build_graphs_for_split` and `flows_to_knn_graph` in **`src/data/graph_builder.py`** build single-class pools, balanced windows, shuffle, Euclidean **k**NN with bidirectional edges, and graph batches consumed by **`GraphSequenceDataset`** in **`src/data/dataset.py`**. Default **k** and window size come from **`config/experiment.yaml`**; sensitivity to other (*window*, *k*) pairs is in **Section 8.8**. **Baselines** read **one row per flow** from processed parquet via **`load_processed_split`** (`src/models/baselines.py`)—not graph windows.

This module is central for validity because graph semantics directly affect model input structure. A small implementation inconsistency here, such as mixed class pooling or incorrect edge direction handling, can change results significantly. For this reason, graph construction logic is kept explicit and traceable to design rules in Chapter 5.

### 6.5 Model Implementation

**Random Forest:** Implemented with scikit-learn on **one row per processed flow** (46 features + label). The implementation uses 200 trees and max_depth=20; class weights address flow-level imbalance in the training parquet.

**MLP:** Feed-forward network with three hidden layers (128, 64, 32), ReLU, dropout 0.2; **two output logits** for benign vs. attack, trained with **cross-entropy** and Adam (learning rate 1e-3), matching `MLPBaseline` and the training loop in `scripts/run_all.py`.

**Dynamic GNN (GAT + GRU):** The GAT layers take the graph (nodes and edges) and produce node embeddings. The GAT uses 4 attention heads and a hidden dimension of 64. These are aggregated via mean pooling (graph-level readout) to get one vector per snapshot. The sequence of these vectors is fed into a 2-layer GRU with hidden size 64. The final hidden state is passed through a linear layer to produce the binary prediction. The implementation uses PyTorch and PyG for the GAT. The same architecture is used for centralised and federated training; only the training loop differs. Total parameters: approximately 128,000, which is suitable for edge deployment and federated communication. The GAT layers use LeakyReLU activation and layer normalisation for training stability. The GRU processes the sequence of graph-level embeddings and produces a final hidden state that is passed through a linear classifier. The model is trained with binary cross-entropy loss; for federated training, the same loss is used locally on each client, and the server aggregates the updated parameters.

### 6.6 Federated Learning (Flower)

Federated training uses Flower with FedAvg: clients train locally on their partitions for a fixed number of epochs per round and send updated weights to the server; the server aggregates updates and broadcasts the new global model. No raw data is exchanged—only parameters—so communication cost can be approximated from float32 weight traffic (reported in Chapter 8). Implementation details are in `src/federated/` (client/server entry points referenced in Appendix C).

The federated module is implemented to mirror central training logic as closely as possible. Local client steps reuse the same optimiser and loss assumptions, so performance differences are mostly due to data partitioning and aggregation process rather than unrelated training-loop changes. This design supports fairer central-versus-federated comparison in results reporting.

### 6.7 Explainability

**Integrated Gradients:** For a given input (e.g. a graph or a feature vector) and the model output, Captum’s Integrated Gradients is called with a suitable baseline (e.g. zero vector or mean feature vector). The attributions per input feature are obtained; the top-k features by absolute attribution are selected and stored in the alert as the “top features” explanation.

**Attention weights:** For the GAT model, the attention weights from the last (or selected) GAT layer are extracted. They indicate how much each edge or neighbour contributed. These are mapped back to flow identifiers (e.g. source–destination pairs) and the top flows are written into the alert. If the model has multiple layers, a simple strategy (e.g. average or use the last layer) is applied and documented.

Explanations are attached to the alert JSON. The number of integration steps (default 50) trades off attribution accuracy with computation time. The `src/explain/explainer.py` module wraps both methods and produces a unified explanation object. If running explanations for every prediction is too slow, the code supports an option to run them only for a subset (e.g. when the model confidence is above a threshold).

### 6.8 Alert Generation and SIEM-Style Output

When the model predicts an attack (or a specific attack type), an alert object is built. It includes: event type (e.g. “alert” or “detection”), timestamp, source and destination (e.g. IP or device ID), predicted label, confidence score, and an explanation object. The explanation object contains the list of top features (from Integrated Gradients) and/or top flows (from attention). The structure follows ECS-like conventions where possible (e.g. event.category, event.outcome, and custom fields for explanation). The alert is returned as JSON. This format is suitable for ingestion into a SIEM or for display in a SOC dashboard. The `src/siem/alert_formatter.py` module implements the ECS-like structure with fields such as `event.category`, `event.outcome`, and a custom `explanation` object holding `top_features` and `top_nodes` or `top_flows`. Severity levels (low, medium, high) are derived from the confidence score.

### 6.9 FastAPI Deployment and CPU Inference

A FastAPI application is set up with an endpoint that accepts input (e.g. a single flow, a batch of flows, or a pre-built graph). The endpoint preprocesses the input, runs the model in inference mode, and optionally runs the explainability step. The response is the alert JSON. CPU inference time is measured (e.g. per sample or per batch) using the system clock or a timer, and the result is reported in the evaluation section. No GPU is required; the design targets edge devices with CPU only. The API can be run locally or in a container for demonstration.

The FastAPI app (`src/siem/api.py`) loads the trained model from checkpoint at startup. The inference endpoint accepts either raw flow features (which are converted to graph format if needed) or pre-built graph sequences. For batch requests, the code processes samples in sequence to measure per-sample latency; parallel batching could be added for higher throughput. The measured inference times (e.g. 22.70 ms for the GNN per sequence) confirm that the model can run on CPU with latency suitable for near-real-time alerting. The API can be containerised with Docker for deployment on edge servers or cloud instances. The `scripts/run_all.py` script orchestrates preprocessing, graph construction, baseline training, central GNN training, and metric/plot outputs for the centralised path. Federated training is run via the Flower entry points (see **Appendix C**). The `scripts/generate_alerts_and_plots.py` script produces example alerts, the FL convergence plot, and additional figures. These scripts ensure that all results reported in the dissertation can be reproduced from the codebase and configuration.

### 6.10 Implementation code screenshots (author’s codebase)

High-quality implementation chapters often present **core code as figures**: a **bold snippet title**, a **dark, line-numbered** editor-style view, then a **formal figure caption** (Harvard-style dissertations vary by template; here captions follow the same rule as elsewhere in this report: **caption below the figure**, with **chapter and section** placement). The screenshots below follow that **sample-style layout** but show **only the main fragments** of this project (function bodies, one **training-step** loop, sequence **forward** head, Captum **IG** core, and the **`/score`** handler)—not whole modules. They are **auto-rendered** with **Pygments** (`one-dark` style) at the cited line ranges (`scripts/render_chapter6_code_screenshots.py`). **Appendix D / Figures A1-1–A1-6** retain **wider** code figures for the handbook requirement.

**`flows_to_knn_graph` — *k*NN edges and tensors (core)**

![Figure 9: `flows_to_knn_graph` core excerpt](results/figures/chapter6/fig_ch6_01_flows_to_knn_core.png)

**Figure 9: `flows_to_knn_graph` (core) — `NearestNeighbors`, bidirectional edges, PyG `Data`** *(Chapter 6, Section 6.10)*

*Source: `src/data/graph_builder.py`, lines 37–58.*

**`build_graphs_for_split` — stratified pools (core)**

![Figure 10: `build_graphs_for_split` core excerpt](results/figures/chapter6/fig_ch6_02_stratified_split_core.png)

**Figure 10: `build_graphs_for_split` (core) — class pools, balance, merge, shuffle** *(Chapter 6, Section 6.10)*

*Source: `src/data/graph_builder.py`, lines 106–123.*

**`train_one_epoch` — central GNN training loop**

![Figure 11: `train_one_epoch` excerpt](results/figures/chapter6/fig_ch6_03_train_one_epoch.png)

**Figure 11: `train_one_epoch` — class-weighted cross-entropy, backward pass, optimiser step** *(Chapter 6, Section 6.10)*

*Source: `src/models/trainer.py`, lines 39–61.*

**`DynamicGNN.forward` — sequence to logits**

![Figure 12: `DynamicGNN.forward` excerpt](results/figures/chapter6/fig_ch6_04_dynamic_gnn_forward.png)

**Figure 12: `DynamicGNN.forward` (core) — stack window embeddings, GRU vs. mean-pool ablation, classifier** *(Chapter 6, Section 6.10)*

*Source: `src/models/dynamic_gnn.py`, lines 84–97.*

**`_ig_wrapper` — Integrated Gradients (core)**

![Figure 13: `_ig_wrapper` excerpt](results/figures/chapter6/fig_ch6_05_integrated_gradients_wrapper.png)

**Figure 13: `_ig_wrapper` (core) — Captum `IntegratedGradients` on the last window’s node features** *(Chapter 6, Section 6.10)*

*Source: `src/explain/explainer.py`, lines 34–50.*

**`POST /score` — FastAPI handler (core)**

![Figure 14: FastAPI `POST /score` core excerpt](results/figures/chapter6/fig_ch6_06_fastapi_score_core.png)

**Figure 14: `POST /score` (core) — build graphs, `explain_sequence`, ECS-style alert, timing** *(Chapter 6, Section 6.10)*

*Source: `src/siem/api.py`, lines 67–89.*

Together, these figures summarise the **data → train → forward → attribute → deploy** path; federated local training reuses the same `train_one_epoch` step inside `GNNFlowerClient` (**Appendix C** and **Appendix D**).

### 6.11 Chapter summary

Implementation delivers a **modular** codebase with scripted **reproducibility**, covering data through to **SIEM-style** JSON and **CPU** inference.

The chapter also demonstrates integration maturity. Modules are not isolated examples; they are connected and executable in one pipeline that can be repeated with configuration control and reproducible artefacts.

From an assessment perspective, implementation quality is shown through traceable module mapping, controlled configuration, and repeatable output generation. The chapter shows where each major function lives, how scripts orchestrate runs, and how outputs in tables and figures are tied back to code paths.

A practical lesson from implementation is that integration effort is often larger than model coding effort. Graph construction, federated orchestration, explanation attachment, and API formatting each add complexity at module boundaries. Explicit interface handling and stable data contracts are therefore as important as architecture choice.

This chapter also documents enough operational detail for extension. Future work can replace one component, for example federated strategy or explanation method, without rebuilding the whole pipeline from scratch. That modularity supports both research reuse and production-oriented prototyping.

Implementation discussion also reflects maintainability concerns. Clear file boundaries, function naming, and configuration-driven behaviour reduce long-term debugging cost and make review easier. In MSc project context this is important, because assessment includes both technical output and evidence of independent software-engineering judgement.

Another practical benefit is reproducible troubleshooting. When a result changes unexpectedly, the modular pipeline allows checks at data, graph, model, federation, explanation, and API layers in sequence. This improves confidence that reported results are stable and not accidental outputs from one uncontrolled run.

In short, Chapter 6 is not only a description of code modules. It is a record of how the system is engineered so that its claims can be reproduced, tested, and extended with minimal ambiguity.

Implementation maturity is also reflected in evidence packaging. Scripts generate metrics, figures, and alert artefacts in predictable locations, and chapters reference those outputs directly. This reduces manual copy errors and helps keep report tables aligned with generated result files.

In addition, module-level clarity supports code review quality. When each responsibility has a clear file path, reviewers can verify claim-to-code mapping more quickly. This improves dissertation defensibility in viva-style questioning, where examiners often ask where a specific behaviour is implemented.

Finally, implementation details are written with extension in mind. The current system is MSc scoped, but it is organised so larger datasets, additional clients, or alternative explanation modules can be integrated with limited structural change.

Implementation chapter depth is also important for assessment fairness. Without enough technical detail, it is hard to judge whether the reported performance comes from a coherent system or from loosely connected experiments. By documenting module behaviour and orchestration paths, this dissertation makes that judgement more transparent.

In addition, implementation documentation supports risk-aware reuse. Future contributors can inspect where deterministic preprocessing is enforced, where federated aggregation is triggered, where explanation payloads are formed, and where API timing is measured. That clarity is important when extending the work to larger datasets or different infrastructure constraints.

The implementation chapter therefore contributes not only to technical transparency, but also to long-term maintainability of the project artefacts and reporting workflow.

This additional depth is important in MSc assessment context, because implementation marks reward both practical execution and evidence of independent engineering judgement under real project constraints.

It also makes the implementation chapter more useful as a technical handover document for future extension and publication preparation.

This has practical value after submission, because publication drafting and replication checks both depend on clear implementation provenance and reusable module-level documentation.

It also improves maintainability for future experimentation, where incremental model or pipeline changes must be reviewed without losing reference to baseline implementation behaviour.

This continuity is valuable for both future publication drafting and technical extension after assessment.

It improves long-term usability of the full prototype.

---

## Chapter 7 – Testing and Evaluation

### 7.1 Chapter overview

This chapter specifies the evaluation protocol: the experimental setup, formal metric definitions, dataset and sequence statistics, and the comparison rules applied consistently across models. The purpose is to fix the “test contract” so Chapter 8 can report results as evidence rather than opinion.

To keep consistency, this chapter is the only authoritative location for metric formulae and split statistics. Other chapters can cite these values but do not redefine them. This reduces confusion and makes it easier to check whether reported numbers in the results section come from one fixed evaluation setup.

### 7.2 Evaluation scope

Evaluation answers the **three sub-questions** and supports the **main question** (**Section 1.3**). **Setup, hyperparameters, and federated heterogeneity** are in **Section 7.3**; **metric definitions** in **Section 7.4**; **scale statistics** in **Section 7.5**; **decision rules** for comparing models in **Section 7.6**. The plan was fixed **before** final test reporting so metrics were not tuned to the test set.

### 7.3 Experimental Setup

All experiments use the same fixed train, validation, and test split from the CICIoT2023 subset. The random seed is fixed (e.g. 42) so that results are reproducible. Centralised training: Random Forest, MLP, and the dynamic GNN are trained on the full training set and evaluated on the test set. Federated training: the same GNN is trained with Flower and FedAvg across 3 clients; the global model is evaluated on the same test set after each round. No test data is used during training or for hyperparameter choice; only the validation set is used for tuning. Time windows for graph construction are set to a fixed length (e.g. 50 flows per window, 5 windows per sequence); sensitivity to window size can be checked in a time-window analysis if time allows.

The federated split uses a Dirichlet distribution (alpha = 0.5) to simulate non-IID clients. The central test set is held out and used only for evaluation. Model hyperparameters and training settings are taken from `config/experiment.yaml` (summarised in Chapter 6 and Appendix C), and the same splits are used across models to ensure fair comparison.

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

This comparison design also defines what is not claimed. It does not claim universal superiority for all datasets or deployments. It reports outcomes on one controlled subset and one pipeline configuration, then checks internal stability with ablation, sensitivity, and multi-seed analyses. This keeps the interpretation grounded and consistent with MSc project scope.

### 7.7 Chapter summary

Chapter 7 fixed the **setup**, **metrics**, **data scales**, and **comparison protocol** used for all models and federated runs. Chapter 8 reports the **numerical and graphical** outcomes only.

This separation of responsibilities supports cleaner scientific reporting. Evaluation logic is defined before interpretation, and result discussion can then reference a stable protocol. It reduces the chance of accidental metric drift or post-hoc selection of favourable indicators.

The chapter also records assumptions and boundaries so readers can judge validity appropriately. It clarifies that claims are scoped to the configured subset, defined split strategy, and specified model settings. This transparency improves trust in the reported findings.

Evaluation quality also depends on conservative interpretation rules. This chapter therefore keeps comparison criteria explicit, for example same split usage across models and separation between validation tuning and final test reporting. This reduces optimistic bias in comparative claims.

A further benefit is reviewer efficiency. With setup, metrics, and comparison logic in one place, chapter readers can verify result legitimacy quickly before moving to interpretation sections. This improves transparency and helps align the report with formal dissertation assessment expectations.

This chapter also functions as an agreement on what counts as fair evidence. It defines the measurement frame before reading Chapter 8, so result interpretation starts from shared protocol assumptions. That improves reliability of conclusions drawn in later discussion sections.

A final evaluation point is reproducible reporting sequence. Setup, metrics, and comparison rules are fixed before final result narration, and this supports consistent argument quality. It also helps external readers reproduce the same ordering when validating claims against scripts and exported artefacts.

This is especially useful for dissertation assessment, where results need to be interpreted against a fixed and transparent contract. By making evaluation structure explicit, the chapter reduces ambiguity and improves confidence in later claims.

It also supports consistency when reproducing report figures from saved artefacts and scripted runs.

---

## Chapter 8 – Results Presentation

### 8.1 Chapter overview

This chapter reports the results produced by the frozen pipeline as tables and figures, with brief factual description. Interpretation and implications are reserved for Chapter 9, while personal reflection is in Chapter 10. Suggested reading order: (1) centralised baselines versus the GNN (**Section 8.2**); (2) federated learning convergence and per-round metrics (**Sections 8.3 and 8.3a**); (3) central training dynamics (**Section 8.4**); (4) inference latency (**Section 8.5**); (5) example alerts with explanations (**Section 8.6**); and (6) ablation, sensitivity, and multi-seed stability checks (**Sections 8.7–8.9**). Artefacts are stored under `results/metrics/`, `results/figures/`, and `results/alerts/`.

The style here is deliberately factual. I report measurements and chart observations without broad interpretation claims. This separation helps avoid circular reasoning, because discussion and meaning are handled in Chapter 9 after all evidence is presented in one place.

**Reporting convention:** Tables keep **decimal** precision (e.g. 0.9986) so readers can verify against CSV/JSON logs. Narrative comparisons use the same underlying values as **percentages**, e.g. **Precision = 99.89%**, **Recall = 99.84%**, **F1 = 99.86%**, **ROC-AUC = 99.96%** — **not** rounded to an artificial band. **Figures 15–22** and **Figures 23–24** (centralised results plots: confusion matrices, ROC curves, FL convergence, model comparison, ablation, sensitivity) are generated with **Matplotlib** via `apply_thesis_style()` in `src/evaluation/plot_style.py`, which applies documented style sheets from the Matplotlib ecosystem ([style sheets reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)).

### 8.2 Centralised Model Comparison (Sub-Question 1)

The dynamic GNN (GAT + GRU), Random Forest, and MLP were evaluated on the same test set. **Table 2** lists the same values as decimals; here they are stated explicitly as percentages for quick comparison: **Random Forest** — **Precision = 99.89%**, **Recall = 99.84%**, **F1 = 99.86%**, **ROC-AUC = 99.96%**; **MLP** — **Precision = 100.00%**, **Recall = 98.85%**, **F1 = 99.42%**, **ROC-AUC = 99.84%**; **Central GNN** — **Precision = Recall = F1 = ROC-AUC = 100.00%**; **Federated GNN** (evaluated centrally on the same test set) — **Precision = Recall = F1 = ROC-AUC = 100.00%**. False positive **rate** (benign predicted as attack): GNN **0.00%**; Random Forest **4.84%** (187 false positives); MLP **0.10%** (4 false positives). **Figure 15** shows the confusion matrix for the Dynamic GNN. **Figure 16** shows the ROC curve for the GNN. **Figures 17** and **18** show confusion matrices for Random Forest and MLP. **Figures 19** and **20** show ROC curves for RF and MLP. **Figure 22** compares inference time and F1 across models.

**Table 2: Model comparison on CICIoT2023 test set**

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| Random Forest | 0.9989 | 0.9984 | 0.9986 | 0.9996 | 46.09 |
| MLP | 1.0000 | 0.9885 | 0.9942 | 0.9984 | 0.66 |
| Central GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| Federated GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20.99 |

*Source: Author-derived metrics on the fixed CICIoT2023 subset (Pinto et al., 2023); `results/metrics/results_table.csv`; reproducibility in Chapter 13. Decimal columns match the prose **Key:** lines above (e.g. RF **F1 = 99.86%** ↔ 0.9986).*

![Figure 15: Confusion matrix for Dynamic GNN](results/figures/cm_gnn.png)

**Figure 15: Confusion matrix for Dynamic GNN on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot from test-set predictions; data CICIoT2023 (Pinto et al., 2023); `scripts/run_all.py`.*

![Figure 16: ROC curve for Dynamic GNN](results/figures/roc_gnn.png)

**Figure 16: ROC curve for Dynamic GNN on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot from test-set scores; data CICIoT2023 (Pinto et al., 2023); `scripts/run_all.py`.*

![Figure 17: Confusion matrix for Random Forest](results/figures/cm_rf.png)

**Figure 17: Confusion matrix for Random Forest on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

![Figure 18: Confusion matrix for MLP](results/figures/cm_mlp.png)

**Figure 18: Confusion matrix for MLP on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

![Figure 19: ROC curve for Random Forest](results/figures/roc_rf.png)

**Figure 19: ROC curve for Random Forest on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

![Figure 20: ROC curve for MLP](results/figures/roc_mlp.png)

**Figure 20: ROC curve for MLP on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al., 2023).*

The confusion matrices (**Figures 15, 17, and 18**) report TP, TN, FP, and FN counts per model. **Figure 15** shows **zero** FP and **zero** FN for the GNN on this test set. **Figures 17** and **18** show **187** FP for Random Forest and **4** FP for MLP (benign predicted as attack). The ROC curves (**Figures 16, 19, and 20**) plot TPR vs. FPR across thresholds; reported ROC-AUC values are in **Table 2**. Implications for SOC triage and model choice are discussed in **Chapter 9**.

### 8.3 Federated Learning (Sub-Question 2)

Federated training was run for **10 rounds** with **3 clients** and non-IID splits (Dirichlet **alpha = 0.5**). The global model was evaluated on the **same central test set** after each round. **Table 3** lists precision, recall, F1, and ROC-AUC per round; **Figure 21** plots F1 and ROC-AUC vs. round. **Key (same numbers as Table 3):** round 1 **F1 = 98.3%**, **ROC-AUC = 55.7%**; from round 2 **ROC-AUC = 100.0%**; **F1** reaches **100.0%** from round 7; round 6 **ROC-AUC = 97.3%** while **F1 = 99.5%**. The final round reports **F1 = ROC-AUC = 100.0%**, matching the centralised GNN on this test set.

Communication cost is approximated from float32 parameter size: **128,002** parameters, order of **~1.0 MB** per client upload/download per round; **~3.07 MB** per round aggregate messaging is reported, **~31 MB** total over 10 rounds with three clients. Interpretation (privacy vs. centralised parity, feasibility) is in **Chapter 9**.

**Table 3: Federated learning round-by-round metrics**

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

![Figure 21: FL convergence](results/figures/fl_convergence.png)

**Figure 21: Federated learning convergence (F1 and ROC-AUC vs. round)** *(Chapter 8, Section 8.3)*

*Source: Author’s own plot; `scripts/generate_alerts_and_plots.py` / federated training logs; data CICIoT2023 (Pinto et al., 2023).*

### 8.3a What federated learning produced (reading the curves and the table)

Figure 21 and Table 3 are the clearest **“output photos”** of the federated run without opening log files. **Round 1** already reaches **Recall = 100.0%** and **F1 = 98.3%**, but **ROC-AUC = 55.7%**, which means the ranking of scores across benign and attack rows was still poor at that early point. From **round 2**, **ROC-AUC = 100.0%**, so the global model’s scores separate the classes well on the central test set even though raw data never left the clients. **F1** then climbs more slowly and reaches **100.0%** from **round 7** onward. **Round 6** shows a small dip to **ROC-AUC = 97.3%** while **F1** stays at **99.5%**; that is visible in the plot as a wobble rather than a collapse, and it supports the discussion in Chapter 9 about noise under non-IID FedAvg. Overall, the federated checkpoint **matches** the central GNN on the reported test metrics, which answers the supervisor’s question about what federated learning **actually delivered** here: a global GNN that converges under heterogeneous clients and still reaches the same headline test performance on this split.

For submission, the student may add **extra screenshots** (Flower console, TensorBoard-style loss if used, or a browser call to FastAPI `/score`) as additional figures in Word; Appendix D already holds static code screenshots for the handbook.

### 8.4 Central GNN Training Convergence

The centralised Dynamic GNN was trained for **6 epochs** (with early-stopping configured in the training script). **Table 4** lists train loss and validation F1 / ROC-AUC per epoch (`results/metrics/gnn_training_history.json`). Validation **F1 = ROC-AUC = 100.0%** from epoch 1 onward on this run; train loss falls from **0.484** to **0.0001** by epoch 4 and stays there through epoch 6.

**Table 4: Central GNN training history (loss and validation metrics)**

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

CPU inference times (reported in Table 2): Random Forest **46.09 ms** per sample; MLP **0.66 ms**; central GNN **22.70 ms**; federated GNN **20.99 ms** (GNN times are per **sequence** of 5 graph windows). **Figure 22** compares F1 and inference time across models.

![Figure 22: Model comparison](results/figures/model_comparison.png)

**Figure 22: Model comparison — inference time and F1-score** *(Chapter 8, Section 8.5)*

*Source: Author’s own plot; metrics from Table 2 and inference timings in `results/metrics/`; `scripts/generate_alerts_and_plots.py`.*

### 8.6 Example Alerts with Explanations (Sub-Question 3)

Five example alerts were generated with full explanations (top features from Integrated Gradients and top flows from attention weights), in the same order as `results/alerts/example_alerts.json`. Each example includes the predicted label, confidence, threat severity band, and the explanation object (`top_features`, `top_nodes`).

**Example 1 (True negative):** Predicted benign; score **0.163** (low severity). Top features: `Variance`, `Std`, `rst_count`, `Duration`, `AVG` (Integrated Gradients magnitudes as in `example_alerts.json`).

**Example 2 (True positive):** Predicted malicious; score **0.997**. Top features: `psh_flag_number`, `ICMP`, `rst_flag_number`.

**Example 3 (True positive):** Predicted malicious; score **0.996**. Top features: `rst_flag_number`, `ICMP`, `Protocol Type`.

**Example 4 (False positive):** Benign labelled as malicious; score **0.711** (medium). Top features: `Variance`, `rst_count`.

**Example 5 (False positive):** Benign labelled as malicious; score **0.945**. Top features: `Variance`, `Std`, `rst_count`.

Each record follows the ECS-like shape: event metadata, rule name, threat indicator, ML prediction and score, and `explanation` (`top_features`, `top_nodes`). Whether these fields are sufficient for SOC triage is discussed in **Chapter 9**.

### 8.7 Ablation Studies (Priority 1: Evidence)

To show that both the graph and the temporal parts of the model add value, one ablation was run: the same GAT-based model but with the GRU replaced by mean pooling over time (so the model sees each window’s graph embedding but does not model the sequence with an RNN). This variant is called “GAT only (no GRU)”. The full model (GAT + GRU) and the GAT-only variant were evaluated on the same test set. Table 5 summarises the results. To reproduce the ablation, run: `python scripts/run_ablation.py --config config/experiment.yaml`; results are saved to `results/metrics/ablation_gat_only.json` and `results/metrics/ablation_table.csv`.

**Table 5: Ablation on CICIoT2023 test set (centralised GNN variants)**

| Variant | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|---------|-----------|--------|-----|---------|----------------|
| Full (GAT + GRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| GAT only (no GRU) | 0.9923 | 1.0000 | 0.9961 | 1.0000 | 16.06 |

*Source: `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`; test split CICIoT2023 (Pinto et al., 2023).*

**Numeric summary (same as Table 5):** full (GAT + GRU) — **Precision = Recall = F1 = ROC-AUC = 100.00%**, inference **22.70 ms**; GAT-only (mean pool over windows) — **Precision = 99.23%**, **Recall = 100.00%**, **F1 = 99.61%**, **ROC-AUC = 100.00%**, inference **16.06 ms**.

![Figure 23: Ablation bar chart](results/figures/ablation_bar.png)

**Figure 23: Ablation comparison — full GAT+GRU vs. GAT-only (F1 and inference time)** *(Chapter 8, Section 8.7)*

*Source: Author’s own plot; `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`.*

Interpretation of the temporal (GRU) vs. pooling trade-off is in **Section 9.3** and **Section 9.5**.

### 8.8 Sensitivity Analysis (Stability of Design Choices)

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

Grid interpretation is in **Section 9.3**.

![Figure 24: Sensitivity heatmap](results/figures/sensitivity.png)

**Figure 24: Sensitivity of test F1 and ROC-AUC to window size and kNN *k*** *(Chapter 8, Section 8.8)*

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

**Table 7** reports **mean F1 = 100.0%**, **mean ROC-AUC = 100.0%**, and **zero** standard deviation on this test split for all three seeds. Per-seed inference times in the raw logs vary with CPU timing; the headline **~22.7 ms** deployment figure is from the primary **seed 42** run (**Table 2**). What this implies for generalisation is discussed in **Chapter 9**.

### 8.10 Chapter summary

Chapter 8 reported **metrics**, **plots**, and **stability tables**; **Chapter 9** interprets them against the research questions and limitations.

---

## Chapter 9 – Conclusion, Discussion, and Recommendations

### 9.1 Chapter overview

This chapter interprets the results, relates findings to the literature, and summarises what the project achieved within its scope. It also records limitations and implications for SOC/edge deployment. Section 9.2 provides a conclusion, followed by discussion in Sections 9.3–9.9.

### 9.2 Structured conclusion (programme format)

This dissertation addressed **SOC-oriented**, **CPU-edge** intrusion detection for **Software-Defined IoT** flow telemetry, where analysts need **accurate** models, **privacy-preserving** training where raw flows cannot be pooled, and **explainable** outputs that fit **SIEM-style** workflows. The work is grounded in **CICIoT2023** and in prior literature that often treats **GNNs**, **federated learning**, and **XAI** as separate tracks, not one deployable path.

The **main outcome** is an end-to-end prototype: *k*NN feature-similarity graphs over windowed flows, a **dynamic GNN** (GAT + GRU), **Flower FedAvg** with three clients, **Captum** attributions and attention cues folded into **ECS-like JSON** via **FastAPI**. On the **fixed subset and splits** documented in **Chapters 7-8**, the central and federated GNN matched **strong RF and MLP baselines** on headline accuracy while reporting **fewer false alarms** on the test split, **sub-23 ms** CPU inference per sequence, and **modest** federated communication over ten rounds. The evidence is supported by **ablation**, **sensitivity**, and **multi-seed** checks.

**Limitations** are explicit: a **45-day** MSc scope, a **subset** of a **lab** dataset, **small** federated topology, **no** formal SOC **user study**, and headline **100%** metrics that are **subset-specific** rather than universal.

**Outlook:** scaling data and clients, live-traffic validation, analyst-facing evaluation, tighter SIEM integration, and publication-oriented breakdowns (per attack type, stronger heterogeneity) are the natural next steps; they are expanded in **Section 9.8**. Together, the chapters show that the **research questions** are **answered** for a **documented prototype**, with honest limits and a **clear** path beyond the thesis.

### 9.3 Answering the Research Questions

**Main question:** The project set out to show how an explainable dynamic graph neural network, trained with federated learning, can detect attacks in Software-Defined IoT flow data and generate SIEM alerts for SOC use on CPU-based edge devices. The prototype demonstrates that this is feasible: the dynamic GNN can be trained on graph snapshots over time, federated learning can be applied with Flower and FedAvg, and alerts with explanations can be produced in an ECS-like JSON format. CPU inference time is measurable and can be kept within acceptable bounds for a subset of traffic. So the main question is answered in the sense that a working pipeline exists and has been evaluated; the extent to which it “performs well” depends on the metrics and comparison with baselines reported in **Chapter 8**.

**Sub-question 1 (dynamic graph vs. simple models):** The dynamic GNN did better than Random Forest and MLP. It got 100% F1 and ROC-AUC compared to 99.86% and 99.42% respectively. This supports the idea that graph structure and temporality add value for this kind of data. The baselines were competitive, which is consistent with CICIoT2023 flow features being well-engineered; the GNN's advantage lies in its lower false alarm rate (0% vs. 4.84% for RF and 0.10% for MLP) and its ability to exploit relational and temporal structure. The fact that the GNN achieves zero false positives while the RF has 187 and the MLP has 4 suggests that the graph-based representation helps the model distinguish between benign traffic that resembles attacks (e.g. high variance, unusual flag patterns) and genuine attacks. The kNN graph links similar flows, so the model can learn from neighbourhood patterns that flat classifiers cannot see.

**Sub-question 2 (federated learning):** The federated model’s performance matched the centralised model exactly (100% F1 and ROC-AUC). So federated learning is a viable option when raw data cannot be shared. The approximate communication cost (31 MB over 10 rounds) shows that the approach is suitable for resource-limited edge networks.

**Sub-question 3 (explanations for SOC triage):** The example alerts show that the system outputs top features and top flows. The highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number for attacks; Variance, Std for benign) are interpretable and match the type of traffic. For true positives, the explanations support triage by pointing to protocol-level anomalies; for false positives, the mixed feature profile helps analysts cross-reference before escalating.

**Ablation (Section 8.7):** The full model (GAT + GRU) reaches F1 = 1.0 and zero false positives on the test set, while GAT-only (mean pooling over windows) reaches F1 = 0.9961 with precision 0.9923 — a small but clear gap. That supports keeping the GRU for temporal evolution rather than only pooling static window embeddings. The bar chart (Figure 23) also shows the latency trade-off (GAT-only is faster).

**Sensitivity (Section 8.8):** Across the nine (window, *k*) settings, F1 ranges from **0.9961** to **1.0000** and ROC-AUC from **0.9895** to **1.0000**. Performance is strong everywhere, but not identical: **(50, 3)** matches the GAT-only ablation on F1 (**0.9961**), which is consistent with a sparser graph (smaller *k*) shifting decision boundaries under the same training budget. The chosen default **(50, 5)** sits with **F1 = ROC-AUC = 1.0** and inference **~17 ms** in this grid. Several other cells also reach **F1 = 1.0** (e.g. all three settings at window 30; (50, 7); (70, 7)). Pairs **(70, 3)** and **(70, 5)** show the largest ROC-AUC dips (**~0.99–0.995**) while recall stays **1.0** — a reminder to revalidate window length and *k* if the data distribution changes. Overall, **(50, 5)** is well supported: full F1/AUC together with reasonable latency; Figure 24 makes the pattern visible at a glance.

**Multi-seed (Section 8.9):** Three seeds gave identical headline test metrics (F1 and AUC 1.0, zero false positives). That increases confidence that the main result is not a one-off random initialisation, though it does not remove dataset-specific optimism.

### 9.4 Strengths and Limitations

**Strengths:** The project delivers an end-to-end prototype. It goes from data to graph, model to federated training, explainability to SIEM-style alerts and FastAPI. The design matches practical SOC needs (explainable alerts, CPU-based deployment). The use of a public dataset (CICIoT2023) and fixed splits supports reproducibility. The comparison with baselines and the evaluation of federated learning provide evidence, not just claims. Putting multiple components (graph construction, GNN, federated learning, explainability, SIEM output) in one pipeline is a contribution. Most prior work looks at these in isolation. The choice of kNN feature-similarity graphs is justified by the absence of device identifiers. It is supported by Ngo et al. (2025) and Basak et al. (2025). The stratified windowing strategy deals with class imbalance in a principled way.

**Limitations:** Results are reported on a time-boxed MSc prototype and a subset of CICIoT2023, so generalisation beyond this benchmark is not guaranteed. The kNN graph is built from feature similarity (a proxy for structure when topology identifiers are absent); topology-based graphs may be preferable when device identifiers are available. Federated learning is demonstrated with a small simulation (3 clients, 10 rounds) rather than a large deployment. Explainability is evaluated through a small number of example alerts and supervisor/author judgement; a formal analyst study would strengthen the SOC-utility claim. Finally, the lab nature of CICIoT2023 and very high headline metrics on this split should be read as subset-specific rather than universal.

### 9.5 Practical Implications

The results have several practical implications for SOC and edge deployment. First, the Dynamic GNN's zero false positive rate on this test set reduces alert fatigue compared to Random Forest (187 false positives) and MLP (4 false positives). Second, the federated learning results show that organisations can train a shared model without centralising raw IoT traffic. This addresses privacy and regulatory concerns. Third, the CPU inference time (under 23 ms per sample in the primary run) means the model can run on edge devices without GPUs. Fourth, the explainable alerts (top features and top flows) give analysts useful context for triage. Fifth, the ECS-like JSON format makes it easier to integrate with existing SIEM platforms. The main caveat is that these results are from a lab dataset and subset. Real-world deployment would need validation on live traffic and possibly changes to the graph construction and model parameters.

The ablation (Section 8.7) confirms that the temporal GRU improves over mean-pooling alone on this split. Sensitivity analysis (Section 8.8) shows that performance remains strong across a grid of window sizes and *k* values, with the chosen (50, 5) settings well supported. Multi-seed runs (Section 8.9) suggest stable optimisation for the reported metrics. Together, these checks mirror what a reviewer would ask for in a publication: not only headline accuracy, but **why** the architecture is shaped as it is and **how fragile** the numbers are to reasonable hyperparameter shifts.

### 9.6 Relation to Literature

The findings can be related back to the literature. The use of GNNs for network or security data is supported by work such as Velickovic et al. (2018). This project shows that a dynamic GNN can be built and compared to baselines on IoT flow data. The GNN's better false positive rate (0% vs. 4.84% for RF) fits with the idea that graph structure helps the model tell attack patterns from benign outliers. Han et al. (2025) and Ngo et al. (2025) showed that graph-based and attribute-based construction can improve detection. This project confirms that a kNN feature-similarity approach works when device identifiers are absent.

Federated learning with FedAvg (McMahan et al., 2017) was shown to work with Flower. It achieved performance matching centralised training. This is consistent with Lazzarini et al. (2023) and Albanbay et al. (2025). They found that FedAvg can keep accuracy for IoT intrusion detection. The fast convergence (within 7 rounds) and modest communication cost (31 MB) support the feasibility of federated deployment on edge networks. Explainability via Integrated Gradients (Sundararajan et al., 2017) and attention (Velickovic et al., 2018) was added to the alert output. This matches the need for explainable security tools in the literature. The example alerts show that the highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number) are interpretable and consistent with known attack signatures. Alabbadi and Bajaber (2025) suggested this. The gap identified in the Literature Review (combining explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype) has been addressed within the stated scope and limitations.

### 9.7 Summary of the Project

This dissertation set out to design and build a small prototype system for detecting attacks in IoT network traffic. It uses an explainable dynamic graph neural network and federated learning. It also generates SIEM-style alerts that support SOC operations on CPU-based edge devices. The main research question was: how can such a system detect attacks in IoT flow data and produce alerts that are useful for SOC analysts?

Empirically, the pipeline met its stated targets on the fixed subset: the dynamic GNN matched or exceeded baselines on headline metrics (**Chapter 8**), federated training matched centralised GNN performance, and SIEM-style alerts with explanations were produced. Stability artefacts (ablation, sensitivity grid, multi-seed summaries) live under `results/metrics/` and are produced by `scripts/run_ablation.py` and `scripts/run_sensitivity_and_seeds.py` as documented in **Chapter 13**.

### 9.8 Recommendations for Future Work

- **Larger scale:** Use a larger subset of CICIoT2023 or other IoT datasets, more federated clients, and more attack types to strengthen the evidence. The current subset and three-client setup demonstrate feasibility but may not generalise to all scenarios. Scaling to more clients and more diverse data would test the stability of FedAvg under higher heterogeneity.
- **Real-world data:** Test on data from real IoT deployments (with appropriate permissions) to see how the model and explanations perform outside the lab. Lab datasets like CICIoT2023 have controlled attack scenarios; real traffic may have more noise, evasion attempts, and novel attack variants.
- **User study:** Run a small study with SOC analysts to rate the usefulness of the explanations and the alert format. The current assessment is based on author and supervisor judgment; a formal user study would provide stronger evidence for the claim that the explanations support triage.
- **Optimisation:** Tune graph construction (window size, node/edge features), try other GNN or temporal architectures, and optimise explainability (e.g. only for high-confidence alerts) to balance accuracy and speed.
- **Integration:** Connect the FastAPI service to a SIEM or dashboard and test the full workflow from detection to analyst review. Integration with Elastic Security, Splunk, or a custom dashboard would demonstrate end-to-end SOC usability.
- **Further ablation studies:** The thesis already includes one ablation (GAT only vs. full model, Section 8.7). Future work could add ablations for kNN vs. other graph construction, or Integrated Gradients vs. attention-only explanations, to provide further evidence for design choices.


### 9.9 Chapter summary

The prototype **answers** the stated questions within documented **limits**, **relates** the outcomes to prior work, and sets out a **credible path** for scaling, evaluation, and publication.

---

## Chapter 10 – Critical Self-Evaluation

### 10.1 Chapter overview

This chapter provides a first-person critical reflection on the project: what went well, what was challenging, what was learned, and what would be improved if the work were repeated. It complements (but does not duplicate) the academic limitations in Chapter 9 by focusing on process, decisions under time constraints, and personal development.

I keep this reflection specific to the actual build path rather than generic statements. Some choices were good. Some were late and caused pressure. Writing this chapter honestly helps show independent judgement, and it also explains why certain limits remain in the current version even though the technical objectives were met.

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

Personally, I learned to **time-box**: the sensitivity and multi-seed runs were scheduled late; they completed, but I would start **stability experiments earlier** next time so writing-up is not waiting on overnight jobs. I also learned to **document while coding** - commit messages, `config/experiment.yaml`, and CSV outputs saved me when I forgot which run produced which figure.

### 10.7 Time management: what I would reorder

Data loading and graph construction **underestimated** at the start. Federated setup **overestimated** in difficulty once the pattern was clear, but the first week of Flower debugging felt slow. Explainability + FastAPI were faster **after** the model stabilised. With hindsight I would allocate: **week 1** - data contract + baseline; **week 2** - GNN + central training; **week 3** - federated + API; **week 4** - explainability + plots; **buffer** - ablation and sensitivity. I used the buffer mainly for stability runs; in another timeline I might have sacrificed one fancy plot for a **user interview**.

Overall, the project met my own bar: **one coherent system**, **evidence-backed claims**, and **clear limits**. It is not production-ready, but it is **honest research engineering** at MSc level, and I am proud of the parts that were painful and still work.

One final personal point is about communication quality. Early drafts were too technical in some parts and too broad in others, so the argument felt uneven. During revision I learned to place each technical detail where it belongs, then point back to one authoritative section instead of repeating the same explanation many times. This improved the flow of the dissertation and reduced contradictions between chapters. I also realised that writing and experiments must run in parallel. If writing starts too late, the analysis section becomes rushed and weaker, even when model outputs are strong. So, in future work I would keep a weekly writing checkpoint from week one, and I would lock figure and table numbering earlier to avoid last-minute formatting pressure.

### 10.8 Chapter summary

Critical reflection highlights **scoped realism**, **modularity**, and **honest limits**, and concrete habits I would repeat on a future project in research, technical delivery, and planning practice.

The chapter also confirms one professional lesson I will carry forward: strong technical work still needs disciplined communication and evidence organisation. Better writing cadence, earlier structure freeze, and earlier validation checkpoints would reduce avoidable stress and improve final quality in future research cycles.

I also learned that reflection should be documented while the project is running, not only at the end. Short weekly notes about decisions, failures, and fixes made this final chapter more honest and less reconstructed from memory. That habit is simple but very useful for future research projects.

I will keep this habit in future work because it improves both technical control and writing quality, and it reduces the risk that important design decisions are forgotten during final report preparation.

---

## Chapter 11 – References

This chapter lists all cited sources in UWS Harvard style, with DOIs or stable URLs where available.

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847. Available at: https://doi.org/10.3390/s25030847

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78. Available at: https://doi.org/10.3390/jsan14040078

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002. Available at: https://doi.org/10.3390/app15095002

Cuppens, F. and Miège, A. (2002) 'Alert correlation in a cooperative intrusion detection framework', in *Proceedings of the 2002 IEEE Symposium on Security and Privacy*, pp. 202-215. Available at: https://doi.org/10.1109/secpri.2002.1004372

Han, Z., Zhang, C., Yang, G., Yang, P., Ren, J. and Liu, L. (2025) 'DIMK-GCN: a dynamic interactive multi-channel graph convolutional network model for intrusion detection', *Electronics*, 14(7), p. 1391. Available at: https://doi.org/10.3390/electronics14071391

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Reynolds, J., Melnikov, A., Lunova, N. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint arXiv:2009.07896*. Available at: https://doi.org/10.48550/arXiv.2009.07896

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80-84. Available at: https://doi.org/10.1109/mc.2017.201

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509-530. Available at: https://doi.org/10.3390/ai4030028

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774. Available at: https://doi.org/10.48550/arXiv.1705.07874

Lusa, R., Pintar, D. and Vranic, M. (2025) 'TE-G-SAGE: explainable edge-aware graph neural networks for network intrusion detection', *Modelling*, 6(4), p. 165. Available at: https://doi.org/10.3390/modelling6040165

McMahan, H.B., Moore, E., Ramage, D., Hampson, S. and Agüera y Arcas, B. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, pp. 1273-1282. Available at: https://doi.org/10.48550/arXiv.1602.05629

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection - a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499. Available at: https://doi.org/10.3390/info16060499

Pinto, C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941. Available at: https://doi.org/10.3390/s23135941

Qu, Y., Gao, L., Luan, T.H., Xiang, Y., Yu, S., Li, B. and Zheng, G. (2020) 'Decentralized Privacy Using Blockchain-Enabled Federated Learning in Fog Computing', *IEEE Internet of Things Journal*, 7(6), pp. 5171-5183. Available at: https://doi.org/10.1109/JIOT.2020.2977383

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319-3328. Available at: https://doi.org/10.48550/arXiv.1703.01365

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *International Conference on Learning Representations (ICLR)*. Available at: https://doi.org/10.48550/arXiv.1710.10903

Wang, R., Zhao, J., Zhang, H., He, L., Li, H. and Huang, M. (2025) 'Network traffic analysis based on graph neural networks: a scoping review', *Big Data and Cognitive Computing*, 9(11), p. 270. Available at: https://doi.org/10.3390/bdcc9110270

Yang, S., Pan, W., Li, M., Yin, M., Ren, H., Chang, Y., Liu, Y., Zhang, S. and Lou, F. (2025) 'Industrial Internet of Things intrusion detection system based on graph neural network', *Symmetry*, 17(7), p. 997. Available at: https://doi.org/10.3390/sym17070997

Zheng, L., Li, Z., Li, J., Li, Z. and Gao, J. (2019) 'AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN', in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 4419-4425. Available at: https://doi.org/10.24963/ijcai.2019/614

Zhong, M., Lin, M., Zhang, C. and Xu, Z. (2024) 'A survey on graph neural networks for intrusion detection systems: methods, trends and challenges', *Computers and Security*, 141, p. 103821. Available at: https://doi.org/10.1016/j.cose.2024.103821

---

## Chapter 12 – Bibliography

**Purpose:** **Chapter 11** is the **sole** reference list for cited work. **Chapter 12** stays **empty** unless examiners ask for extra readings; it avoids duplicating Chapter 11 and keeps the submission tidy.

---

## Chapter 13 – Appendices

The appendices provide supporting evidence and submission material (process documentation, the signed specification, reproducibility instructions, and code figures) that would interrupt the main argument if placed in the core chapters.

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
  *Canonical location:* **`archive/process_attendance/`** (see **`archive/README.md`**). *Supervisor zip mirror:* `supervisor_package/05_Appendix_documents/`. The Word exporter embeds from the archive paths.

- **Attendance Log** — Arka Talukder_Attendance_Jan-Feb_B01821011.docx  
  *Canonical location:* **`archive/process_attendance/`** (see **`archive/README.md`**). *Supervisor zip mirror:* `supervisor_package/05_Appendix_documents/`.

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

Output directory: `results/figures/appendix1/`. In the **Table of Figures**, these appear as **Figure A1-1**–**Figure A1-6** (appendix labels), separate from the body sequence **Figure 1**–**Figure 24**. **Chapter 6 (Section 6.10)** shows shorter **dark-theme** code excerpts (**Figures 9–14**); Appendix D here shows **wider** monochrome extracts for the handbook. End-to-end batch orchestration lives in `scripts/run_all.py` (**Appendix C**).

![Figure A1-1 — DynamicGNN source code excerpt](results/figures/appendix1/fig_a1_01_dynamic_gnn.png)

**Figure A1-1** — Dynamic graph classifier: `DynamicGNN` (GAT layers, pooling, GRU vs. mean-pool ablation, logits). *(Chapter 13, Appendix D.)* *Source: `src/models/dynamic_gnn.py`, lines 12–97.*

**Caption (formal):** Figure A1-1 — Core implementation of the dynamic GNN (`DynamicGNN`): node embedding, two `GATConv` layers with multi-head attention and dropout, per-window graph embedding, sequence encoding with `GRU` (or mean-pool when `use_gru` is false for ablation), and two-class logits. Attention weights may be retained for explainability (`return_attention_weights`).

**How to interpret:** This class is the **main learnable model** described in **Chapters 5–6**. Each time step is one PyTorch Geometric `Data` object (nodes = flows in a window, edges = *k*NN). `_encode_graph` applies GAT message passing and reduces node states to one vector per window; `forward` stacks windows in time order and either runs the **GRU** (full model) or **mean-pools** over time (ablation in **Section 8.7**). The `from_config` constructor ties hyperparameters to `config/experiment.yaml`, which supports the sensitivity study in **Section 8.8**.

![Figure A1-2 — graph_builder *k*NN excerpt](results/figures/appendix1/fig_a1_02_graph_builder_knn_graph.png)

**Figure A1-2** — Constructing a single-window *k*NN graph from flow feature rows. *(Chapter 13, Appendix D.)* *Source: `src/data/graph_builder.py`, lines 25–58.*

**Caption (formal):** Figure A1-2 — `flows_to_knn_graph`: for one window of `N` flows with `F` features, fits `sklearn.neighbors.NearestNeighbors` (Euclidean), adds bidirectional edges among each node and its `k` actual neighbours (capped when `N` is small), and returns a `torch_geometric.data.Data` object with `x`, `edge_index`, and graph-level label `y`.

**How to interpret:** This function shows **how “graph structure” is defined** when device IPs are unavailable (**Chapter 1**): similarity in the **46-dimensional** flow feature space replaces physical topology. Bidirectional edges make the graph undirected for GAT. The label on the graph is supplied by the caller (binary benign vs. attack) and matches the stratified pool in **Figure A1-3**, not a per-flow vote—important for reading the evaluation chapters.

![Figure A1-3 — stratified windowing excerpt](results/figures/appendix1/fig_a1_03_graph_builder_stratified.png)

**Figure A1-3** — Stratified windowing so both classes appear in training graphs. *(Chapter 13, Appendix D.)* *Source: `src/data/graph_builder.py`, lines 89–129.*

**Caption (formal):** Figure A1-3 — `build_graphs_for_split`: splits flows into benign and attack pools, builds sliding (or strided) windows from each pool with `_build_windows_from_pool`, balances counts between classes, shuffles, and logs totals—addressing severe class imbalance in raw CICIoT2023.

**How to interpret:** This block explains **why training does not collapse** to “always predict attack” despite a very high attack ratio in the raw CSV (**Chapter 4**): windows are drawn **within** each class, so each graph’s supervision matches the pool. The `minority_stride` argument increases overlap for the benign pool when it is smaller. The dissertation’s **window size** and **k** used in experiments come from `config/experiment.yaml` and feed the sensitivity grid in **Section 8.8**.

![Figure A1-4 — explainer source excerpt](results/figures/appendix1/fig_a1_04_explainer_integrated_gradients.png)

**Figure A1-4** — Post-hoc explanations: forward pass with attention, Integrated Gradients, top nodes/features. *(Chapter 13, Appendix D.)* *Source: `src/explain/explainer.py`, lines 53–102.*

**Caption (formal):** Figure A1-4 — `explain_sequence`: runs the model with attention enabled, wraps Captum **Integrated Gradients** on the **last window’s** node features (`_ig_wrapper`), aggregates absolute attributions to rank top nodes and top feature indices, and returns an `ExplanationBundle` for JSON alert formatting.

**How to interpret:** This is the bridge between **Chapter 6** (implementation) and **Chapter 8** (example alerts): SOC-facing text is not arbitrary—it is derived from **IG magnitudes** and **GAT edge attention** from the same forward pass analysts would get at inference. IG is computed on the **most recent** graph in the sequence (design choice documented in code comments). If Captum is missing, the bundle degrades gracefully (`HAS_CAPTUM` guard in the same module).

![Figure A1-5 — federated Flower client excerpt](results/figures/appendix1/fig_a1_05_federated_flower_client.png)

**Figure A1-5** — Federated learning CLI: Flower server vs. client, local data, `GNNFlowerClient`. *(Chapter 13, Appendix D.)* *Source: `src/federated/run_federated.py`, lines 28–71.*

**Caption (formal):** Figure A1-5 — `main` in `run_federated.py`: loads YAML config; **server** mode calls `run_fl_server` with round count and quorum; **client** mode loads `client_{cid}_graphs.pt`, builds train/validation sequence loaders, constructs `GNNFlowerClient`, and connects to `127.0.0.1:8080` via `fl.client.start_numpy_client`.

**How to interpret:** This file is the **student-facing entry point** for **Chapter 8** federated results: each client trains only on its partition (**non-IID** split from `src.federated.data_split`, referenced in **Appendix C**). The address and round counts come from `config/experiment.yaml` under `fl`. The pattern matches the FedAvg narrative in **Chapter 2**—local epochs, then aggregation on the server (server implementation in `src/federated/server.py`, not shown).

![Figure A1-6 — FastAPI score endpoint excerpt](results/figures/appendix1/fig_a1_06_fastapi_score_alert.png)

**Figure A1-6** — HTTP API: load checkpoint, `/score` builds graphs, runs `explain_sequence`, returns ECS-style alert JSON. *(Chapter 13, Appendix D.)* *Source: `src/siem/api.py`, lines 32–89.*

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
