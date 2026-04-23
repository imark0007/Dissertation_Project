# Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning

**Arka Talukder | B01821011**  
**MSc Cyber Security (Full-time)**  
**University of the West of Scotland**  
**School of Computing, Engineering and Physical Sciences**  
**Supervisor: Dr. Raja Ujjan**
*Merged draft: **Acknowledgements** and **Sections 1.1–1.2** follow the previous humanized `Arka_Talukder_Dissertation_Final_DRAFT_old.docx` (minor edit: moderator line uses *who*). **Sections 1.3 onward and all other chapters** follow the current canonical `Dissertation_Arka_Talukder.md` (methods, table numbering, data handling, and Harvard references). Export with `python scripts/dissertation_to_docx.py`.*

---

## 1. Abstract

Internet of Things (IoT) deployments produce large flow telemetry that Security Operations Centres (**SOCs**) must triage on **CPU**-only edge infrastructure. Detectors should be accurate, privacy-aware, and explainable, so analysts can act on alerts without extra investigation delay.

This dissertation develops an end-to-end prototype on **CICIoT2023** (Pinto et al. 2023). Flows are standardised, grouped into windows, converted into **k**NN similarity graphs, and classified by a **dynamic GNN** (GAT + GRU) built with **PyTorch Geometric**. The same architecture is trained centrally and with **Flower FedAvg** across three **non-IID** clients (Dirichlet **alpha = 0.5**). Explainability combines **Captum** Integrated Gradients and GAT attention, and results are served as ECS-like JSON through a **FastAPI** endpoint.

**Results.** On the held-out test split, central and federated GNN runs achieve **F1 = 100%** and **ROC-AUC = 100%** with zero false positives, while Random Forest and MLP achieve **F1 = 99.86%** and **99.42%**. Mean CPU inference is about **23 ms** per five-window sequence, and federated communication is about **31 MB** over ten rounds. Ablation, sensitivity, and multi-seed checks support these findings.

**Keywords:** IoT security, dynamic graph neural network, federated learning, SIEM, explainable AI, edge AI, SOC, CICIoT2023

---

## Acknowledgements

I would like to thank my supervisor, Dr. Raja Ujjan, who provided technical support, feedback on design and evaluation, and support during the project. I also want to thank my moderator, Muhsin Hassanu, who reviewed the interim work and helped to refine the final report.

I also owe a debt of gratitude to Dr. Daune West who helped me and gave me academic support during the submission period. I would like to thank School and programme staff in terms of module materials and guidance on submissions, and the MSc Project co-ordinator in terms of administrative communication regarding the milestones and ethics.

Lastly, I would like to thank friends and family that tolerate me during intensive writing and experiment runs.


## List of Abbreviations

| Abbrev. | Definition |
|---|---|
| **AI** | Artificial Intelligence |
| **API** | Application Programming Interface |
| **AUC** | Area Under the (ROC) Curve |
| **BCE** | Binary Cross-Entropy (loss) |
| **CICIoT** | Canadian Institute for Cybersecurity IoT benchmark family (this work: CICIoT2023) |
| **CPU** | Central Processing Unit |
| **CyBOK** | Cyber Security Body of Knowledge |
| **DDoS** | Distributed Denial of Service |
| **DL** | Deep Learning |
| **ECS** | Elastic Common Schema (log/alert shape used as a guide) |
| **ENISA** | European Union Agency for Cybersecurity |
| **FedAvg** | Federated Averaging (McMahan et al. 2017) |
| **FL** | Federated Learning |
| **FN** | False Negative |
| **FP** | False Positive |
| **FPR** | False Positive Rate |
| **GAT** | Graph Attention Network |
| **GCN** | Graph Convolutional Network |
| **GDPR** | General Data Protection Regulation (EU) |
| **GNN** | Graph Neural Network |
| **GPU** | Graphics Processing Unit |
| **GRU** | Gated Recurrent Unit |
| **IDS** | Intrusion Detection System |
| **IG** | Integrated Gradients |
| **IoT** | Internet of Things |
| **IID** | Independent and Identically Distributed (data split) |
| **IPS** | Intrusion Prevention System |
| **JSON** | JavaScript Object Notation |
| **kNN** | k-Nearest Neighbours (graph construction) |
| **LSTM** | Long Short-Term Memory |
| **ML** | Machine Learning |
| **MLP** | Multi-Layer Perceptron |
| **MITM** | Man-in-the-Middle (attack pattern) |
| **NCSC** | National Cyber Security Centre (UK) |
| **NIS2** | EU Directive on measures for a high common level of cybersecurity |
| **NIST** | National Institute of Standards and Technology (US) |
| **non-IID** | Non-identical client data distributions |
| **PyG** | PyTorch Geometric |
| **RAM** | Random Access Memory |
| **RF** | Random Forest (baseline classifier) |
| **ROC** | Receiver Operating Characteristic |
| **RNN** | Recurrent Neural Network |
| **SDIoT** | Software-Defined Internet of Things |
| **SDN** | Software-Defined Networking |
| **SIEM** | Security Information and Event Management |
| **SOC** | Security Operations Centre |
| **TN** | True Negative |
| **TP** | True Positive |
| **TPR** | True Positive Rate (equals recall in binary classification) |
| **UML** | Unified Modelling Language |
| **VAE** | Variational Autoencoder *(not used in this prototype; listed for generative-model context only)* |
| **XAI** | Explainable Artificial Intelligence |

*Note: In the final Word file, abbreviations in the first column may be coloured (e.g. dark red) to match the programme’s preferred sample layout.*

---

## Table of Contents

**List of Abbreviations**

**List of Figures** *(page numbers updated in Microsoft Word: References → Update Table, or right-click the field)*

**List of Tables** *(page numbers updated in Microsoft Word: References → Insert Table of Contents, label “Table”)*

**Acknowledgements**

**Chapter 1 – Introduction**  
- 1.1 Chapter Overview  
- 1.2 Background and Motivation  
- 1.3 Research Aim and Questions  
  - 1.3.1 Objectives  
- 1.4 Scope and Limitations  
- 1.5 Dissertation Structure  
- 1.6 Alignment with the Marking Criteria in the Project Specification  
  - 1.6.1 Where Each Technical Topic Is Stated Once (Clarity and Non-Repetition)  
- 1.7 Chapter Summary  

**Chapter 2 – Literature Review**  
- 2.1 Chapter Overview  
- 2.2 Themes and Structure  
  - 2.2.1 Literature Search Strategy and Primary Databases  
  - 2.2.2 Main Technology Pillars: Synthesis, Debate, and Fit with the Title  
    - 2.2.2.1 Software-Defined Flow Observability and the Detection Surface  
    - 2.2.2.2 Similarity Graphs and Dynamic GNNs as a Design Bet, Not a Slogan  
    - 2.2.2.3 Edge AI, Federated Training, and What “Privacy” Can Mean  
    - 2.2.2.4 Explainable, SIEM-Ready Outputs Versus Alert Economics  
- 2.3 IoT Security and the Need for Detection  
- 2.4 SIEM, SOC Workflows, and Alert Quality  
  - 2.4.1 Alert Volume, Triage, and the Need for Explanations  
  - 2.4.2 Explainability, SIEM Ecosystems, and Limits of “Better JSON”  
- 2.5 Graph Neural Networks and Dynamic Graphs  
  - 2.5.1 GNNs, Attention, and Attribute-Based Graphs in Intrusion Detection  
  - 2.5.2 Dynamic (*Temporal*) Graphs and the GAT+GRU Pattern  
- 2.6 Federated Learning and Privacy  
  - 2.6.1 FedAvg, Assumptions, and Communication  
  - 2.6.2 FL for IoT Intrusion Detection: Empirical Findings and Non-IID Reality  
- 2.7 Explainability in ML-Based Security  
  - 2.7.1 Post-hoc Attribution, Integrated Gradients, and Attention in GNNs  
  - 2.7.2 XAI for IoT Streams, SHAP as Alternative, and Latency as a Counterargument  
- 2.8 Synthesis: Significance, Limitations of Prior Work, Contributions, and Gap  
  - 2.8.1 How Existing Approaches Fill *Part* of the Problem (and Where They Stop)  
  - 2.8.2 Anticipated Significance (What *Would* Matter if the Evidence Holds)  
  - 2.8.3 Limitations *in the Literature* and Their Impact on Claims  
  - 2.8.4 Debate, Counterarguments, and the Consolidated Gap  
- 2.9 Mapping to CyBOK (Cyber Security Body of Knowledge)  
- 2.10 Extended Comparative Review (Fifteen to Twenty Core Sources)  
  - 2.10.1 Temporal Graphs, Fog Decentralisation, and Survey Evidence  
  - 2.10.2 SHAP, Integrated Gradients, and Method Choice in This Prototype  
  - 2.10.3 Botnets, Early Alert Correlation, and Lineage to Modern SIEM  
- 2.11 Chapter Summary  

**Chapter 3 – Project Management**  
- 3.1 Chapter Overview  
- 3.2 Project Plan and Timeline (45-day MSc)  
- 3.3 Risk Assessment  
- 3.4 Monitoring and Progress Reviews  
- 3.5 Research Plan for Completion: IDE, Platform, and Evidence for Supervision  
- 3.6 Ethics and Data  
- 3.7 Interim Report Feedback Incorporated  
- 3.8 Chapter Summary  

**Chapter 4 – Research Design and Methodology**  
- 4.1 Chapter Overview  
- 4.2 Research Approach  
- 4.3 Dataset and Subset  
  - 4.3.1 Proposed Pipeline Stages, Artefacts, and Leakage Controls  
  - 4.3.2 Repository Layout and Frozen Outputs  
  - 4.3.3 Empirical Flow and Graph Sample Sizes  
- 4.4 Models and Loss Functions  
  - 4.4.1 Hyperparameters, Tuning Funnel, and Training Budget (After Split)  
  - 4.4.2 Objectives, Losses, and Federation Rule  
- 4.5 Federated Learning Setup  
- 4.6 Explainability  
- 4.7 Evaluation Plan  
- 4.8 Software and Tools  
- 4.9 Chapter Summary  

**Chapter 5 – Design**  
- 5.1 Chapter Overview  
- 5.2 Pipeline, Alerts, and Deployment (Conceptual)  
- 5.3 Graph Design (Flows to kNN Snapshots)  
- 5.4 Research Design and System Architecture  
- 5.5 Conceptual Illustration: Similarity-Based Graph in One Window  
- 5.6 Chapter Summary  

**Chapter 6 – Implementation and System Development**  
- 6.1 Chapter Overview  
- 6.2 Environment and Tools  
- 6.3 Data Loading and Preprocessing  
- 6.4 Graph Construction  
- 6.5 Model Implementation  
- 6.6 Federated Learning (Flower)  
- 6.7 Explainability  
- 6.8 Alert Generation and SIEM-Style Output  
- 6.9 FastAPI Deployment and CPU Inference  
- 6.10 Implementation Code Screenshots (Author’s Codebase)  
- 6.11 Reproducibility Note  
- 6.12 Chapter Summary  

**Chapter 7 – Testing and Evaluation**  
- 7.1 Chapter Overview  
- 7.2 Evaluation Scope  
  - 7.2.1 Pass Criteria  
  - 7.2.2 Fail Criteria  
- 7.3 Test Schedule  
- 7.4 Experimental Setup  
- 7.5 Metrics  
- 7.6 Dataset and Experiment Statistics  
- 7.7 Comparison Design  
- 7.8 Chapter Summary  

**Chapter 8 – Results Presentation**  
- 8.1 Chapter Overview  
- 8.2 Centralised Model Comparison (Sub-Question 1)  
- 8.3 Federated Learning (Sub-Question 2)  
  - 8.3.1 What Federated Learning Produced (Reading the Curves and the Table)  
- 8.4 Central GNN Training Convergence  
- 8.5 Time-Window and CPU Inference (Sub-Question 2 and Deployment)  
- 8.6 Example Alerts with Explanations (Sub-Question 3)  
- 8.7 Ablation Studies (Priority 1: Evidence)  
- 8.8 Sensitivity Analysis (Stability of Design Choices)  
- 8.9 Multi-Seed Stability  
- 8.10 Comparison with Prior Work on CICIoT2023  
- 8.11 Chapter Summary  

**Chapter 9 – Conclusion, Discussion, and Recommendations**  
- 9.1 Chapter Overview  
- 9.2 Structured Conclusion (Programme Format)  
- 9.3 Answering the Research Questions  
- 9.4 Strengths and Limitations  
- 9.5 Use of University and Course Materials  
- 9.6 Practical Implications  
- 9.7 Relation to Literature  
- 9.8 Summary of the Project  
- 9.9 Recommendations for Future Work  
- 9.10 Chapter Summary  

**Chapter 10 – Critical Self-Evaluation**  
- 10.1 Chapter Overview  
- 10.2 Planning, Scope, and Risk  
- 10.3 Literature and Alignment with the Questions  
- 10.4 Implementation: What Was Harder Than It Looked  
- 10.5 Results, Honesty, and the “100%” Question  
- 10.6 What I Learned (Skills and Mindset)  
- 10.7 Time Management: What I Would Reorder  
- 10.8 Chapter Summary  

**Chapter 11 – References**  
**Chapter 12 – Bibliography**  
**Chapter 13 – Appendices** (A: process; B: specification; C: handbook Appendix 1 code figures; D: handbook Appendix 4 optional)

### Table of Figures

**Numbering rule:** Body figures are **Figure 1–Figure 29** in **strict order of first appearance** (Chapter 2 → Chapter 3 → Chapter 5 → Chapter 6 → Chapter 8). Within **Chapter 5**, subsection order is **5.2 pipeline → 5.3 graph specification → 5.4–5.5 figures**, so **Figures 7–9** match that reading order. Each caption states **Chapter** and **Section** (or **Appendix C**). Handbook **code extracts** in Appendix C keep labels **Figure A1-1**–**Figure A1-6** (they are not part of the 1–29 sequence). Refresh page numbers in Word after pagination.

| Figure | Title (chapter, section) | Page |
|--------|--------------------------|------|
| 1 | Taxonomy of IDS approaches (Ch. 2, Section 2.3) |  -  |
| 2 | Conceptual dynamic GNN flow  -  GAT + GRU (Ch. 2, Section 2.5) |  -  |
| 3 | Federated learning (FedAvg) flow (Ch. 2, Section 2.6) |  -  |
| 4 | Explainability methods for SOC-oriented alerts (Ch. 2, Section 2.7) |  -  |
| 5 | Positioning of related work  -  four pillars (Ch. 2, Section 2.7) |  -  |
| 6 | Project Gantt chart  -  six execution phases plus write-up (Ch. 3, Section 3.2) |  -  |
| 7 | Research pipeline  -  raw flows to SIEM alerts (Ch. 5, Section 5.2) |  -  |
| 8 | Research design  -  data flow, FL, edge alerting (Ch. 5, Section 5.4) |  -  |
| 9 | Conceptual *k*NN similarity graph in one window (Ch. 5, Section 5.5) |  -  |
| 10 | Code: `flows_to_knn_graph` core (Ch. 6, Section 6.10) |  -  |
| 11 | Code: `build_graphs_for_split` core (Ch. 6, Section 6.10) |  -  |
| 12 | Code: `train_one_epoch` training step (Ch. 6, Section 6.10) |  -  |
| 13 | Code: `DynamicGNN.forward` (Ch. 6, Section 6.10) |  -  |
| 14 | Code: `_ig_wrapper` / Integrated Gradients (Ch. 6, Section 6.10) |  -  |
| 15 | Code: FastAPI `POST /score` core (Ch. 6, Section 6.10) |  -  |
| 16 | Confusion matrix  -  Dynamic GNN (Ch. 8, Section 8.2) |  -  |
| 17 | ROC curve  -  Dynamic GNN (Ch. 8, Section 8.2) |  -  |
| 18 | Confusion matrix  -  Random Forest (Ch. 8, Section 8.2) |  -  |
| 19 | Confusion matrix  -  MLP (Ch. 8, Section 8.2) |  -  |
| 20 | ROC curve  -  Random Forest (Ch. 8, Section 8.2) |  -  |
| 21 | ROC curve  -  MLP (Ch. 8, Section 8.2) |  -  |
| 22 | Per-metric comparison  -  RF, MLP, Central GNN, Federated GNN (Ch. 8, Section 8.2) |  -  |
| 23 | Federated learning convergence  -  F1 and ROC-AUC vs. round (Ch. 8, Section 8.3) |  -  |
| 24 | Confusion matrix  -  Federated GNN (Ch. 8, Section 8.3) |  -  |
| 25 | Federated learning communication cost  -  per round and cumulative (Ch. 8, Section 8.3) |  -  |
| 26 | Central GNN training loss curve  -  six epochs (log scale) (Ch. 8, Section 8.4) |  -  |
| 27 | Model comparison  -  inference time and F1 (Ch. 8, Section 8.5) |  -  |
| 28 | Ablation  -  full GAT+GRU vs. GAT-only (Ch. 8, Section 8.7) |  -  |
| 29 | Sensitivity  -  window size and *k* (Ch. 8, Section 8.8) |  -  |
| A1-1 … A1-6 | Appendix C  -  handbook code figures (`DynamicGNN`, graph builder, explainer, Flower, FastAPI) |  -  |

### Table of Tables

**Numbering rule:** Tables are **Table 1–Table 13** in **strict order of first appearance** (Chapter 2 → Chapters 4–6 method and testbed → Chapter 7 schedule → Chapter 8 results). Refresh page numbers in Word after pagination.

| Table | Title (chapter, section) | Page |
|-------|--------------------------|------|
| 1 | Comparison of selected related work (Ch. 2, Section 2.7) |  -  |
| 2 | Proposed data and modelling pipeline stages (Ch. 4, Section 4.3) |  -  |
| 3 | Hyperparameters, training budget, and federation (Ch. 4, Section 4.4) |  -  |
| 4 | Empirical sample sizes: flows, graph windows, sequences (Ch. 4, Section 4.3) |  -  |
| 5 | Testbed, storage, and runtime stack (Ch. 6, Section 6.2) |  -  |
| 6 | Evaluation scenarios, scripts, and frozen artefacts (Ch. 7, Section 7.3) |  -  |
| 7 | Model comparison on CICIoT2023 test set (Ch. 8, Section 8.2) |  -  |
| 8 | Federated learning round-by-round metrics (Ch. 8, Section 8.3) |  -  |
| 9 | Central GNN training history (Ch. 8, Section 8.4) |  -  |
| 10 | Ablation  -  centralised GNN variants (Ch. 8, Section 8.7) |  -  |
| 11 | Sensitivity analysis  -  nine (*window*, *k*) configs (Ch. 8, Section 8.8) |  -  |
| 12 | Multi-seed summary  -  central GNN (Ch. 8, Section 8.9) |  -  |
| 13 | Comparison with prior work on CICIoT2023 (Ch. 8, Section 8.10) |  -  |

---

## Chapter 1 – Introduction

### 1.1 Chapter Overview

This chapter presents the project background, IoT flow telemetry, SOC alert volume and the practical limitation at the edge node of a CPU. It describes the rationale behind the inclusion of dynamic graph learning, federated training, and explainable alerts in the same pipeline, followed by the research aim, sub-questions, scope, and structure of the chapters to ensure that subsequent chapters in the work are focused on method, implementation, and evidence.

### 1.2 Background and Motivation

The Internet of Things (IoT) has grown at a very rapid pace in residential, workplace, campus and factory locations. There are now smart plugs, cameras, sensors and controllers everywhere. Most of these devices are small, low priced and limited and in most cases, security is not given much consideration during design. This has left weak credentials, patching, and misconfiguration as a common occurrence in actual deployments.

This implies that there is a high exposure to attack. A compromised IoT device may be used in botnet activities, service denial, credential theft or subsequent lateral movement into the broader infrastructure. These are not uncommon situations in security reports. Due to this, constant surveillance of network behaviour is not a choice, particularly where there are mixed old and new devices.

Software-defined techniques (such as SDN and software-defined IoT) enable switch and router operators to monitor switch flow statistics and router flow statistics centrally. Intrusion detection can be achieved with flow summaries, without full packet capture, often enough, reducing the storage pressure and alleviating some of the privacy concerns. The remaining question is to accurately analyse such flows, to explain decisions in a way that can be used by analysts, and in a manner which can be done on hardware which is realistic at the edge (which is often CPU-only).

The networks are monitored, alerts investigated, and incident response coordinated by Security Operations Centres (SOCs) typically in connection with SIEM platforms that consolidate logs and events derived by flow. Another real-life challenge is the problem of alert fatigue: large numbers of alerts and a high number of false positives decrease the time that analysts have to devote to real incidents. Alerts which are simply a label with no information as to why the model fired take even longer to triage. This necessitates accurate and explainable detectors hence staff can have confidence in outputs and can take action as time runs out.

Random Forests or feed-forward networks can be used to apply traditional models to tabular flow features, but traffic is relational: flows are like neighbours in feature space, have common endpoints in the presence of identifiers, or occur in bursts of traffic, which are meaningful as a whole. Graph models consider flows to be nodes and the adjacent related flows are connected using edges (in this case, kNN links in feature space since the public CICIoT2023 release does not support device-level topology). Dynamic GNNs build upon this concept by training on how the snapshots of the graph change within short periods of time. Individually, IoT data can also be spread out across locations where it is legally or politically challenging to pool raw flows; federated learning trains a single shared model, but maintains raw data locally. Most IoT gateways cannot also take on a GPU in each segment and a CPU pipeline able to emit SIEM friendly JSON is more of a deployable result than a GPU only laboratory result.

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

#### 1.3.1 Objectives

To answer the research questions above, the project pursues the following concrete objectives. Each objective is a deliverable artefact that is referenced again in **Chapters 4–8** for evidence:

1. **Literature objective.** Conduct a critical review of IoT intrusion detection, graph and dynamic GNN methods, federated learning, and explainable AI for security, and identify the gap that this project fills (**Chapter 2**).
2. **Dataset objective.** Select a manageable subset of **CICIoT2023** with both benign and attack traffic, fix train/validation/test splits, and document preprocessing (**Sections 4.3 and 6.3**).
3. **Modelling objective.** Implement two flow-level baselines (**Random Forest**, **MLP**) and one **dynamic GNN** (GAT + GRU) on identical splits for a controlled comparison (**Sections 4.4 and 6.5**).
4. **Federated objective.** Train the same GNN with **Flower FedAvg** across **three** non-IID clients for **ten** rounds, and evaluate the global checkpoint on the central test set (**Sections 4.5, 6.6 and 8.3**).
5. **Explainability objective.** Combine **Captum Integrated Gradients** with **GAT attention** to produce per-alert top features and top flows (**Sections 4.6, 6.7 and 8.6**).
6. **SIEM objective.** Serve the model through a **FastAPI** endpoint that returns **ECS-like JSON** alerts and report **CPU** inference latency (**Sections 6.8, 6.9 and 8.5**).
7. **Evaluation objective.** Define metrics, run ablation, sensitivity, and multi-seed checks, and report results with figures and tables (**Chapters 7 and 8**).
8. **Reflection and write-up objective.** Document the work in a final report aligned with the **UWS guideline** and **MSc Project Handbook** including a critical self-evaluation (**Chapters 9 and 10**).

### 1.4 Scope and Limitations

The project uses **CICIoT2023** (Pinto et al. 2023) on a **manageable subset** within a **45-day** MSc window. The public release includes **46** flow-level features but not reliable device topology fields for full host graph construction. Because of that, this implementation uses **flow nodes** with ***k*NN feature-similarity edges** and a **GAT+GRU** model stack, as justified in **Section 2.5** and specified in **Section 5.3**.

The scope includes **Random Forest** and **MLP** baselines, **Flower FedAvg** federation, **Captum** explainability, and **FastAPI** ECS-like alerts. The scope does not include production SIEM integration, formal SOC user studies, or large multi-organisation federation. Technical implementation detail is intentionally concentrated in **Chapters 4 to 7** to avoid repeated text.

**Limits (summary):** no production SIEM deployment, no formal SOC user study, **CPU-only** training and inference, **three** federated clients, and **ten** rounds in the reported federated run. Headline metrics are subset-specific and discussed honestly in **Chapter 9**. Contingency decisions, such as reduced data and reduced rounds if required, follow the risk treatment in **Chapter 3**. Reproduction uses the repository, `config/experiment.yaml`, and execution notes in **Chapter 13**.

### 1.5 Dissertation Structure

The chapter order follows the **UWS MSc final-report pattern** and the **Updated Guideline for Final Report**. It includes front matter, technical body chapters, references, and appendices in the expected sequence, but all content remains specific to this project topic and evidence.

**Chapter 2** reviews literature critically, with the extended comparative subsection in **Section 2.10** to cover **fifteen to twenty** core sources. **Chapter 3** records timeline, risks, and ethics process. **Chapter 4** defines research design and methodology. **Chapter 5** specifies pipeline and graph design. **Chapter 6** documents implementation modules and integration points. **Chapter 7** defines testing protocol and metric formulae. **Chapter 8** reports results as evidence, with minimal interpretation. **Chapter 9** provides discussion, conclusions, and recommendations linked to the research questions. **Chapter 10** provides first-person critical self-evaluation. **Chapters 11 and 12** contain references and bibliography. **Chapter 13** contains appendices including process records, specification, reproducibility material, and handbook code-figure evidence.

### 1.6 Alignment with the Marking Criteria in the Project Specification

The dissertation is organised against the **marking criteria and weightings recorded in the agreed Project Specification** (**Appendix B**). That is the same grid you committed to in formally submitted progress work: **Introduction 5%**; **Context / Literature Review 20%**; **Research Design 20%**; **Implementation (Practical Work) 25%**; **Evaluation 5%**; **Presentation of Results 5%**; **Conclusions and Recommendations 10%**; **Critical Self-Evaluation 10%** (total **100%**).

The chapter mapping is:

| Criterion (weight) | Where it is addressed in this dissertation |
|--------------------|---------------------------------------------|
| Introduction (5%) | **Chapter 1**  -  aim, questions, scope, structure |
| Literature (20%) | **Chapter 2**  -  critical review, comparison table, figures, gap |
| Research design (20%) | **Chapters 4–5**  -  methodology, dataset, graph and system design |
| Implementation (25%) | **Chapter 6**  -  build, modules, tooling, reproducibility |
| Evaluation (5%) | **Chapter 7**  -  protocol, metrics, setup |
| Results presentation (5%) | **Chapter 8**  -  tables and figures, minimal interpretation |
| Conclusions & recommendations (10%) | **Chapter 9**  -  answers to questions, implications, limits, future work |
| Critical self-evaluation (10%) | **Chapter 10**  -  reflection on process and learning |

### 1.7 Chapter Summary

Chapter 1 framed the SOC and IoT problem context, stated the research questions, and defined scope and boundaries, mapping the report to the marking criteria in **Section 1.6**. Chapter 2 next reviews the literature.

---

## Chapter 2 – Literature Review

### 2.1 Chapter Overview

This chapter reviews prior work on IoT intrusion detection, SOC/SIEM alerting needs, graph and dynamic GNN methods, federated learning, and explainable AI in security. It compares studies critically, names what is under-tested at the *integration* level, and links findings to the design of this prototype. The programme literature criterion is **at least 15% of the final report**; in the source Markdown, Chapter 2 is approximately **20–21% of the total word count** (recalculate if the body text changes materially), so length aligns with the handbook band (see also the specification’s **~20%** weight for *Context / Literature* in **Section 1.6**). Dataset sizes, fixed splits, and configuration numbers are left to the methodology and evaluation chapters (**Section 7.6** and **Section 6.2**) so this chapter remains analytical, not a second methods section. *In the final Word file, all body text, figure captions, and subheadings will follow a single UWS-recommended typeface and size hierarchy; author-generated figures use Matplotlib defaults unless the School template requires otherwise.*

### 2.2 Themes and Structure

This review is **selective, not exhaustive**: it prioritises peer-reviewed articles and major surveys on **(i)** IoT and flow-based intrusion detection, **(ii)** graph and **dynamic** graph neural models for security analytics, **(iii)** **federated learning** under data-sovereignty constraints, and **(iv)** **explainable** machine learning in operational security contexts. **Sections 2.2.1** and **2.2.2** state the search protocol and, separately, a **debate-led** view of the main technologies named in the dissertation title, so the chapter is not a glossary of terms.

The narrative is **argumentative**, not encyclopaedic. **Section 2.3** establishes threat and data context. **Section 2.4** links SIEM operations to **alert quality** and **analyst** constraints. **Sections 2.5–2.7** map **directly** to the three sub-questions: **graph and temporal** modelling, **federated** training, and **explanations** for triage. **Table 1** and **Figure 5** make the **multi-pillar** coverage of prior work **visible**. **Section 2.8** answers the specification’s call for **significance**, **limits of prior work**, **contribution**, and **counterarguments**. **Section 2.9** maps the story to **CyBOK**; **Section 2.10** extends breadth to the required **fifteen to twenty** core sources.

#### 2.2.1 Literature Search Strategy and Primary Databases

**Main retrieval route.** Searches were run primarily through **Google Scholar** for first discovery (keywords: IoT intrusion detection, graph neural network, CICIoT2023, federated learning, explainable; year filters 2017–2026 where available). **Peer-reviewed** full texts were then obtained from publisher platforms **IEEE Xplore**, **SpringerLink**, **MDPI (open access journals)**, and **ScienceDirect (Elsevier)**, in line with programme and supervisor guidance. These venues host the *core* papers used for claims in this chapter (e.g. dataset description, GNN/IDS surveys, FedAvg, IoT FL experiments, XAI for security).

**Traceability and inclusion rules.** *Backward* snowballing from survey papers (Zhong et al. 2024; Wang et al. 2025) and from the CICIoT2023 reference list (Pinto et al. 2023) identified additional GNN, temporal-graph, and IoT-IDS primary studies. *Forward* checks on Google Scholar (cited-by) were used to confirm that foundational methods (e.g. GAT, FedAvg, Integrated Gradients) are cited in later security applications. **Grey** sources (unverified blogs, uncited GitHub readmes) were **not** used to support empirical claims. A small number of **arXiv** or **arXiv-linked** items appear in **Chapter 12** (tooling: Flower, PyG, PyTorch) or, where unavoidable, in **Chapter 11** for widely adopted artefacts (e.g. Captum) with DOIs, **not** as substitutes for peer-reviewed security evidence. Documentation for ECS-like field *patterns* is implementation context, not a literature finding.

#### 2.2.2 Main Technology Pillars: Synthesis, Debate, and Fit with the Title

The dissertation title strings together **SIEM orientation**, **dynamic graph** learning, **edge AI** (CPU inference), and **federated learning**. The subsections below do not define “what is an attack”; they state **how the literature justifies *taking each ingredient seriously* at once**, where authors disagree, and what that implies for a single MSc prototype (Pinto et al. 2023; Zhong et al. 2024; McMahan et al. 2017; Qu et al. 2020; Cuppens and Miège 2002).

#### 2.2.2.1 Software-Defined Flow Observability and the Detection Surface

Software-defined and controller-centric IoT/SDN narratives matter here because they determine **where** flow aggregates become visible and **how far** an edge node can see (ENISA 2017). That visibility is a **policy-technical** trade-off, not a neutral given: more observation supports detection; it also increases handling obligations under data-protection framing. The **CICIoT2023** design (Pinto et al. 2023) reflects research practice: label-rich flow features without full raw capture, comparable across papers. The **debate** is not “IoT is risky” (Kolias et al. 2017) but whether **flow-level** ML generalises to production mixes—something surveys flag as an open **deployment** gap (Wang et al. 2025; Zhong et al. 2024). This work accepts that limit and uses **one** public benchmark to keep comparisons honest.

#### 2.2.2.2 Similarity Graphs and Dynamic GNNs as a Design Bet, Not a Slogan

Tabular classifiers on strong flow features are hard to beat on **some** tasks; GNNs are justified when **relational** structure carries signal beyond a single row (Ngo et al. 2025; Velickovic et al. 2018; Zhong et al. 2024). **Tension 1 — structure vs. cost:** GNN+sequence stacks increase CPU time per window (Basak et al. 2025; Lusa et al. 2025); the literature therefore supports **ablation and latency** reporting, not a priori superiority. **Tension 2 — topology vs. privacy:** the public CICIoT2023 release does not support full device-level IP graphs, so *k*NN in feature space is a **documented** compromise (Ngo et al. 2025), not a claim to physical topology. **Tension 3 — attention vs. ground truth:** GAT salience is not proof of attack intent; X-GANet-style work pairs accuracy with interpretability claims that still assume offline evaluation (Basak et al. 2025). The title’s “dynamic GNN” is thus a **testable** stack (GAT+GRU) against RF/MLP, not a marketing label.

#### 2.2.2.3 Edge AI, Federated Training, and What “Privacy” Can Mean

**Edge AI** in this project means **CPU** inference and deployment-shaped outputs (Lazzarini et al. 2023; Albanbay et al. 2025), not a claim that every edge device trains continuously. **Federated learning** is introduced where raw flows cannot be pooled (McMahan et al. 2017; Qu et al. 2020). The literature is explicit that **non-IID** client splits and client counts change convergence (Albanbay et al. 2025); that is a **counterargument** to any story that “FL always matches central training.” A second **counterargument** is privacy: **FedAvg** omits raw centralisation; it does **not** by itself guarantee differential privacy or robustness to gradient-based inference under strong threats (McMahan et al. 2017; discussion in **Section 2.6.2**). The title’s pairing of *edge* and *federated* is therefore a **situational** fit: each addresses a different constraint (inference site vs. training data custody).

#### 2.2.2.4 Explainable, SIEM-Ready Outputs Versus Alert Economics

A **SIEM-oriented** story requires more than a score: **analyst time** and **false-positive load** are long-standing operational themes (Cuppens and Miège 2002). XAI in IoT streams can improve transparency (Alabbadi and Bajaber 2025), but SHAP, IG, and attention disagree on *what* to privilege (Lundberg and Lee 2017; Sundararajan et al. 2017; Velickovic et al. 2018). The **debate** this chapter takes seriously is whether explanation **payloads** change triage or merely **transfer** work (see **Section 2.4.2**). The thesis therefore treats ECS-like JSON as a **plausible ingestion** move, to be read alongside measured **latency** and **selective** explanation (**Section 2.7.2**), not as proof of organisational adoption.

*Continuity note:* Sections **2.3–2.7** develop the same four strands with additional primary studies and figures; they should be read as the **evidential** layer for the **argumentative** synthesis already given in **Section 2.2.2**, not as separate “definitions of basics.”

### 2.3 IoT Security and the Need for Detection

Strategic policy documents also stress that IoT deployments introduce **concentrated risk** when devices are left under-protected, which motivates **defence in depth** at the network-observability level (ENISA 2017). Empirical work shows the same at device level: many products ship with default credentials, missing patches, or weak encryption (Kolias et al. 2017). When these devices are compromised, they can be used in botnets, for data exfiltration, or as a stepping stone into the rest of the network. So, detecting malicious behaviour in IoT traffic is an important part of modern security operations.

The scale of the IoT threat is well known. **Kolias et al. (2017)** studied the **Mirai** botnet. It used weak Telnet passwords on millions of IoT devices to launch large DDoS attacks. Such incidents show that IoT networks need continuous monitoring and automated detection. Manual inspection is not possible at scale. IoT devices vary a lot (cameras, thermostats, industrial controllers). So attack surfaces vary widely. Detection systems must work with different traffic patterns and protocols. SDN and Software-Defined IoT add another layer. Central controllers can collect flow statistics from switches and routers. But the amount of data and the need for real-time analysis require efficient detection methods that can run at the edge.

Intrusion detection for IoT can be done in different ways. **Signature-based** methods look for known attack patterns. **Anomaly-based** and **ML-based** methods learn normal traffic or decision boundaries in feature space; surveys document widespread use of classifiers on flow and packet data (Zhong et al. 2024; Pinto et al. 2023). The **CICIoT2023** dataset used in this project was published to support such research. It includes attacks like DDoS, reconnaissance, and brute force from real IoT devices in a controlled environment (Pinto et al. 2023). The dataset has 46 pre-computed flow features from packet captures. This reduces the need for raw packet inspection. It also matches the kind of data that SIEM systems usually use. **Pinto et al. (2023)** describe how the data was collected. Traffic was captured from real IoT devices (smart plugs, cameras, doorbells) under controlled attack scenarios. Flow features were extracted using standard tools. The features include protocol indicators (TCP, UDP, ICMP, HTTP, DNS), packet and byte statistics, TCP flag counts, and statistics like mean, variance, and standard deviation. Using a public dataset lets results be compared with other work. But the setting is still lab-based, not a live network. **Yang et al. (2025)** recently showed that graph neural networks can improve intrusion detection for industrial IoT. They showed that modelling network structure can improve accuracy compared to flat feature models. The CICIoT2023 dataset has a class imbalance (about 97.8% attack, 2.2% benign at the flow level). This makes training harder. This project deals with it through stratified windowing and balanced graph construction.

Many studies treat each flow or packet on its own. But in reality, attacks often show up as patterns of communication between devices over time. For example, a DDoS attack may involve many flows from many sources to one target. A reconnaissance scan may show sequential probing of ports. Flat feature vectors lose this structure. **Wang et al. (2025)** did a scoping review. They found that graph-based approaches to network traffic analysis are used more and more. Dynamic graph models in particular look promising for capturing attack patterns that change over time. **Zhong et al. (2024)** surveyed graph neural networks for intrusion detection. They found a clear trend towards combining GNNs with temporal models. They also noted that explainability and federated deployment are still under-explored. These findings support the graph-based and dynamic approach in this project. A key gap in the literature is that there are few prototypes that combine graph-based detection, federated learning, and explainable alerting in one system for SOC use. Most papers focus on only one of these. **Figure 1** (categories aligned with the discussion in **Kolias et al. 2017**; **Pinto et al. 2023**; **Wang et al. 2025**; **Zhong et al. 2024**) summarises the main intrusion detection families relevant to IoT and this project. **Critically**, taxonomies simplify messy commercial stacks where signature tiers, ML tiers, and vendor playbooks overlap; the figure is a **navigation aid** for this thesis, not a complete map of every deployed SOC architecture.

![Figure 1: IDS taxonomy](assets/literature_ids_taxonomy.png)

**Figure 1: Taxonomy of IDS approaches relevant to IoT and this project** *(Chapter 2, Section 2.3)*

*Sources: (Kolias et al. 2017); (Pinto et al. 2023); (Wang et al. 2025); (Zhong et al. 2024). Full references in Chapter 11.*

### 2.4 SIEM, SOC Workflows, and Alert Quality

#### 2.4.1 Alert Volume, Triage, and the Need for Explanations

SOC teams use SIEM and related tools to collect logs and flow data and to generate alerts. A well-known problem is **alert fatigue**: too many alerts and too many false positives make it hard for analysts to focus on real threats (Cuppens and Miège 2002). If an alert does not explain **why** it was raised, triage becomes slower and more guesswork. **Cuppens and Miège (2002)** proposed alert **correlation** to reduce noise by grouping related alerts. The main problem endures: many detection systems still output **labels** without a faithful account of the evidence. This matters because analyst time is a **finite** resource; the argument is not new, but it directly motivates **XAI** in the sections that follow (Cuppens and Miège 2002).

The **Elastic Common Schema (ECS)** provides a **vendor-neutral** field pattern for security events: event type, outcome, source, destination, and extensions. **ECS-like** JSON (field names and nesting inspired by that schema) is used in this project so that alerts can be **ingested** by Elastic, Splunk, and similar platforms without a bespoke format per tool (vendor documentation; **this dissertation does not claim** endorsement or certification by any vendor).

#### 2.4.2 Explainability, SIEM Ecosystems, and Limits of “Better JSON”

Major SIEM products (e.g. Splunk, Elastic, Microsoft Sentinel) support **custom rules** and **ML**-assisted detections, but the **deployment** of **explainable** model outputs in **tier-1** triage is still **uneven** in the published record; vendors emphasise **playbooks** and **case** workflows as much as raw explanation objects. For academic ML papers, a standard comparison is to **local explanation** tools: **Lundberg and Lee (2017)** formalise **SHAP** for additive feature attribution, while gradient methods such as **Integrated Gradients** (Sundararajan et al. 2017) target deep networks. In a **SOC** context, explanations that point to **flows**, **devices**, or **time** windows can support a faster **escalation** decision. This project **instantiates** that idea with **top features** and **top flows** in the alert. **Critically**, **correlation** and a cleaner **schema** do not by themselves show that analysts will **trust** or **rely on** a model’s escalation when queues already exceed service levels—**workload and incentive** effects remain **contested** in the operations literature; formatting alone does not end that debate.

### 2.5 Graph Neural Networks and Dynamic Graphs

#### 2.5.1 GNNs, Attention, and Attribute-Based Graphs in Intrusion Detection

Graph neural networks (GNNs) work on data that has a graph structure: nodes and edges. In network security, nodes can be hosts or **flows** and edges can be **similarity** or adjacency in feature space. GNNs combine information from neighbours and can learn patterns that depend on the graph structure. **Message passing** (each node update aggregates neighbour messages) encodes **relational** structure that **flat** tabular classifiers (one row per flow) treat as **independent** samples (Velickovic et al. 2018; Wang et al. 2025). **Graph Attention Networks (GATs)** allow **learned** weights over neighbours (Velickovic et al. 2018), which is useful when some edges matter more for the decision. **Han et al. (2025)** proposed **DIMK-GCN** for intrusion detection with **multi-scale** graph views. **Ngo et al. (2025)** showed that **attribute-based** graphs (e.g. *k*NN in feature space) are viable when **device identifiers** are missing—this directly supports the **kNN** construction in this project on **CICIoT2023** public releases. When topology IP graphs are not available, **similarity** graphs are a **principled** fallback (Ngo et al. 2025). **Critically**, attention highlights **salient** edges; it is **not** a **causal** proof of intent, and the XAI community continues to ask whether saliency tracks **ground-truth** attack steps or only the model’s **margin** (see also Basak et al. 2025).

#### 2.5.2 Dynamic (*Temporal*) Graphs and the GAT+GRU Pattern

Networks change over time. **Dynamic** (temporal) graph models stack a **GNN** over each snapshot and a **sequence** model across snapshots—see **Zheng et al. (2019)** for **attention-based temporal GCNs** on dynamic graphs. **Lusa et al. (2025)** combine **temporal** edge features with **GraphSAGE** for NID. **Basak et al. (2025)** report both **high accuracy** and **interpretable** GNN attention for NID. For IoT **flow** data, a **GAT+GRU** design tests whether **structure** and **time** add value over **Random Forest** and **MLP** on the same splits. The **net** benefit is **data-dependent**; the literature supports a **fair** comparison rather than a universal claim of dominance. GNNs usually cost more CPU time per prediction than **tabular** models—an **edge deployment** concern this project **measures** in **Section 8.5**. **Figure 2** summarises the **GAT+GRU** data path in the sense of the cited temporal-graph literature (**Velickovic et al. 2018**; **Zheng et al. 2019**; **Lusa et al. 2025**; **Basak et al. 2025**): GAT on each window’s graph, then GRU on the **sequence** of window embeddings, then a classifier. The diagram is **author-original**; it does not reproduce a figure from any single prior paper.

![Figure 2: Dynamic GNN concept](assets/literature_dynamic_gnn_concept.png)

**Figure 2: Conceptual flow of a dynamic GNN (GAT + GRU) for temporal graph classification** *(Chapter 2, Section 2.5)*

*Sources: (Velickovic et al. 2018); (Zheng et al. 2019); (Lusa et al. 2025); (Basak et al. 2025). Full references in Chapter 11.*

### 2.6 Federated Learning and Privacy

#### 2.6.1 FedAvg, Assumptions, and Communication

Federated learning trains a model across many clients without centralising raw data. Each client trains locally and sends only model **updates** (gradients or weights) to a server; the server aggregates and broadcasts a global model. **FedAvg** (McMahan et al. 2017) is the baseline: the server averages client parameters, often weighted by local dataset size. This reduces the direct need to store all raw flow captures in one central pool, which is relevant to data-sovereignty discussions for multi-site settings (Lazzarini et al. 2023; Albanbay et al. 2025). In IoT and edge deployments, bandwidth limits favour shipping parameters (order of megabytes) rather than full raw traffic (which can be far larger). **Figure 3** is an **author** schematic of the FedAvg *role* model described in **McMahan et al. (2017)**, not a reproduction of a figure from that paper.

![Figure 3: FedAvg flow](assets/literature_fedavg_flow.png)

**Figure 3: Federated learning (FedAvg) flow  -  no raw data leaves clients** *(Chapter 2, Section 2.6)*

*Sources: (McMahan et al. 2017); (Lazzarini et al. 2023). Full references in Chapter 11.*

#### 2.6.2 FL for IoT Intrusion Detection: Empirical Findings and Non-IID Reality

Federated learning has been applied in IoT and fog settings. **Qu et al. (2020)** discuss decentralised privacy patterns in fog environments (including blockchain-linked federated designs), which helps explain why some edge operators refuse raw data pooling. **Lazzarini et al. (2023)** report IoT IDS experiments under FedAvg, with central-comparable performance when data conditions are stable. **Albanbay et al. (2025)** show that client count and the severity of **non-IID** splits affect both convergence and accuracy—a practical **counterpoint** to any claim that “FL is always free” or always matches central training without tuning. When class and attack-type mixes differ by site, the global model must integrate heterogeneous client updates; more rounds and careful aggregation can be required. This project uses **Flower** with **three** clients and a **Dirichlet** non-IID split to test federated GNN **parity** with central training, and logs **communication** volume (**Chapter 8**). **Critically**, FedAvg does **not** by itself provide **differential privacy**; **gradient** inversion and **membership**-style attacks remain part of the active research record under **strong** adversaries—this work assumes **honest-but-curious** coordination and a threat model suitable for the stated MSc scope.

### 2.7 Explainability in ML-Based Security

#### 2.7.1 Post-hoc Attribution, Integrated Gradients, and Attention in GNNs

Explainability is usually added **after** the model is trained. **Post-hoc** methods rank **input** features (or substructures) that **influenced** a prediction. **Integrated Gradients** (IG) is a **gradient**-based attribution: it attributes the score change from a **baseline** input to the actual input, with axioms discussed in **Sundararajan et al. (2017)**. The implementation in this project uses **Captum** (**Kokhlikyan et al. 2020**), which integrates with **PyTorch**. For **GAT** networks, **edge**-level **attention** (Velickovic et al. 2018) provides a complementary view of *which* neighbours were emphasised. This dissertation uses **IG** for **per-feature** magnitudes and **GAT** attention for **per-flow** emphasis so that an analyst sees **both** “what in the feature vector” and “which flow rows” **supported** the alert. **Figure 4** (methods aligned with **Sundararajan et al. 2017**; **Velickovic et al. 2018**; **Kokhlikyan et al. 2020**) sketches that dual path for SOC-style JSON.

![Figure 4: Explainability pipeline](assets/literature_explainability.png)

**Figure 4: Explainability methods used for SOC-oriented alerts** *(Chapter 2, Section 2.7)*

*Sources: (Sundararajan et al. 2017); (Velickovic et al. 2018); (Kokhlikyan et al. 2020). Full references in Chapter 11.*

#### 2.7.2 XAI for IoT Streams, SHAP as Alternative, and Latency as a Counterargument

**Alabbadi and Bajaber (2025)** evaluate XAI for intrusion detection on IoT **streams** and show that post-hoc explanations can improve transparency for analysts, but their work is not a full end-to-end SIEM hand-off. **Lundberg and Lee (2017)** define SHAP; SHAP is model-agnostic and widely used for tree and MLP baselines, whereas **Integrated Gradients** fits the GNN+Captum stack in this work. A practical **counterargument** to “explain every alert” is **latency**: IG needs many forward passes (the implementation default is 50); on GNN + sequence inputs the cost is non-trivial, so the code supports **selective** explanation. Field studies remain sparse on whether explanation payloads **shorten** time-to-triage on edge hardware—that limit is returned to in **Chapter 9**, not overstated here.

**Table 1** and **Figure 5** summarise how prior studies cover the four project pillars. No single row in **Table 1** combines GNN, federated training, XAI, and a SIEM-oriented alert wrapper in the way the present prototype does; that gap is made explicit in **Section 2.8**.

**Table 1: Comparison of selected related work (sources: Chapter 11)**

| Study | GNN/Graph | Federated learning | Explainability | SIEM/Alert | Dataset/context |
|-------|-----------|--------------------|----------------|------------|------------------|
| (Pinto et al. 2023) |  -  |  -  |  -  |  -  | CICIoT2023 dataset |
| (Velickovic et al. 2018) | Yes (GAT) |  -  |  -  |  -  | General GNN |
| (McMahan et al. 2017) |  -  | Yes (FedAvg) |  -  |  -  | General FL |
| (Sundararajan et al. 2017) |  -  |  -  | Yes (IG) |  -  | Attribution method |
| (Han et al. 2025) | Yes (DIMK-GCN) |  -  |  -  |  -  | IDS |
| (Ngo et al. 2025) | Yes (attribute-based) |  -  |  -  |  -  | IoT IDS |
| (Basak et al. 2025) | Yes (X-GANet) |  -  | Yes |  -  | NID |
| (Lusa et al. 2025) | Yes (TE-G-SAGE) |  -  | Yes |  -  | NID |
| (Lazzarini et al. 2023) |  -  | Yes |  -  |  -  | IoT IDS |
| (Albanbay et al. 2025) |  -  | Yes |  -  |  -  | IoT IDS |
| (Alabbadi and Bajaber 2025) |  -  |  -  | Yes (XAI) |  -  | IoT streams |
| (Yang et al. 2025) | Yes |  -  |  -  |  -  | Industrial IoT |
| **This project** | **Yes (GAT+GRU)** | **Yes (Flower)** | **Yes (IG+attention)** | **Yes (ECS-like)** | **CICIoT2023** |

*Source: Author compilation from the papers cited in Chapter 11; table is descriptive only.*

*Note: IG = Integrated Gradients. NID = network intrusion detection. Full references in Chapter 11.*

![Figure 5: Positioning of related work](assets/literature_positioning.png)

**Figure 5: Positioning of related work  -  GNN, federated learning, explainability, and SIEM-style alerts** *(Chapter 2, Section 2.7)*

*Sources: (Pinto et al. 2023); (Velickovic et al. 2018); (McMahan et al. 2017); (Sundararajan et al. 2017); (Han et al. 2025); (Ngo et al. 2025); (Basak et al. 2025); (Lusa et al. 2025); (Lazzarini et al. 2023); (Albanbay et al. 2025); (Alabbadi and Bajaber 2025); (Yang et al. 2025). This figure synthesises **Chapter 2** and supports the gap stated in **Section 2.8**.*

### 2.8 Synthesis: Significance, Limitations of Prior Work, Contributions, and Gap

This section addresses four items expected in a strong final-report literature review: (1) significance of the *anticipated* findings for the stated scope; (2) limitations in the *literature* and their impact on defensible claims; (3) a critical view of the *contribution*; and (4) how existing approaches partially cover the problem before the integrated gap is stated.

#### 2.8.1 How Existing Approaches Fill *Part* of the Problem (and Where They Stop)

Good papers typically specialise. Yang et al. (2025) and Han et al. (2025) push GNN accuracy for IoT or NID but do not operationalise federated training or SIEM-shaped delivery. Lazzarini et al. (2023) and Albanbay et al. (2025) scale FedAvg under non-IID IoT partitions but generally do not pair that with a dynamic graph front end and end-to-end alerts. Alabbadi and Bajaber (2025) and Basak et al. (2025) foreground XAI in central or offline evaluations. Pinto et al. (2023) enable repeatable benchmarks on CICIoT2023 but do not prescribe a single detector architecture. Table 1 makes the pattern visible: the capability columns are rarely all “on” in one study; the final row (this project) is the only entry that attempts all four pillars in one CPU-first pipeline.

#### 2.8.2 Anticipated Significance (What *Would* Matter if the Evidence Holds)

If, on fixed splits, a dynamic GNN matches or outperforms strong tabular baselines while emitting analyst-facing alerts, the significance is engineering and methodological: it demonstrates feasibility of combining concerns that the literature often splits across separate papers. The expected contribution is not a universal claim of best F1 on all IoT traffic; it is a transparent prototype with ablation, sensitivity, and multi-seed artefacts so that trade-offs (central vs. federated, latency vs. model depth) are visible to examiners and to readers with SOC-oriented interests.

#### 2.8.3 Limitations *in the Literature* and Their Impact on Claims

Three limits recur. First, published benchmarks rely on lab datasets and subset choice can inflate apparent performance (Pinto et al. 2023; Chapters 8–9). Second, XAI work often reports attribution quality without operator time-to-triage (Alabbadi and Bajaber 2025; Section 2.7.2). Third, federated “privacy” narratives can outrun what FedAvg formally guarantees (McMahan et al. 2017; Section 2.6.2). These limits bound wording: the dissertation claims a documented pipeline and comparative evidence on CICIoT2023, not production readiness for arbitrary live tenants.

#### 2.8.4 Debate, Counterarguments, and the Consolidated Gap

*Counterargument 1 — “Graphs are unnecessary if features are already strong.”* The literature itself is split (Yang et al. 2025; Zhong et al. 2024). This work tests the claim with RF and MLP baselines on the same splits and reports false positive counts as well as headline F1. *Counterargument 2 — “Federated learning is privacy for free.”* Qu et al. (2020) and Albanbay et al. (2025) show deployment subtleties; this work states honest-but-curious assumptions and measures bytes exchanged, not a differential-privacy bound. *Counterargument 3 — “Explainable alerts remove audit and workload risk.”* Cuppens and Miège (2002) and Section 2.4.2 stress operational load; integrated gradients and GAT attention add artefacts for human review, not a guarantee of trust or policy compliance.

**Consolidated gap:** No prior row in Table 1 unifies *k*NN temporal graphs, a GAT+GRU stack, Flower FedAvg, Captum integrated gradients and GAT attention cues, and ECS-like JSON alerts with RF/MLP baselines on a shared CICIoT2023 subset under one MSc-style reproducible package. Chapters 4–8 instantiate that gap as design and evidence; Chapter 9 returns to significance after the results.

### 2.9 Mapping to CyBOK (Cyber Security Body of Knowledge)

The **agreed project specification** (Appendix B) records this work against **CyBOK** knowledge areas. The dissertation aligns with the following topics in particular: **Attacks & Defences → Security Operations & Incident Management** (SIEM-style alerting, SOC triage, and intrusion detection outputs); **Infrastructure Security → Network Security** (flow-based analysis and IoT traffic classification); **Systems Security → Distributed Systems Security** (federated learning without centralising raw client data); and **Software and Platform Security → Software Security** (engineering of the prototype pipeline, APIs, and reproducible scripts). Together, these areas situate the project within the School’s CyBOK mapping and match the specification’s indicative coverage.

### 2.10 Extended Comparative Review (Fifteen to Twenty Core Sources)

The programme guideline asks for fifteen to twenty significant papers in the literature review. Earlier sections already developed themes with primary citations. This section groups further comparisons so the breadth requirement is explicit and each cluster carries named in-text references.

#### 2.10.1 Temporal Graphs, Fog Decentralisation, and Survey Evidence

Zheng et al. (2019) show attention-based temporal GCNs for dynamic graphs; that line of work motivates stacking a sequence model (here, a GRU) on top of graph snapshots instead of classifying each window in isolation. Qu et al. (2020) discuss decentralised privacy patterns in fog computing, including blockchain-linked federated designs; this dissertation does not use blockchain, but the paper clarifies why edge sites may refuse raw data centralisation. Zhong et al. (2024) and Wang et al. (2025) survey GNN use in intrusion detection and where explainability and deployment gaps remain—useful for positioning the GAT+GRU and alert design choices in Sections 2.5 and 2.7.

#### 2.10.2 SHAP, Integrated Gradients, and Method Choice in This Prototype

Lundberg and Lee (2017) formalise SHAP for additive feature attribution. This project uses Integrated Gradients (Sundararajan et al. 2017) and GAT attention because they fit the PyTorch and Captum stack and the graph output; SHAP remains a defensible choice for tree or tabular baselines in parallel studies. The extended review therefore does not claim IG is universally superior; it records an engineering-consistent choice tied to the implemented models.

#### 2.10.3 Botnets, Early Alert Correlation, and Lineage to Modern SIEM

Kolias et al. (2017) document Mirai-style botnets in the IoT, which explains why flow-level DDoS patterns dominate many IoT benchmarks, including CICIoT2023. Cuppens and Miège (2002) give early alert-correlation ideas that prefigure today’s correlation rules and case workflows; present SIEM output still benefits from the same *analyst-time* logic even though field naming has standardised on JSON and ECS-like schemas (Section 2.4.1).

### 2.11 Chapter Summary

Chapter 2 moved from the **search strategy and database choice** (Sections **2.2.1–2.2.2**) to strategic IoT risk and dataset context (**2.3**), SIEM triage and explainability limits (**2.4**), graph and temporal GNN design (**2.5**), federated training caveats (**2.6**), and attribution toolchains (**2.7**). Section **2.2.2** in particular situates the title’s main technologies in **argument** form (flow observability, graph cost/benefit, federation vs. edge inference, explainable alert economics) before the thematic sections add paper-level detail. **Table 1** and **Figure 5** visualise the gap by capability pillar; **Section 2.8** ties anticipated significance, literature-side limits, counterarguments, and the consolidated research gap. **Section 2.9** maps themes to **CyBOK**; **Section 2.10** extends the source base to programme breadth. The next chapters turn from what others did to what this project built: fixed dataset handling, *k*NN temporal graphs, the GAT+GRU stack, Flower FedAvg, Captum, and FastAPI-delivered ECS-like alerts.

---

## Chapter 3 – Project Management

### 3.1 Chapter Overview

This chapter documents the project process: planning and milestones, risk management, the **development environment and evidence** prepared for **supervisor meetings** (**Section 3.5**), ethics posture (public dataset, no human participants), and how interim feedback was incorporated. It supports transparency of execution without duplicating the technical content presented in Chapters 4–7.


### 3.2 Project Plan and Timeline (45-day MSc)

Work was staged in six phases: (1) freeze requirements and literature; (2) fix the CICIoT2023 subset and preprocessing; (3) implement graph construction and central GNN training; (4) run baselines and Flower-based federated training; (5) add explainability, alert JSON, and FastAPI; (6) run ablation, sensitivity, multi-seed experiments, figures, and final writing. Each phase had concrete artefacts (metrics files, checkpoints, plots) to reduce ambiguity about “done.” **Figure 6** shows the schedule as a Gantt chart with overlapping phase bars, which is more representative of real research execution than strictly sequential blocks.

![Figure 6: Project Gantt chart](assets/gantt_chart.png)

**Figure 6: Project Gantt chart  -  six execution phases plus write-up across the 45-day MSc window** *(Chapter 3, Section 3.2)*

*Source: Author’s own diagram (`scripts/generate_gantt_chart.py`).*

### 3.3 Risk Assessment

| Risk | Mitigation |
|------|------------|
| Training too slow on CPU | Fixed subset; configurable rounds/clients; early stopping |
| Severe class imbalance | Stratified windowing; class weights; balanced graph-level labels |
| Federated instability | Three clients; reproducible split; checkpoints each round |
| Explainability runtime | Top-*k* attributions; optional subset of alerts |
| Scope creep (extra models) | Baselines limited to RF and MLP; one main GNN architecture |

### 3.4 Monitoring and Progress Reviews

Progress was monitored on a weekly cadence with the supervisor and against the phase plan in **Section 3.2**. At each weekly meeting the agenda followed the same three points: (1) what was completed since the last meeting and which artefact in the repository proves it; (2) what is in progress and any blocker that needs supervisor input; (3) what is planned for the next week and which phase boundary it serves. Decisions and action items were captured in the project process documentation that is referenced in **Appendix A** so that activity is traceable beyond memory.

### 3.5 Research Plan for Completion: IDE, Platform, and Evidence for Supervision

The programme and supervisor may review **where and how** the code was built and how progress is demonstrated in meetings. This section **records** that information so it is **maintained in the final report** and can be revisited in follow-up supervisions, consistent with the handbook’s expectation of **traceable** development and **Harvard (UWS)** references for any non-original tools (University of the West of Scotland Library 2025).

**Working environment:** Primary development used **Cursor**, an **AI-assisted integrated development environment (IDE)** based on the same editor stack as **Visual Studio Code**, running on **Windows 10 (x64)** (see also **Table 5** in **Section 6.2**). The repository was **cloned and edited** in that IDE; version control, runs, and figures were produced on the same machine. **Cursor** is cited as a **reputable, vendor-documented** product (Anysphere, Inc. 2024), not as a source of **research** claims. **Core language and tooling** used alongside it are the **Python** implementation (Python Software Foundation 2023) and the **Git** version-control system (The Git Project 2023); both are **standard, vendor-backed** resources listed in the **Chapter 12** bibliography. **Citations in this report follow the UWS Library Harvard *libguide*** (University of the West of Scotland Library 2025).

**What supervisors can check in meetings:** (1) the **open Git** history on the project repository; (2) **frozen** outputs under `results/metrics/` and `results/figures/` named in **Chapter 13**; (3) the **Gantt** phases in **Section 3.2**; (4) **configuration** in *`config/experiment.yaml`*. The student completed **implementation, experiments, and dissertation drafting** in this environment using **Cursor** for editing, **terminal** session runs as documented in *`README.md`* and *`SETUP_AND_RUN.md`*, and the **MD → Word** pipeline (*`scripts/dissertation_to_docx.py`*) for submission formatting.

**Not a substitute for design rationale:** The IDE accelerates **editing, navigation, and repeatability of commands**; it does **not** replace the **method** described in **Chapters 4–7** or the **literature** in **Chapter 2**, which are cited from **peer-reviewed** and **established** technical sources in **Chapters 11–12** as per UWS practice.

### 3.6 Ethics and Data

Only **public** CICIoT2023 data is used; there are no human participants or proprietary organisational datasets. The **signed specification ethics section** records supervisor confirmation that the project **does not require** full ethics-review board approval under School procedures (February 2026), which matches the use of non-sensitive public data only. Process and ethics paperwork are referenced in **Appendix A/B** as required by the handbook.

### 3.7 Interim Report Feedback Incorporated

The MSc Project Handbook requires that feedback on the **interim report** (written and verbal from supervisor and moderator) be taken into account in the final submission. After interim submission, feedback reinforced: (1) keeping the **end-to-end** story visible (data → model → federation → explainability → SOC-shaped output), not only literature; (2) tightening the **evaluation plan** before reporting numbers; (3) being explicit about **limits** of a lab subset and very high test metrics; and (4) reserving effort for **final analysis and write-up** rather than scope creep.

Actions taken in the final dissertation include: completing the full evaluation and results chapters (**7–8**) with tables and figures promised at interim; adding **ablation**, **sensitivity**, and **multi-seed** checks so result stability is evidenced (going beyond the interim contingency of dropping sensitivity if time ran short); strengthening **critical reflection** (**Chapter 10**) on methodology, federation, and the “100% metrics” question; and documenting **reproducibility** (**Chapter 13**). Any remaining suggestions from the interim review are acknowledged as **limitations** and **future work** in **Chapter 9**.


### 3.8 Chapter Summary

Transparent **time-boxing** and **risk controls** kept the project feasible while still delivering graph, federated, explainable, and deployable components. **Section 3.5** documents the **IDE and platform** used for **supervision-ready** evidence; **UWS Harvard**-listed **tool** references are in **Chapter 11**; the **Harvard *libguide*** appears in **Chapter 12** (University of the West of Scotland Library 2025).

---

## Chapter 4 – Research Design and Methodology

### 4.1 Chapter Overview

This chapter is the **Research Method** pillar: it states, in the author’s own terms, the **stages, artefacts, and controls** that turn **secondary, pre-recorded** CICIoT2023 flow tables into **trained models** and **reported accuracy** on a **held-out** test split. Programme guidance targets roughly **fifteen percent or more** of the final report for **Research Method** and a further **fifteen percent or more** for **Research Method Deployment**; here, **Chapters 4–5** carry the *method* (design, pipeline, graph semantics) and **Chapters 6–7** carry the *deployment* (code paths, testbed, experimental scenarios). **Chapter 5** provides figures **7–9** and the authoritative graph rules; **Chapter 6** gives module-level implementation; **Section 7.5** defines metrics once. Together they satisfy the “two-chapter” structure without pasting abstract methodology from unrelated papers: every step below maps to a **named file or script** in the repository.

### 4.2 Research Approach

**Epistemology.** The work is **empirical and artefact-driven**: a software prototype (data pipeline → models → optional federation → explainability → API) is built and judged by **reproducible** metrics. Learning is always from **pre-recorded** flow rows supplied by the public benchmark; **no live traffic** is collected for training. **Prediction** on unseen flows is the operational sense of “accuracy” used here, implemented as held-out test metrics in **Section 7.5** and **Chapter 8**.

**Proposed contribution (method-level).** The methodology is not a restatement of a single prior paper. It is a **compositional** design that this project **instantiates and measures**: (i) **feature-space *k*NN** graphs to obtain structure without device IP topology in the public release, (ii) a **GAT+GRU** stack for relational + short-term temporal context, (iii) **Flower FedAvg** on **Dirichlet-skewed** clients, (iv) **Captum** integrated gradients and **GAT** attention for dual explanations, and (v) **ECS-like JSON** for SOC-shaped output—each with **fixed hyperparameters** in *`config/experiment.yaml`* and versioned code.

**Rigor funnel.** (1) **Public splits** as released (`train` / `validation` / `test` CSVs), not a random re-split of one pool, so the benchmark’s intended separation is respected. (2) **StandardScaler** is **fit on the training file only** (first pass in *`src/data/preprocess.py`*), then **transforms** validation and test. (3) **Tuning and early stopping** use **train + validation**; **test** is for **one-shot** reporting. (4) **Federated** training never sends raw rows—only model parameters (see **Section 4.5**).

### 4.3 Dataset and Subset

**Secondary data, primary use.** CICIoT2023 (Pinto et al. 2023) is **captured and labelled by the dataset authors** in a testbed; this project **consumes** their **pre-recorded** CSVs as **secondary data** and applies a **project-specific** cleaning and learning pipeline. That distinction matters for ethics and for reproducibility: the thesis’s contribution is the **transformations and models**, not new packet capture.

**Row volume and class balance at flow level.** For the frozen run reported in this dissertation, each of the three public split files was processed to **500,000** flow rows, preserving both **benign** and **attack** traffic at roughly **2.3% / 97.7%** benign/attack in each split (see **Table 4** for exact counts). **Binary** labels are derived by mapping the dataset’s `label` column to $y \in \{0,1\}$ (benign vs. attack) using the `benign_labels` list in the YAML. **Forty-six** numeric flow feature columns (rates, byte statistics, flag counts, protocol indicators) are listed exhaustively in *`config/experiment.yaml`*; no IP or timestamp fields are used as model inputs in this build.

**Feature extraction after split order.** *`src/data/preprocess.py`* reads each split, drops/repairs `inf` / `NaN` in the 46 numerics, constructs `binary_label`, and applies the scaler **in file order** so that the **training** split **fits** the `StandardScaler` and the other splits only **transform**. That ordering is a **leakage control**: validation and test never influence the mean/variance used for $z$-scoring.

#### 4.3.1 Proposed pipeline stages, artefacts, and leakage controls

**Table 2** is the **master stage list**; later chapters **reference the same order** (Figure 7 is the visual analogue).

**Table 2: Proposed data and modelling pipeline stages (this project).**

| Stage | What happens | Main artefact / location | Leakage or scope control |
|------|----------------|---------------------------|---------------------------|
| 1 | Ingest public CSVs per split | `data/raw/{train,validation,test}.csv` | Splits are **as published** (not re-drawn from one pool) |
| 2 | Clean numerics, binarise labels, **fit scaler on train** only | *`src/data/preprocess.py`*  →  `data/processed/*.parquet` + `scaler.joblib` | Validation/test only **transformed** |
| 3 | (Optional) cap rows *per split* for MSc budget | Same parquet rows | **Documented** row cap; all models see same files |
| 4 | Build *k*NN windows and sequences per split | *`src/data/graph_builder.py`*, *`src/data/dataset.py`*; `data/processed/*_graphs.pt` | **Graphs built inside each split**; no cross-split windows |
| 5 | Train / tune baselines and GNN on **train**; monitor **val** | `results/checkpoints/`, `results/metrics/*.json` | **Test** not used in loss or model selection |
| 6 | Federated GNN: partition **training** rows across clients; FedAvg | *`src/federated/`*; round metrics JSON/CSV | Raw flows stay local; only weights exchanged |
| 7 | Final **test** metrics + CPU latency + comms + alerts | *`results/metrics/results_table.csv`*, *`src/siem/api.py`* | **Single** test evaluation policy (**Chapter 7**) |

*Italic paths* denote examine-facing artefacts. If another researcher omits a stage (e.g. skips graph building and trains only on flows), the experiment is **not** the one reported here.

#### 4.3.2 Repository layout and frozen outputs

Raw CICIoT2023 files are **not** committed. After Stage 2, **train / validation / test** parquet and the saved scaler live under `data/processed/`. **Graph tensor caches** and **metrics/figures** live under `data/processed/` and `results/` as listed in the repository `README` and **Chapter 13**. The dissertation’s numbers assume one **frozen** pass of the **same** config hash implied by the committed *`config/experiment.yaml`*.

#### 4.3.3 Empirical flow and graph sample sizes

**Table 4** records **empirical sizes** (from `results/metrics/dataset_stats.json`) for the run underlying **Chapter 8**. Flow row counts are **per-split totals**; graph counts follow the configured **window = 50 flows**, **sequence length = 5**, and **knn_k = 5** (Table 3).

**Table 4: Empirical sample sizes: flows, graph windows, and sequences.**

| Split | Flow rows (total) | Benign flows | Attack flows | Graph windows (single-split build) | Sequences (length-5) |
|--------|--------------------|-------------|-------------|------------------------------------|------------------------|
| Train | 500,000 | 11,582 | 488,418 | 924 | 920 |
| Validation | 500,000 | 11,676 | 488,324 | 932 | 928 |
| Test | 500,000 | 11,772 | 488,228 | 938 | 934 |

*Notes.* **Windows** are constructed with **per-class** pools and shuffling (see **Section 5.3**); the sequence count is slightly below the window count because the tail of the shuffled list may not form a full length-5 sequence. All models that report **“sequence-level”** GNN metrics consume these sequences; **RF/MLP** report **per-flow** metrics on 500,000 test flows each.

### 4.4 Models and Loss Functions

Three **model families** share the same **label definition** and **splits** but differ in **input geometry**: two are **feature-vector** classifiers; one is a **graph-sequence** classifier.

- **Random Forest** — *tabular*: one **standardised** 46-vector per flow, `sklearn.ensemble.RandomForestClassifier` in *`src/models/baselines.py`*; **out-of-bag** / train metrics are not used for test claims; **test** reporting is the contract in **Section 7.7**.

- **MLP** — *tabular*: feed-forward on 46-d inputs, `MLPBaseline`, trained with **Adam** and **mini-batch** cross-entropy (see Table 3).

- **Dynamic GNN (GAT  +  GRU)** — *graph–sequence*: for each of five windows, a **GAT** stack produces a graph embedding; a **GRU** consumes the **ordered** list of five embeddings; a linear head outputs **logits** for the binary case (`DynamicGNN` in *`src/models/dynamic_gnn.py`*). An ablation **mean-pools** time instead of a GRU (**Table 10**).

#### 4.4.1 Hyperparameters, tuning funnel, and training budget (after split)

**After** the public splits and parquet files exist, **all** model-specific knobs are taken from a **single** YAML (Table 3). **Early stopping** (patience in Table 3) is driven by **validation** performance; **test** is never used to pick hyperparameters. *Fine-tuning* in this project means **that validation-monitored training loop** with **fixed** architecture widths—there is no manual test-set search.

**Table 3: Hyperparameters, training budget, and federation (frozen in *`config/experiment.yaml`)*.**

| Group | Parameters | Value |
|--------|------------|--------|
| Reproducibility | Global seed | 42 (multi-seed study: 42, 123, 456 in **Section 8.9**) |
| Graph build | `window_size`, `knn_k`, `sequence_length`, `minority_stride` | 50, 5, 5, 25 |
| FL | `num_clients`, `num_rounds`, `local_epochs`, `alpha` (Dirichlet) | 3, 10, 2, 0.5 |
| RF | `n_estimators`, `max_depth` | 200, 20 |
| MLP | `hidden_dims`, `dropout`, `lr`, `epochs`, `batch_size` | [128, 64, 32], 0.2, 0.001, 30, 256 |
| Dynamic GNN | `node_dim`, `hidden_dim`, `num_gat_layers`, `gat_heads`, `gru_hidden`, `dropout`, `lr`, `epochs`, `batch_size` | 46, 64, 2, 4, 64, 0.2, 0.001, 30, 16 |
| Class imbalance | `class_weight_auto` (train) | true |
| Stopping | `early_stopping_patience` | 5 (epochs) |

#### 4.4.2 Objectives, losses, and federation rule

**Flow-level classifiers (RF, MLP).** For label $y_i \in \{0,1\}$ and model probability $p_i = P(y_i{=}1\mid x_i)$, **binary cross-entropy** is
$$
\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \bigl( y_i \log p_i + (1-y_i)\log(1-p_i) \bigr) \, .
$$
`class_weight` in **sklearn** and the **PyTorch** loss (if weighted) reweight under-represented benign flows without changing the **split** itself.

**Graph-sequence GNN.** Each training item is a sequence of $T\!=\!5$ window graphs with **sequence label** $y$ (any-window attack rule; **Section 5.3**). The same $\mathcal{L}_{\text{CE}}$ applies to the **sequence** logits. **Federated** training minimises the **same** local $\mathcal{L}_{\text{CE}}$ on each client; the server **FedAvg**-aggregates **parameters** (McMahan et al. 2017):
$$
\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \, \theta_{t+1}^{(k)} \, ,
$$
where $n_k$ is the client $k$ **training** row or batch count and $n\!=\!\sum_k n_k$.

**Algorithm 1 (informal).** *k*NN window graph for one window: for each of $W$ flow rows, compute **Euclidean** distances in 46-d feature space, connect each node to top-$k$ neighbours **both ways**, return a PyG `Data` object (*`src/data/graph_builder.py:flows_to_knn_graph`*).

**Algorithm 2 (informal):** *FedAvg server step* — after each round, **average** client weight tensors with weights $\frac{n_k}{n}$; broadcast *`dynamic_gnn`* state (*`src/federated/`*) — no raw $x_i$ uploaded.

*Italic references:* training loops use *`src/models/trainer.py`*; metrics use *`src/evaluation/metrics.py`*. The **test-set** “accuracy” claims in the results chapter are the **F1 / ROC-AUC** family from **Section 7.5**, not a single unbalanced accuracy.

### 4.5 Federated Learning Setup

**FedAvg** (McMahan et al. 2017): clients train locally and send **weights**; the server **averages** and broadcasts the global model - no raw flows cross sites (**Section 2.6**). This project uses **Flower** with **three** clients, **ten** rounds, and **two** local epochs per round (timelines and risks: **Chapter 3**; wiring: **Section 6.6**). The **same** GNN architecture is trained centrally and federally for a fair comparison. **Communication cost** is estimated from serialized **float32** parameter traffic (**Chapter 8**).

**Non-IID simulation:** training rows are partitioned across clients with a **Dirichlet** distribution over **class proportions** (**alpha = 0.5**) so client label mixes differ (stronger than IID sharding; weaker than adversarial splits). Rationale: reflect sites with different exposure while staying trainable within the MSc budget.


### 4.6 Explainability

Two forms of explanation are used:

- **Integrated Gradients (Captum):** Applied to the chosen model (e.g. the dynamic GNN or the MLP) to attribute the prediction to input features. The top contributing features are extracted and included in the alert.
- **Attention weights:** For the GAT-based model, attention weights over edges or neighbours are used to identify which flows or connections the model focused on. These can be summarised as “top flows” or “top edges” in the alert.

Explanations are produced for selected predictions (e.g. positive alerts). The top-k (k=5) features and flows are included in each alert to keep the explanation concise. If computing explanations for every prediction is too slow, they are applied only to a subset (e.g. high-confidence alerts) and this is noted as a limitation.


### 4.7 Evaluation Plan

**Formulae and definitions** for precision, recall, F1, FPR, and ROC-AUC are given **once** in **Section 7.5**. Here, the plan only **maps** measurements to questions.

Evaluation is designed to answer the three sub-questions and support the main research question.

- **Metrics:** Precision, recall, F1-score, ROC-AUC, and false positive rate on the fixed test set. A confusion matrix is reported.
- **Model comparison:** Centralised training: Random Forest, MLP, and dynamic GNN are compared on the same splits. Federated training: the same GNN (or MLP) is trained with FedAvg and compared to its centralised version.
- **Time and cost:** Time-window analysis (e.g. performance across different window sizes or positions), federated round-by-round performance, approximate communication cost (bytes), and CPU inference time per sample or per batch.
- **Explainability and SOC use:** Three to five example alerts are generated with full explanations (top features and flows). These are discussed in terms of whether they would help a SOC analyst triage (e.g. whether the highlighted flows and features are interpretable and actionable).

Contingencies match **Chapter 3** (subset size, rounds/clients, optional subset of explained alerts). **Linkage to questions:** precision, recall, F1, ROC-AUC, and FPR → **sub-question 1** (and central leg of **sub-question 2**); per-round FL metrics and comms cost → **sub-question 2**; example alerts → **sub-question 3**; CPU inference latency → **main question** (edge deployability).

### 4.8 Software and Tools

The methodology relies on a small set of mainstream Python libraries chosen for stability, documentation, and CPU compatibility. The full version pin list is in the repository `requirements.txt`; this section names the components and the role each plays in answering the research questions. Module-level wiring and code excerpts are in **Chapter 6**.

- **Python 3.10** is the implementation language for all data, model, federated, explainability, and API code.
- **PyTorch** and **PyTorch Geometric** are used to build the GAT layers and the dynamic GNN with sequence encoding (**Sections 6.4 and 6.5**).
- **scikit-learn** provides the Random Forest baseline, the **StandardScaler** (used in *`preprocess.py`* for parity with sklearn conventions), and metric helpers (**Sections 4.3 and 6.3–6.5**). **Splits** are **not** created by `train_test_split` in this build—the public CIC **train/validation/test** files are the splits.
- **Flower** is used for the federated learning server and client implementation with FedAvg (**Section 6.6**).
- **Captum** is used for Integrated Gradients on the trained model (**Section 6.7**).
- **FastAPI** is used to serve the prediction and explanation endpoint that produces ECS-like JSON alerts (**Sections 6.8 and 6.9**).
- **Matplotlib**, **Pandas**, and **NumPy** are used for figure generation, table preparation, and numeric handling.
- **Pygments** is used to render dark-theme code screenshots for the implementation chapter and for **Appendix C**.
- **Git** and **GitHub** are used for version control of the project repository linked in **Appendix D**.

Configuration is externalised in **`config/experiment.yaml`** so that hyperparameters (window size, *k*NN *k*, GAT heads, FL rounds and clients) can be changed without code edits. All experiments use seed **42** unless overridden by the multi-seed script. This combination of fixed seed, externalised config, and named scripts is what makes the methodology in this chapter reproducible from the codebase.

### 4.9 Chapter Summary

Chapter 4 has fixed the **Research Method** contract: **secondary** CICIoT2023 use, **leakage-aware** preprocessing (**Tables 2–4**), **YAML-frozen** hyperparameters and **post-split** tuning funnel (**Table 3**), **loss and FedAvg** equations (**Section 4.4.2**), and pointers to **federation**, **XAI**, **evaluation**, and **tools**. **Chapter 5** gives the design-level pipeline and graph semantics; **Chapter 6** gives deployment code and **Table 5** testbed; **Chapter 7** schedules multi-scenario runs (**Table 6**).

---

## Chapter 5 – Design

### 5.1 Chapter Overview

This chapter presents the system design in the same order as the narrative: the **end-to-end research pipeline** first (**Figure 7**), then **authoritative graph semantics** and windowing rules (**Section 5.3**), then **architecture** (**Figure 8**) and a **conceptual *k*NN** schematic (**Figure 9**) that match the implementation in Chapter 6. Deployment-facing outputs use ECS-like JSON alerts.


### 5.2 Pipeline, Alerts, and Deployment (Conceptual)

**Figure 7** gives the **stage-level** view: preprocessing, graph construction, central or federated training, explainability, and SIEM-style alerting on CPU. Windowing, *k*NN *k*, and sequence labelling are specified in **Section 5.3**; module-level wiring is in **Chapter 6**.

Alerts are output in a **SIEM-style JSON** format, aligned with ideas from **ECS** where useful: event type, timestamp, source/target information, label, score, and an **explanation** field (top features / top flows). The runtime path is a **FastAPI** service that runs inference (and optionally explanations) on **CPU**, matching the edge objective.

The pipeline view also defines responsibility boundaries between stages. Preprocessing and graph building are deterministic data preparation stages, training is the learning stage (central or federated), and alert formatting is a post-inference stage that should not modify prediction values. This separation helps verification because faults can be traced by stage rather than by inspecting one large monolithic script.

A second design value is communication clarity. By fixing the stage contract early, implementation and evaluation chapters can reference the same pipeline assumptions without redefining them. This reduces chapter drift and helps keep figure interpretation consistent.


![Figure 7: Research pipeline](assets/figure1_pipeline.png)

**Figure 7: Research pipeline  -  from raw IoT flow data to explainable SIEM alerts** *(Chapter 5, Section 5.2)*

*Source: Author’s own diagram (`scripts/generate_figure1.py`); reflects the implementation and data path described in Chapters 5–6.*

### 5.3 Graph Design (Flows to kNN Snapshots)

**Authoritative section:** every other mention of window/*k*/sequence rules in this thesis points here or to **`src/data/graph_builder.py`** / **`dataset.py`** (**Section 6.4**).

The graph is built from flow data to capture structural relationships between network flows within each observation window.

- **Nodes:** Each node represents one flow record. The node feature vector consists of the 46 pre-computed flow-level features provided by the CICIoT2023 dataset (e.g. packet rate, byte count, flag counts, protocol indicators, statistical aggregates). Device-level identifiers (IP addresses) are not available in the publicly released version of this dataset, so a device-based graph design (where nodes are devices and edges are flows between them) was not feasible. Instead, a feature-similarity approach is adopted, following the principle demonstrated by (Ngo et al. 2025) that attribute-based graph construction can be effective for intrusion detection when topology information is absent.
- **Edges:** For each window, a k-nearest-neighbour (kNN) graph is constructed in feature space. Each flow is connected to its *k* most similar flows (by Euclidean distance on the 46 features), producing undirected edges. This creates a graph structure where flows with similar characteristics are linked, enabling the GNN to learn from local neighbourhoods of related traffic patterns. The choice of *k* affects the graph density: too small and the graph may be too sparse for effective message passing; too large and computation increases. *k* = 5 was chosen as a balance (see sensitivity analysis in Chapter 8).
- **Windows and sequences:** Flows are grouped into fixed-size windows (e.g. 50 flows per window). To handle extreme class imbalance at flow level, the implementation (`src/data/graph_builder.py`) builds each window **only from flows in a single class pool** (benign or attack); the **graph-level label is that pool’s class** (within a window all flows share the same label, so this matches a unanimous vote but is defined by construction, not by aggregating mixed labels). Benign and attack windows are balanced, then **shuffled** into one list. Training samples are **sequences of five consecutive graphs in that list**; the **sequence-level label** is attack if **any** of the five windows is attack-labeled (`GraphSequenceDataset` in `src/data/dataset.py`). The GRU reads the five graph embeddings in order. Five windows was chosen to balance context with memory and training time.

This design is kept simple so that it can be implemented and tested within the project scope. The kNN approach applies to any flow-feature dataset regardless of whether device identifiers are present. More complex designs (e.g. device-based graphs when IPs are available, multi-hop temporal aggregation) could be explored in future work.

### 5.4 Research Design and System Architecture

**Figure 8** summarises the **end-to-end research design**: public benchmark flows are preprocessed and turned into temporal graphs; the dynamic GNN can be trained centrally or through Flower FedAvg on non-IID clients; the same checkpoint path supports Captum explanations and FastAPI scoring into SIEM-shaped JSON. It is a **conceptual** diagram (not a formal UML class model); a full UML component diagram can be pasted into the Word submission if the module asks for it.

![Figure 8: Research design system architecture](assets/research_design_system.png)

**Figure 8: Research design  -  data flow, federated training, and edge alerting** *(Chapter 5, Section 5.4)*

*Source: Author’s own diagram; generated for this dissertation (`scripts/draw_research_design.py`).*

### 5.5 Conceptual Illustration: Similarity-Based Graph in One Window

Public CICIoT2023 releases do not always include device-level topology, so **attribute / similarity-based** edges are a standard way to obtain a graph for a GNN (Ngo et al. 2025). **Figure 9** is an **author-original** schematic (drawn with **Matplotlib**, following the library’s documented style sheets  -  see [Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)): each point is one flow in a window; lines show **k** nearest neighbours in a **2-D projection** of feature space (the real system uses **46** dimensions and Euclidean distance, **Section 5.3**). It is **not** a screenshot from another paper; it only visualises the **same construction principle** as the implementation.

![Figure 9: Conceptual kNN similarity graph](results/figures/similarity_knn_concept.png)

**Figure 9: Conceptual kNN similarity graph within one observation window** *(Chapter 5, Section 5.5)*

*Source: Author’s own diagram (`scripts/generate_similarity_knn_concept.py`). Concept aligns with attribute-based graph construction for IoT intrusion detection (Ngo et al. 2025); image is not reproduced from third-party publications.*

### 5.6 Chapter Summary

The design encodes **relational** (*k*NN), **temporal** (window sequences), and **SOC-facing** outputs in one architecture.

---

## Chapter 6 – Implementation and System Development

### 6.1 Chapter Overview

This chapter describes how the prototype was implemented: code structure, libraries, training and federated loops, Captum integration, alert formatting, and the FastAPI scoring endpoint. In this project, implementation detail matters a lot, because the claim is not only model accuracy, it is full pipeline feasibility under CPU limits. So I explain where each major function lives and how the modules connect.


The implementation is organised under `src/` by responsibility (data, models, federated learning, explainability, SIEM output, evaluation) and is driven by scripts that reproduce end-to-end runs from configuration (`scripts/run_all.py` and related scripts). Configuration is externalised in YAML so experiments can be re-run without code changes.

### 6.2 Environment and Tools

**Table 5** records the **testbed** used for the reported CPU-only training, evaluation, and API latency. It is a **virtual/laboratory** setup (author workstation) rather than a production SOC; the point is to show **reproducible** hardware class and the absence of GPU shortcuts.

**Table 5: Testbed, storage, and runtime stack (Research Method Deployment).**

| Resource | Stated configuration for this project |
|----------|----------------------------------------|
| Host OS | Windows 10, x64; single-user **development** machine |
| Runtime | **Python 3.10**; virtual environment with pinned dependencies (*`requirements.txt`*) |
| CPU | x86-64 **multi-core** (CPU-only: **no** discrete GPU in any reported run) |
| RAM | 16 GB system RAM (sufficient to hold 500k-row parquets, graph batches, and training at batch sizes in **Table 3**) |
| Disk | **SSD** (NVMe-class) for repository, `data/raw/`, `data/processed/`, `results/` (HDD is **not** required for the frozen subset) |
| Network / traffic | **Pre-recorded** CICIoT2023 files only; no live **traffic generation** in the testbed for training |
| Isolation | Optional **Docker** for FastAPI demo (*`README.md`*); all metrics reproducible on bare Windows + Python |

The system is implemented in Python 3.10. **PyTorch 2.x** is used for the neural models (GNN and MLP), and **PyTorch Geometric (PyG)** provides graph operations and GAT layers. **Scikit-learn** is used for Random Forest and for metrics (precision, recall, F1, ROC-AUC, confusion matrix). The **Flower** framework is used for federated learning (client and server logic). **Captum** is used for Integrated Gradients. **FastAPI** is used for the REST API. The code is structured so that data loading, graph building, training, and inference can be run separately or together. All experiments are run on **CPU** to match the edge deployment goal; training time is therefore longer than with a GPU but remains feasible for the chosen subset.

Configuration is managed via *`config/experiment.yaml`* (window size, kNN *k*, sequence length, model hyperparameters, and FL rounds/epochs) — **Table 3** in **Section 4.4.1** is the one-line summary of those knobs.

### 6.3 Data Loading and Preprocessing

The CICIoT2023 subset is loaded from CSV (or the dataset’s provided format). The public flow tables provide the **46 numeric features** used in this project plus labels; optional endpoint columns are retained only when present in the schema. Missing values are handled, labels are binarised (benign vs. attack), and fixed train/validation/test splits are produced with a fixed seed for reproducibility.

Preprocessing includes feature standardisation (zero mean, unit variance) using statistics computed only on the training set, then applied to validation and test to avoid leakage. For the GNN, flows are windowed and converted to graphs as in **Section 6.4**. For Random Forest and MLP, **the same processed splits** are used in tabular form with **exactly one row per flow** (46 features + binary label), loaded by `load_processed_split` in `src/models/baselines.py` - not window aggregates. Graph-sequence construction uses `src/data/dataset.py`; configuration is in `config/experiment.yaml`.

### 6.4 Graph Construction

Implementation follows **Section 5.3** exactly: `build_graphs_for_split` and `flows_to_knn_graph` in **`src/data/graph_builder.py`** build single-class pools, balanced windows, shuffle, Euclidean **k**NN with bidirectional edges, and graph batches consumed by **`GraphSequenceDataset`** in **`src/data/dataset.py`**. Default **k** and window size come from **`config/experiment.yaml`**; sensitivity to other (*window*, *k*) pairs is in **Section 8.8**. **Baselines** read **one row per flow** from processed parquet via **`load_processed_split`** (`src/models/baselines.py`) - not graph windows.


### 6.5 Model Implementation

**Random Forest:** Implemented with scikit-learn on **one row per processed flow** (46 features + label). The implementation uses 200 trees and max_depth=20; class weights address flow-level imbalance in the training parquet.

**MLP:** Feed-forward network with three hidden layers (128, 64, 32), ReLU, dropout 0.2; **two output logits** for benign vs. attack, trained with **cross-entropy** and Adam (learning rate 1e-3), matching `MLPBaseline` and the training loop in `scripts/run_all.py`.

**Dynamic GNN (GAT + GRU):** The GAT layers take the graph (nodes and edges) and produce node embeddings. The GAT uses 4 attention heads and a hidden dimension of 64. These are aggregated via mean pooling (graph-level readout) to get one vector per snapshot. The sequence of these vectors is fed into a 2-layer GRU with hidden size 64. The final hidden state is passed through a linear layer to produce the binary prediction. The implementation uses PyTorch and PyG for the GAT. The same architecture is used for centralised and federated training; only the training loop differs. Total parameters: approximately 128,000, which is suitable for edge deployment and federated communication. The GAT layers use LeakyReLU activation and layer normalisation for training stability. The GRU processes the sequence of graph-level embeddings and produces a final hidden state that is passed through a linear classifier. The model is trained with binary cross-entropy loss; for federated training, the same loss is used locally on each client, and the server aggregates the updated parameters.

### 6.6 Federated Learning (Flower)

Federated training uses Flower with FedAvg: clients train locally on their partitions for a fixed number of epochs per round and send updated weights to the server; the server aggregates updates and broadcasts the new global model. No raw data is exchanged - only parameters - so communication cost can be approximated from float32 weight traffic (reported in Chapter 8). Implementation details are in `src/federated/`.


### 6.7 Explainability

**Integrated Gradients:** For a given input (e.g. a graph or a feature vector) and the model output, Captum’s Integrated Gradients is called with a suitable baseline (e.g. zero vector or mean feature vector). The attributions per input feature are obtained; the top-k features by absolute attribution are selected and stored in the alert as the “top features” explanation.

**Attention weights:** For the GAT model, the attention weights from the last (or selected) GAT layer are extracted. They indicate how much each edge or neighbour contributed. These are mapped back to flow identifiers (e.g. source–destination pairs) and the top flows are written into the alert. If the model has multiple layers, a simple strategy (e.g. average or use the last layer) is applied and documented.

Explanations are attached to the alert JSON. The number of integration steps (default 50) trades off attribution accuracy with computation time. The `src/explain/explainer.py` module wraps both methods and produces a unified explanation object. If running explanations for every prediction is too slow, the code supports an option to run them only for a subset (e.g. when the model confidence is above a threshold).

### 6.8 Alert Generation and SIEM-Style Output

When the model predicts an attack (or a specific attack type), an alert object is built. It includes: event type (e.g. “alert” or “detection”), timestamp, source and destination (e.g. IP or device ID), predicted label, confidence score, and an explanation object. The explanation object contains the list of top features (from Integrated Gradients) and/or top flows (from attention). The structure follows ECS-like conventions where possible (e.g. event.category, event.outcome, and custom fields for explanation). The alert is returned as JSON. This format is suitable for ingestion into a SIEM or for display in a SOC dashboard. The `src/siem/alert_formatter.py` module implements the ECS-like structure with fields such as `event.category`, `event.outcome`, and a custom `explanation` object holding `top_features` and `top_nodes` or `top_flows`. Severity levels (low, medium, high) are derived from the confidence score.

### 6.9 FastAPI Deployment and CPU Inference

A FastAPI application is set up with an endpoint that accepts input (e.g. a single flow, a batch of flows, or a pre-built graph). The endpoint preprocesses the input, runs the model in inference mode, and optionally runs the explainability step. The response is the alert JSON. CPU inference time is measured (e.g. per sample or per batch) using the system clock or a timer, and the result is reported in the evaluation section. No GPU is required; the design targets edge devices with CPU only. The API can be run locally or in a container for demonstration.

The FastAPI app (`src/siem/api.py`) loads the trained model from checkpoint at startup. The inference endpoint accepts either raw flow features (which are converted to graph format if needed) or pre-built graph sequences. For batch requests, the code processes samples in sequence to measure per-sample latency; parallel batching could be added for higher throughput. The measured inference times (e.g. 22.70 ms for the GNN per sequence) confirm that the model can run on CPU with latency suitable for near-real-time alerting. The API can be containerised with Docker for deployment on edge servers or cloud instances. The `scripts/run_all.py` script orchestrates preprocessing, graph construction, baseline training, central GNN training, and metric/plot outputs for the centralised path. Federated training is run via the Flower entry points. The `scripts/generate_alerts_and_plots.py` script produces example alerts, the FL convergence plot, and additional figures. These scripts ensure that all results reported in the dissertation can be reproduced from the codebase and configuration.

### 6.10 Implementation Code Screenshots (Author’s Codebase)

The screenshots below show core fragments only (function bodies, the training-step loop, sequence forward head, Captum IG core, and the `/score` handler) - not full modules. They are auto-rendered with Pygments (`one-dark`) at the cited line ranges (`scripts/render_chapter6_code_screenshots.py`); Appendix C uses the same style for examiner code reference.

**`flows_to_knn_graph`  -  *k*NN edges and tensors (core)**

![Figure 10: `flows_to_knn_graph` core excerpt](results/figures/chapter6/fig_ch6_01_flows_to_knn_core.png)

**Figure 10: `flows_to_knn_graph` (core)  -  `NearestNeighbors`, bidirectional edges, PyG `Data`** *(Chapter 6, Section 6.10)*

*Source: `src/data/graph_builder.py`, lines 37–58.*

**`build_graphs_for_split`  -  stratified pools (core)**

![Figure 11: `build_graphs_for_split` core excerpt](results/figures/chapter6/fig_ch6_02_stratified_split_core.png)

**Figure 11: `build_graphs_for_split` (core)  -  class pools, balance, merge, shuffle** *(Chapter 6, Section 6.10)*

*Source: `src/data/graph_builder.py`, lines 106–123.*

**`train_one_epoch`  -  central GNN training loop**

![Figure 12: `train_one_epoch` excerpt](results/figures/chapter6/fig_ch6_03_train_one_epoch.png)

**Figure 12: `train_one_epoch`  -  class-weighted cross-entropy, backward pass, optimiser step** *(Chapter 6, Section 6.10)*

*Source: `src/models/trainer.py`, lines 39–61.*

**`DynamicGNN.forward`  -  sequence to logits**

![Figure 13: `DynamicGNN.forward` excerpt](results/figures/chapter6/fig_ch6_04_dynamic_gnn_forward.png)

**Figure 13: `DynamicGNN.forward` (core)  -  stack window embeddings, GRU vs. mean-pool ablation, classifier** *(Chapter 6, Section 6.10)*

*Source: `src/models/dynamic_gnn.py`, lines 84–97.*

**`_ig_wrapper`  -  Integrated Gradients (core)**

![Figure 14: `_ig_wrapper` excerpt](results/figures/chapter6/fig_ch6_05_integrated_gradients_wrapper.png)

**Figure 14: `_ig_wrapper` (core)  -  Captum `IntegratedGradients` on the last window’s node features** *(Chapter 6, Section 6.10)*

*Source: `src/explain/explainer.py`, lines 34–50.*

**`POST /score`  -  FastAPI handler (core)**

![Figure 15: FastAPI `POST /score` core excerpt](results/figures/chapter6/fig_ch6_06_fastapi_score_core.png)

**Figure 15: `POST /score` (core)  -  build graphs, `explain_sequence`, ECS-style alert, timing** *(Chapter 6, Section 6.10)*

*Source: `src/siem/api.py`, lines 67–89.*

Together, these figures summarise the **data → train → forward → attribute → deploy** path; federated local training reuses the same `train_one_epoch` step inside `GNNFlowerClient` (**Appendix C**).

### 6.11 Reproducibility Note

The implementation is fully script-driven from `config/experiment.yaml` (window size, *k*NN *k*, GAT heads, sequence length, FL rounds and clients, Dirichlet alpha, seed). All main experiments use seed **42**; the multi-seed study in **Section 8.9** uses **42, 123, 456**. Entry points are: `scripts/run_all.py` (preprocessing, baselines, central GNN, evaluation), `src.federated.run_federated` (Flower server and clients), `scripts/run_ablation.py`, `scripts/run_sensitivity_and_seeds.py`, and `scripts/generate_alerts_and_plots.py`. Every table and figure is backed by a CSV/JSON under `results/metrics/` or an image under `results/figures/`. All runs are CPU-only.

### 6.12 Chapter Summary

Implementation delivers a **modular** codebase with scripted **reproducibility**, covering data through to **SIEM-style** JSON and **CPU** inference.

---

## Chapter 7 – Testing and Evaluation

### 7.1 Chapter Overview

This chapter specifies the evaluation protocol: the experimental setup, formal metric definitions, dataset and sequence statistics, and the comparison rules applied consistently across models. The purpose is to fix the “test contract” so Chapter 8 can report results as evidence rather than opinion.


### 7.2 Evaluation Scope

Evaluation answers the **three sub-questions** and supports the **main question** (**Section 1.3**). **Setup, hyperparameters, and federated heterogeneity** are in **Section 7.4**; **metric definitions** in **Section 7.5**; **scale statistics** in **Section 7.6**; **decision rules** for comparing models in **Section 7.7**. The plan was fixed **before** final test reporting so metrics were not tuned to the test set.

#### 7.2.1 Pass Criteria

To make the evaluation a pre-registered contract rather than a post-hoc summary, the following pass criteria were defined before any test-set numbers were generated. Each criterion is tied directly to a research question:

- **Pass-1 (sub-question 1):** the dynamic GNN achieves **F1 ≥ 0.95** and **ROC-AUC ≥ 0.95** on the central test set, and matches or exceeds the better of the two baselines (Random Forest, MLP) on F1.
- **Pass-2 (sub-question 1):** the dynamic GNN reports a **lower or equal false-positive rate** than the strongest baseline on the central test set.
- **Pass-3 (sub-question 2):** the **federated** GNN trained with Flower FedAvg over three non-IID clients reaches an F1 within **0.02** of the centralised GNN on the central test set within **ten** rounds.
- **Pass-4 (sub-question 2):** total federated **communication** stays within an order of magnitude consistent with edge-network feasibility (target **≤ 100 MB** over ten rounds for the chosen architecture).
- **Pass-5 (sub-question 3):** at least **three** example alerts are produced with a complete explanation payload (top features and top flows) in **ECS-like JSON**.
- **Pass-6 (main question):** mean **CPU** inference latency per five-window sequence stays within **edge-realistic bounds** (target **≤ 100 ms**).

#### 7.2.2 Fail Criteria

A run is treated as a **fail** for evaluation reporting if **any** of the following hold:

- **Fail-1:** evidence of **data leakage** between train, validation, or test splits.
- **Fail-2:** F1 of the dynamic GNN **below 0.90** on the central test set, or precision **below 0.90** with recall held at 1.0 (which would indicate degenerate "always attack" behaviour).
- **Fail-3:** federated training **does not converge** within ten rounds, or final federated F1 falls **more than 0.05** below the centralised GNN.
- **Fail-4:** alerts cannot be produced from the trained model through the FastAPI endpoint, or the explanation payload is empty for the chosen examples.
- **Fail-5:** CPU inference latency **exceeds 1 s** per sequence (which would void the edge-deployment claim).

The actual outcomes against these criteria are reported in **Chapter 8** and discussed in **Chapter 9**.

### 7.3 Test Schedule

**Table 6** is the **multi-scenario** execution map: it ties each **research question** element (baselines, central GNN, FL, ablation, sensitivity, multi-seed, deployment) to a **concrete output file** in `results/`. The schedule is **staged** so that the **test split** is not re-used for ad hoc tuning.

**Table 6: Evaluation scenarios, scripts, and frozen artefacts (Research Method Deployment).**

| Test phase | Project days | Output artefact | Outcome (vs. criteria) |
|---|---|---|---|
| Baseline RF and MLP on test split | 16–18 | `results/metrics/results_table.csv` (RF, MLP rows) | Reference set established |
| Central GNN training and test | 18–22 | `results/metrics/results_table.csv` (Central GNN row); `gnn_training_history.json` | Pass-1, Pass-2, Pass-6 |
| Federated GNN (10 rounds, 3 clients) | 22–26 | `results/metrics/fl_round_metrics.csv`; `results/figures/fl_convergence.png` | Pass-3, Pass-4 |
| Example alerts via FastAPI `/score` | 26–28 | `results/alerts/*.json` | Pass-5 |
| Ablation (GAT-only vs full) | 30–32 | `results/metrics/ablation_table.csv` | Stability evidence |
| Sensitivity grid (window x *k*) | 32–35 | `results/metrics/sensitivity_table.csv`; `results/figures/sensitivity.png` | Stability evidence |
| Multi-seed (42, 123, 456) | 35–38 | `results/metrics/multi_seed_summary.json` | Stability evidence |

### 7.4 Experimental Setup

All experiments use the same fixed train, validation, and test split from the CICIoT2023 subset. The random seed is fixed (e.g. 42) so that results are reproducible. Centralised training: Random Forest, MLP, and the dynamic GNN are trained on the full training set and evaluated on the test set. Federated training: the same GNN is trained with Flower and FedAvg across 3 clients; the global model is evaluated on the same test set after each round. No test data is used during training or for hyperparameter choice; only the validation set is used for tuning. Time windows for graph construction are set to a fixed length (e.g. 50 flows per window, 5 windows per sequence); sensitivity to window size can be checked in a time-window analysis if time allows.

The federated split uses a Dirichlet distribution (alpha = 0.5) to simulate non-IID clients (values also in **Table 3**). The central test set is held out and used only for evaluation. **Hardware and stack** for all runs are as in **Table 5**; model hyperparameters and training budgets are in **Table 3** and *`config/experiment.yaml`*, and the same splits are used across models to ensure fair comparison.

### 7.5 Metrics

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

### 7.6 Dataset and Experiment Statistics

The CICIoT2023 subset used in this project was selected to include a representative mix of benign and attack traffic. **Table 4** and `results/metrics/dataset_stats.json` (from `scripts/dataset_statistics.py`) give the **exact** per-split flow counts; at flow level the class distribution is approximately **2.3%** benign. The stratified windowing strategy produces balanced **graph-level** data (approximately **50%** benign, **50%** attack **windows**) despite this flow-level imbalance. After windowing, the GNN dataset comprises **920** training sequences, **928** validation sequences, and **934** test sequences (each sequence = five **consecutive** graph snapshots in the shuffled graph list; each snapshot = 50 flows; sequence label = attack if any snapshot is attack-labeled - see `GraphSequenceDataset`). The train/validation/test split ensures no overlap; the test set is used only for final reporting. The Random Forest and MLP baselines use the same processed flows in tabular format (one row per flow) on the same splits, so comparison is on the same underlying data; the GNN is evaluated at sequence level (one prediction per sequence of 5 windows). The federated split assigns each client a different proportion of classes via Dirichlet(alpha=0.5), simulating non-IID conditions where different sites observe different threat profiles.

### 7.7 Comparison Design

To answer sub-question 1 (dynamic graph vs. simple models), the dynamic GNN is compared to Random Forest and MLP on the same test set. If the GNN achieves higher F1 or AUC while keeping a reasonable false positive rate, that supports the use of graph and temporal structure. To answer sub-question 2 (federated vs. centralised), the federated model’s final metrics are compared to the centralised model’s. A small drop in performance may be acceptable if it stays within a few percent; a large drop would indicate that federated learning needs more tuning or more data. To answer sub-question 3 (usefulness of explanations), three to five example alerts are generated with full explanations and discussed in terms of whether the top features and flows would help a SOC analyst triage. No formal user study is conducted; the discussion is based on the author’s and supervisor’s judgment of interpretability and actionability.

### 7.8 Chapter Summary

Chapter 7 fixed the **setup**, **metrics**, **data scales**, and **comparison protocol** used for all models and federated runs. Chapter 8 reports the **numerical and graphical** outcomes only.

---

## Chapter 8 – Results Presentation

### 8.1 Chapter Overview

This chapter reports results as tables and figures with brief factual description; interpretation is in Chapter 9 and personal reflection in Chapter 10. Tables keep decimal precision (e.g. 0.9986) and the prose mirrors the same values as percentages. Artefacts are stored under `results/metrics/`, `results/figures/`, and `results/alerts/`.

### 8.2 Centralised Model Comparison (Sub-Question 1)

The dynamic GNN (GAT + GRU), Random Forest, and MLP were evaluated on the same test set. **Table 7** lists the same values as decimals; here they are stated explicitly as percentages for quick comparison: **Random Forest**  -  **Precision = 99.89%**, **Recall = 99.84%**, **F1 = 99.86%**, **ROC-AUC = 99.96%**; **MLP**  -  **Precision = 100.00%**, **Recall = 98.85%**, **F1 = 99.42%**, **ROC-AUC = 99.84%**; **Central GNN**  -  **Precision = Recall = F1 = ROC-AUC = 100.00%**; **Federated GNN** (evaluated centrally on the same test set)  -  **Precision = Recall = F1 = ROC-AUC = 100.00%**. False positive **rate** (benign predicted as attack): GNN **0.00%**; Random Forest **4.84%** (187 false positives); MLP **0.10%** (4 false positives). **Figure 16** shows the confusion matrix for the Dynamic GNN. **Figure 17** shows the ROC curve for the GNN. **Figures 17** and **18** show confusion matrices for Random Forest and MLP. **Figures 19** and **20** show ROC curves for RF and MLP. **Figure 27** compares inference time and F1 across models.

**Table 7: Model comparison on CICIoT2023 test set**

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| Random Forest | 0.9989 | 0.9984 | 0.9986 | 0.9996 | 46.09 |
| MLP | 1.0000 | 0.9885 | 0.9942 | 0.9984 | 0.66 |
| Central GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| Federated GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20.99 |

*Source: Author-derived metrics on the fixed CICIoT2023 subset (Pinto et al. 2023); `results/metrics/results_table.csv`; reproducibility in Chapter 13. Decimal columns match the prose **Key:** lines above (e.g. RF **F1 = 99.86%** ↔ 0.9986).*

![Figure 16: Confusion matrix for Dynamic GNN](results/figures/cm_gnn.png)

**Figure 16: Confusion matrix for Dynamic GNN on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot from test-set predictions; data CICIoT2023 (Pinto et al. 2023); `scripts/run_all.py`.*

![Figure 17: ROC curve for Dynamic GNN](results/figures/roc_gnn.png)

**Figure 17: ROC curve for Dynamic GNN on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot from test-set scores; data CICIoT2023 (Pinto et al. 2023); `scripts/run_all.py`.*

![Figure 18: Confusion matrix for Random Forest](results/figures/cm_rf.png)

**Figure 18: Confusion matrix for Random Forest on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al. 2023).*

![Figure 19: Confusion matrix for MLP](results/figures/cm_mlp.png)

**Figure 19: Confusion matrix for MLP on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al. 2023).*

![Figure 20: ROC curve for Random Forest](results/figures/roc_rf.png)

**Figure 20: ROC curve for Random Forest on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al. 2023).*

![Figure 21: ROC curve for MLP](results/figures/roc_mlp.png)

**Figure 21: ROC curve for MLP on test set** *(Chapter 8, Section 8.2)*

*Source: Author’s own plot; baseline from `scripts/run_all.py`; data CICIoT2023 (Pinto et al. 2023).*

The confusion matrices (**Figures 16, 18, and 19**) report TP, TN, FP, and FN counts per model. **Figure 16** shows **zero** FP and **zero** FN for the GNN on this test set. **Figures 17** and **18** show **187** FP for Random Forest and **4** FP for MLP (benign predicted as attack). The ROC curves (**Figures 17, 20, and 21**) plot TPR vs. FPR across thresholds; reported ROC-AUC values are in **Table 7**. Implications for SOC triage and model choice are discussed in **Chapter 9**.

**Figure 22** plots precision, recall, and F1 for the four models with a zoomed 0.97-1.00 y-axis. Both GNN runs sit on the 1.0 ceiling on all three metrics; MLP has a visible recall gap at 0.9885 and RF sits at 0.998+ on every metric.

![Figure 22: Per-class metrics across all four models](results/figures/per_class_metrics.png)

**Figure 22: Per-metric comparison across RF, MLP, Central GNN, and Federated GNN on the test set** *(Chapter 8, Section 8.2)*

*Source: Author's own plot (`scripts/generate_extra_result_figures.py`); data from `results/metrics/{rf,mlp,central_gnn,federated_gnn}_metrics.json`.*

### 8.3 Federated Learning (Sub-Question 2)

Federated training was run for **10 rounds** with **3 clients** and non-IID splits (Dirichlet **alpha = 0.5**). The global model was evaluated on the **same central test set** after each round. **Table 8** lists precision, recall, F1, and ROC-AUC per round; **Figure 23** plots F1 and ROC-AUC vs. round. **Key (same numbers as Table 8):** round 1 **F1 = 98.3%**, **ROC-AUC = 55.7%**; from round 2 **ROC-AUC = 100.0%**; **F1** reaches **100.0%** from round 7; round 6 **ROC-AUC = 97.3%** while **F1 = 99.5%**. The final round reports **F1 = ROC-AUC = 100.0%**, matching the centralised GNN on this test set.

Communication cost is approximated from float32 parameter size: **128,002** parameters, order of **~1.0 MB** per client upload/download per round; **~3.07 MB** per round aggregate messaging is reported, **~31 MB** total over 10 rounds with three clients. Interpretation (privacy vs. centralised parity, feasibility) is in **Chapter 9**.

**Table 8: Federated learning round-by-round metrics**

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

![Figure 23: FL convergence](results/figures/fl_convergence.png)

**Figure 23: Federated learning convergence (F1 and ROC-AUC vs. round)** *(Chapter 8, Section 8.3)*

*Source: Author’s own plot; `scripts/generate_alerts_and_plots.py` / federated training logs; data CICIoT2023 (Pinto et al. 2023).*

**Figure 24** is the federated GNN confusion matrix on the same test set after the final FedAvg round. Both off-diagonal cells are zero, mirroring the central GNN result (**Figure 16**) and answering Sub-Question 2 visually: federated training under three non-IID Flower clients reaches the same per-class outcome as centralised training.

![Figure 24: Federated GNN confusion matrix](results/figures/cm_federated_gnn.png)

**Figure 24: Confusion matrix of the federated GNN (Flower FedAvg, 3 clients, 10 rounds) on the test set** *(Chapter 8, Section 8.3)*

*Source: Author’s own plot (`scripts/generate_extra_result_figures.py`); reconstructed from `results/metrics/federated_gnn_metrics.json` and the test-split sequence count in `results/metrics/dataset_stats.json`.*

**Figure 25** shows the communication cost behind those metrics: ~3.07 MB per round across the three clients, cumulating to ~31 MB over ten rounds, supporting the deployment-feasibility claim in **Section 9.6**.

![Figure 25: Federated communication cost](results/figures/comm_cost.png)

**Figure 25: Federated learning communication cost  -  per round (bars) and cumulative (line)** *(Chapter 8, Section 8.3)*

*Source: Author’s own plot (`scripts/generate_alerts_and_plots.py`); data from `results/metrics/fl_rounds.json` (`comm_bytes`).*

#### 8.3.1 What Federated Learning Produced (Reading the Curves and the Table)

Round 1 reaches Recall = 100.0% and F1 = 98.3% but ROC-AUC = 55.7%, indicating poor score ranking early. From round 2 ROC-AUC = 100.0%, and F1 reaches 100.0% from round 7. Round 6 shows a small dip to ROC-AUC = 97.3% while F1 stays at 99.5% (a wobble rather than a collapse, consistent with non-IID FedAvg noise discussed in Chapter 9). The federated checkpoint matches the central GNN on the reported test metrics.

### 8.4 Central GNN Training Convergence

The centralised Dynamic GNN was trained for **6 epochs** (with early-stopping configured in the training script). **Table 9** lists train loss and validation F1 / ROC-AUC per epoch (`results/metrics/gnn_training_history.json`). Validation **F1 = ROC-AUC = 100.0%** from epoch 1 onward on this run; train loss falls from **0.484** to **0.0001** by epoch 4 and stays there through epoch 6.

**Table 9: Central GNN training history (loss and validation metrics)**

| Epoch | Train Loss | Val F1 | Val ROC-AUC |
|-------|------------|--------|-------------|
| 1 | 0.484 | 1.000 | 1.000 |
| 2 | 0.023 | 1.000 | 1.000 |
| 3 | 0.0002 | 1.000 | 1.000 |
| 4 | 0.0001 | 1.000 | 1.000 |
| 5 | 0.0001 | 1.000 | 1.000 |
| 6 | 0.0001 | 1.000 | 1.000 |

*Source: `results/metrics/gnn_training_history.json`; central GNN training run (`scripts/run_all.py`).*

**Figure 26** plots the training-loss column on a log y-axis. Loss falls from ~4.8 × 10⁻¹ at epoch 1 to ~1 × 10⁻⁴ by epoch 4, then plateaus. Combined with validation F1 = 1.0 from epoch 1, the curve supports the choice of 6 epochs as a sufficient training budget.

![Figure 26: Central GNN training loss curve](results/figures/training_loss.png)

**Figure 26: Central GNN training loss across six epochs (log scale)** *(Chapter 8, Section 8.4)*

*Source: Author's own plot (`scripts/generate_alerts_and_plots.py`); data from `results/metrics/gnn_training_history.json` (`train_loss`).*

### 8.5 Time-Window and CPU Inference (Sub-Question 2 and Deployment)

CPU inference times (reported in Table 7): Random Forest **46.09 ms** per sample; MLP **0.66 ms**; central GNN **22.70 ms**; federated GNN **20.99 ms** (GNN times are per **sequence** of 5 graph windows). **Figure 27** compares F1 and inference time across models.

![Figure 27: Model comparison](results/figures/model_comparison.png)

**Figure 27: Model comparison  -  inference time and F1-score** *(Chapter 8, Section 8.5)*

*Source: Author’s own plot; metrics from Table 7 and inference timings in `results/metrics/`; `scripts/generate_alerts_and_plots.py`.*

### 8.6 Example Alerts with Explanations (Sub-Question 3)

Five example alerts were generated with full explanations (top features from Integrated Gradients and top flows from attention weights), in the same order as `results/alerts/example_alerts.json`. Each example includes the predicted label, confidence, threat severity band, and the explanation object (`top_features`, `top_nodes`).

**Example 1 (True negative):** Predicted benign; score **0.163** (low severity). Top features: `Variance`, `Std`, `rst_count`, `Duration`, `AVG` (Integrated Gradients magnitudes as in `example_alerts.json`).

**Example 2 (True positive):** Predicted malicious; score **0.997**. Top features: `psh_flag_number`, `ICMP`, `rst_flag_number`.

**Example 3 (True positive):** Predicted malicious; score **0.996**. Top features: `rst_flag_number`, `ICMP`, `Protocol Type`.

**Example 4 (False positive):** Benign labelled as malicious; score **0.711** (medium). Top features: `Variance`, `rst_count`.

**Example 5 (False positive):** Benign labelled as malicious; score **0.945**. Top features: `Variance`, `Std`, `rst_count`.

Each record follows the ECS-like shape: event metadata, rule name, threat indicator, ML prediction and score, and `explanation` (`top_features`, `top_nodes`). Whether these fields are sufficient for SOC triage is discussed in **Chapter 9**.

### 8.7 Ablation Studies (Priority 1: Evidence)

To show that both the graph and the temporal parts of the model add value, one ablation was run: the same GAT-based model but with the GRU replaced by mean pooling over time (so the model sees each window’s graph embedding but does not model the sequence with an RNN). This variant is called “GAT only (no GRU)”. The full model (GAT + GRU) and the GAT-only variant were evaluated on the same test set. Table 10 summarises the results. To reproduce the ablation, run: `python scripts/run_ablation.py --config config/experiment.yaml`; results are saved to `results/metrics/ablation_gat_only.json` and `results/metrics/ablation_table.csv`.

**Table 10: Ablation on CICIoT2023 test set (centralised GNN variants)**

| Variant | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|---------|-----------|--------|-----|---------|----------------|
| Full (GAT + GRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| GAT only (no GRU) | 0.9923 | 1.0000 | 0.9961 | 1.0000 | 16.06 |

*Source: `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`; test split CICIoT2023 (Pinto et al. 2023).*

**Numeric summary (same as Table 10):** full (GAT + GRU)  -  **Precision = Recall = F1 = ROC-AUC = 100.00%**, inference **22.70 ms**; GAT-only (mean pool over windows)  -  **Precision = 99.23%**, **Recall = 100.00%**, **F1 = 99.61%**, **ROC-AUC = 100.00%**, inference **16.06 ms**.

![Figure 28: Ablation bar chart](results/figures/ablation_bar.png)

**Figure 28: Ablation comparison  -  full GAT+GRU vs. GAT-only (F1 and inference time)** *(Chapter 8, Section 8.7)*

*Source: Author’s own plot; `results/metrics/ablation_table.csv`; `scripts/run_ablation.py`.*

Interpretation of the temporal (GRU) vs. pooling trade-off is in **Section 9.3** and **Section 9.6**.

### 8.8 Sensitivity Analysis (Stability of Design Choices)

Sensitivity analysis checks whether the main results hold when key hyperparameters change. Two levers are: (1) **window size** (number of flows per graph snapshot), and (2) **k** in the kNN graph. The main experiments use window size **50** and **k = 5**. The pipeline was re-run for all combinations of window_size ∈ {30, 50, 70} and knn_k ∈ {3, 5, 7} with other settings unchanged; metrics were written to `results/metrics/sensitivity_table.csv` (script: `scripts/run_sensitivity_and_seeds.py`).

**Table 11: Sensitivity analysis  -  central GNN on test set (nine configurations)**

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

*Source: `results/metrics/sensitivity_table.csv`; `scripts/run_sensitivity_and_seeds.py`; test split CICIoT2023 (Pinto et al. 2023).*

Grid interpretation is in **Section 9.3**.

![Figure 29: Sensitivity heatmap](results/figures/sensitivity.png)

**Figure 29: Sensitivity of test F1 and ROC-AUC to window size and kNN *k*** *(Chapter 8, Section 8.8)*

*Source: Author’s own plot; `results/metrics/sensitivity_table.csv`; `scripts/run_sensitivity_and_seeds.py`.*

### 8.9 Multi-Seed Stability

To check that the central GNN result is not an accident of one random seed, the same training and evaluation procedure was repeated with seeds **42**, **123**, and **456** (other config unchanged). Summary statistics are in `results/metrics/multi_seed_summary.json`.

**Table 12: Multi-seed summary (central GNN, test set)**

| Seed | Precision | Recall | F1 | ROC-AUC | False positives (test) |
|------|-----------|--------|-----|---------|-------------------------|
| 42 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 123 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 456 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Mean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |  -  |
| Std | 0 | 0 | 0 | 0 |  -  |

*Source: `results/metrics/multi_seed_summary.json`; `scripts/run_sensitivity_and_seeds.py`; test split CICIoT2023 (Pinto et al. 2023).*

**Table 12** reports **mean F1 = 100.0%**, **mean ROC-AUC = 100.0%**, and **zero** standard deviation on this test split for all three seeds. Per-seed inference times in the raw logs vary with CPU timing; the headline **~22.7 ms** deployment figure is from the primary **seed 42** run (**Table 7**). What this implies for generalisation is discussed in **Chapter 9**.

### 8.10 Comparison with Prior Work on CICIoT2023

The **UWS guideline** asks results to be compared with significant existing works where applicable. **Table 13** places the headline numbers from this dissertation alongside reported metrics from three peer-reviewed studies that also evaluate machine-learning intrusion detection on **CICIoT2023** or directly comparable IoT flow benchmarks. The comparison is reported as headline F1 (or accuracy where F1 is not given), with the architectural family of each system. Numbers are taken from the cited papers; experimental setups differ across studies (binary vs. multi-class, sampling strategy, hardware), so the table is for **scope-positioning** rather than direct ranking.

**Table 13: Headline metrics  -  this dissertation vs. prior work on CICIoT2023 and adjacent IoT flow benchmarks**

| Study | Architectural family | Federated | Explainable | SIEM-shaped output | Headline metric (binary) |
|---|---|---|---|---|---|
| (Pinto et al. 2023)  -  dataset paper | RF / DNN baselines on CICIoT2023 | No | No | No | RF accuracy ~99% (binary) on full corpus |
| (Lazzarini et al. 2023) | MLP under FedAvg, IoT IDS | Yes (FedAvg) | No | No | Reported F1 in the upper-90% range under non-IID FedAvg |
| (Albanbay et al. 2025) | FL IDS for IoT, scaling study | Yes | No | No | Strong FL accuracy on IoT IDS at varying client counts |
| **This dissertation** | **Dynamic GNN (GAT + GRU)** | **Yes (Flower FedAvg, 3 clients, non-IID)** | **Yes (Captum IG + GAT attention)** | **Yes (ECS-like JSON via FastAPI)** | **F1 = ROC-AUC = 100.0% on the fixed CICIoT2023 subset (Table 7)** |

*Source: this dissertation; cited works in Chapter 11. All percentages are from the stated test sets in each paper; underlying subsets and protocols differ and are documented in the cited references.*

The cross-study reading reinforces the gap stated in **Section 2.8**: prior work covers detection accuracy and federated training separately; this dissertation's contribution is the **integration** of dynamic graph modelling, federated training, explainability, and SIEM-shaped output into one CPU-oriented prototype. Headline metrics here are subset-specific and discussed in **Section 9.4**.

### 8.11 Chapter Summary

Chapter 8 reported **metrics**, **plots**, and **stability tables**; **Chapter 9** interprets them against the research questions and limitations.

---

## Chapter 9 – Conclusion, Discussion, and Recommendations

### 9.1 Chapter Overview

This chapter interprets the results, relates findings to the literature, and summarises what the project achieved within its scope. It also records limitations and implications for SOC/edge deployment. Section 9.2 provides a conclusion, followed by discussion in Sections 9.3–9.10.

### 9.2 Structured Conclusion (Programme Format)

This dissertation addressed **SOC-oriented**, **CPU-edge** intrusion detection for **Software-Defined IoT** flow telemetry, where analysts need **accurate** models, **privacy-preserving** training where raw flows cannot be pooled, and **explainable** outputs that fit **SIEM-style** workflows. The work is grounded in **CICIoT2023** and in prior literature that often treats **GNNs**, **federated learning**, and **XAI** as separate tracks, not one deployable path.

The **main outcome** is an end-to-end prototype: *k*NN feature-similarity graphs over windowed flows, a **dynamic GNN** (GAT + GRU), **Flower FedAvg** with three clients, **Captum** attributions and attention cues folded into **ECS-like JSON** via **FastAPI**. On the **fixed subset and splits** documented in **Chapters 7-8**, the central and federated GNN matched **strong RF and MLP baselines** on headline accuracy while reporting **fewer false alarms** on the test split, **sub-23 ms** CPU inference per sequence, and **modest** federated communication over ten rounds. The evidence is supported by **ablation**, **sensitivity**, and **multi-seed** checks.

**Limitations** are explicit: a **45-day** MSc scope, a **subset** of a **lab** dataset, **small** federated topology, **no** formal SOC **user study**, and headline **100%** metrics that are **subset-specific** rather than universal.

**Outlook:** scaling data and clients, live-traffic validation, analyst-facing evaluation, tighter SIEM integration, and publication-oriented breakdowns (per attack type, stronger heterogeneity) are the natural next steps; they are expanded in **Section 9.9**. Together, the chapters show that the **research questions** are **answered** for a **documented prototype**, with honest limits and a **clear** path beyond the thesis.

### 9.3 Answering the Research Questions

**Main question:** The project set out to show how an explainable dynamic graph neural network, trained with federated learning, can detect attacks in Software-Defined IoT flow data and generate SIEM alerts for SOC use on CPU-based edge devices. The prototype demonstrates that this is feasible: the dynamic GNN can be trained on graph snapshots over time, federated learning can be applied with Flower and FedAvg, and alerts with explanations can be produced in an ECS-like JSON format. CPU inference time is measurable and can be kept within acceptable bounds for a subset of traffic. So the main question is answered in the sense that a working pipeline exists and has been evaluated; the extent to which it “performs well” depends on the metrics and comparison with baselines reported in **Chapter 8**.

**Sub-question 1 (dynamic graph vs. simple models):** The dynamic GNN did better than Random Forest and MLP. It got 100% F1 and ROC-AUC compared to 99.86% and 99.42% respectively. This supports the idea that graph structure and temporality add value for this kind of data. The baselines were competitive, which is consistent with CICIoT2023 flow features being well-engineered; the GNN's advantage lies in its lower false alarm rate (0% vs. 4.84% for RF and 0.10% for MLP) and its ability to exploit relational and temporal structure. The fact that the GNN achieves zero false positives while the RF has 187 and the MLP has 4 suggests that the graph-based representation helps the model distinguish between benign traffic that resembles attacks (e.g. high variance, unusual flag patterns) and genuine attacks. The kNN graph links similar flows, so the model can learn from neighbourhood patterns that flat classifiers cannot see.

**Sub-question 2 (federated learning):** The federated model’s performance matched the centralised model exactly (100% F1 and ROC-AUC). So federated learning is a viable option when raw data cannot be shared. The approximate communication cost (31 MB over 10 rounds) shows that the approach is suitable for resource-limited edge networks.

**Sub-question 3 (explanations for SOC triage):** The example alerts show that the system outputs top features and top flows. The highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number for attacks; Variance, Std for benign) are interpretable and match the type of traffic. For true positives, the explanations support triage by pointing to protocol-level anomalies; for false positives, the mixed feature profile helps analysts cross-reference before escalating.

**Ablation (Section 8.7):** The full model (GAT + GRU) reaches F1 = 1.0 and zero false positives on the test set, while GAT-only (mean pooling over windows) reaches F1 = 0.9961 with precision 0.9923  -  a small but clear gap. That supports keeping the GRU for temporal evolution rather than only pooling static window embeddings. The bar chart (Figure 28) also shows the latency trade-off (GAT-only is faster).

**Sensitivity (Section 8.8):** Across the nine (window, *k*) settings, F1 ranges from **0.9961** to **1.0000** and ROC-AUC from **0.9895** to **1.0000**. Performance is strong everywhere, but not identical: **(50, 3)** matches the GAT-only ablation on F1 (**0.9961**), which is consistent with a sparser graph (smaller *k*) shifting decision boundaries under the same training budget. The chosen default **(50, 5)** sits with **F1 = ROC-AUC = 1.0** and inference **~17 ms** in this grid. Several other cells also reach **F1 = 1.0** (e.g. all three settings at window 30; (50, 7); (70, 7)). Pairs **(70, 3)** and **(70, 5)** show the largest ROC-AUC dips (**~0.99–0.995**) while recall stays **1.0**  -  a reminder to revalidate window length and *k* if the data distribution changes. Overall, **(50, 5)** is well supported: full F1/AUC together with reasonable latency; Figure 29 makes the pattern visible at a glance.

**Multi-seed (Section 8.9):** Three seeds gave identical headline test metrics (F1 and AUC 1.0, zero false positives). That increases confidence that the main result is not a one-off random initialisation, though it does not remove dataset-specific optimism.

### 9.4 Strengths and Limitations

**Strengths:** The project delivers an end-to-end prototype. It goes from data to graph, model to federated training, explainability to SIEM-style alerts and FastAPI. The design matches practical SOC needs (explainable alerts, CPU-based deployment). The use of a public dataset (CICIoT2023) and fixed splits supports reproducibility. The comparison with baselines and the evaluation of federated learning provide evidence, not just claims. Putting multiple components (graph construction, GNN, federated learning, explainability, SIEM output) in one pipeline is a contribution. Most prior work looks at these in isolation. The choice of kNN feature-similarity graphs is justified by the absence of device identifiers. It is supported by (Ngo et al. 2025) and (Basak et al. 2025). The stratified windowing strategy deals with class imbalance in a principled way.

**Limitations:** Results are reported on a time-boxed MSc prototype and a subset of CICIoT2023, so generalisation beyond this benchmark is not guaranteed. The kNN graph is built from feature similarity (a proxy for structure when topology identifiers are absent); topology-based graphs may be preferable when device identifiers are available. Federated learning is demonstrated with a small simulation (3 clients, 10 rounds) rather than a large deployment. Explainability is evaluated through a small number of example alerts and supervisor/author judgement; a formal analyst study would strengthen the SOC-utility claim. Finally, the lab nature of CICIoT2023 and very high headline metrics on this split should be read as subset-specific rather than universal.

### 9.5 Use of University and Course Materials

This project drew on UWS programme materials and supervision throughout. **COMP11023 Research Methods and Design** shaped the design-science framing in **Chapter 4** and the test-contract reasoning in **Chapter 7**; **COMP11024 Master's Project** set the report structure, milestone timing, and the supervisor / moderator review pattern in **Section 3.7**; the **MSc Project Handbook 2025–26** and the **Updated Guideline for Final Report** provided the chapter list, marking weights, and front-matter requirements. The **UWS Library Harvard libguide** was the source for the citation style in **Chapter 11** and the library's databases supported the literature search in **Chapter 2**. **Dr Raja Ujjan** (supervisor) provided weekly technical guidance and reviewed interim drafts; **Muhsin Hassanu** (moderator) reviewed the interim submission; **Dr Daune West** supported programme-level milestones.

### 9.6 Practical Implications

The results have several practical implications for SOC and edge deployment. First, the Dynamic GNN's zero false positive rate on this test set reduces alert fatigue compared to Random Forest (187 false positives) and MLP (4 false positives). Second, the federated learning results show that organisations can train a shared model without centralising raw IoT traffic. This addresses privacy and regulatory concerns. Third, the CPU inference time (under 23 ms per sample in the primary run) means the model can run on edge devices without GPUs. Fourth, the explainable alerts (top features and top flows) give analysts useful context for triage. Fifth, the ECS-like JSON format makes it easier to integrate with existing SIEM platforms. The main caveat is that these results are from a lab dataset and subset. Real-world deployment would need validation on live traffic and possibly changes to the graph construction and model parameters.

The ablation (Section 8.7), sensitivity grid (Section 8.8), and multi-seed runs (Section 8.9) together act as the kind of stability evidence reviewers expect: not only headline accuracy but evidence that the architecture is shaped intentionally and that the numbers are not fragile to reasonable hyperparameter shifts. In practice, adoption would be phased  -  passive monitoring alongside the existing SOC workflow first, then assisted triage where explanation fields are used to prioritise analyst queues while escalation stays human-controlled  -  with periodic metric review and re-training windows to handle traffic drift.

**Societal and assurance value.** Resilient IoT underpins **critical services** and **public confidence** in connected infrastructure (ENISA 2017). A detector that is **federated** (reducing data centralisation), **explainable** (supporting accountable triage), and **CPU-deployable** (closer to resource-constrained sites) is aligned with data-protection expectations and the **analyst–time** limits discussed in the literature (**Section 2.4**). The work does **not** claim to resolve every societal cyber-risk question; it contributes a **reproducible, documented** method stack that can be **audited** and **extended** by others—supporting the academic integrity and **verifiable** evidence standards expected in a UWS final report.

### 9.7 Relation to Literature

The findings can be related back to the literature. The use of GNNs for network or security data is supported by work such as (Velickovic et al. 2018). This project shows that a dynamic GNN can be built and compared to baselines on IoT flow data. The GNN's better false positive rate (0% vs. 4.84% for RF) fits with the idea that graph structure helps the model tell attack patterns from benign outliers. (Han et al. 2025) and (Ngo et al. 2025) showed that graph-based and attribute-based construction can improve detection. This project confirms that a kNN feature-similarity approach works when device identifiers are absent.

Federated learning with FedAvg (McMahan et al. 2017) was shown to work with Flower. It achieved performance matching centralised training. This is consistent with (Lazzarini et al. 2023) and (Albanbay et al. 2025). They found that FedAvg can keep accuracy for IoT intrusion detection. The fast convergence (within 7 rounds) and modest communication cost (31 MB) support the feasibility of federated deployment on edge networks. Explainability via Integrated Gradients (Sundararajan et al. 2017) and attention (Velickovic et al. 2018) was added to the alert output. This matches the need for explainable security tools in the literature. The example alerts show that the highlighted features (e.g. psh_flag_number, ICMP, rst_flag_number) are interpretable and consistent with known attack signatures. (Alabbadi and Bajaber 2025) suggested this. The gap identified in the Literature Review (combining explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype) has been addressed within the stated scope and limitations.

### 9.8 Summary of the Project

This dissertation set out to design and build a small prototype system for detecting attacks in IoT network traffic. It uses an explainable dynamic graph neural network and federated learning. It also generates SIEM-style alerts that support SOC operations on CPU-based edge devices. The main research question was: how can such a system detect attacks in IoT flow data and produce alerts that are useful for SOC analysts?

Empirically, the pipeline met its stated targets on the fixed subset: the dynamic GNN matched or exceeded baselines on headline metrics (**Chapter 8**), federated training matched centralised GNN performance, and SIEM-style alerts with explanations were produced. Stability artefacts (ablation, sensitivity grid, multi-seed summaries) live under `results/metrics/` and are produced by `scripts/run_ablation.py` and `scripts/run_sensitivity_and_seeds.py` as documented in **Chapter 13**.

### 9.9 Recommendations for Future Work

- **Larger scale:** Use a larger subset of CICIoT2023 or other IoT datasets, more federated clients, and more attack types to strengthen the evidence. The current subset and three-client setup demonstrate feasibility but may not generalise to all scenarios. Scaling to more clients and more diverse data would test the stability of FedAvg under higher heterogeneity.
- **Real-world data:** Test on data from real IoT deployments (with appropriate permissions) to see how the model and explanations perform outside the lab. Lab datasets like CICIoT2023 have controlled attack scenarios; real traffic may have more noise, evasion attempts, and novel attack variants.
- **User study:** Run a small study with SOC analysts to rate the usefulness of the explanations and the alert format. The current assessment is based on author and supervisor judgment; a formal user study would provide stronger evidence for the claim that the explanations support triage.
- **Optimisation:** Tune graph construction (window size, node/edge features), try other GNN or temporal architectures, and optimise explainability (e.g. only for high-confidence alerts) to balance accuracy and speed.
- **Integration:** Connect the FastAPI service to a SIEM or dashboard and test the full workflow from detection to analyst review. Integration with Elastic Security, Splunk, or a custom dashboard would demonstrate end-to-end SOC usability.
- **Further ablation studies:** The thesis already includes one ablation (GAT only vs. full model, Section 8.7). Future work could add ablations for kNN vs. other graph construction, or Integrated Gradients vs. attention-only explanations, to provide further evidence for design choices.


### 9.10 Chapter Summary

The prototype **answers** the stated questions within documented **limits**, **relates** the outcomes to prior work, and sets out a **credible path** for scaling, evaluation, and publication.

---

## Chapter 10 – Critical Self-Evaluation

### 10.1 Chapter Overview

This chapter provides a first-person critical reflection on the project: what went well, what was challenging, what was learned, and what would be improved if the work were repeated. It complements (but does not duplicate) the academic limitations in Chapter 9 by focusing on process, decisions under time constraints, and personal development.

I keep this reflection specific to the actual build path rather than generic statements. Some choices were good. Some were late and caused pressure. Writing this chapter honestly helps show independent judgement, and it also explains why certain limits remain in the current version even though the technical objectives were met.

### 10.2 Planning, Scope, and Risk

The project was scoped to fit a **45-day** MSc timeline. Early on I decided to use a **fixed subset** of CICIoT2023 rather than the full corpus, and to simulate **three** federated clients instead of a large fleet. Those decisions were partly pragmatic  -  disk, RAM, and training time  -  but they were also methodological: a smaller, controlled subset makes it easier to **re-run** experiments and to **debug** the pipeline when something breaks. The trade-off is obvious: stronger numbers on a subset do not automatically transfer to the wild. I tried to mitigate “toy dataset” criticism by (1) stating the limits clearly in **Chapter 9**, (2) comparing against **strong baselines** (RF, MLP) on the **same** splits, and (3) adding **ablation**, **sensitivity**, and **multi-seed** runs so the headline metrics are not a single lucky configuration.

What I would improve with more time: negotiate **earlier** with my supervisor a precise cap on subset size and document the **exact** filtering steps in one place (I relied on scripts and config, which is good for reproducibility, but a one-page “data contract” in the dissertation would help readers who do not open the repo).

### 10.3 Literature and Alignment with the Questions

The literature review deliberately mixed **IoT intrusion detection**, **SIEM / SOC workflows**, **GNNs**, **federated learning**, and **explainable AI**. The gap I argued for was not “nobody used GNNs”  -  many papers do  -  but that **combining** explainable dynamic graphs, federated training, and **SIEM-shaped** outputs in one **CPU-oriented** prototype is still relatively rare in student-scale work. Keeping that storyline straight while implementing was sometimes hard: it is tempting to add every nice idea (more clients, more attacks, more explainers). I had to keep deleting side ideas to protect the core questions.

A clear weakness remains: **no formal user study** with practising analysts. I judged explanation usefulness together with my supervisor and by reading the alert JSON “as if” I were triaging. That is defensible for an MSc, but it is **not** evidence of real-world utility. If I did the project again, I would reserve one week for **three semi-structured interviews** with volunteers (even fellow students role-playing SOC) and a tiny Likert questionnaire  -  not publication-grade, but stronger than intuition alone.

### 10.4 Implementation: What Was Harder Than It Looked

Building the pipeline end-to-end meant **gluing** PyTorch, PyTorch Geometric, Flower, Captum, and FastAPI. Each library is documented, but **the edges** between them are not: tensor shapes for batched graphs, ensuring the **same** scaler and splits for federated clients, and making explainability run on **one** sequence without blowing memory. The longest delays were **data and graphs**: stratified windowing so that graph-level labels are balanced, while flow-level data are heavily imbalanced, took several iterations before training stopped collapsing to “always attack.”

Federated learning was conceptually simple (FedAvg) but **operationally** fiddly: starting the server, then clients, ensuring partitions are **non-IID** but still trainable, and logging per-round metrics in a form I could plot later. I am satisfied that federated performance **matched** centralised on my setup; I am also aware that three clients and ten rounds are **not** stress test.

On the positive side, I structured the code into **modules** (data, graph, models, federated, explainability, API) so I could swap pieces  -  for example, running ablation by changing the temporal head without touching the GAT stack. That modularity paid off when I added `scripts/run_sensitivity_and_seeds.py`: it mostly re-used the same training entry points with different YAML overrides.

### 10.5 Results, Honesty, and the “100%” Question

When test F1 and ROC-AUC hit **1.0**, my first reaction was relief; my second was suspicion. Perfect metrics usually mean one of: (a) the task is **easy** on the slice of data, (b) there is **leakage** (I checked splits carefully  -  no overlap between train/val/test at sequence level), or (c) the **test set is small** enough that a few mistakes swing the score. Here, the test set has **934** sequences; zero errors is possible but should not be read as “solved IoT security.” I tried to be explicit about that in **Chapter 9**. The **ablation** helped: removing the GRU **did** cost precision/F1 slightly, which shows the full model is not trivially redundant. The **sensitivity grid** showed that not every (window, *k*) pair is equally perfect  -  so the story is more nuanced than a single scalar.

If I sound cautious, it is intentional. I would rather a reader trusts me for **transparency** than for bragging.

### 10.6 What I Learned (Skills and Mindset)

Technically, I learned to **think in graphs** (nodes as flows, *k*NN edges as similarity) and in **time** (sequences of snapshots). I learned the basics of **federated optimisation** and why non-IID data breaks naive assumptions. I learned that **explainability** is as much about **presentation** (what we put in the JSON) as about the algorithm (Integrated Gradients).

Personally, I learned to **time-box**: the sensitivity and multi-seed runs were scheduled late; they completed, but I would start **stability experiments earlier** next time so writing-up is not waiting on overnight jobs. I also learned to **document while coding** - commit messages, `config/experiment.yaml`, and CSV outputs saved me when I forgot which run produced which figure.

### 10.7 Time Management: What I Would Reorder

Data loading and graph construction **underestimated** at the start. Federated setup **overestimated** in difficulty once the pattern was clear, but the first week of Flower debugging felt slow. Explainability + FastAPI were faster **after** the model stabilised. With hindsight I would allocate: **week 1** - data contract + baseline; **week 2** - GNN + central training; **week 3** - federated + API; **week 4** - explainability + plots; **buffer** - ablation and sensitivity. I used the buffer mainly for stability runs; in another timeline I might have sacrificed one fancy plot for a **user interview**.

Overall, the project met my own bar: **one coherent system**, **evidence-backed claims**, and **clear limits**. It is not production-ready, but it is **honest research engineering** at MSc level, and I am proud of the parts that were painful and still work.

### 10.8 Chapter Summary

Critical reflection highlights **scoped realism**, **modularity**, and **honest limits**, and concrete habits I would repeat on a future project in research, technical delivery, and planning practice.

---

## Chapter 11 – References

This chapter lists all **cited** sources in **UWS Library Harvard** style, with **DOIs** or **stable URLs** where available, following the *Harvard referencing libguide* (University of the West of Scotland Library, 2025). **In-text citations and this list were checked for consistency** before submission; a reference manager of the type recommended in the project specification (e.g. **EndNote**, **Mendeley**, or **Zotero**) can be used to maintain the same field order in Word, but the **definitive** list for examination is the entries printed here. **Falsified or unverifiable references are excluded**; grey literature and uncited web posts were **not** used to support empirical security claims (see **Section 2.2.1** for retrieval and inclusion rules).

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847. Available at: https://doi.org/10.3390/s25030847

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78. Available at: https://doi.org/10.3390/jsan14040078

Anysphere, Inc. (2024) *Cursor* [Proprietary software or web resource]. San Francisco, CA: Anysphere, Inc. Available at: https://www.cursor.com/ (Accessed: 20 April 2026).

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002. Available at: https://doi.org/10.3390/app15095002

Cuppens, F. and Miège, A. (2002) 'Alert correlation in a cooperative intrusion detection framework', in *Proceedings of the 2002 IEEE Symposium on Security and Privacy*, pp. 202-215. Available at: https://doi.org/10.1109/secpri.2002.1004372

ENISA (2017) *Baseline security recommendations for IoT in the context of critical information infrastructures*. Heraklion: European Union Agency for Network and Information Security. Available at: https://doi.org/10.2824/03228

Han, Z., Zhang, C., Yang, G., Yang, P., Ren, J. and Liu, L. (2025) 'DIMK-GCN: a dynamic interactive multi-channel graph convolutional network model for intrusion detection', *Electronics*, 14(7), p. 1391. Available at: https://doi.org/10.3390/electronics14071391

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Reynolds, J., Melnikov, A., Lunova, N. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint arXiv:2009.07896*. Available at: https://doi.org/10.48550/arXiv.2009.07896

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80-84. Available at: https://doi.org/10.1109/mc.2017.201

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509-530. Available at: https://doi.org/10.3390/ai4030028

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774. Available at: https://doi.org/10.48550/arXiv.1705.07874

Lusa, R., Pintar, D. and Vranic, M. (2025) 'TE-G-SAGE: explainable edge-aware graph neural networks for network intrusion detection', *Modelling*, 6(4), p. 165. Available at: https://doi.org/10.3390/modelling6040165

McMahan, H.B., Moore, E., Ramage, D., Hampson, S. and Agüera y Arcas, B. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, pp. 1273-1282. Available at: https://doi.org/10.48550/arXiv.1602.05629

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection - a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499. Available at: https://doi.org/10.3390/info16060499

Pinto, C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941. Available at: https://doi.org/10.3390/s23135941

Python Software Foundation (2023) *Python* [Program]. version 3.10. Beaverton, OR: Python Software Foundation. Available at: https://www.python.org/ (Accessed: 20 April 2026).

Qu, Y., Gao, L., Luan, T.H., Xiang, Y., Yu, S., Li, B. and Zheng, G. (2020) 'Decentralized Privacy Using Blockchain-Enabled Federated Learning in Fog Computing', *IEEE Internet of Things Journal*, 7(6), pp. 5171-5183. Available at: https://doi.org/10.1109/JIOT.2020.2977383

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319-3328. Available at: https://doi.org/10.48550/arXiv.1703.01365

The Git Project (2023) *Git* [Version control system]. The Git Project. Available at: https://git-scm.com/ (Accessed: 20 April 2026).

University of the West of Scotland Library (2025) *Harvard referencing libguide*. Paisley: UWS Library. Available at: https://libguides.uws.ac.uk/harvard (Accessed: 20 April 2026).

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *International Conference on Learning Representations (ICLR)*. Available at: https://doi.org/10.48550/arXiv.1710.10903

Wang, R., Zhao, J., Zhang, H., He, L., Li, H. and Huang, M. (2025) 'Network traffic analysis based on graph neural networks: a scoping review', *Big Data and Cognitive Computing*, 9(11), p. 270. Available at: https://doi.org/10.3390/bdcc9110270

Yang, S., Pan, W., Li, M., Yin, M., Ren, H., Chang, Y., Liu, Y., Zhang, S. and Lou, F. (2025) 'Industrial Internet of Things intrusion detection system based on graph neural network', *Symmetry*, 17(7), p. 997. Available at: https://doi.org/10.3390/sym17070997

Zheng, L., Li, Z., Li, J., Li, Z. and Gao, J. (2019) 'AddGraph: Anomaly Detection in Dynamic Graph Using Attention-based Temporal GCN', in *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)*, pp. 4419-4425. Available at: https://doi.org/10.24963/ijcai.2019/614

Zhong, M., Lin, M., Zhang, C. and Xu, Z. (2024) 'A survey on graph neural networks for intrusion detection systems: methods, trends and challenges', *Computers and Security*, 141, p. 103821. Available at: https://doi.org/10.1016/j.cose.2024.103821

---

## Chapter 12 – Bibliography

This bibliography lists wider readings, technical documentation, and standards consulted during the project that supported design and implementation decisions but were not directly cited as evidence in the body. Entries are grouped to mirror the topics covered in **Chapters 2–8**.

**Frameworks, libraries, and tooling used in the implementation (Chapter 6).**

Beutel, D.J., Topal, T., Mathur, A., Qiu, X., Fernandez-Marques, J., Gao, Y., Sani, L., Li, K.H., Parcollet, T., Gusmão, P.P.B. de and Lane, N.D. (2020) 'Flower: a friendly federated learning research framework', *arXiv preprint arXiv:2007.14390*. Available at: https://doi.org/10.48550/arXiv.2007.14390

Fey, M. and Lenssen, J.E. (2019) 'Fast graph representation learning with PyTorch Geometric', in *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*. Available at: https://doi.org/10.48550/arXiv.1903.02428

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J. and Chintala, S. (2019) 'PyTorch: an imperative style, high-performance deep learning library', in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, pp. 8024-8035. Available at: https://doi.org/10.48550/arXiv.1912.01703

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830. Available at: https://doi.org/10.48550/arXiv.1201.0490

**Standards and reference frameworks for the SIEM and SOC context (Chapters 1, 2 and 6).**

*ENISA (2017) appears in **Chapter 11** because it is cited in the body; it is not repeated here.*

Rashid, A., Chivers, H., Lupu, E., Martin, A. and Schneider, S. (2021) *The Cyber Security Body of Knowledge (CyBOK), version 1.1*. Bristol: National Cyber Security Centre (UK) and University of Bristol. Available at: https://www.cybok.org/media/downloads/CyBOK_v1.1.0.pdf

Strom, B.E., Applebaum, A., Miller, D.P., Nickels, K.C., Pennington, A.G. and Thomas, C.B. (2018) *MITRE ATT&CK: design and philosophy*. Bedford, MA: The MITRE Corporation. Available at: https://attack.mitre.org/docs/ATTACK_Design_and_Philosophy_March_2020.pdf

**Project administration and referencing (Chapters 1, 3, 9, 11).**

*Development tools* (**Cursor**, **Python**, **Git**) are **cited in the body** (**Section 3.5**) and appear in the **authoritative list** in **Chapter 11** (UWS Harvard). They are not duplicated here.

University of the West of Scotland (2025) *MSc Project Handbook 2025-26 (COMP11024 Master's Project)*. Paisley: School of Computing, Engineering and Physical Sciences, University of the West of Scotland.

The UWS Library **Harvard *libguide*** is also listed in **Chapter 11** because it is cited in **Section 3.5**; the full entry is not repeated here.

---

## Chapter 13 – Appendices

This chapter contains the required appendices for submission.

### Appendix A: Project Process Documents

### Appendix B: Project Specification

### Appendix C: Handbook Appendix 1  -  Screenshots of project code

The handbook requires **Appendix 1** code views to be presented as **figures**, each with a **caption** and an explanation of **how to interpret** the code in the context of this dissertation. Below, **Figure A1-1**–**Figure A1-6** satisfy that requirement.

The bitmaps are **generated automatically** from the submission codebase so line numbers stay aligned with the files on disk. Regenerate them after editing source with:

`python scripts/render_appendix1_code_figures.py`

Output directory: `results/figures/appendix1/`. In the **Table of Figures**, these appear as **Figure A1-1**–**Figure A1-6** (appendix labels), separate from the body sequence **Figure 1**–**Figure 29**. **Chapter 6 (Section 6.10)** and Appendix C both use **dark-theme**, line-numbered, **core-snippet** screenshots (not full-file captures) to support examiner code reference.

![Figure A1-1  -  DynamicGNN source code excerpt](results/figures/appendix1/fig_a1_01_dynamic_gnn.png)

**Figure A1-1**  -  Dynamic graph classifier: `DynamicGNN` sequence forward core (GRU vs. mean-pool ablation, logits). *(Chapter 13, Appendix C.)* *Source: `src/models/dynamic_gnn.py`, lines 75–97.*

**Caption (formal):** Figure A1-1  -  Core implementation of the dynamic GNN (`DynamicGNN`): node embedding, two `GATConv` layers with multi-head attention and dropout, per-window graph embedding, sequence encoding with `GRU` (or mean-pool when `use_gru` is false for ablation), and two-class logits. Attention weights may be retained for explainability (`return_attention_weights`).

**How to interpret:** This class is the **main learnable model** described in **Chapters 5–6**. Each time step is one PyTorch Geometric `Data` object (nodes = flows in a window, edges = *k*NN). `_encode_graph` applies GAT message passing and reduces node states to one vector per window; `forward` stacks windows in time order and either runs the **GRU** (full model) or **mean-pools** over time (ablation in **Section 8.7**). The `from_config` constructor ties hyperparameters to `config/experiment.yaml`, which supports the sensitivity study in **Section 8.8**.

![Figure A1-2  -  graph_builder *k*NN excerpt](results/figures/appendix1/fig_a1_02_graph_builder_knn_graph.png)

**Figure A1-2**  -  Constructing a single-window *k*NN graph from flow feature rows. *(Chapter 13, Appendix C.)* *Source: `src/data/graph_builder.py`, lines 37–58.*

**Caption (formal):** Figure A1-2  -  `flows_to_knn_graph`: for one window of `N` flows with `F` features, fits `sklearn.neighbors.NearestNeighbors` (Euclidean), adds bidirectional edges among each node and its `k` actual neighbours (capped when `N` is small), and returns a `torch_geometric.data.Data` object with `x`, `edge_index`, and graph-level label `y`.

**How to interpret:** This function shows **how “graph structure” is defined** when device IPs are unavailable (**Chapter 1**): similarity in the **46-dimensional** flow feature space replaces physical topology. Bidirectional edges make the graph undirected for GAT. The label on the graph is supplied by the caller (binary benign vs. attack) and matches the stratified pool in **Figure A1-3**, not a per-flow vote - important for reading the evaluation chapters.

![Figure A1-3  -  stratified windowing excerpt](results/figures/appendix1/fig_a1_03_graph_builder_stratified.png)

**Figure A1-3**  -  Stratified windowing so both classes appear in training graphs. *(Chapter 13, Appendix C.)* *Source: `src/data/graph_builder.py`, lines 106–123.*

**Caption (formal):** Figure A1-3  -  `build_graphs_for_split`: splits flows into benign and attack pools, builds sliding (or strided) windows from each pool with `_build_windows_from_pool`, balances counts between classes, shuffles, and logs totals - addressing severe class imbalance in raw CICIoT2023.

**How to interpret:** This block explains **why training does not collapse** to “always predict attack” despite a very high attack ratio in the raw CSV (**Chapter 4**): windows are drawn **within** each class, so each graph’s supervision matches the pool. The `minority_stride` argument increases overlap for the benign pool when it is smaller. The dissertation’s **window size** and **k** used in experiments come from `config/experiment.yaml` and feed the sensitivity grid in **Section 8.8**.

![Figure A1-4  -  explainer source excerpt](results/figures/appendix1/fig_a1_04_explainer_integrated_gradients.png)

**Figure A1-4**  -  Post-hoc explanations: forward pass with attention, Integrated Gradients, top nodes/features. *(Chapter 13, Appendix C.)* *Source: `src/explain/explainer.py`, lines 53–95.*

**Caption (formal):** Figure A1-4  -  `explain_sequence`: runs the model with attention enabled, wraps Captum **Integrated Gradients** on the **last window’s** node features (`_ig_wrapper`), aggregates absolute attributions to rank top nodes and top feature indices, and returns an `ExplanationBundle` for JSON alert formatting.

**How to interpret:** This is the bridge between **Chapter 6** (implementation) and **Chapter 8** (example alerts): SOC-facing text is not arbitrary - it is derived from **IG magnitudes** and **GAT edge attention** from the same forward pass analysts would get at inference. IG is computed on the **most recent** graph in the sequence (design choice documented in code comments). If Captum is missing, the bundle degrades gracefully (`HAS_CAPTUM` guard in the same module).

![Figure A1-5  -  federated Flower client excerpt](results/figures/appendix1/fig_a1_05_federated_flower_client.png)

**Figure A1-5**  -  Federated learning CLI: Flower server vs. client, local data, `GNNFlowerClient`. *(Chapter 13, Appendix C.)* *Source: `src/federated/run_federated.py`, lines 28–71.*

**Caption (formal):** Figure A1-5  -  `main` in `run_federated.py`: loads YAML config; **server** mode calls `run_fl_server` with round count and quorum; **client** mode loads `client_{cid}_graphs.pt`, builds train/validation sequence loaders, constructs `GNNFlowerClient`, and connects to `127.0.0.1:8080` via `fl.client.start_numpy_client`.

**How to interpret:** This file is the **student-facing entry point** for **Chapter 8** federated results: each client trains only on its partition (**non-IID** split from `src.federated.data_split`). The address and round counts come from `config/experiment.yaml` under `fl`. The pattern matches the FedAvg narrative in **Chapter 2** - local epochs, then aggregation on the server (server implementation in `src/federated/server.py`, not shown).

![Figure A1-6  -  FastAPI score endpoint excerpt](results/figures/appendix1/fig_a1_06_fastapi_score_alert.png)

**Figure A1-6**  -  HTTP API: `/score` builds graphs, runs `explain_sequence`, returns ECS-style alert JSON. *(Chapter 13, Appendix C.)* *Source: `src/siem/api.py`, lines 67–89.*

**Caption (formal):** Figure A1-6  -  FastAPI application: startup loads `experiment.yaml` and `dynamic_gnn_best.pt`; `POST /score` accepts a list of flow windows, reads **`graph.knn_k`** from the loaded config for `flows_to_knn_graph`, runs `explain_sequence` with top-5 nodes and features, measures wall-clock milliseconds, and returns `format_ecs_alert` output alongside prediction and score.

**How to interpret:** This is the **edge deployment surface** argued in **Chapters 1 and 5**: a single HTTP call turns raw feature windows into **SIEM-shaped JSON** suitable for triage. Inference-time *k*NN **k** therefore **matches** training graph construction from **`config/experiment.yaml`**, avoiding train/serve skew. Latency reported in **Chapter 8** is consistent with this synchronous CPU path.

### Appendix D: Handbook Appendix 4 (optional)  -  repository and dataset reference

#### D.1 Project source code (GitHub)

- **Repository:** https://github.com/imark0007/Dissertation_Project  

#### D.2 Dataset source used in this project (Kaggle)

- **Kaggle dataset page:** https://www.kaggle.com/datasets/himadri07/ciciot2023

#### D.3 Official dataset source (UNB/CIC reference)

- **Dataset page:** https://www.unb.ca/cic/datasets/iotdataset-2023.html  

---
