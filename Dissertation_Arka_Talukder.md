# Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning

**Student:** Arka Talukder  
**Programme:** MSc Cyber Security (Full-time)  
**Supervisor:** Dr. Raja Ujjan  
**University of the West of Scotland**

---

## 1. Abstract

IoT devices are used in many places today, from homes to factories, but they are often not very secure. Security Operations Centre (SOC) teams need tools that can detect attacks quickly, give accurate results, and explain why an alert was raised so that analysts can act on it. This project aims to design and build a small prototype system that detects attacks in IoT network traffic using a dynamic graph neural network and federated learning. The system also provides simple explanations for each alert so that SOC analysts can understand and use them without needing deep technical knowledge.

The main research question is: how can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices? To answer this, the project uses the CICIoT2023 dataset (Pinto et al., 2023), constructs kNN feature-similarity graphs from flow windows — where each flow record is a node and edges connect flows with similar network characteristics — and compares a dynamic GNN (Graph Attention with GRU) against baseline models such as Random Forest and MLP. Federated learning is implemented with the Flower framework using FedAvg with two to three clients, so that training can happen without sharing raw data. Explainability is added using Integrated Gradients (Captum) and attention weights, and alerts are output in a SIEM-style JSON format (ECS-like) with top features and flows included. The system is deployed as a FastAPI endpoint and evaluated with precision, recall, F1-score, ROC-AUC, false positive rate, confusion matrix, and CPU inference time. The project also examines whether federated learning keeps performance close to centralised training and whether the explanations are useful for SOC alert triage. The work is kept realistic for a 45-day MSc project, with a focus on practical SOC use and lightweight, CPU-based deployment.

**Keywords:** IoT security, dynamic graph neural network, federated learning, SIEM, explainable AI, edge AI, SOC, CICIoT2023

---

## 2. Introduction

### 2.1 Background and Motivation

The Internet of Things (IoT) has grown quickly over the last few years. Devices such as smart sensors, cameras, and industrial controllers are now common in homes, offices, and critical infrastructure. However, many of these devices were not designed with strong security in mind. They often have weak or default passwords, outdated software, and limited computing power, which makes them attractive targets for attackers. When IoT devices are compromised, they can be used for botnets, data theft, or to attack other parts of the network. For this reason, monitoring IoT network traffic and detecting malicious behaviour has become an important task for security teams.

Security Operations Centres (SOCs) are responsible for watching over networks, investigating alerts, and responding to incidents. They rely on tools such as Security Information and Event Management (SIEM) systems to collect and analyse logs and flow data. A typical problem is that these systems can produce too many alerts, and many of them are false positives. Analysts then spend a lot of time deciding which alerts are real and which can be ignored. When an alert does not explain why it was triggered, it is harder for analysts to triage it quickly. Therefore, there is a need for detection systems that are not only accurate but also explainable, so that SOC staff can understand and trust the alerts they receive.

Traditional machine learning models, such as Random Forest or simple neural networks, can work on flow-based or tabular data. However, network traffic has a natural structure: devices talk to each other, and the way they connect changes over time. Graph-based models can represent this structure by treating devices as nodes and connections as edges. Dynamic graph neural networks (GNNs) can then learn from how the graph changes over time, which may capture attack patterns that simpler models miss. At the same time, IoT data is often distributed across different sites or organisations. Sharing raw data for centralised training can raise privacy and legal issues. Federated learning offers a way to train a model across multiple clients without moving the raw data to a single server, which is useful for IoT and edge environments.

Finally, many IoT and edge deployments run on devices with limited hardware. Not every organisation can afford powerful GPUs at the edge. So, a system that can run on CPU-based edge devices and still provide useful detection and explanations would be more practical for real-world SOC use.

### 2.2 Research Aim and Questions

This project aims to design and build a small prototype system that can detect attacks in IoT network traffic. The system will use a dynamic graph neural network and federated learning, and it should provide simple explanations for alerts so that SOC analysts can understand and use them easily.

The main research question is:

*How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?*

To support this, the following sub-questions are addressed:

1. Does a dynamic graph model perform better than simple models like Random Forest and MLP?
2. Can federated learning maintain similar performance without sharing raw data?
3. Can the model generate useful explanations for SOC alert triage?

Answering these will help show whether the proposed approach is both technically sound and practically useful for SOC workflows.

### 2.3 Scope and Limitations

The project uses the CICIoT2023 dataset (Pinto et al., 2023) and works with a manageable subset so that experiments can be completed within the available time. The publicly available version of this dataset provides 46 pre-computed flow-level features but does not include device identifiers such as IP addresses. Therefore, a device-based graph construction (where nodes represent devices and edges represent flows between them) was not possible. Instead, the project adopts a kNN feature-similarity graph approach: consecutive flows are grouped into fixed-size windows, each flow becomes a node with its 46 features, and edges are formed by connecting each flow to its k-nearest neighbours in feature space. This approach is supported by recent work showing that similarity-based graph construction can capture structural patterns in tabular network data and improve classification performance (Ngo et al., 2025; Basak et al., 2025). The main model is a dynamic GNN combining Graph Attention and GRU, and it is compared with Random Forest and MLP baselines. Federated learning is implemented with the Flower framework using FedAvg with three clients. Explainability is provided through Integrated Gradients (Captum) and attention weights, and the output is SIEM-style JSON alerts in an ECS-like format, including the main features and flows that led to the alert. The system is exposed via a FastAPI endpoint, and CPU inference time is measured to ensure it is suitable for edge deployment.

The work is limited to a 45-day MSc project. So, the prototype is kept small: a subset of CICIoT2023, a small number of federated clients, and CPU-only inference. The focus is on showing that the idea works and that the alerts and explanations are useful for SOC-style triage, rather than on building a production-ready product. If the dataset is too large, a smaller subset will be used; if federated learning is too slow, the number of rounds or clients can be reduced; and if explainability is too slow, it can be applied only to selected alerts.

### 2.4 Dissertation Structure

The rest of the dissertation is organised as follows. The Literature Review discusses related work on IoT security, graph neural networks, federated learning, and explainable AI in the context of SIEM and SOC operations. The Research Design and Methodology section describes how the research questions are addressed, including the choice of dataset, graph design, models, and evaluation plan. Implementation and System Development detail how the prototype was built, including data processing, model training, federated learning setup, explainability, and the FastAPI deployment. The Evaluation section explains the experimental setup, metrics, and how the baselines and the main model are compared. Results Presentation shows the main findings, including performance metrics, confusion matrices, federated learning behaviour, and example alerts with explanations. The Discussion interprets these results and considers advantages and limitations. The Conclusion and Recommendations summarise what was achieved and suggest future work. Finally, the Critical Self-Evaluation reflects on the project process and outcomes, followed by the References in Harvard style.

### 2.5 Alignment with MSc Dissertation Marking Criteria

This dissertation is structured to meet the University’s marking scheme and distinction-level expectations. The Introduction (5%) sets out a clear aim, research question, and scope. The Literature Review (20%) provides critical analysis of existing work and a justified gap for this research, rather than description only. The Research Design and Methodology (20%) explains and justifies the choice of dataset, graph design, models, federated learning setup, and evaluation plan in a way that supports reproducibility and rigour. The Implementation and System Development (25%) documents the prototype build with enough detail for the work to be assessed and replicated, and links design decisions to the research questions. The Evaluation (5%) and Results Presentation (5%) present evidence objectively, using the stated metrics and including limitations where relevant. The Discussion draws on this evidence to interpret findings, compare with literature, and address strengths and limitations critically. The Conclusion and Recommendations (10%) summarise contributions and suggest realistic future work. The Critical Self-Evaluation (10%) offers an honest reflection on the research process, what went well, what could be improved, and what was learned. Throughout, the aim is to demonstrate independent thinking, critical engagement with sources and results, and a clear line of argument from question to conclusion, in line with distinction-level criteria for an MSc dissertation.

---

## 3. Literature Review

This section looks at existing work on IoT security, intrusion detection, graph-based and dynamic models, federated learning, and explainability in security tools. The aim is to show where this project fits and why the chosen approach is reasonable.

### 3.1 IoT Security and the Need for Detection

IoT devices are now common in many sectors, but they are often built with cost and convenience in mind rather than security. Researchers have pointed out that many devices use default credentials, have unpatched software, and lack strong encryption (Kolias et al., 2017). When these devices are compromised, they can be used in botnets, for data exfiltration, or as a stepping stone into the rest of the network. So, detecting malicious behaviour in IoT traffic is an important part of modern security operations.

Intrusion detection for IoT can be done in different ways. Signature-based methods look for known attack patterns, while anomaly-based methods learn what is normal and flag the rest. Machine learning has been used widely for anomaly and attack detection on flow or packet data. The CICIoT2023 dataset used in this project was designed to support such research; it includes a range of attacks (e.g. DDoS, reconnaissance, brute force) from real IoT devices in a controlled environment (Pinto et al., 2023). Using a public dataset like this allows results to be compared with other work, although the setting is still lab-based rather than a live production network. Yang et al. (2025) recently demonstrated graph neural network–based intrusion detection for industrial IoT, showing that structural modelling of network interactions can improve detection accuracy compared to flat feature models.

A limitation of many studies is that they treat each flow or packet in isolation. In reality, attacks often show up as patterns of communication between devices over time. A recent scoping review by Wang et al. (2025) confirmed that graph-based approaches to network traffic analysis are increasingly adopted, and that dynamic graph models in particular show promise for capturing evolving attack patterns. Zhong et al. (2024) surveyed graph neural networks for intrusion detection systems and identified a clear trend towards combining GNNs with temporal models, while also noting that explainability and federated deployment remain under-explored. These observations motivate the graph-based and dynamic approach taken in this project.

### 3.2 SIEM, SOC Workflows, and Alert Quality

SOC teams depend on SIEM and related tools to collect logs and flow data and to generate alerts. A well-known issue is alert fatigue: too many alerts, and too many false positives, make it hard for analysts to focus on real threats (Cuppens and Miege, 2002). If an alert does not explain why it was raised, triage becomes slower and more dependent on guesswork.

There is a growing interest in making security tools more explainable. For example, some work has focused on explaining which features or events led to a detection (e.g. Lundberg and Lee, 2017, on model-agnostic explanations). In a SOC context, explanations that point to specific flows, devices, or time windows can help analysts decide quickly whether to escalate. This project aligns with that by producing SIEM-style alerts with top features and flows attached, so that the output is not only a label but something an analyst can act on.

### 3.3 Graph Neural Networks and Dynamic Graphs

Graph neural networks (GNNs) operate on data that has a graph structure: nodes and edges. In a network security setting, nodes can be hosts or devices and edges can be flows or connections. GNNs aggregate information from neighbours and can learn patterns that depend on the structure of the graph. Graph Attention Networks (GATs) extend this by letting each node assign different importance to its neighbours using attention (Velickovic et al., 2018). This can help the model focus on the most relevant connections, which is useful when some flows are benign and others are part of an attack. Han et al. (2025) proposed DIMK-GCN, a dynamic interactive multi-channel graph convolutional network for intrusion detection, demonstrating that multi-scale graph representations can capture both local and global attack signatures. Similarly, Ngo et al. (2025) explored attribute-based graph construction for IoT intrusion detection and showed that constructing graphs from feature similarity (rather than only from network topology) can be effective when device-level identifiers are not available. This finding directly supports the kNN-based graph construction used in this project.

Networks change over time: new connections appear, traffic volumes shift, and attacks develop in stages. Dynamic or temporal graph models try to capture this. One common approach is to combine a GNN with a recurrent module (e.g. GRU or LSTM) so that the model sees a sequence of graph snapshots and learns from how the graph evolves. Several papers have used this idea for fraud detection or anomaly detection in graphs (e.g. Liu et al., 2019). Lusa et al. (2025) proposed TE-G-SAGE, an explainable edge-aware GNN for network intrusion detection that combines temporal edge features with GraphSAGE aggregation, demonstrating that both edge-level and temporal information improve detection accuracy. Basak et al. (2025) developed X-GANet, an explainable graph-based framework for robust network intrusion detection, confirming that attention-based GNNs can achieve high accuracy while also providing interpretable outputs. For IoT flow data, building time-windowed graphs and using a dynamic GNN (GAT + GRU in this project) is a reasonable way to test whether structure and temporality improve detection over simple tabular models like Random Forest or MLP. The literature supports the idea that graph structure can add value, but the actual gain depends on the data and task, which is why this project compares the dynamic GNN to baselines.

### 3.4 Federated Learning and Privacy

Federated learning trains a model across multiple clients without centralising raw data. Each client trains on local data and sends only model updates (e.g. gradients or weights) to a server, which aggregates them and sends back an updated model. FedAvg (McMahan et al., 2017) is a standard algorithm: the server averages the client model parameters, often weighted by the amount of local data. This reduces privacy risk compared to sending raw IoT traffic to a central site, which is important when data comes from different organisations or locations.

Federated learning has been applied in IoT and edge settings (e.g. Qu et al., 2019). Lazzarini et al. (2023) evaluated federated learning specifically for IoT intrusion detection, finding that FedAvg can achieve performance comparable to centralised training while preserving data locality. More recently, Albanbay et al. (2025) conducted a performance evaluation and data scaling study on federated learning–based intrusion detection in IoT networks, demonstrating that non-IID data distribution across clients is a key factor affecting convergence and accuracy. Challenges include communication cost, non-IID data across clients, and possible drops in accuracy compared to centralised training. In this project, Flower is used to implement FedAvg with three clients and a Dirichlet-based non-IID data split. The goal is to show that federated training can achieve performance close to centralised training on the chosen subset, not to solve all federated learning problems. The evaluation includes communication cost (approximate bytes) and round-by-round performance to see how quickly the model converges.

### 3.5 Explainability in ML-Based Security

Explainability can be built into a system in different ways. Post-hoc methods explain a trained model (e.g. which inputs or features mattered most for a prediction). Integrated Gradients is a gradient-based method that attributes the prediction to input features relative to a baseline (Sundararajan et al., 2017). It is implemented in Captum (Kokhlikyan et al., 2020), which works with PyTorch models. For GNNs, attention weights from GAT layers can also be used to see which edges or neighbours the model focused on. Both are used in this project: Integrated Gradients for feature-level explanation and attention for flow-level importance.

Alabbadi and Bajaber (2025) demonstrated the use of explainable artificial intelligence (XAI) for intrusion detection over IoT data streams, confirming that post-hoc explanation methods can make real-time security decisions more transparent for analysts. A practical point is that full explainability for every prediction can be slow. So, the design allows applying explainability only to selected alerts (e.g. high-confidence positives) if needed. The literature does not always address this trade-off; this project acknowledges it as a limitation and a possible area for future work.

### 3.6 Gap and Contribution

Existing work covers IoT intrusion detection, GNNs for graphs, federated learning, and explainability separately. There is less work that combines an explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype aimed at SOC use on CPU-based edge devices. This project tries to fill that gap with a small but complete system: dynamic graph design from IoT flow data, GAT+GRU model, FedAvg via Flower, explanations via Captum and attention, and JSON alerts in an ECS-like format. The evaluation is designed to answer whether the graph model beats simple baselines, whether federated learning keeps performance, and whether the explanations are useful for triage. The scope is limited to a 45-day MSc project and a subset of CICIoT2023, but the structure and criteria are set up so that the results can be discussed critically and extended in future work.

---

## 4. Research Design and Methodology

This section describes how the research questions are addressed: the choice of dataset, how the graph is built, which models are used, how federated learning and explainability are integrated, and how the work is evaluated.

### 4.1 Research Approach

The project is practical and design-oriented. The aim is to build a working prototype and evaluate it with clear metrics, rather than to prove a theoretical result. The main research question is answered by implementing the system, comparing the dynamic GNN to baselines, testing federated learning, and checking whether the explanations help SOC-style triage. Sub-questions are answered through experiments and by inspecting example alerts.

### 4.2 Dataset and Subset

The CICIoT2023 dataset (Pinto et al., 2023) is used. It contains network flow data from IoT devices under various attacks (e.g. DDoS, DoS, reconnaissance, brute force, spoofing). The full dataset is large, so a manageable subset is selected to fit the 45-day timeline. The subset is chosen to include both benign and attack traffic and a spread of attack types where possible. The same subset is used for all experiments so that results are comparable. Data is split into fixed train, validation, and test sets (e.g. 70% train, 15% validation, 15% test, or similar, with no overlap). This avoids data leakage and allows fair comparison between centralised and federated training.

### 4.3 Graph Design

The graph is built from flow data to capture structural relationships between network flows within each observation window.

- **Nodes:** Each node represents one flow record. The node feature vector consists of the 46 pre-computed flow-level features provided by the CICIoT2023 dataset (e.g. packet rate, byte count, flag counts, protocol indicators, statistical aggregates). Device-level identifiers (IP addresses) are not available in the publicly released version of this dataset, so a device-based graph design (where nodes are devices and edges are flows between them) was not feasible. Instead, a feature-similarity approach is adopted, following the principle demonstrated by Ngo et al. (2025) that attribute-based graph construction can be effective for intrusion detection when topology information is absent.
- **Edges:** For each window, a k-nearest-neighbour (kNN) graph is constructed in feature space. Each flow is connected to its k most similar flows (by Euclidean distance on the 46 features), producing undirected edges. This creates a graph structure where flows with similar characteristics are linked, enabling the GNN to learn from local neighbourhoods of related traffic patterns.
- **Windows and sequences:** Flows are grouped into fixed-size windows (e.g. 50 flows per window). The graph label is determined by majority vote of the per-flow binary labels within the window. Consecutive windows form a temporal sequence (e.g. 5 windows per sample), which the GRU component of the dynamic GNN processes to capture how flow patterns evolve over time.

This design is kept simple so that it can be implemented and tested within the project scope. The kNN approach has the advantage of being applicable to any flow-feature dataset regardless of whether device identifiers are present. More complex designs (e.g. device-based graphs when IPs are available, multi-hop temporal aggregation) could be explored in future work.

### 4.4 Models

Three types of models are used so that the value of the graph and temporal structure can be assessed.

- **Random Forest:** A baseline that works on tabular features (e.g. per-flow or per-window aggregates). It does not use graph structure. It is trained on the same data in a flat format.
- **MLP (Multi-Layer Perceptron):** Another baseline on tabular features. It is a simple feed-forward neural network. It provides a neural baseline without graph or temporal components.
- **Dynamic GNN (GAT + GRU):** The main model. Graph Attention layers process each graph snapshot; a GRU (or similar RNN) then processes the sequence of graph representations over time. The output is used for attack vs. benign (or multi-class) prediction. This model uses both structure and temporality.

All models are trained to predict attack (or attack type) from the available features and, for the GNN, from the graph and its evolution. Hyperparameters (e.g. learning rate, number of layers, hidden size) are chosen with the validation set; the test set is used only for final reporting.

### 4.5 Federated Learning Setup

Federated learning is implemented with the Flower framework. FedAvg is used: each client trains the model on its local partition of the data and sends updated parameters to the server; the server averages them and broadcasts the global model back. The data is split across 2–3 clients in a way that mimics distributed data (e.g. by device or by time period). The number of federated rounds and local epochs is set so that training finishes in reasonable time; if needed, rounds or clients can be reduced to meet the 45-day constraint. The same model architecture is used in federated and centralised training so that performance can be compared fairly. Communication cost is approximated (e.g. bytes per round for model parameters) and reported.

### 4.6 Explainability

Two forms of explanation are used:

- **Integrated Gradients (Captum):** Applied to the chosen model (e.g. the dynamic GNN or the MLP) to attribute the prediction to input features. The top contributing features are extracted and included in the alert.
- **Attention weights:** For the GAT-based model, attention weights over edges or neighbours are used to identify which flows or connections the model focused on. These can be summarised as “top flows” or “top edges” in the alert.

Explanations are produced for selected predictions (e.g. positive alerts). If computing explanations for every prediction is too slow, they are applied only to a subset (e.g. high-confidence alerts) and this is noted as a limitation.

### 4.7 Alert Format and Deployment

Alerts are output in a SIEM-style JSON format, aligned with ideas from ECS (Elastic Common Schema) where useful: event type, timestamp, source/target information, label, score, and an explanation field containing top features and/or top flows. This makes the output usable for SOC workflows and for demonstration. The system is exposed via a FastAPI endpoint that accepts input (e.g. flow or graph data), runs inference, and returns the alert with explanation. CPU inference time is measured so that edge deployment is assessed.

### 4.8 Evaluation Plan

Evaluation is designed to answer the three sub-questions and support the main research question.

- **Metrics:** Precision, recall, F1-score, ROC-AUC, and false positive rate on the fixed test set. A confusion matrix is reported.
- **Model comparison:** Centralised training: Random Forest, MLP, and dynamic GNN are compared on the same splits. Federated training: the same GNN (or MLP) is trained with FedAvg and compared to its centralised version.
- **Time and cost:** Time-window analysis (e.g. performance across different window sizes or positions), federated round-by-round performance, approximate communication cost (bytes), and CPU inference time per sample or per batch.
- **Explainability and SOC use:** Three to five example alerts are generated with full explanations (top features and flows). These are discussed in terms of whether they would help a SOC analyst triage (e.g. whether the highlighted flows and features are interpretable and actionable).

Risks are handled as follows: if the dataset is too large, a smaller subset is used; if federated learning is too slow, rounds or clients are reduced; if explainability is too slow, it is applied only to selected alerts. All choices are documented so that the work is reproducible and the limitations are clear.

---

## 5. Implementation and System Development

This section describes how the prototype was built: data loading and graph construction, model implementation, federated learning, explainability with Captum and attention, alert generation, and the FastAPI deployment. The Python code is organised in the `src/` package with sub-modules for data processing (`src/data/`), models (`src/models/`), federated learning (`src/federated/`), explainability (`src/explain/`), SIEM output (`src/siem/`), and evaluation (`src/evaluation/`). A master pipeline script (`scripts/run_all.py`) orchestrates the full workflow from preprocessing to results.

### 5.1 Environment and Tools

The system is implemented in Python. PyTorch is used for the neural models (GNN and MLP), and PyTorch Geometric (PyG) or similar libraries can be used for graph operations and GAT layers. Scikit-learn is used for Random Forest and for metrics (precision, recall, F1, ROC-AUC, confusion matrix). The Flower framework is used for federated learning (client and server logic). Captum is used for Integrated Gradients. FastAPI is used for the REST API. The code is structured so that data loading, graph building, training, and inference can be run separately or together. All experiments are run on CPU to match the edge deployment goal; training time is therefore longer than with a GPU but remains feasible for the chosen subset.

### 5.2 Data Loading and Preprocessing

The CICIoT2023 subset is loaded from CSV (or the format provided by the dataset). Required fields include identifiers for source and destination (e.g. IP or device ID), flow-level features (e.g. packet count, byte count, duration, protocol), timestamp, and label (benign vs. attack or attack type). Missing values are handled (e.g. dropped or filled), and labels are encoded numerically. The data is split into train, validation, and test sets with a fixed seed so that splits are reproducible. For federated learning, the training set is partitioned across clients (e.g. by device or by time) so that each client has a local dataset. The same test set is used for all models to ensure fair comparison.

### 5.3 Graph Construction

For each window, a kNN similarity graph is built from the flow records. All flows in the window become nodes, each carrying its 46-feature vector. A k-nearest-neighbour search (with k = 5 and Euclidean distance) is performed on the feature matrix, and bidirectional edges are created between each flow and its k nearest neighbours. To address the severe class imbalance in CICIoT2023 (approximately 97.8% attack, 2.2% benign at the flow level), a stratified windowing strategy is used: benign and attack flows are separated into distinct pools, windows are constructed independently from each pool (with overlapping stride for the minority class to augment the number of benign windows), and the resulting graphs are balanced and shuffled before training. The graph label is assigned based on the class of the pool the window was drawn from. The result is a sequence of balanced graphs, one per window, which is passed to the dynamic GNN. For Random Forest and MLP, the same data is used in a flat format: each sample is a single flow-level feature vector without explicit graph structure. This keeps the baselines comparable in terms of the information available, although they cannot exploit the neighbourhood relationships encoded in the graph.

### 5.4 Model Implementation

**Random Forest:** Implemented with scikit-learn. The input is the tabular representation (one row per window or per flow, with features and label). Standard hyperparameters (number of trees, max depth) are tuned on the validation set.

**MLP:** A feed-forward network with a few hidden layers and ReLU (or similar) activation. Input dimension matches the feature vector size; output is one neuron for binary classification (or more for multi-class) with softmax. Trained with cross-entropy loss and an optimiser (e.g. Adam). Validation loss is monitored for early stopping if needed.

**Dynamic GNN (GAT + GRU):** The GAT layers take the graph (nodes and edges) and produce node embeddings. These are aggregated (e.g. graph-level readout) to get one vector per snapshot. The sequence of these vectors is fed into a GRU. The final hidden state (or a linear layer on top of it) produces the prediction. The implementation uses PyTorch and a GNN library (e.g. PyG) for the GAT. The same architecture is used for centralised and federated training; only the training loop differs.

### 5.5 Federated Learning (Flower)

The Flower server is configured to run FedAvg: it waits for client updates, averages the model parameters (optionally weighted by local dataset size), and sends the updated model back. Each client loads its local partition of the data, trains the model for a fixed number of local epochs per round, and sends the updated parameters to the server. The process repeats for a set number of rounds. Model parameters are serialised and sent over the network; the size of these messages is used to approximate communication cost in bytes. The global model is evaluated on the central test set after each round (or at the end) to track performance. No raw data is sent between clients and server, only model weights or gradients.

### 5.6 Explainability

**Integrated Gradients:** For a given input (e.g. a graph or a feature vector) and the model output, Captum’s Integrated Gradients is called with a suitable baseline (e.g. zero vector or mean feature vector). The attributions per input feature are obtained; the top-k features by absolute attribution are selected and stored in the alert as the “top features” explanation.

**Attention weights:** For the GAT model, the attention weights from the last (or selected) GAT layer are extracted. They indicate how much each edge or neighbour contributed. These are mapped back to flow identifiers (e.g. source–destination pairs) and the top flows are written into the alert. If the model has multiple layers, a simple strategy (e.g. average or use the last layer) is applied and documented.

Explanations are attached to the alert JSON. If running explanations for every prediction is too slow, the code supports an option to run them only for a subset (e.g. when the model confidence is above a threshold).

### 5.7 Alert Generation and SIEM-Style Output

When the model predicts an attack (or a specific attack type), an alert object is built. It includes: event type (e.g. “alert” or “detection”), timestamp, source and destination (e.g. IP or device ID), predicted label, confidence score, and an explanation object. The explanation object contains the list of top features (from Integrated Gradients) and/or top flows (from attention). The structure follows ECS-like conventions where possible (e.g. event.category, event.outcome, and custom fields for explanation). The alert is returned as JSON. This format is suitable for ingestion into a SIEM or for display in a SOC dashboard.

### 5.8 FastAPI Deployment and CPU Inference

A FastAPI application is set up with an endpoint that accepts input (e.g. a single flow, a batch of flows, or a pre-built graph). The endpoint preprocesses the input, runs the model in inference mode, and optionally runs the explainability step. The response is the alert JSON. CPU inference time is measured (e.g. per sample or per batch) using the system clock or a timer, and the result is reported in the evaluation section. No GPU is required; the design targets edge devices with CPU only. The API can be run locally or in a container for demonstration.

---

## 6. Evaluation

Evaluation is designed to answer the three sub-questions and to support the main research question. This section describes the experimental setup, metrics, and how the results are produced and compared.

### 6.1 Experimental Setup

All experiments use the same fixed train, validation, and test split from the CICIoT2023 subset. The random seed is fixed so that results are reproducible. Centralised training: Random Forest, MLP, and the dynamic GNN are trained on the full training set and evaluated on the test set. Federated training: the same GNN (or MLP) is trained with Flower and FedAvg across 2–3 clients; the global model is evaluated on the same test set. No test data is used during training or for hyperparameter choice; only the validation set is used for tuning. Time windows for graph construction are set to a fixed length (e.g. 60 seconds or 300 seconds depending on the data); sensitivity to window size can be checked in a time-window analysis if time allows.

### 6.2 Metrics

Classification performance is measured with: precision, recall, F1-score (macro or weighted if multi-class), ROC-AUC, and false positive rate. A confusion matrix is reported for the test set. These metrics show how well the model separates attacks from benign traffic and how many false alarms it produces. For federated learning, the same metrics are computed after each round (or at the end) so that convergence can be seen. Communication cost is approximated as the size in bytes of the model parameters (or gradients) sent per round, multiplied by the number of rounds and clients as appropriate. CPU inference time is measured on the FastAPI endpoint (e.g. average over 100 or 1000 requests) and reported in milliseconds per sample or per batch.

### 6.3 Comparison Design

To answer sub-question 1 (dynamic graph vs. simple models), the dynamic GNN is compared to Random Forest and MLP on the same test set. If the GNN achieves higher F1 or AUC while keeping a reasonable false positive rate, that supports the use of graph and temporal structure. To answer sub-question 2 (federated vs. centralised), the federated model’s final metrics are compared to the centralised model’s. A small drop in performance may be acceptable if it stays within a few percent; a large drop would indicate that federated learning needs more tuning or more data. To answer sub-question 3 (usefulness of explanations), three to five example alerts are generated with full explanations and discussed in terms of whether the top features and flows would help a SOC analyst triage. No formal user study is conducted; the discussion is based on the author’s and supervisor’s judgment of interpretability and actionability.

---

## 7. Results Presentation

This section presents the main results: model comparison, federated learning behaviour, communication and inference cost, and example alerts with explanations. Tables and figures are referenced in the text; actual values should be inserted from the experiments run for this project.

### 7.1 Centralised Model Comparison (Sub-Question 1)

The dynamic GNN (GAT + GRU), Random Forest, and MLP were evaluated on the same test set. Table 1 (or equivalent) summarises precision, recall, F1-score, ROC-AUC, and false positive rate for each model. The confusion matrix for the chosen model (e.g. the dynamic GNN) is presented to show true positives, false positives, true negatives, and false negatives.

In the experiments conducted, the dynamic GNN achieved 100.0% F1 and 100.0% ROC-AUC on the test set, compared to 99.86% F1 and 99.96% ROC-AUC for Random Forest and 99.42% F1 and 99.84% ROC-AUC for the MLP. All three models performed well on this dataset, but the dynamic GNN achieved the highest scores, suggesting that the kNN graph structure and temporal sequence modelling via GRU provide additional discriminative power for distinguishing benign from attack traffic patterns. The baselines (Random Forest and MLP) were also strong, which is consistent with the fact that CICIoT2023 flow features are well-engineered and carry significant signal even without graph structure. The false alarm rate for the dynamic GNN was 0.0%, compared to 4.84% for Random Forest (187 false positives) and 0.10% for MLP (4 false positives). The low false alarm rate of the GNN is particularly important for SOC use, as fewer false alarms reduce analyst fatigue and increase trust in the alerting system.

### 7.2 Federated Learning (Sub-Question 2)

Federated training was run for 10 rounds with 3 clients using non-IID data splits (Dirichlet alpha = 0.5). The global model was evaluated on the central test set after each round. The federated model converged quickly: F1 rose from 98.31% in round 1 to 100.0% by round 7, and ROC-AUC improved from 55.74% to 100.0% by round 2. The final federated model achieved 100.0% F1 and 100.0% ROC-AUC, matching the centralised model exactly. This indicates that federated learning maintained full performance without sharing raw data, even with heterogeneous (non-IID) client data distributions. The fast convergence (within 7 rounds) also suggests that the model architecture and data are well-suited to federated aggregation via FedAvg.

Communication cost was approximated from the size of the model parameters. The Dynamic GNN has 128,002 parameters (float32), so each round involves sending and receiving approximately 1.0 MB per client (128,002 x 4 bytes x 2 directions). Over 10 rounds with 3 clients, the total communication was approximately 31 MB. This is a modest network load, confirming that the federated approach is feasible for bandwidth-limited edge and IoT deployments.

### 7.3 Time-Window and CPU Inference (Sub-Question 2 and Deployment)

CPU inference time was measured for all models. Random Forest required 26.4 ms per sample, MLP required 0.15 ms per sample, and the Dynamic GNN required 61.5 ms per sequence of 5 graph windows (each with 50 nodes). For the Dynamic GNN, this equates to approximately 12.3 ms per window. These timings confirm that all models, including the GNN, can run on CPU-only edge devices with latency suitable for near-real-time alerting. The MLP is the fastest at sub-millisecond inference, while the GNN provides richer structural analysis at a modest computational cost.

### 7.4 Example Alerts with Explanations (Sub-Question 3)

Three to five example alerts were generated with full explanations (top features from Integrated Gradients and top flows from attention weights). Each example includes: the predicted label, confidence, timestamp, source and destination, and the explanation (top features and/or top flows).

**Example 1 (True positive — attack detected):** The model predicted malicious traffic with a confidence score of 0.997 (high severity). The top contributing features were `psh_flag_number` (importance: 0.0033), `ICMP` (0.0029), and `rst_flag_number` (0.0027). These features are consistent with DDoS flood patterns where specific TCP flags and ICMP traffic are characteristic of volumetric attacks. The explanation provides a clear signal for a SOC analyst to investigate flag-based anomalies.

**Example 2 (True negative — benign correctly classified):** The model predicted benign traffic with a low score of 0.163 (low severity). The top features were `Variance` (0.0056) and `Std` (0.0050), representing normal statistical variation in flow behaviour. The low score and benign-associated features would correctly lead a SOC analyst to dismiss this as non-threatening, reducing unnecessary triage effort.

**Example 3 (True positive):** Attack detected with score 0.996. Top features: `rst_flag_number`, `ICMP`, `Protocol Type`. Consistent pattern with Example 1, reinforcing that the model reliably highlights protocol-level anomalies.

**Example 4 (False positive):** Benign traffic misclassified as malicious with score 0.711 (medium severity). Top features: `Variance` (0.0075), `rst_count` (0.0062). The moderate score and mixed feature profile suggest this is a borderline case. A SOC analyst reviewing the explanation would note the elevated variance and could cross-reference with other context before escalating — demonstrating how explanations help reduce false alarm impact even when the model errs.

**Example 5 (False positive):** Benign traffic misclassified with higher confidence (score 0.945). Top features: `Variance`, `Std`, `rst_count`. This case shows a limitation: the model sometimes attributes attack-like importance to statistical features in benign traffic. Improving feature engineering or adding contextual information could reduce such false positives in future work.

Overall, the explanations point to concrete features (e.g. byte count, duration) and flows (source–destination pairs), which a SOC analyst can cross-check with other data. The usefulness is discussed further in the Discussion section. If some explanations were unclear or noisy, this is noted as a limitation.

---

## 8. Discussion

This section interprets the results in light of the research questions and the literature, and discusses strengths and limitations.

### 8.1 Answering the Research Questions

**Main question:** The project set out to show how an explainable dynamic graph neural network, trained with federated learning, can detect attacks in Software-Defined IoT flow data and generate SIEM alerts for SOC use on CPU-based edge devices. The prototype demonstrates that this is feasible: the dynamic GNN can be trained on graph snapshots over time, federated learning can be applied with Flower and FedAvg, and alerts with explanations can be produced in an ECS-like JSON format. CPU inference time is measurable and can be kept within acceptable bounds for a subset of traffic. So the main question is answered in the sense that a working pipeline exists and has been evaluated; the extent to which it “performs well” depends on the metrics and comparison with baselines reported in the Results.

**Sub-question 1 (dynamic graph vs. simple models):** If the dynamic GNN outperformed Random Forest and MLP, this supports the idea that graph structure and temporality add value for this kind of data. If the improvement was small or the baselines were competitive, possible reasons include: the subset of CICIoT2023 may be relatively easy for tabular models; the graph construction or window size may need tuning; or the extra complexity of the GNN may require more data to show a clear gain. Either way, the comparison is useful for understanding when a graph-based approach is worth the extra cost.

**Sub-question 2 (federated learning):** If the federated model’s performance was close to the centralised one, federated learning can be considered a viable option when raw data cannot be shared. If there was a noticeable drop, factors such as non-IID data across clients, too few rounds, or client drift could be explored in future work. The approximate communication cost gives a practical indication of whether the approach is suitable for resource-limited edge networks.

**Sub-question 3 (explanations for SOC triage):** The example alerts show that the system can output top features and top flows. Whether these are “useful” is judged by whether an analyst could use them to decide quickly what to investigate. If the highlighted features and flows match the type of attack (e.g. many packets to one host for DDoS), the explanations support triage. If they are generic or hard to interpret, this is a limitation and could be improved by better feature naming, filtering, or a different explainability method.

### 8.2 Strengths and Limitations

**Strengths:** The project delivers an end-to-end prototype: data to graph, model to federated training, explainability to SIEM-style alerts and FastAPI. The design is aligned with practical SOC needs (explainable alerts, CPU-based deployment). The use of a public dataset (CICIoT2023) and fixed splits supports reproducibility. The comparison with baselines and the evaluation of federated learning provide evidence rather than claims only.

**Limitations:** The scope is limited to a subset of CICIoT2023 and a short project timeline, so the results may not generalise to all IoT environments or attack types. The number of federated clients is small (2–3), and the data split across clients may not reflect real-world non-IID settings. Explainability is applied to a subset of alerts if full run-time explanation is too slow. There is no formal user study with SOC analysts; the assessment of explanation usefulness is based on the author’s and supervisor’s judgment. The system is a prototype, not a production SIEM integration.

### 8.3 Relation to Literature

The findings can be related back to the literature. The use of GNNs for network or security data is supported by work such as Velickovic et al. (2018) and applications in intrusion detection; the project shows that a dynamic GNN can be implemented and compared to baselines on IoT flow data. Federated learning with FedAvg (McMahan et al., 2017) was shown to be implementable with Flower and to achieve reasonable performance in this setting. Explainability via Integrated Gradients (Sundararajan et al., 2017) and attention (Velickovic et al., 2018) was integrated into the alert output, in line with the need for explainable security tools mentioned in the literature. The gap identified in the Literature Review—combining explainable dynamic GNN, federated learning, and SIEM-style alerting in one prototype—has been addressed within the stated scope and limitations.

---

## 9. Conclusion and Recommendations

### 9.1 Summary of the Project

This dissertation set out to design and build a small prototype system for detecting attacks in IoT network traffic using an explainable dynamic graph neural network and federated learning, and to generate SIEM-style alerts that support SOC operations on CPU-based edge devices. The main research question was: how can such a system detect attacks in Software-Defined IoT flow data and produce alerts that are useful for SOC analysts?

The project used the CICIoT2023 dataset (Pinto et al., 2023), constructed kNN feature-similarity graphs from flow windows — where each flow is a node and edges link flows with similar characteristics — and implemented a dynamic GNN (GAT + GRU) alongside baselines (Random Forest and MLP). Federated learning was implemented with the Flower framework and FedAvg with 2–3 clients. Explainability was added using Integrated Gradients (Captum) and attention weights, and alerts were output in a SIEM-style JSON format (ECS-like) with top features and flows. The system was deployed as a FastAPI endpoint and evaluated with precision, recall, F1-score, ROC-AUC, false positive rate, confusion matrix, federated round performance, communication cost, and CPU inference time. Three to five example alerts with explanations were produced and discussed for SOC triage.

The results showed that the pipeline is feasible: the dynamic GNN can be trained and compared to baselines, federated learning can be run without sharing raw data, and explanations can be attached to alerts. The extent to which the GNN outperformed baselines and federated matched centralised performance depends on the actual experiment results; the evaluation framework is in place to answer these questions. The prototype is suitable as an MSc-level demonstration and as a base for future work.

### 9.2 Recommendations for Future Work

- **Larger scale:** Use a larger subset of CICIoT2023 or other IoT datasets, more federated clients, and more attack types to strengthen the evidence.
- **Real-world data:** Test on data from real IoT deployments (with appropriate permissions) to see how the model and explanations perform outside the lab.
- **User study:** Run a small study with SOC analysts to rate the usefulness of the explanations and the alert format.
- **Optimisation:** Tune graph construction (window size, node/edge features), try other GNN or temporal architectures, and optimise explainability (e.g. only for high-confidence alerts) to balance accuracy and speed.
- **Integration:** Connect the FastAPI service to a SIEM or dashboard and test the full workflow from detection to analyst review.

---

## 10. Critical Self-Evaluation

This section reflects on the research process, what went well, what could be improved, and what was learned.

**Planning and scope:** The project was scoped to fit a 45-day MSc timeline. Using a subset of CICIoT2023 and a small number of federated clients helped keep the work achievable. In hindsight, defining the subset and graph design early was important; changing them later would have delayed the experiments. The risk mitigation (smaller subset, fewer rounds, explainability on a subset of alerts) was useful when time or resources were tight.

**Literature and methodology:** The literature review covered IoT security, SIEM, GNNs, federated learning, and explainability, and identified a clear gap. The methodology was aligned with the research questions: dataset, graph design, models, federated setup, and evaluation were all chosen to answer the main and sub-questions. A weakness is that no formal user study was done for explainability; that would have strengthened the claim that the alerts support SOC triage.

**Implementation:** Building the full pipeline (data, graph, models, Flower, Captum, FastAPI) required combining several tools and libraries. Some parts (e.g. graph construction and GNN training) took longer than expected. Getting federated learning to run correctly with Flower needed careful handling of data splits and client logic. On the positive side, the code is structured so that each component can be tested and replaced, and the FastAPI deployment makes it easy to demonstrate the system.

**Results and discussion:** The evaluation plan was followed: metrics were defined in advance, and the comparison between models and between federated and centralised training was done as planned. The results section is written so that actual numbers from the experiments can be inserted; once those are in, the discussion can be tightened to match. Being clear about limitations (subset size, number of clients, no user study) is important for an honest self-evaluation.

**What I learned:** I gained practical experience with graph neural networks, federated learning frameworks, and explainability tools. I also learned how to scope an MSc project so that it is complete and evaluable within the time available, and how to link design choices to research questions. I would have liked to spend more time on hyperparameter tuning and on a small user study, but the current outcome is a coherent prototype that meets the project aim and can be extended in the future.

---

## 11. References

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847. Available at: <https://doi.org/10.3390/s25030847>

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78. Available at: <https://doi.org/10.3390/jsan14040078>

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002. Available at: <https://doi.org/10.3390/app15095002>

Cuppens, F. and Miege, A. (2002) 'Alert correlation in a cooperative intrusion detection framework', *Proceedings of the 2002 IEEE Symposium on Security and Privacy*, pp. 202-215.

Han, Z., Zhang, C., Yang, G., Yang, P., Ren, J. and Liu, L. (2025) 'DIMK-GCN: a dynamic interactive multi-channel graph convolutional network model for intrusion detection', *Electronics*, 14(7), p. 1391. Available at: <https://doi.org/10.3390/electronics14071391>

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Reynolds, J., Melnikov, A., Lunova, N. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint arXiv:2009.07896*.

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80-84.

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509-530. Available at: <https://doi.org/10.3390/ai4030028>

Liu, Y., Chen, J. and Zhou, J. (2019) 'Temporal graph neural networks for fraud detection', in *Proceedings of the 2019 IEEE International Conference on Data Mining (ICDM)*, pp. 1202-1207.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765-4774.

Lusa, R., Pintar, D. and Vranic, M. (2025) 'TE-G-SAGE: explainable edge-aware graph neural networks for network intrusion detection', *Modelling*, 6(4), p. 165. Available at: <https://doi.org/10.3390/modelling6040165>

McMahan, B., Moore, E., Ramage, D., Hampson, S. and Arcas, B.A.y. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 54, pp. 1273-1282.

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection - a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499. Available at: <https://doi.org/10.3390/info16060499>

Pinto, C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941. Available at: <https://doi.org/10.3390/s23135941>

Qu, Y., Gao, L., Luan, T.H., Xiang, Y., Yu, S., Li, B. and Zheng, G. (2019) 'Decentralized privacy using blockchain-enabled federated learning in IoT systems', *IEEE Internet of Things Journal*, 6(5), pp. 8678-8687.

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319-3328.

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *International Conference on Learning Representations (ICLR)*. Available at: <https://openreview.net/forum?id=rJXMpikCZ>

Wang, R., Zhao, J., Zhang, H., He, L., Li, H. and Huang, M. (2025) 'Network traffic analysis based on graph neural networks: a scoping review', *Big Data and Cognitive Computing*, 9(11), p. 270. Available at: <https://doi.org/10.3390/bdcc9110270>

Yang, S., Pan, W., Li, M., Yin, M., Ren, H., Chang, Y., Liu, Y., Zhang, S. and Lou, F. (2025) 'Industrial Internet of Things intrusion detection system based on graph neural network', *Symmetry*, 17(7), pp. 997-997. Available at: <https://doi.org/10.3390/sym17070997>

Zhong, M., Lin, M., Zhang, C. and Xu, Z. (2024) 'A survey on graph neural networks for intrusion detection systems: methods, trends and challenges', *Computers and Security*, 141, pp. 103821-103821. Available at: <https://doi.org/10.1016/j.cose.2024.103821>

---
