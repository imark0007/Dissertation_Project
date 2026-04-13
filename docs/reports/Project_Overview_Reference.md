# Full Project Overview and Reference Document

**Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning**

**Author:** Arka Talukder | **Student Number:** B01821011 | **Programme:** MSc Cyber Security (Full-time)  
**Supervisor:** Dr. Raja Ujjan | **University of the West of Scotland**

*This document provides a complete reference of the project for analysis by AI agents or for anyone to understand the whole project. Upload this PDF to any AI agent as context when discussing or extending this work.*

---

## 1. Project Summary

This MSc dissertation project designs and builds a prototype system that detects attacks in IoT network traffic using a dynamic graph neural network (GNN) with federated learning and explainability. The system generates SIEM-style alerts for Security Operations Centre (SOC) use on CPU-based edge devices.

**Main research question:** How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?

**Sub-questions:**
1. Does a dynamic graph model perform better than simple models (Random Forest, MLP)?
2. Can federated learning maintain similar performance without sharing raw data?
3. Can the model generate useful explanations for SOC alert triage?

**Keywords:** IoT security, dynamic graph neural network, federated learning, SIEM, explainable AI, edge AI, SOC, CICIoT2023

---

## 2. Background and Motivation

- IoT devices are often insecure (default passwords, weak encryption). When compromised, they can be used in botnets or for data theft.
- SOCs use SIEM systems but face alert fatigue (too many alerts, false positives). Unexplained alerts slow triage.
- Network traffic has structure (devices connect over time). Graph-based models can capture this; dynamic GNNs learn from graph evolution.
- IoT data is often distributed; federated learning trains without centralising raw data.
- Edge deployment needs CPU-only inference; the system targets lightweight, practical SOC use.
- Scope: 45-day MSc project; prototype kept small and achievable.

---

## 3. Dataset

**CICIoT2023** (Pinto et al., 2023)
- 46 numeric flow-level features (no IP addresses or device IDs in the public version)
- Labels: Benign + 34 attack classes (DDoS, DoS, reconnaissance, brute force, spoofing, etc.)
- Pre-split: train.csv, test.csv, validation.csv (approx. 1.6 GB train)
- Heavy class imbalance at flow level (approx. 97.8% attack, 2.2% benign)
- Flow features include: flow_duration, packet/byte counts, TCP flags, protocol indicators (HTTP, DNS, ICMP, etc.), statistical aggregates (Min, Max, AVG, Std, Variance, etc.)

---

## 4. Graph Design (kNN Feature-Similarity)

Because device identifiers are absent, a device-based graph is not possible. Instead:

- **Nodes:** Each flow record; 46 features per node
- **Edges:** k-nearest neighbours in feature space (Euclidean distance, k=5)
- **Windows:** 50 flows per window; graph label from pool (stratified windowing for class balance)
- **Sequences:** 5 consecutive windows per sample for the GRU
- **Stratified windowing:** Benign and attack flows separated; windows built from each pool; minority_stride=25 for augmentation; balanced and shuffled before training
- Supported by Ngo et al. (2025) and Basak et al. (2025) for attribute-based graph construction when topology is absent

---

## 5. Architecture and Models

**Pipeline:** Raw flows -> StandardScaler -> kNN graphs (50 flows/window) -> Sequences of 5 windows -> GATConv (2 layers, 4 heads) -> Global mean pool -> GRU -> Classifier

**Models:**
| Model | Input | Purpose |
|-------|-------|---------|
| Random Forest | Flat tabular features | Baseline (n_estimators=200, max_depth=20) |
| MLP | Flat tabular features | Neural baseline (hidden 128,64,32) |
| Dynamic GNN (GAT + GRU) | Graph sequences | Main model (node_dim=46, hidden=64, 2 GAT layers, 4 heads) |

**Federated learning:** Flower, FedAvg, 3 clients, Dirichlet non-IID split (alpha=0.5), 10 rounds, 2 local epochs

**Explainability:** Integrated Gradients (Captum) + GAT attention weights -> top features and top flows in alerts

**SIEM output:** ECS-formatted JSON alerts via FastAPI (POST /score endpoint)

---

## 6. Key Configuration (config/experiment.yaml)

- Graph: window_size=50, knn_k=5, sequence_length=5, minority_stride=25
- FL: num_clients=3, num_rounds=10, alpha=0.5, local_epochs=2
- GNN: node_dim=46, hidden_dim=64, 2 GAT layers, 4 heads, GRU hidden=64, batch_size=16
- RF: n_estimators=200, max_depth=20
- MLP: hidden_dims=[128,64,32], dropout=0.2, batch_size=256
- Training: early_stopping_patience=5, class_weight_auto=true

---

## 7. Evaluation Metrics

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 x (Precision x Recall) / (Precision + Recall)
- False alarm rate (FAR) = FP / (FP + TN)
- ROC-AUC, confusion matrix, CPU inference time, communication cost (FL)
- Test set held out; validation set for hyperparameter choice; fixed seed for reproducibility

---

## 8. Results (from experiments)

| Model | Precision | Recall | F1 | ROC-AUC | Inference (ms) |
|-------|-----------|--------|-----|---------|----------------|
| Random Forest | 0.9989 | 0.9984 | 0.9986 | 0.9996 | 46.09 |
| MLP | 1.0000 | 0.9885 | 0.9942 | 0.9984 | 0.66 |
| Central GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.70 |
| Federated GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20.99 |

- Federated model matched centralised; F1 converged from 98.31% (round 1) to 100% by round 7
- Communication cost: approx. 31 MB over 10 rounds (3 clients); GNN has 128,002 parameters (float32)
- GNN achieved 0% false alarm rate vs 4.84% (RF) and 0.10% (MLP); suitable for SOC use
- All models run on CPU; GNN approx. 12 ms per window

---

## 9. Example Alerts with Explanations

- True positive (attack): score 0.997; top features psh_flag_number, ICMP, rst_flag_number (DDoS-like)
- True negative (benign): score 0.163; top features Variance, Std (normal variation)
- False positive: score 0.711; top features Variance, rst_count (borderline case; explanations help analyst decide)

---

## 10. Repository Structure

config/experiment.yaml - All hyperparameters
data/raw/ - CICIoT2023 CSVs
data/processed/ - Normalised parquets + scaler.joblib
data/graphs/ - PyG graph objects per split
src/data/preprocess.py - Clean, normalise, binary labels
src/data/graph_builder.py - kNN graph construction, stratified windowing
src/data/dataset.py - GraphSequenceDataset, DataLoaders
src/models/dynamic_gnn.py - GAT + GRU + classifier
src/models/baselines.py - RF + MLP
src/models/trainer.py - Training loop, early stopping, class weights
src/federated/data_split.py - Dirichlet non-IID splitting
src/federated/client.py, server.py, run_federated.py - Flower FL
src/explain/explainer.py - Captum IG + GAT attention
src/siem/alert_formatter.py, api.py - ECS JSON, FastAPI
src/evaluation/metrics.py - Precision, Recall, F1, ROC, CM plots
scripts/run_all.py - Full pipeline
scripts/run_fl_simulation.py - FL simulation
scripts/generate_alerts_and_plots.py - Example alerts, FL plots
scripts/md_to_docx.py - Interim report to Word
notebooks/01_explore_ciciot.ipynb, 02_federated_vs_central.ipynb, 03_alert_examples.ipynb
tests/test_api.py - API tests

---

## 11. How to Run

Full pipeline: python scripts/run_all.py --config config/experiment.yaml
Quick test (10k rows): python scripts/run_all.py --config config/experiment.yaml --nrows 10000
Federated learning: Run server + 3 clients in separate terminals (after split_and_save)
API: uvicorn src.siem.api:app --reload -> POST /score for inference + ECS alert
Tests: python tests/test_api.py

---

## 12. Strengths and Limitations

Strengths: End-to-end prototype; public dataset; reproducible; CPU-based; explainable alerts; FL without raw data sharing; GNN achieved lowest false alarm rate

Limitations: Subset of CICIoT2023; small number of FL clients (3); no formal user study; prototype not production SIEM; explainability can be slow (applied to subset of alerts); results may not generalise to all IoT environments

---

## 13. Future Work

- Larger dataset, more clients, more attack types
- Real-world IoT deployment testing
- User study with SOC analysts
- Graph construction tuning; other GNN architectures
- SIEM/dashboard integration

---

## 14. Key References

Pinto et al. (2023) - CICIoT2023 dataset
Velickovic et al. (2018) - Graph Attention Networks
McMahan et al. (2017) - FedAvg
Sundararajan et al. (2017) - Integrated Gradients
Kokhlikyan et al. (2020) - Captum
Ngo et al. (2025), Basak et al. (2025) - kNN/attribute-based graph construction for IDS
Lazzarini et al. (2023), Albanbay et al. (2025) - Federated learning for IoT IDS
Alabbadi and Bajaber (2025) - XAI for IoT intrusion detection

---

Document generated for project reference. Use this as a single source to understand the full project when uploading to AI agents or sharing with others.
