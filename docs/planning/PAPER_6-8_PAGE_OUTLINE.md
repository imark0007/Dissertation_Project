# 6–8 Page Paper Outline

**Title (example):** IoT Network Traffic Detection Using Dynamic Graph Neural Networks with Federated Learning and Explainability for SOC Environments

---

## Abstract (150–200 words)
- **Problem**: Detecting malicious IoT traffic in Security Operations Centers (SOCs) while preserving privacy and supporting analyst decisions.
- **Approach**: Dynamic Graph Neural Networks (GNNs) on time-windowed flow data (devices as nodes, flows as edges), with Federated Learning (FedAvg) for multi-site training and Captum-based explainability for alerts.
- **Results**: (Fill: F1/ROC-AUC vs RF/MLP; central vs federated; inference time; SIEM JSON output.)
- **Conclusion**: Dynamic GNN with FL and explainability is viable for IoT traffic detection and SOC integration.

---

## 1. Introduction (~1 page)
- **Paragraph 1**: IoT growth and security challenges; need for automated, interpretable detection.
- **Paragraph 2**: GNNs for relational/flow data; limitations of centralised training (privacy, data locality).
- **Paragraph 3**: Our contributions:
  - Dynamic GNN for time-series IoT flow data.
  - Federated training with non-IID simulation (2–3 clients).
  - Explainability (top flows and features) and SIEM-ready JSON alerts.
- **Paragraph 4**: Paper structure.

---

## 2. Methodology (~2 pages)

### 2.1 Data and Preprocessing
- Dataset: CICIoT2023 (or similar); binary labels (benign vs malicious).
- Windowing (e.g. 60 s), sequence length (e.g. 10 windows).
- Graph construction: devices = nodes, flows = edges; aggregated features per edge/node.

### 2.2 Model Architecture
- **Baselines**: Random Forest (scikit-learn), MLP (PyTorch).
- **Dynamic GNN**: Per-window graph → GraphSAGE or GAT → graph-level embedding → sequence of embeddings → GRU → binary classifier.
- Brief equation or diagram reference.

### 2.3 Federated Learning
- FedAvg (Flower); 2–3 clients; non-IID split (e.g. by device group or day).
- Local epochs, rounds; communication of model parameters.

### 2.4 Explainability and SIEM Output
- Captum Integrated Gradients (or attention) for attributions.
- Top-k flows and top-k features per alert; JSON schema for SIEM.

### 2.5 Evaluation
- Metrics: Precision, Recall, F1, ROC-AUC; confusion matrix.
- FL: metrics per round; communication cost (bytes).
- Inference time on CPU (edge feasibility).

---

## 3. Results (~1.5 pages)

### 3.1 Model Comparison
- **Table 1**: RF | MLP | Dynamic GNN (central) | Dynamic GNN (federated) — Precision, Recall, F1, ROC-AUC, Inference (ms).
- Short commentary: GNN vs baselines; central vs federated gap.

### 3.2 Federated Learning
- **Figure**: Metric (e.g. loss or F1) vs round; optional: bytes per round.
- Comment on non-IID impact and convergence.

### 3.3 Explainability and SIEM
- **Figure/Table**: One example alert with top-5 flows and top-5 features.
- **Snippet**: SIEM JSON alert structure (key fields only).

---

## 4. Discussion (~0.5 page)
- FL: data heterogeneity, communication cost, scalability.
- Explainability: usefulness for analysts; limitations (attribution interpretation).
- Limitations: single dataset, simulated FL, binary task only.

---

## 5. Conclusion (~0.5 page)
- Summary: Dynamic GNN + FL + explainability for IoT traffic detection and SIEM integration.
- Future work: more clients, real federated deployment, multi-class, other datasets.

---

## References
- CICIoT2023; PyTorch Geometric; Flower; Captum; relevant GNN/FL/XAI security papers.

---

## Suggested Figures
| Fig | Content |
|-----|--------|
| 1 | System pipeline: Data → Preprocessing → Dynamic GNN/FL → Explainability → SIEM |
| 2 | Dynamic GNN: graph per window → GNN → GRU → classifier |
| 3 | Results table (model comparison) |
| 4 | ROC curves (RF, MLP, GNN central, GNN federated) |
| 5 | FL: metric vs round and/or communication cost |
| 6 | Example explainability output (top flows, top features) |
| 7 | SIEM JSON sample |

Fill sections with your actual experiments and numbers; keep abstract and conclusion aligned with the final results.
