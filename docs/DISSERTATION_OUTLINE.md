# Dissertation Outline: IoT Network Traffic Detection via Dynamic GNN, FL, and Explainability

## Full Dissertation Structure

### 1. Introduction
- **Problem**: IoT traffic detection in SOC environments; need for privacy-preserving and explainable methods.
- **Objectives**:
  - Graph-based model for time-series IoT flow data (devices → nodes, flows → edges).
  - Comparison of RF, MLP, and Dynamic GNN (central and federated).
  - Federated Learning with non-IID data (2–3 clients).
  - Explainability (top flows and features) and SIEM-ready alert output.
- **Contributions**: Dynamic GNN design, FL deployment, explainability pipeline, SIEM JSON API.
- **Thesis statement**: Dynamic GNNs with FL and explainability improve detection and operational usability in SOCs.

### 2. Related Work
- IoT security and traffic classification.
- Graph Neural Networks for network security.
- Federated Learning for intrusion detection.
- Explainable AI in security (XAI, SOC workflows).
- CICIoT and similar datasets.

### 3. Methods
- **Data**: CICIoT2023; binary task (benign vs malicious); relevant features (device IDs, flow stats).
- **Preprocessing**: Windowing, aggregation per (src, dst), sequence construction.
- **Models**:
  - Baseline: Random Forest, MLP.
  - Dynamic GNN: Graph per window (GraphSAGE/GAT), GRU over graph embeddings, classifier.
- **Federated Learning**: FedAvg (Flower); non-IID split (by device group/day); 2–3 clients.
- **Explainability**: Captum (Integrated Gradients); top flows and top features per alert.
- **Evaluation**: Precision, Recall, F1, ROC-AUC, confusion matrix; FL rounds and communication cost; inference time.

### 4. Results
- **Results table**: RF vs MLP vs Dynamic GNN (central) vs Dynamic GNN (federated) — F1, ROC-AUC, inference time.
- **FL**: Central vs federated performance; communication cost (bytes per round).
- **Explainability**: Example alerts with top flows and features.
- **SIEM**: JSON alert schema and sample output.

### 5. Discussion
- Federated learning challenges (non-IID, convergence).
- Explainability insights for analysts.
- Limitations (dataset, number of clients, single binary task).
- Edge inference feasibility (inference time on CPU).

### 6. Conclusion
- Summary of findings; future work (more clients, real deployment, multi-class).

### References & Appendices
- Reproducibility: seeds, config, code structure.
- Appendix: Full hyperparameters; extra figures (ROC, confusion matrices).

---

## Paper-Ready Draft (6–8 pages)

### Abstract
- One paragraph: problem, approach (Dynamic GNN + FL + explainability), main result (e.g. F1, FL vs central, SIEM output).

### 1. Introduction (≈1 page)
- Motivation, objectives, contributions.

### 2. Methodology (≈2 pages)
- Data and preprocessing; Dynamic GNN architecture; FL setup; explainability; evaluation metrics.

### 3. Results (≈1.5 pages)
- Table: model comparison (RF, MLP, GNN central, GNN federated).
- FL rounds and communication cost.
- Explainability example; SIEM JSON snippet.

### 4. Discussion (≈0.5 page)
- FL trade-offs; explainability value; limitations.

### 5. Conclusion (≈0.5 page)
- Summary and future work.

### References

### Key Figures for Publication
1. **System overview**: Data → preprocessing → Dynamic GNN / FL → explainability → SIEM.
2. **Dynamic GNN architecture**: Graph per window → SAGE/GAT → GRU → classifier.
3. **Results table**: Precision, Recall, F1, ROC-AUC, inference time.
4. **ROC curves**: Central vs federated (or RF/MLP vs GNN).
5. **FL**: Loss/metrics vs round; communication cost.
6. **Explainability**: Example alert with top flows and top features (table or small graph).
7. **SIEM JSON**: Sample alert payload.

---

*Use this outline to draft chapters and the 6–8 page paper; fill in actual numbers and figures from your experiments.*
