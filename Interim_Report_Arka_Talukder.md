# MSc Interim Report

**Project Title:** Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning

**Author:** Arka Talukder  
**Student Number:** B01821011  
**Programme:** MSc Cyber Security (Full-time)  
**Supervisor:** Dr. Raja Ujjan  
**University of the West of Scotland**

**Word count:** approximately 4,500 words (maximum 5,000 words)

---

## Introduction

This interim report is submitted at around the halfway stage of my MSc project. It describes in detail how I am conducting my research and gives my supervisor and moderator a clear view of the whole project. It is not a first draft of the final report, and it is not the first few chapters of the final report. The report has three required parts: (1) a summary literature review, (2) research methodology, and (3) a plan for completion. No final results are included here; results will be gathered, analysed and presented in the final report.

---

## 1. Summary Literature Review

This section critically summarises the current academic literature in my subject area and provides the academic framework for the research I am undertaking.

### 1.1 IoT Security Landscape and Software-Defined IoT

The rapid proliferation of Internet of Things (IoT) devices has introduced significant security challenges across consumer, industrial, and critical infrastructure domains. IoT devices frequently ship with default credentials, limited cryptographic capability, and infrequent firmware updates, making them attractive targets for adversaries (Kolias et al., 2017). Once compromised, these devices serve as nodes in large-scale botnets such as Mirai, capable of launching volumetric distributed denial-of-service (DDoS) attacks exceeding 1 Tbps (Antonakakis et al., 2017). The heterogeneity and scale of IoT ecosystems further complicate defence, as traditional perimeter-based security models struggle to monitor thousands of constrained devices with diverse communication patterns.

Software-Defined IoT (SDIoT) architectures address this complexity by decoupling the control plane from the data plane, allowing centralised network management through SDN controllers such as OpenFlow-based platforms (Bera, Misra and Vasilakos, 2017). In SDIoT environments, the SDN controller has visibility over all forwarding decisions and can export flow-level telemetry—including packet counts, byte counts, flag distributions, and protocol breakdowns—for every communication pair traversing the network. This centralised flow collection capability makes SDIoT environments particularly well-suited to machine-learning-based intrusion detection, as the controller provides a natural aggregation point for network-wide feature extraction without requiring agents on resource-constrained end devices (Bera, Misra and Vasilakos, 2017). The flow-level feature set used in this project, drawn from the CICIoT2023 dataset (Pinto et al., 2023), mirrors the telemetry that an SDN controller would produce, grounding the research in a realistic SDIoT deployment scenario.

However, Bera, Misra and Vasilakos (2017) focused primarily on architectural benefits of SDN for IoT management rather than demonstrating ML-based detection within that architecture, leaving a gap between the SDIoT vision and practical intrusion detection implementations. This project bridges that gap by building a detection system that operates on flow-level features consistent with SDN controller exports.

### 1.2 Machine Learning for Network Intrusion Detection

Machine learning has become the dominant paradigm for network intrusion detection, with models trained on flow-level features to classify traffic as benign or malicious. Random Forest classifiers have demonstrated strong performance on tabular network datasets due to their resistance to overfitting and implicit feature selection through ensemble averaging (Ahmad et al., 2021). On the CICIoT2023 dataset specifically, Pinto et al. (2023) reported that Random Forest achieved classification accuracy above 99% for binary detection, establishing a strong baseline for comparison.

Deep learning approaches, particularly Multi-Layer Perceptrons (MLPs), offer the advantage of learning non-linear decision boundaries without manual feature engineering (Shone et al., 2018). Shone et al. (2018) demonstrated that deep autoencoders combined with shallow classifiers achieved competitive performance on the NSL-KDD dataset. However, their evaluation was limited to NSL-KDD, which is significantly smaller and less diverse than CICIoT2023, and they did not consider flow-level temporal dynamics or relational structure between network entities. Both Random Forest and MLP approaches treat each flow independently, discarding any relational or temporal context between flows—a limitation that graph-based methods can address.

A key limitation shared across these tabular approaches is that they assume flow independence. In real networks, flows are contextually related: a reconnaissance scan precedes exploitation; command-and-control traffic correlates temporally with data exfiltration. Ignoring these relationships limits detection capability for multi-stage attacks (Caville et al., 2022).

### 1.3 Graph Neural Networks for Intrusion Detection

Graph Neural Networks (GNNs) have emerged as a promising approach for network intrusion detection because they can model relational structure between network entities. In a typical formulation, nodes represent network flows or hosts, edges represent communication relationships, and the GNN learns representations that incorporate neighbourhood context through message-passing mechanisms (Caville et al., 2022).

Graph Attention Networks (GATs) extend this framework by learning to assign different importance weights to different neighbours, allowing the model to focus on the most relevant connections during aggregation (Velickovic et al., 2018). This attention mechanism provides two benefits: improved classification performance through selective aggregation, and interpretable attention coefficients that indicate which neighbouring flows most influenced a detection decision.

Basak et al. (2025) applied GNN-based intrusion detection using attribute-similarity graphs on the UNSW-NB15 dataset, where nodes were constructed from flow features and edges from feature-space proximity. They reported F1 scores exceeding 97% and demonstrated that GNN-based approaches could match or exceed tree-based baselines. However, their work used static graphs that treated the entire dataset as a single snapshot, ignoring the temporal evolution of network behaviour. Furthermore, UNSW-NB15 is a dated dataset with fewer attack categories than CICIoT2023, limiting the generalisability of their findings to modern IoT threat landscapes.

Ngo et al. (2025) advanced this line of research by applying GNNs to IoT-specific datasets with kNN-based graph construction, reporting that k-nearest-neighbour edges in feature space effectively captured behavioural similarity between flows even when device topology was unavailable. Their approach validated k=5 as a reasonable default for feature-similarity graphs. However, Ngo et al. (2025) also employed static graph snapshots without temporal modelling, meaning their system could not capture how attack patterns evolve over time windows—a critical capability for detecting slow-and-low attacks or multi-phase campaigns.

This project extends both works by introducing temporal modelling through a GRU layer that processes sequences of graph snapshots, enabling the system to learn from graph evolution rather than individual static snapshots.

### 1.4 Federated Learning for Privacy-Preserving IoT Security

Federated learning (FL) enables collaborative model training across distributed data sources without centralising raw data, addressing both privacy concerns and regulatory constraints such as GDPR (McMahan et al., 2017). In the Federated Averaging (FedAvg) algorithm, each client trains on local data and transmits only model parameter updates to a central server, which aggregates them through weighted averaging. This paradigm is particularly relevant for IoT security, where traffic data may be generated across multiple organisational boundaries or geographic locations and cannot be legally or practically pooled.

Lazzarini et al. (2023) applied federated learning to network intrusion detection using the UNSW-NB15 dataset with a federated MLP architecture. They reported that the federated model matched centralised performance within 2% F1 after sufficient training rounds. However, their study used only two clients with IID (independent and identically distributed) data partitions, which represents an optimistic scenario. In real-world IoT deployments, different network segments observe different traffic distributions—an enterprise IoT subnet dominated by sensor telemetry will have fundamentally different flow characteristics than a consumer smart-home network. This non-IID heterogeneity is known to degrade FedAvg convergence (Kairouz et al., 2021).

Albanbay et al. (2025) directly addressed the non-IID challenge by evaluating federated intrusion detection under Dirichlet-distributed data splits with varying alpha values. They demonstrated that lower alpha values (higher heterogeneity) caused up to 8% F1 degradation compared to IID settings, with convergence requiring approximately twice as many communication rounds. Their work, however, used flat tabular models and did not explore whether graph-structured models exhibit different sensitivity to non-IID conditions. Since graph models aggregate information across neighbourhoods, they may exhibit different convergence dynamics under non-IID splits—a question this project investigates by combining GNNs with Dirichlet non-IID federated learning (alpha=0.5).

### 1.5 Explainable AI for Security Operations

Security Operations Centres (SOCs) face chronic alert fatigue, with analysts processing thousands of alerts daily, many of which are false positives or lack sufficient context for efficient triage (Alahmadi et al., 2022). Machine learning models that produce only a binary classification or confidence score without explanation exacerbate this problem, as analysts cannot assess whether the model's reasoning aligns with known attack patterns or reflects spurious correlations.

Explainable AI (XAI) methods address this by attributing model decisions to specific input features. Integrated Gradients (Sundararajan, Taly and Yan, 2017) computes feature attributions by accumulating gradients along a straight-line path from a baseline input to the actual input, satisfying desirable axiomatic properties including sensitivity (every feature that changes the prediction receives non-zero attribution) and implementation invariance (attributions depend only on the function, not the model architecture). Captum (Kokhlikyan et al., 2020) provides a production-quality implementation of Integrated Gradients for PyTorch models, which this project uses.

Alabbadi and Bajaber (2025) applied SHAP-based explanations to IoT intrusion detection models, demonstrating that feature attribution improved analyst confidence in alert triage decisions by identifying which flow characteristics drove each detection. However, their study relied on SHAP's kernel-based approximation for deep models, which is computationally expensive—potentially prohibitive for real-time edge deployment. SHAP computation for a single prediction on a deep model can exceed 10 seconds, which is impractical for a system targeting sub-second alert generation. Integrated Gradients, by contrast, requires only a single forward and backward pass with interpolation, making it substantially faster and more suitable for edge CPU deployment.

Additionally, GAT attention weights provide a complementary form of explanation: they indicate which neighbouring flows in the graph most influenced the classification of a given node. Combining feature-level attributions (Integrated Gradients) with structural explanations (attention weights) provides richer context for SOC analysts than either method alone—a combination not explored by Alabbadi and Bajaber (2025).

### 1.6 Research Gap and Contribution

Table 1 summarises the capabilities of the key related works against the four pillars of this project.

**Table 1: Comparison of Related Work Against Project Pillars**


| Study                       | Dynamic GNN | Federated Learning | Explainability   | SIEM Integration |
| --------------------------- | ----------- | ------------------ | ---------------- | ---------------- |
| Basak et al. (2025)         | Static GNN  | ✗                  | ✓ (partial)      | ✗                |
| Ngo et al. (2025)           | Static GNN  | ✗                  | ✗                | ✗                |
| Lazzarini et al. (2023)     | ✗           | ✓ (IID only)       | ✗                | ✗                |
| Albanbay et al. (2025)      | ✗           | ✓ (non-IID)        | ✗                | ✗                |
| Alabbadi and Bajaber (2025) | ✗           | ✗                  | ✓ (SHAP)         | ✗                |
| This project                | ✓ (GAT+GRU) | ✓ (non-IID)        | ✓ (IG+Attention) | ✓ (ECS JSON)     |


No existing study in the reviewed literature combines all four elements—dynamic graph neural network with temporal modelling, federated learning under non-IID conditions, dual explainability (feature attribution plus structural attention), and SIEM-compatible alert generation—in a single prototype evaluated on CICIoT2023. This project fills that specific gap by integrating all four capabilities into a CPU-deployable edge system suitable for SOC operations.

---

## 2. Research Methodology

This section describes in some detail how I intend to conduct (and am already conducting) my research. It makes clear the basis of my research method, how I intend to gather, analyse and interpret the results, the verifiable academic goal of the study, and how I will present the data to prove the rigour of the process and my interpretation of the results.

### 2.1 Research Approach and Basis

This project follows a quantitative, experimental methodology in which multiple intrusion detection models are designed, implemented, compared, and explained under controlled conditions. The research is grounded in the positivist paradigm: objective performance metrics (precision, recall, F1, ROC-AUC, false alarm rate, inference latency) are measured on a held-out test set to evaluate each model against clearly defined hypotheses.

**Primary research question:** How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?

**Sub-questions and hypotheses:**

- **H1:** The dynamic GNN (GAT+GRU) achieves a higher F1 score and lower false alarm rate than Random Forest and MLP baselines on CICIoT2023 binary classification.
- **H2:** The federated GNN achieves F1 within 2% of the centralised GNN, demonstrating that privacy-preserving training does not substantially degrade detection performance.
- **H3:** Integrated Gradients and GAT attention weights produce feature attributions and flow-level explanations that identify attack-relevant features (such as TCP flags and protocol indicators) in generated SIEM alerts.

### 2.2 Dataset and Preprocessing

The CICIoT2023 dataset (Pinto et al., 2023) contains flow-level records with 46 numeric features and labels covering benign traffic plus 34 attack categories spanning DDoS, DoS, reconnaissance, brute force, and spoofing. The dataset is pre-split into training, test, and validation CSV files totalling approximately 1.6 GB for training.

**Preprocessing steps:**

1. Load raw CSVs and remove any rows with missing or infinite values.
2. Map the 35 original labels to binary classes: 0 (Benign) and 1 (Attack).
3. Apply StandardScaler (fitted on training data only) to normalise all 46 features to zero mean and unit variance; save the fitted scaler for consistent transformation of validation and test sets.
4. Export normalised data as Parquet files for efficient subsequent loading.

Binary classification was chosen to simplify the prototype scope while still testing the core detection capability. The binary formulation is also more representative of the primary SOC question—"Is this traffic malicious?"—before deeper categorisation. Multi-class extension is identified as future work.

### 2.3 Graph Construction and Design Decisions

Because the public CICIoT2023 dataset does not contain device identifiers or IP addresses, a device-topology graph cannot be constructed. Instead, following the attribute-similarity approach validated by Ngo et al. (2025) and Basak et al. (2025), graphs are built using feature-space proximity:

- **Nodes:** Each flow record becomes a node with its 46 normalised features as the node feature vector.
- **Edges:** k-nearest neighbours (kNN) in Euclidean feature space, with k=5, following Ngo et al. (2025) who validated this value as balancing graph connectivity with computational cost.
- **Windows:** 50 flows per window, chosen to balance sufficient graph density for meaningful message-passing with computational feasibility on CPU. Each window produces one PyTorch Geometric graph object.
- **Sequences:** 5 consecutive windows per sample, fed sequentially to the GRU to capture temporal evolution of network behaviour across 250 flows.
- **Stratified windowing:** To address class imbalance (approximately 97.8% attack, 2.2% benign at flow level), benign and attack flows are separated into pools; windows are built from each pool independently with minority_stride=25 for augmentation; sequences are balanced and shuffled before training.

**Justification of key parameters:**

- **k=5:** Validated by Ngo et al. (2025); higher k values increase computation quadratically while providing diminishing returns in neighbourhood information.
- **window_size=50:** Provides 50 nodes per graph—large enough for the 2-layer GAT to learn meaningful neighbourhood representations, small enough for sub-second CPU inference.
- **sequence_length=5:** Captures temporal context across 250 flows; longer sequences increase GRU memory requirements disproportionately for a CPU-based prototype.
- **minority_stride=25:** 50% overlap for benign windows creates approximately 2× augmentation, partially offsetting the extreme class imbalance without excessive duplication.

### 2.4 Model Architectures

Four models are trained and compared, representing increasing structural sophistication:

- **Random Forest (Baseline 1):** 200 decision trees with maximum depth 20, trained on flat 46-dimensional feature vectors. Selected as a strong non-neural baseline that is fast to train and resistant to overfitting. Each flow is classified independently.
- **MLP (Baseline 2):** Three hidden layers (128, 64, 32 neurons) with ReLU activation and 20% dropout, trained on the same flat features. Selected to isolate the contribution of neural learning from graph structure—any GNN improvement over MLP demonstrates the value of relational modelling.
- **Centralised Dynamic GNN (Main Model):** The core architecture comprises: (1) Two GAT convolutional layers (4 attention heads, hidden dimension 64) that learn neighbourhood-aware node representations through multi-head attention; (2) Global mean pooling that aggregates all node representations into a single graph-level embedding per window; (3) A GRU recurrent layer (hidden dimension 64) that processes the sequence of 5 graph embeddings to capture temporal dynamics; (4) A fully connected classifier that produces binary attack/benign probability.
- **Federated Dynamic GNN:** Identical architecture to the centralised GNN, but trained using FedAvg across 3 simulated clients with Dirichlet Non-IID data splitting (alpha=0.5) for 10 communication rounds with 2 local training epochs per round. Alpha=0.5 creates moderate heterogeneity—each client observes a different distribution of attack types, simulating realistic IoT deployment scenarios where different network segments face different threat profiles (Albanbay et al., 2025). Flower (Beutel et al., 2020) is used as the federated learning framework.

### 2.5 Data Gathering and Analysis

All experiments use the pre-defined CICIoT2023 train/validation/test splits to ensure comparability with published benchmarks. The validation set is used for hyperparameter selection and early stopping (patience=5 epochs); the test set is held out and used only for final evaluation. A fixed random seed ensures reproducibility across all experiments.

Class-weighted loss (automatically computed inversely proportional to class frequency) is applied during all neural model training to prevent the classifier from trivially predicting the majority class.

### 2.6 Evaluation Metrics

Performance is evaluated using the following metrics, chosen for their relevance to SOC operations:

**Precision** (Eq. 1):

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (Eq. 2):

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-score** (Eq. 3):

$$F_1 = \frac{2 \times (\text{Precision} \times \text{Recall})}{\text{Precision} + \text{Recall}}$$

**False Alarm Rate (FAR)** (Eq. 4):

$$\text{FAR} = \frac{FP}{FP + TN}$$

**Additionally:**

- **ROC-AUC:** Area under the Receiver Operating Characteristic curve, measuring discrimination ability across all classification thresholds.
- **Confusion matrix:** Visual breakdown of TP, TN, FP, FN counts.
- **CPU inference time:** Milliseconds per sample, measured on CPU to validate edge deployment feasibility.
- **Communication cost:** Total bytes transmitted between clients and server during federated training (number of parameters × 4 bytes × 2 directions × rounds × clients).

False alarm rate is particularly critical for SOC deployment: a system with high recall but high FAR will overwhelm analysts with false positives, negating the operational benefit.

### 2.7 Explainability and SIEM Integration

Explainability is implemented through two complementary mechanisms:

1. **Integrated Gradients (Captum):** For each flagged flow, feature attributions are computed by accumulating gradients along a path from a zero-valued baseline to the actual input. The top-5 contributing features are extracted and included in the alert.
2. **GAT attention weights:** The learned attention coefficients from the final GAT layer identify which neighbouring flows in the graph most influenced the classification, providing structural context.

Alerts are formatted as ECS (Elastic Common Schema) JSON objects and served via a FastAPI endpoint (POST /score). Each alert includes: timestamp, alert severity, detection score, classification label, top contributing features with attribution scores, and top influential neighbouring flows. This format is directly ingestible by standard SIEM platforms (Elastic Security, Splunk).

Figure 1 illustrates the complete research pipeline from raw data to SIEM alert generation.

**Figure 1: Research Pipeline**

![Figure 1: Research Pipeline - From Raw IoT Flow Data to Explainable SIEM Alerts](assets/figure1_pipeline.png)

### 2.8 Academic Worth and Verifiable Goal

The academic contribution of this project is threefold:

1. **Architectural novelty:** It is the first prototype, to the author's knowledge, that combines a dynamic GNN (GAT+GRU) with federated learning, dual explainability, and SIEM integration in a single system evaluated on CICIoT2023 (Table 1, Section 1.6).
2. **Empirical evidence:** It provides controlled experimental comparison between flat models, centralised GNN, and federated GNN on the same dataset with identical preprocessing, directly testing whether graph structure and federated training add value.
3. **Practical applicability:** It demonstrates that the complete pipeline—from flow ingestion to explained SIEM alert—runs on CPU within latency bounds acceptable for edge deployment, bridging the gap between research prototypes and operational SOC tools.

The research goal is verifiable: each hypothesis (H1–H3) maps to specific, measurable metrics that can be objectively evaluated on the held-out test set.

### 2.9 Anticipated Critical Reflection

Several limitations are anticipated and will be critically discussed in the final dissertation:

- **Dataset representativeness:** CICIoT2023, while recent and comprehensive, is generated in a controlled testbed. Results may not directly transfer to all production IoT environments with different device populations and traffic patterns.
- **Graph construction validity:** kNN feature-similarity graphs are a proxy for true network topology. The assumption that flows similar in feature space are behaviourally related may not hold universally, particularly for encrypted traffic where feature distributions are less discriminative.
- **Federated learning scale:** Three simulated clients on a single machine do not capture real-world communication latency, client dropout, or Byzantine failures. The moderate non-IID setting (alpha=0.5) represents only one point on the heterogeneity spectrum.
- **Software-Defined IoT scope:** While the flow features used are consistent with SDN controller exports, the prototype does not integrate with an actual SDN controller or OpenFlow switch. The SDIoT framing provides architectural context rather than a demonstrated SDN deployment.
- **Explainability validation:** Without a formal user study with SOC analysts, the claim that explanations "support" triage remains based on qualitative assessment of output format and feature relevance rather than measured analyst performance improvement.
- **Prototype maturity:** The system is a research prototype, not a production SIEM. It lacks authentication, persistent storage, alert correlation, and the many features required for operational deployment.

These limitations define clear boundaries for the conclusions that can be drawn and identify specific directions for future work.

---

## 3. Plan for Completion

This section briefly describes the current status of my project and in more detail describes how I intend to progress it to completion. It also includes an indication of how I will proceed if the results I collect are in some way deficient.

### 3.1 Current Status

As of the interim submission date, the following components are complete:

- **Data pipeline:** CICIoT2023 downloaded, preprocessed, normalised, and exported as Parquet files with fitted scaler saved.
- **Graph construction:** kNN graph builder, stratified windowing, and sequence dataset implemented and tested.
- **All four models:** Random Forest, MLP, centralised GNN, and federated GNN architectures implemented.
- **Training infrastructure:** Training loops with early stopping, class-weighted loss, and metric logging operational.
- **Federated learning:** Flower server and client scripts implemented; Dirichlet non-IID splitting functional; 10-round simulation completed.
- **Explainability:** Captum Integrated Gradients and GAT attention extraction implemented.
- **SIEM integration:** FastAPI endpoint producing ECS-formatted JSON alerts operational.
- **Initial results:** All four models trained and evaluated on the test set; preliminary metrics collected.

The core engineering work is substantially complete. The remaining work focuses on rigorous evaluation, analysis, and dissertation writing.

### 3.2 Plan to Completion

**Table 2: Completion Timeline**


| Week | Dates (Approx.) | Tasks                                                                                                                         | Deliverables                            |
| ---- | --------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 5    | Current week    | Finalise all experiments; ensure reproducibility with fixed seeds; generate final metrics for all four models                 | Final metric tables, confusion matrices |
| 5–6  | +1 week         | Generate all figures: confusion matrices, ROC curves, FL convergence plot (F1 vs. round), inference time comparison bar chart | All dissertation figures                |
| 6    | +1.5 weeks      | Conduct sensitivity analysis: vary k (3,5,7), window_size (25,50,100) if time permits; document results                       | Sensitivity analysis table (optional)   |
| 6–7  | +2 weeks        | Write Results chapter with all tables and figures; write Discussion chapter interpreting results against hypotheses H1–H3     | Draft Results and Discussion chapters   |
| 7    | +2.5 weeks      | Write Conclusion and Critical Self-Evaluation; revisit literature review for any additional references prompted by results    | Draft Conclusion chapter                |
| 7–8  | +3 weeks        | Incorporate interim report feedback from supervisor; integrate all chapters; polish references and formatting; proofread      | Complete dissertation draft             |
| 8    | Submission week | Final proofread; check word count; verify all figures render correctly; submit                                                | Final dissertation submission           |


### 3.3 Contingency Plans

- **Scenario 1: Results do not support H1 (GNN does not outperform baselines).** This is a valid finding. The discussion will analyse why flat models may suffice for this dataset (e.g., kNN graphs may not add information beyond what features already encode). The contribution shifts to empirical evidence about when graph structure does and does not help.
- **Scenario 2: Federated GNN shows significant degradation (>5% F1 drop).** The analysis will investigate whether non-IID heterogeneity or insufficient communication rounds caused the gap, with recommendations for techniques such as FedProx or increased rounds to mitigate the effect.
- **Scenario 3: Explainability computation is too slow for real-time use.** Integrated Gradients will be applied to a representative subset of alerts (e.g., top 10% by confidence score) rather than all alerts, with the latency trade-off documented. The discussion will recommend approximation methods (e.g., gradient × input) for full-speed deployment.
- **Scenario 4: Time overrun on writing.** The sensitivity analysis (Week 6) is designated as optional and will be dropped first. The core deliverable (four-model comparison with explanations and alerts) remains achievable within the remaining timeline.

---

## References

Ahmad, Z., Shahid Khan, A., Wai Shiang, C., Abdullah, J. and Ahmad, F. (2021) 'Network intrusion detection system: A systematic study of machine learning and deep learning approaches', *Transactions on Emerging Telecommunications Technologies*, 32(1), pp. 1–29.

Alahmadi, B.A., Axon, L., Martinovic, I. and Sherr, M. (2022) '99% False Positives: A Qualitative Study of SOC Analysts' Perspectives on Security Alarms', in *Proceedings of the 31st USENIX Security Symposium*. Boston, MA: USENIX Association, pp. 2783–2800.

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847.

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78.

Antonakakis, M., April, T., Bailey, M., Bernhard, M., Bursztein, E., Cochran, J., Durumeric, Z., Halderman, J.A., Invernizzi, L., Kallitsis, M., Kumar, D., Lever, C., Ma, Z., Mason, J., Menscher, D., Seaman, C., Sullivan, N., Thomas, K. and Zhou, Y. (2017) 'Understanding the Mirai Botnet', in *Proceedings of the 26th USENIX Security Symposium*. Vancouver, BC: USENIX Association, pp. 1093–1110.

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002.

Bera, S., Misra, S. and Vasilakos, A.V. (2017) 'Software-Defined Networking for Internet of Things: A Survey', *IEEE Internet of Things Journal*, 4(6), pp. 1994–2008.

Beutel, D.J., Tober, T., Steiner, D., Aber, S., Becker, T., Naseri, A., Tanner, Y. and Lane, N.D. (2020) 'Flower: A Friendly Federated Learning Framework', *arXiv preprint arXiv:2007.14390*, pp. 1–6.

Caville, E., Lo, W.W., Layeghy, S. and Portmann, M. (2022) 'Anomal-E: A Self-Supervised Network Intrusion Detection System based on Graph Neural Networks', *Knowledge-Based Systems*, 258, pp. 110–125.

Kairouz, P., McMahan, H.B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A.N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R., et al. (2021) 'Advances and Open Problems in Federated Learning', *Foundations and Trends in Machine Learning*, 14(1–2), pp. 1–210.

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., Melnikov, A., Kliushkina, N., Araya, C., Yan, S. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint arXiv:2009.07896*.

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80–84.

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509–530.

McMahan, B., Moore, E., Ramage, D., Hampson, S. and Arcas, B.A. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*. Fort Lauderdale, FL: PMLR, pp. 1273–1282.

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection - a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499.

Pinto, C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941.

Shone, N., Ngoc, T.N., Phai, V.D. and Shi, Q. (2018) 'A Deep Learning Approach to Network Intrusion Detection', *IEEE Transactions on Emerging Topics in Computational Intelligence*, 2(1), pp. 41–50.

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*. Sydney, Australia: PMLR, pp. 3319–3328.

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *International Conference on Learning Representations (ICLR)*. Available at: [https://openreview.net/forum?id=rJXMpikCZ](https://openreview.net/forum?id=rJXMpikCZ)