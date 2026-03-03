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

I am submitting this interim report at the halfway point of my MSc project. In this report I explain how I am doing my research. I also give my supervisor and moderator a clear picture of the whole project. This is not a first draft of the final report. It is also not the first few chapters of the final report. The report has three parts: (1) a summary literature review, (2) research methodology, and (3) a plan for completion. I have not included any final results here. I will gather, analyse and present the results in the final report.

---

## 1. Summary Literature Review

In this section I summarise the main literature in my subject area. I also explain the academic framework for my research.

### 1.1 IoT Security Landscape and Software-Defined IoT

IoT devices are growing fast. This has created many security problems in consumer, industrial and critical infrastructure (Kolias et al., 2017). Many IoT devices use default passwords and have weak security. They also get few firmware updates. This makes them easy targets for attackers. When hacked, these devices can join large botnets like Mirai. Mirai can launch DDoS attacks over 1 Tbps (Antonakakis et al., 2017). The variety and number of IoT devices make defence hard. Old security models find it difficult to monitor thousands of different devices.

Software-Defined IoT (SDIoT) helps with this. It separates the control plane from the data plane. This allows centralised network management through SDN controllers (Bera, Misra and Vasilakos, 2017). In SDIoT, the controller can see all forwarding decisions. It can also export flow data such as packet counts, byte counts and protocol breakdowns. This makes SDIoT a good fit for machine learning intrusion detection. The controller collects data from the whole network without needing software on each device (Bera, Misra and Vasilakos, 2017). My project uses flow-level features from the CICIoT2023 dataset (Pinto Neto et al., 2023). These features are similar to what an SDN controller would produce.

But Bera, Misra and Vasilakos (2017) mainly looked at SDN architecture. They did not show ML-based detection in that setup. So there is a gap between the SDIoT idea and real intrusion detection. My project fills this gap. I built a detection system that works with flow-level features from SDN controllers.

### 1.2 Machine Learning for Network Intrusion Detection

Machine learning is now widely used for network intrusion detection. Models are trained on flow features to classify traffic as benign or malicious. Random Forest works well on network datasets. It does not overfit easily and it selects features through ensemble averaging (Ahmad et al., 2021). On CICIoT2023, Pinto Neto et al. (2023) found Random Forest got over 99% accuracy for binary detection. So I use it as a strong baseline.

Deep learning like MLPs can learn complex patterns without manual feature work (Shone et al., 2018). Shone et al. (2018) used deep autoencoders with classifiers on NSL-KDD. They got good results. But NSL-KDD is smaller and less diverse than CICIoT2023. They also did not look at how flows change over time or how they relate to each other. Both Random Forest and MLP treat each flow on its own. They ignore links between flows. Graph-based methods can fix this.

A big problem with these methods is that they assume flows are independent. In real networks, flows are linked. For example, a scan often comes before an attack. Command-and-control traffic links to data theft. If we ignore these links, we miss multi-stage attacks (Caville et al., 2022).

### 1.3 Graph Neural Networks for Intrusion Detection

Graph Neural Networks (GNNs) are a promising approach for intrusion detection. They can model how network entities relate to each other. Usually, nodes are flows or hosts. Edges show communication links. The GNN learns from its neighbours through message passing (Caville et al., 2022).

Graph Attention Networks (GATs) go further. They learn to give different weights to different neighbours. So the model focuses on the most important connections (Velickovic et al., 2018). This helps in two ways. First, it improves classification. Second, we can see which flows influenced each decision.

Basak et al. (2025) used GNNs for intrusion detection on UNSW-NB15. They built graphs from flow features. Edges linked similar flows. They got F1 over 97%. GNNs matched or beat tree-based models. But they used static graphs. The whole dataset was one snapshot. They did not look at how behaviour changes over time. Also, UNSW-NB15 is older and has fewer attack types than CICIoT2023.

Ngo et al. (2025) used GNNs on IoT datasets. They built graphs with kNN. Edges linked flows that were similar in feature space. This worked even without device topology. They found k=5 is a good default. But they also used static graphs. No temporal modelling. So their system could not see how attacks change over time. That is important for slow attacks or multi-phase campaigns.

My project builds on both. I add a GRU layer that processes sequences of graph snapshots. So the system learns from how the graph changes over time, not just single snapshots.

### 1.4 Federated Learning for Privacy-Preserving IoT Security

Federated learning (FL) lets many clients train a model together without sharing raw data. This helps with privacy and rules like GDPR (McMahan et al., 2017). In FedAvg, each client trains on its own data. It only sends model updates to a server. The server averages them. This fits IoT security well. Traffic data often comes from different organisations or places. We cannot pool it all in one place.

Lazzarini et al. (2023) used federated learning for intrusion detection on UNSW-NB15. Their federated MLP got within 2% F1 of the centralised model. But they only had two clients with IID data. That is an ideal case. In real IoT, different networks see different traffic. A factory sensor network is not like a smart home. This non-IID data hurts FedAvg (Kairouz et al., 2021).

Albanbay et al. (2025) looked at non-IID data. They used Dirichlet splits with different alpha values. Lower alpha (more different data) caused up to 8% F1 drop. It also needed about twice as many rounds to converge. But they used flat models, not graphs. We do not know if graph models behave differently with non-IID data. My project tests this. I combine GNNs with federated learning using alpha=0.5.

### 1.5 Explainable AI for Security Operations

SOCs get too many alerts. Analysts handle thousands each day. Many are false positives or lack context (Alahmadi et al., 2022). ML models that only give a yes/no or a score make this worse. Analysts cannot tell if the model is right or just guessing.

Explainable AI (XAI) helps. It shows which input features drove each decision. Integrated Gradients (Sundararajan, Taly and Yan, 2017) does this by following a path from a baseline to the real input. It has good properties. Captum (Kokhlikyan et al., 2020) implements it for PyTorch. I use Captum in my project.

Alabbadi and Bajaber (2025) used SHAP for IoT intrusion detection. Feature attribution helped analysts trust the alerts. But SHAP is slow for deep models. One prediction can take over 10 seconds. That is too slow for real-time use. Integrated Gradients is faster. It needs only one forward and backward pass. So it fits edge deployment better.

GAT attention weights also help. They show which neighbour flows influenced each classification. I use both: feature attributions from Integrated Gradients and structural explanations from attention. Together they give analysts more context. Alabbadi and Bajaber (2025) did not try this combination.

### 1.6 Research Gap and Contribution

Table 1 compares the main related works against the four pillars of my project.

**Table 1: Comparison of Related Work Against Project Pillars**


| Study                       | Dynamic GNN | Federated Learning | Explainability   | SIEM Integration |
| --------------------------- | ----------- | ------------------ | ---------------- | ---------------- |
| Basak et al. (2025)         | Static GNN  | ✗                  | ✓ (partial)      | ✗                |
| Ngo et al. (2025)           | Static GNN  | ✗                  | ✗                | ✗                |
| Lazzarini et al. (2023)     | ✗           | ✓ (IID only)       | ✗                | ✗                |
| Albanbay et al. (2025)      | ✗           | ✓ (non-IID)        | ✗                | ✗                |
| Alabbadi and Bajaber (2025) | ✗           | ✗                  | ✓ (SHAP)         | ✗                |
| This project                | ✓ (GAT+GRU) | ✓ (non-IID)        | ✓ (IG+Attention) | ✓ (ECS JSON)     |


No study I found combines all four: dynamic GNN with temporal modelling, federated learning with non-IID data, dual explainability, and SIEM alerts. All in one system on CICIoT2023. My project fills this gap. I built a system that has all four and runs on CPU for edge use.

---

## 2. Research Methodology

In this section I describe how I am doing my research. I explain my method, how I gather and analyse data, and my academic goals. I also explain how I will present the data to show my work is rigorous.

### 2.1 Research Approach and Basis

I use a quantitative, experimental approach. I design, build and compare several intrusion detection models. I run them under controlled conditions. I measure objective metrics like precision, recall, F1, ROC-AUC, false alarm rate and inference time. I use a held-out test set. Each model is tested against clear hypotheses.

**Primary research question:** How can an explainable dynamic graph neural network, trained using federated learning, detect attacks in Software-Defined IoT flow data and generate SIEM alerts that support SOC operations on CPU-based edge devices?

**Sub-questions and hypotheses:**

- **H1:** The dynamic GNN (GAT+GRU) achieves a higher F1 score and lower false alarm rate than Random Forest and MLP baselines on CICIoT2023 binary classification.
- **H2:** The federated GNN achieves F1 within 2% of the centralised GNN, demonstrating that privacy-preserving training does not substantially degrade detection performance.
- **H3:** Integrated Gradients and GAT attention weights produce feature attributions and flow-level explanations that identify attack-relevant features (such as TCP flags and protocol indicators) in generated SIEM alerts.

### 2.2 Dataset and Preprocessing

I use the CICIoT2023 dataset (Pinto Neto et al., 2023). It has flow-level records with 46 numeric features. Labels cover benign traffic and 34 attack types including DDoS, DoS, reconnaissance, brute force and spoofing. The data comes pre-split into train, test and validation. The training set is about 1.6 GB.

**Preprocessing steps:**

1. Load raw CSVs and remove any rows with missing or infinite values.
2. Map the 35 original labels to binary classes: 0 (Benign) and 1 (Attack).
3. Apply StandardScaler (fitted on training data only) to normalise all 46 features to zero mean and unit variance; save the fitted scaler for consistent transformation of validation and test sets.
4. Export normalised data as Parquet files for efficient subsequent loading.

I use binary classification to keep the scope manageable. It still tests the main detection task. The main SOC question is: is this traffic malicious? Multi-class can be added later.

### 2.3 Graph Construction and Design Decisions

CICIoT2023 has no device IDs or IP addresses. So I cannot build a device-topology graph. Instead I follow Ngo et al. (2025) and Basak et al. (2025). I build graphs from feature similarity:

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

I use the pre-defined train/validation/test splits. This lets me compare with other papers. I use validation for hyperparameters and early stopping (patience=5). I only use the test set for final evaluation. A fixed seed makes results reproducible.

I use class-weighted loss for all neural models. This stops the model from just predicting the majority class.

### 2.6 Evaluation Metrics

I evaluate performance with these metrics. I chose them because they matter for SOC work:

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

False alarm rate is very important for SOCs. High recall with high FAR means too many false positives. Analysts get overwhelmed.

### 2.7 Explainability and SIEM Integration

I use two methods for explainability:

1. **Integrated Gradients (Captum):** For each flagged flow, feature attributions are computed by accumulating gradients along a path from a zero-valued baseline to the actual input. The top-5 contributing features are extracted and included in the alert.
2. **GAT attention weights:** The learned attention coefficients from the final GAT layer identify which neighbouring flows in the graph most influenced the classification, providing structural context.

Alerts are in ECS JSON format. I serve them via a FastAPI endpoint (POST /score). Each alert has timestamp, severity, score, label, top features and top neighbour flows. This works with Elastic Security and Splunk.

Figure 1 shows the full pipeline from raw data to SIEM alerts.

**Figure 1: Research Pipeline**

![Figure 1: Research Pipeline - From Raw IoT Flow Data to Explainable SIEM Alerts](assets/figure1_pipeline.png)

### 2.8 Academic Worth and Verifiable Goal

My project contributes in three ways:

1. **Architectural novelty:** As far as I know, no other prototype combines dynamic GNN (GAT+GRU), federated learning, dual explainability and SIEM integration. All in one system on CICIoT2023 (Table 1, Section 1.6).
2. **Empirical evidence:** I compare flat models, centralised GNN and federated GNN on the same data. Same preprocessing. This tests whether graphs and federated learning actually help.
3. **Practical applicability:** The full pipeline runs on CPU. Latency is acceptable for edge use. So it bridges research and real SOC tools.

Each hypothesis (H1–H3) maps to clear metrics. I can check them on the test set.

### 2.9 Anticipated Critical Reflection

I expect several limitations. I will discuss them in the final report:

- **Dataset representativeness:** CICIoT2023, while recent and comprehensive, is generated in a controlled testbed. Results may not directly transfer to all production IoT environments with different device populations and traffic patterns.
- **Graph construction validity:** kNN feature-similarity graphs are a proxy for true network topology. The assumption that flows similar in feature space are behaviourally related may not hold universally, particularly for encrypted traffic where feature distributions are less discriminative.
- **Federated learning scale:** Three simulated clients on a single machine do not capture real-world communication latency, client dropout, or Byzantine failures. The moderate non-IID setting (alpha=0.5) represents only one point on the heterogeneity spectrum.
- **Software-Defined IoT scope:** While the flow features used are consistent with SDN controller exports, the prototype does not integrate with an actual SDN controller or OpenFlow switch. The SDIoT framing provides architectural context rather than a demonstrated SDN deployment.
- **Explainability validation:** Without a formal user study with SOC analysts, the claim that explanations "support" triage remains based on qualitative assessment of output format and feature relevance rather than measured analyst performance improvement.
- **Prototype maturity:** The system is a research prototype, not a production SIEM. It lacks authentication, persistent storage, alert correlation, and the many features required for operational deployment.

These limitations define clear boundaries for the conclusions that can be drawn and identify specific directions for future work.

---

## 3. Plan for Completion

This section describes where my project is now. It also describes how I will finish it. I include what I will do if the results are not as expected.

### 3.1 Current Status

At the time of this report, I have completed:

- **Data pipeline:** CICIoT2023 downloaded, preprocessed, normalised, and exported as Parquet files with fitted scaler saved.
- **Graph construction:** kNN graph builder, stratified windowing, and sequence dataset implemented and tested.
- **All four models:** Random Forest, MLP, centralised GNN, and federated GNN architectures implemented.
- **Training infrastructure:** Training loops with early stopping, class-weighted loss, and metric logging operational.
- **Federated learning:** Flower server and client scripts implemented; Dirichlet non-IID splitting functional; 10-round simulation completed.
- **Explainability:** Captum Integrated Gradients and GAT attention extraction implemented.
- **SIEM integration:** FastAPI endpoint producing ECS-formatted JSON alerts operational.
- **Initial results:** All four models trained and evaluated on the test set; preliminary metrics collected.

The main engineering is done. What is left is evaluation, analysis and writing the dissertation.

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

- **Scenario 1: GNN does not beat baselines.** That is still a valid result. I will discuss why flat models might be enough (e.g. kNN graphs may not add much). The contribution becomes evidence about when graphs help and when they do not.
- **Scenario 2: Federated GNN drops a lot (>5% F1).** I will look at whether non-IID data or too few rounds caused it. I may suggest FedProx or more rounds.
- **Scenario 3: Explainability is too slow.** I will use it only for some alerts (e.g. top 10% by confidence). I will document the trade-off. I may suggest faster approximations like gradient × input.
- **Scenario 4: Running out of time.** I will drop the sensitivity analysis first. The main deliverable (four models, explanations, alerts) is still achievable.

---

## References

Ahmad, Z., Shahid Khan, A., Wai Shiang, C., Abdullah, J. and Ahmad, F. (2021) 'Network intrusion detection system: a systematic study of machine learning and deep learning approaches', *Transactions on Emerging Telecommunications Technologies*, 32(1), e4150. doi: 10.1002/ett.4150.

Alahmadi, B.A., Axon, L., Martinovic, I. and Sherr, M. (2022) '99% false positives: a qualitative study of SOC analysts' perspectives on security alarms', in *Proceedings of the 31st USENIX Security Symposium*. Boston, MA, 10-12 August. Berkeley, CA: USENIX Association, pp. 2783-2800. Available at: https://www.usenix.org/conference/usenixsecurity22/presentation/alahmadi (Accessed: 15 June 2025).

Alabbadi, A. and Bajaber, F. (2025) 'An intrusion detection system over the IoT data streams using eXplainable artificial intelligence (XAI)', *Sensors*, 25(3), p. 847. doi: 10.3390/s25030847.

Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025) 'Federated learning-based intrusion detection in IoT networks: performance evaluation and data scaling study', *Journal of Sensor and Actuator Networks*, 14(4), p. 78. doi: 10.3390/jsan14040078.

Antonakakis, M., April, T., Bailey, M., Bernhard, M., Bursztein, E., Cochran, J., Durumeric, Z., Halderman, J.A., Invernizzi, L., Kallitsis, M., Kumar, D., Lever, C., Ma, Z., Mason, J., Menscher, D., Seaman, C., Sullivan, N., Thomas, K. and Zhou, Y. (2017) 'Understanding the Mirai botnet', in *Proceedings of the 26th USENIX Security Symposium*. Vancouver, BC, 16-18 August. Berkeley, CA: USENIX Association, pp. 1093-1110. Available at: https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/antonakakis (Accessed: 15 June 2025).

Basak, M., Kim, D.-W., Han, M.-M. and Shin, G.-Y. (2025) 'X-GANet: an explainable graph-based framework for robust network intrusion detection', *Applied Sciences*, 15(9), p. 5002. doi: 10.3390/app15095002.

Bera, S., Misra, S. and Vasilakos, A.V. (2017) 'Software-defined networking for Internet of Things: a survey', *IEEE Internet of Things Journal*, 4(6), pp. 1994-2008. doi: 10.1109/JIOT.2017.2746186.

Beutel, D.J., Tober, T., Steiner, D., Aber, S., Becker, T., Naseri, A., Tanner, Y. and Lane, N.D. (2020) 'Flower: a friendly federated learning framework', *arXiv preprint*. doi: 10.48550/arXiv.2007.14390.

Caville, E., Lo, W.W., Layeghy, S. and Portmann, M. (2022) 'Anomal-E: a self-supervised network intrusion detection system based on graph neural networks', *Knowledge-Based Systems*, 258, p. 110030. doi: 10.1016/j.knosys.2022.110030.

Kairouz, P., McMahan, H.B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A.N., Bonawitz, K., Charles, Z., Cormode, G., Cummings, R. et al. (2021) 'Advances and open problems in federated learning', *Foundations and Trends in Machine Learning*, 14(1-2), pp. 1-210. doi: 10.1561/2200000083.

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., Melnikov, A., Kliushkina, N., Araya, C., Yan, S. and Reblitz-Richardson, O. (2020) 'Captum: a unified and generic model interpretability library for PyTorch', *arXiv preprint*. doi: 10.48550/arXiv.2009.07896.

Kolias, C., Kambourakis, G., Stavrou, A. and Voas, J. (2017) 'DDoS in the IoT: Mirai and other botnets', *Computer*, 50(7), pp. 80-84. doi: 10.1109/MC.2017.201.

Lazzarini, R., Tianfield, H. and Charissis, V. (2023) 'Federated learning for IoT intrusion detection', *AI*, 4(3), pp. 509-530. doi: 10.3390/ai4030027.

McMahan, B., Moore, E., Ramage, D., Hampson, S. and Arcas, B.A. (2017) 'Communication-efficient learning of deep networks from decentralized data', in *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*. Fort Lauderdale, FL, 20-22 April. PMLR, pp. 1273-1282. Available at: http://proceedings.mlr.press/v54/mcmahan17a.html (Accessed: 15 June 2025).

Ngo, T., Yin, J., Ge, Y.-F. and Wang, H. (2025) 'Optimizing IoT intrusion detection: a graph neural network approach with attribute-based graph construction', *Information*, 16(6), p. 499. doi: 10.3390/info16060499.

Pinto Neto, E.C., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R. and Ghorbani, A.A. (2023) 'CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment', *Sensors*, 23(13), p. 5941. doi: 10.3390/s23135941.

Shone, N., Ngoc, T.N., Phai, V.D. and Shi, Q. (2018) 'A deep learning approach to network intrusion detection', *IEEE Transactions on Emerging Topics in Computational Intelligence*, 2(1), pp. 41-50. doi: 10.1109/TETCI.2017.2772792.

Sundararajan, M., Taly, A. and Yan, Q. (2017) 'Axiomatic attribution for deep networks', in *Proceedings of the 34th International Conference on Machine Learning (ICML)*. Sydney, Australia, 6-11 August. PMLR, pp. 3319-3328. Available at: http://proceedings.mlr.press/v70/sundararajan17a.html (Accessed: 15 June 2025).

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018) 'Graph attention networks', in *Proceedings of the 6th International Conference on Learning Representations (ICLR)*. Vancouver, BC, 30 April - 3 May. Available at: https://openreview.net/forum?id=rJXMpikCZ (Accessed: 15 June 2025).