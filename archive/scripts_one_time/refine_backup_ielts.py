"""
Refine Interim_Report_Arka_Talukder_backup.md to IELTS Band 6 style.
- Easy, student language
- Shorter sentences
- Non-AI format (natural flow, avoid stiff/formal patterns)
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_PATH = PROJECT_ROOT / "Interim_Report_Arka_Talukder_backup.md"

# (old_text, new_text) - keep citations, tables, equations, references unchanged
REPLACEMENTS = [
    # Introduction
    (
        "This interim report is submitted at around the halfway stage of my MSc project. It describes in detail how I am conducting my research and gives my supervisor and moderator a clear view of the whole project. It is not a first draft of the final report, and it is not the first few chapters of the final report. The report has three required parts: (1) a summary literature review, (2) research methodology, and (3) a plan for completion. No final results are included here; results will be gathered, analysed and presented in the final report.",
        "I am submitting this interim report at the halfway point of my MSc project. In this report I explain how I am doing my research. I also give my supervisor and moderator a clear picture of the whole project. This is not a first draft of the final report. It is also not the first few chapters of the final report. The report has three parts: (1) a summary literature review, (2) research methodology, and (3) a plan for completion. I have not included any final results here. I will gather, analyse and present the results in the final report."
    ),
    # Section 1 intro
    (
        "This section critically summarises the current academic literature in my subject area and provides the academic framework for the research I am undertaking.",
        "In this section I summarise the main literature in my subject area. I also explain the academic framework for my research."
    ),
    # 1.1 - IoT paragraph 1 (split long sentence)
    (
        "The rapid proliferation of Internet of Things (IoT) devices has introduced significant security challenges across consumer, industrial, and critical infrastructure domains. IoT devices frequently ship with default credentials, limited cryptographic capability, and infrequent firmware updates, making them attractive targets for adversaries (Kolias et al., 2017). Once compromised, these devices serve as nodes in large-scale botnets such as Mirai, capable of launching volumetric distributed denial-of-service (DDoS) attacks exceeding 1 Tbps (Antonakakis et al., 2017). The heterogeneity and scale of IoT ecosystems further complicate defence, as traditional perimeter-based security models struggle to monitor thousands of constrained devices with diverse communication patterns.",
        "IoT devices are growing fast. This has created many security problems in consumer, industrial and critical infrastructure (Kolias et al., 2017). Many IoT devices use default passwords and have weak security. They also get few firmware updates. This makes them easy targets for attackers. When hacked, these devices can join large botnets like Mirai. Mirai can launch DDoS attacks over 1 Tbps (Antonakakis et al., 2017). The variety and number of IoT devices make defence hard. Old security models find it difficult to monitor thousands of different devices."
    ),
    # 1.1 - SDIoT paragraph
    (
        "Software-Defined IoT (SDIoT) architectures address this complexity by decoupling the control plane from the data plane, allowing centralised network management through SDN controllers such as OpenFlow-based platforms (Bera, Misra and Vasilakos, 2017). In SDIoT environments, the SDN controller has visibility over all forwarding decisions and can export flow-level telemetry—including packet counts, byte counts, flag distributions, and protocol breakdowns—for every communication pair traversing the network. This centralised flow collection capability makes SDIoT environments particularly well-suited to machine-learning-based intrusion detection, as the controller provides a natural aggregation point for network-wide feature extraction without requiring agents on resource-constrained end devices (Bera, Misra and Vasilakos, 2017). The flow-level feature set used in this project, drawn from the CICIoT2023 dataset (Pinto Neto et al., 2023), mirrors the telemetry that an SDN controller would produce, grounding the research in a realistic SDIoT deployment scenario.",
        "Software-Defined IoT (SDIoT) helps with this. It separates the control plane from the data plane. This allows centralised network management through SDN controllers (Bera, Misra and Vasilakos, 2017). In SDIoT, the controller can see all forwarding decisions. It can also export flow data such as packet counts, byte counts and protocol breakdowns. This makes SDIoT a good fit for machine learning intrusion detection. The controller collects data from the whole network without needing software on each device (Bera, Misra and Vasilakos, 2017). My project uses flow-level features from the CICIoT2023 dataset (Pinto Neto et al., 2023). These features are similar to what an SDN controller would produce."
    ),
    # 1.1 - However paragraph
    (
        "However, Bera, Misra and Vasilakos (2017) focused primarily on architectural benefits of SDN for IoT management rather than demonstrating ML-based detection within that architecture, leaving a gap between the SDIoT vision and practical intrusion detection implementations. This project bridges that gap by building a detection system that operates on flow-level features consistent with SDN controller exports.",
        "But Bera, Misra and Vasilakos (2017) mainly looked at SDN architecture. They did not show ML-based detection in that setup. So there is a gap between the SDIoT idea and real intrusion detection. My project fills this gap. I built a detection system that works with flow-level features from SDN controllers."
    ),
    # 1.2 - ML paragraph 1
    (
        "Machine learning has become the dominant paradigm for network intrusion detection, with models trained on flow-level features to classify traffic as benign or malicious. Random Forest classifiers have demonstrated strong performance on tabular network datasets due to their resistance to overfitting and implicit feature selection through ensemble averaging (Ahmad et al., 2021). On the CICIoT2023 dataset specifically, Pinto Neto et al. (2023) reported that Random Forest achieved classification accuracy above 99% for binary detection, establishing a strong baseline for comparison.",
        "Machine learning is now widely used for network intrusion detection. Models are trained on flow features to classify traffic as benign or malicious. Random Forest works well on network datasets. It does not overfit easily and it selects features through ensemble averaging (Ahmad et al., 2021). On CICIoT2023, Pinto Neto et al. (2023) found Random Forest got over 99% accuracy for binary detection. So I use it as a strong baseline."
    ),
    # 1.2 - Deep learning paragraph
    (
        "Deep learning approaches, particularly Multi-Layer Perceptrons (MLPs), offer the advantage of learning non-linear decision boundaries without manual feature engineering (Shone et al., 2018). Shone et al. (2018) demonstrated that deep autoencoders combined with shallow classifiers achieved competitive performance on the NSL-KDD dataset. However, their evaluation was limited to NSL-KDD, which is significantly smaller and less diverse than CICIoT2023, and they did not consider flow-level temporal dynamics or relational structure between network entities. Both Random Forest and MLP approaches treat each flow independently, discarding any relational or temporal context between flows—a limitation that graph-based methods can address.",
        "Deep learning like MLPs can learn complex patterns without manual feature work (Shone et al., 2018). Shone et al. (2018) used deep autoencoders with classifiers on NSL-KDD. They got good results. But NSL-KDD is smaller and less diverse than CICIoT2023. They also did not look at how flows change over time or how they relate to each other. Both Random Forest and MLP treat each flow on its own. They ignore links between flows. Graph-based methods can fix this."
    ),
    # 1.2 - Key limitation
    (
        "A key limitation shared across these tabular approaches is that they assume flow independence. In real networks, flows are contextually related: a reconnaissance scan precedes exploitation; command-and-control traffic correlates temporally with data exfiltration. Ignoring these relationships limits detection capability for multi-stage attacks (Caville et al., 2022).",
        "A big problem with these methods is that they assume flows are independent. In real networks, flows are linked. For example, a scan often comes before an attack. Command-and-control traffic links to data theft. If we ignore these links, we miss multi-stage attacks (Caville et al., 2022)."
    ),
    # 1.3 - GNN intro
    (
        "Graph Neural Networks (GNNs) have emerged as a promising approach for network intrusion detection because they can model relational structure between network entities. In a typical formulation, nodes represent network flows or hosts, edges represent communication relationships, and the GNN learns representations that incorporate neighbourhood context through message-passing mechanisms (Caville et al., 2022).",
        "Graph Neural Networks (GNNs) are a promising approach for intrusion detection. They can model how network entities relate to each other. Usually, nodes are flows or hosts. Edges show communication links. The GNN learns from its neighbours through message passing (Caville et al., 2022)."
    ),
    # 1.3 - GAT
    (
        "Graph Attention Networks (GATs) extend this framework by learning to assign different importance weights to different neighbours, allowing the model to focus on the most relevant connections during aggregation (Velickovic et al., 2018). This attention mechanism provides two benefits: improved classification performance through selective aggregation, and interpretable attention coefficients that indicate which neighbouring flows most influenced a detection decision.",
        "Graph Attention Networks (GATs) go further. They learn to give different weights to different neighbours. So the model focuses on the most important connections (Velickovic et al., 2018). This helps in two ways. First, it improves classification. Second, we can see which flows influenced each decision."
    ),
    # 1.3 - Basak
    (
        "Basak et al. (2025) applied GNN-based intrusion detection using attribute-similarity graphs on the UNSW-NB15 dataset, where nodes were constructed from flow features and edges from feature-space proximity. They reported F1 scores exceeding 97% and demonstrated that GNN-based approaches could match or exceed tree-based baselines. However, their work used static graphs that treated the entire dataset as a single snapshot, ignoring the temporal evolution of network behaviour. Furthermore, UNSW-NB15 is a dated dataset with fewer attack categories than CICIoT2023, limiting the generalisability of their findings to modern IoT threat landscapes.",
        "Basak et al. (2025) used GNNs for intrusion detection on UNSW-NB15. They built graphs from flow features. Edges linked similar flows. They got F1 over 97%. GNNs matched or beat tree-based models. But they used static graphs. The whole dataset was one snapshot. They did not look at how behaviour changes over time. Also, UNSW-NB15 is older and has fewer attack types than CICIoT2023."
    ),
    # 1.3 - Ngo
    (
        "Ngo et al. (2025) advanced this line of research by applying GNNs to IoT-specific datasets with kNN-based graph construction, reporting that k-nearest-neighbour edges in feature space effectively captured behavioural similarity between flows even when device topology was unavailable. Their approach validated k=5 as a reasonable default for feature-similarity graphs. However, Ngo et al. (2025) also employed static graph snapshots without temporal modelling, meaning their system could not capture how attack patterns evolve over time windows—a critical capability for detecting slow-and-low attacks or multi-phase campaigns.",
        "Ngo et al. (2025) used GNNs on IoT datasets. They built graphs with kNN. Edges linked flows that were similar in feature space. This worked even without device topology. They found k=5 is a good default. But they also used static graphs. No temporal modelling. So their system could not see how attacks change over time. That is important for slow attacks or multi-phase campaigns."
    ),
    # 1.3 - This project extends
    (
        "This project extends both works by introducing temporal modelling through a GRU layer that processes sequences of graph snapshots, enabling the system to learn from graph evolution rather than individual static snapshots.",
        "My project builds on both. I add a GRU layer that processes sequences of graph snapshots. So the system learns from how the graph changes over time, not just single snapshots."
    ),
    # 1.4 - FL intro
    (
        "Federated learning (FL) enables collaborative model training across distributed data sources without centralising raw data, addressing both privacy concerns and regulatory constraints such as GDPR (McMahan et al., 2017). In the Federated Averaging (FedAvg) algorithm, each client trains on local data and transmits only model parameter updates to a central server, which aggregates them through weighted averaging. This paradigm is particularly relevant for IoT security, where traffic data may be generated across multiple organisational boundaries or geographic locations and cannot be legally or practically pooled.",
        "Federated learning (FL) lets many clients train a model together without sharing raw data. This helps with privacy and rules like GDPR (McMahan et al., 2017). In FedAvg, each client trains on its own data. It only sends model updates to a server. The server averages them. This fits IoT security well. Traffic data often comes from different organisations or places. We cannot pool it all in one place."
    ),
    # 1.4 - Lazzarini
    (
        "Lazzarini et al. (2023) applied federated learning to network intrusion detection using the UNSW-NB15 dataset with a federated MLP architecture. They reported that the federated model matched centralised performance within 2% F1 after sufficient training rounds. However, their study used only two clients with IID (independent and identically distributed) data partitions, which represents an optimistic scenario. In real-world IoT deployments, different network segments observe different traffic distributions—an enterprise IoT subnet dominated by sensor telemetry will have fundamentally different flow characteristics than a consumer smart-home network. This non-IID heterogeneity is known to degrade FedAvg convergence (Kairouz et al., 2021).",
        "Lazzarini et al. (2023) used federated learning for intrusion detection on UNSW-NB15. Their federated MLP got within 2% F1 of the centralised model. But they only had two clients with IID data. That is an ideal case. In real IoT, different networks see different traffic. A factory sensor network is not like a smart home. This non-IID data hurts FedAvg (Kairouz et al., 2021)."
    ),
    # 1.4 - Albanbay
    (
        "Albanbay et al. (2025) directly addressed the non-IID challenge by evaluating federated intrusion detection under Dirichlet-distributed data splits with varying alpha values. They demonstrated that lower alpha values (higher heterogeneity) caused up to 8% F1 degradation compared to IID settings, with convergence requiring approximately twice as many communication rounds. Their work, however, used flat tabular models and did not explore whether graph-structured models exhibit different sensitivity to non-IID conditions. Since graph models aggregate information across neighbourhoods, they may exhibit different convergence dynamics under non-IID splits—a question this project investigates by combining GNNs with Dirichlet non-IID federated learning (alpha=0.5).",
        "Albanbay et al. (2025) looked at non-IID data. They used Dirichlet splits with different alpha values. Lower alpha (more different data) caused up to 8% F1 drop. It also needed about twice as many rounds to converge. But they used flat models, not graphs. We do not know if graph models behave differently with non-IID data. My project tests this. I combine GNNs with federated learning using alpha=0.5."
    ),
    # 1.5 - SOC intro
    (
        "Security Operations Centres (SOCs) face chronic alert fatigue, with analysts processing thousands of alerts daily, many of which are false positives or lack sufficient context for efficient triage (Alahmadi et al., 2022). Machine learning models that produce only a binary classification or confidence score without explanation exacerbate this problem, as analysts cannot assess whether the model's reasoning aligns with known attack patterns or reflects spurious correlations.",
        "SOCs get too many alerts. Analysts handle thousands each day. Many are false positives or lack context (Alahmadi et al., 2022). ML models that only give a yes/no or a score make this worse. Analysts cannot tell if the model is right or just guessing."
    ),
    # 1.5 - XAI
    (
        "Explainable AI (XAI) methods address this by attributing model decisions to specific input features. Integrated Gradients (Sundararajan, Taly and Yan, 2017) computes feature attributions by accumulating gradients along a straight-line path from a baseline input to the actual input, satisfying desirable axiomatic properties including sensitivity (every feature that changes the prediction receives non-zero attribution) and implementation invariance (attributions depend only on the function, not the model architecture). Captum (Kokhlikyan et al., 2020) provides a production-quality implementation of Integrated Gradients for PyTorch models, which this project uses.",
        "Explainable AI (XAI) helps. It shows which input features drove each decision. Integrated Gradients (Sundararajan, Taly and Yan, 2017) does this by following a path from a baseline to the real input. It has good properties. Captum (Kokhlikyan et al., 2020) implements it for PyTorch. I use Captum in my project."
    ),
    # 1.5 - Alabbadi
    (
        "Alabbadi and Bajaber (2025) applied SHAP-based explanations to IoT intrusion detection models, demonstrating that feature attribution improved analyst confidence in alert triage decisions by identifying which flow characteristics drove each detection. However, their study relied on SHAP's kernel-based approximation for deep models, which is computationally expensive—potentially prohibitive for real-time edge deployment. SHAP computation for a single prediction on a deep model can exceed 10 seconds, which is impractical for a system targeting sub-second alert generation. Integrated Gradients, by contrast, requires only a single forward and backward pass with interpolation, making it substantially faster and more suitable for edge CPU deployment.",
        "Alabbadi and Bajaber (2025) used SHAP for IoT intrusion detection. Feature attribution helped analysts trust the alerts. But SHAP is slow for deep models. One prediction can take over 10 seconds. That is too slow for real-time use. Integrated Gradients is faster. It needs only one forward and backward pass. So it fits edge deployment better."
    ),
    # 1.5 - GAT attention
    (
        "Additionally, GAT attention weights provide a complementary form of explanation: they indicate which neighbouring flows in the graph most influenced the classification of a given node. Combining feature-level attributions (Integrated Gradients) with structural explanations (attention weights) provides richer context for SOC analysts than either method alone—a combination not explored by Alabbadi and Bajaber (2025).",
        "GAT attention weights also help. They show which neighbour flows influenced each classification. I use both: feature attributions from Integrated Gradients and structural explanations from attention. Together they give analysts more context. Alabbadi and Bajaber (2025) did not try this combination."
    ),
    # 1.6 - Table intro
    (
        "Table 1 summarises the capabilities of the key related works against the four pillars of this project.",
        "Table 1 compares the main related works against the four pillars of my project."
    ),
    # 1.6 - Gap conclusion
    (
        "No existing study in the reviewed literature combines all four elements—dynamic graph neural network with temporal modelling, federated learning under non-IID conditions, dual explainability (feature attribution plus structural attention), and SIEM-compatible alert generation—in a single prototype evaluated on CICIoT2023. This project fills that specific gap by integrating all four capabilities into a CPU-deployable edge system suitable for SOC operations.",
        "No study I found combines all four: dynamic GNN with temporal modelling, federated learning with non-IID data, dual explainability, and SIEM alerts. All in one system on CICIoT2023. My project fills this gap. I built a system that has all four and runs on CPU for edge use."
    ),
    # 2 intro
    (
        "This section describes in some detail how I intend to conduct (and am already conducting) my research. It makes clear the basis of my research method, how I intend to gather, analyse and interpret the results, the verifiable academic goal of the study, and how I will present the data to prove the rigour of the process and my interpretation of the results.",
        "In this section I describe how I am doing my research. I explain my method, how I gather and analyse data, and my academic goals. I also explain how I will present the data to show my work is rigorous."
    ),
    # 2.1 - methodology
    (
        "This project follows a quantitative, experimental methodology in which multiple intrusion detection models are designed, implemented, compared, and explained under controlled conditions. The research is grounded in the positivist paradigm: objective performance metrics (precision, recall, F1, ROC-AUC, false alarm rate, inference latency) are measured on a held-out test set to evaluate each model against clearly defined hypotheses.",
        "I use a quantitative, experimental approach. I design, build and compare several intrusion detection models. I run them under controlled conditions. I measure objective metrics like precision, recall, F1, ROC-AUC, false alarm rate and inference time. I use a held-out test set. Each model is tested against clear hypotheses."
    ),
    # 2.2 - dataset
    (
        "The CICIoT2023 dataset (Pinto Neto et al., 2023) contains flow-level records with 46 numeric features and labels covering benign traffic plus 34 attack categories spanning DDoS, DoS, reconnaissance, brute force, and spoofing. The dataset is pre-split into training, test, and validation CSV files totalling approximately 1.6 GB for training.",
        "I use the CICIoT2023 dataset (Pinto Neto et al., 2023). It has flow-level records with 46 numeric features. Labels cover benign traffic and 34 attack types including DDoS, DoS, reconnaissance, brute force and spoofing. The data comes pre-split into train, test and validation. The training set is about 1.6 GB."
    ),
    # 2.2 - binary
    (
        "Binary classification was chosen to simplify the prototype scope while still testing the core detection capability. The binary formulation is also more representative of the primary SOC question—\"Is this traffic malicious?\"—before deeper categorisation. Multi-class extension is identified as future work.",
        "I use binary classification to keep the scope manageable. It still tests the main detection task. The main SOC question is: is this traffic malicious? Multi-class can be added later."
    ),
    # 2.3 - graph intro
    (
        "Because the public CICIoT2023 dataset does not contain device identifiers or IP addresses, a device-topology graph cannot be constructed. Instead, following the attribute-similarity approach validated by Ngo et al. (2025) and Basak et al. (2025), graphs are built using feature-space proximity:",
        "CICIoT2023 has no device IDs or IP addresses. So I cannot build a device-topology graph. Instead I follow Ngo et al. (2025) and Basak et al. (2025). I build graphs from feature similarity:"
    ),
    # 2.5
    (
        "All experiments use the pre-defined CICIoT2023 train/validation/test splits to ensure comparability with published benchmarks. The validation set is used for hyperparameter selection and early stopping (patience=5 epochs); the test set is held out and used only for final evaluation. A fixed random seed ensures reproducibility across all experiments.",
        "I use the pre-defined train/validation/test splits. This lets me compare with other papers. I use validation for hyperparameters and early stopping (patience=5). I only use the test set for final evaluation. A fixed seed makes results reproducible."
    ),
    # 2.5 - class weighted
    (
        "Class-weighted loss (automatically computed inversely proportional to class frequency) is applied during all neural model training to prevent the classifier from trivially predicting the majority class.",
        "I use class-weighted loss for all neural models. This stops the model from just predicting the majority class."
    ),
    # 2.6 - metrics intro
    (
        "Performance is evaluated using the following metrics, chosen for their relevance to SOC operations:",
        "I evaluate performance with these metrics. I chose them because they matter for SOC work:"
    ),
    # 2.6 - FAR
    (
        "False alarm rate is particularly critical for SOC deployment: a system with high recall but high FAR will overwhelm analysts with false positives, negating the operational benefit.",
        "False alarm rate is very important for SOCs. High recall with high FAR means too many false positives. Analysts get overwhelmed."
    ),
    # 2.7 - explainability
    (
        "Explainability is implemented through two complementary mechanisms:",
        "I use two methods for explainability:"
    ),
    # 2.7 - alerts
    (
        "Alerts are formatted as ECS (Elastic Common Schema) JSON objects and served via a FastAPI endpoint (POST /score). Each alert includes: timestamp, alert severity, detection score, classification label, top contributing features with attribution scores, and top influential neighbouring flows. This format is directly ingestible by standard SIEM platforms (Elastic Security, Splunk).",
        "Alerts are in ECS JSON format. I serve them via a FastAPI endpoint (POST /score). Each alert has timestamp, severity, score, label, top features and top neighbour flows. This works with Elastic Security and Splunk."
    ),
    # 2.7 - Figure
    (
        "Figure 1 illustrates the complete research pipeline from raw data to SIEM alert generation.",
        "Figure 1 shows the full pipeline from raw data to SIEM alerts."
    ),
    # 2.8 - threefold
    (
        "The academic contribution of this project is threefold:",
        "My project contributes in three ways:"
    ),
    # 2.8 - point 1
    (
        "**Architectural novelty:** It is the first prototype, to the author's knowledge, that combines a dynamic GNN (GAT+GRU) with federated learning, dual explainability, and SIEM integration in a single system evaluated on CICIoT2023 (Table 1, Section 1.6).",
        "**Architectural novelty:** As far as I know, no other prototype combines dynamic GNN (GAT+GRU), federated learning, dual explainability and SIEM integration. All in one system on CICIoT2023 (Table 1, Section 1.6)."
    ),
    # 2.8 - point 2
    (
        "**Empirical evidence:** It provides controlled experimental comparison between flat models, centralised GNN, and federated GNN on the same dataset with identical preprocessing, directly testing whether graph structure and federated training add value.",
        "**Empirical evidence:** I compare flat models, centralised GNN and federated GNN on the same data. Same preprocessing. This tests whether graphs and federated learning actually help."
    ),
    # 2.8 - point 3
    (
        "**Practical applicability:** It demonstrates that the complete pipeline—from flow ingestion to explained SIEM alert—runs on CPU within latency bounds acceptable for edge deployment, bridging the gap between research prototypes and operational SOC tools.",
        "**Practical applicability:** The full pipeline runs on CPU. Latency is acceptable for edge use. So it bridges research and real SOC tools."
    ),
    # 2.8 - verifiable
    (
        "The research goal is verifiable: each hypothesis (H1–H3) maps to specific, measurable metrics that can be objectively evaluated on the held-out test set.",
        "Each hypothesis (H1–H3) maps to clear metrics. I can check them on the test set."
    ),
    # 2.9 - limitations intro
    (
        "Several limitations are anticipated and will be critically discussed in the final dissertation:",
        "I expect several limitations. I will discuss them in the final report:"
    ),
    # 3.1 intro
    (
        "As of the interim submission date, the following components are complete:",
        "At the time of this report, I have completed:"
    ),
    # 3.1 - core work
    (
        "The core engineering work is substantially complete. The remaining work focuses on rigorous evaluation, analysis, and dissertation writing.",
        "The main engineering is done. What is left is evaluation, analysis and writing the dissertation."
    ),
    # 3.2 intro
    (
        "This section briefly describes the current status of my project and in more detail describes how I intend to progress it to completion. It also includes an indication of how I will proceed if the results I collect are in some way deficient.",
        "This section describes where my project is now. It also describes how I will finish it. I include what I will do if the results are not as expected."
    ),
    # 3.3 - Scenario 1
    (
        "**Scenario 1: Results do not support H1 (GNN does not outperform baselines).** This is a valid finding. The discussion will analyse why flat models may suffice for this dataset (e.g., kNN graphs may not add information beyond what features already encode). The contribution shifts to empirical evidence about when graph structure does and does not help.",
        "**Scenario 1: GNN does not beat baselines.** That is still a valid result. I will discuss why flat models might be enough (e.g. kNN graphs may not add much). The contribution becomes evidence about when graphs help and when they do not."
    ),
    # 3.3 - Scenario 2
    (
        "**Scenario 2: Federated GNN shows significant degradation (>5% F1 drop).** The analysis will investigate whether non-IID heterogeneity or insufficient communication rounds caused the gap, with recommendations for techniques such as FedProx or increased rounds to mitigate the effect.",
        "**Scenario 2: Federated GNN drops a lot (>5% F1).** I will look at whether non-IID data or too few rounds caused it. I may suggest FedProx or more rounds."
    ),
    # 3.3 - Scenario 3
    (
        "**Scenario 3: Explainability computation is too slow for real-time use.** Integrated Gradients will be applied to a representative subset of alerts (e.g., top 10% by confidence score) rather than all alerts, with the latency trade-off documented. The discussion will recommend approximation methods (e.g., gradient × input) for full-speed deployment.",
        "**Scenario 3: Explainability is too slow.** I will use it only for some alerts (e.g. top 10% by confidence). I will document the trade-off. I may suggest faster approximations like gradient × input."
    ),
    # 3.3 - Scenario 4
    (
        "**Scenario 4: Time overrun on writing.** The sensitivity analysis (Week 6) is designated as optional and will be dropped first. The core deliverable (four-model comparison with explanations and alerts) remains achievable within the remaining timeline.",
        "**Scenario 4: Running out of time.** I will drop the sensitivity analysis first. The main deliverable (four models, explanations, alerts) is still achievable."
    ),
]


def main():
    content = BACKUP_PATH.read_text(encoding="utf-8")
    count = 0
    for old, new in REPLACEMENTS:
        if old in content:
            content = content.replace(old, new)
            count += 1
    BACKUP_PATH.write_text(content, encoding="utf-8")
    print(f"Refined {count} sections in {BACKUP_PATH.name}")


if __name__ == "__main__":
    main()
