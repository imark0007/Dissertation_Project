"""Apply remaining interim report fixes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "Interim_Report_Arka_Talukder.md"

def main():
    content = PATH.read_text(encoding="utf-8")

    # Fix 3: Fix corrupted mermaid and replace with proper flowchart
    old_fig = """**Figure 1. Research pipeline flowchart.**

\\mermaid
flowchart LR
    subgraph Input
        A[CICIoT2023]
    end
    subgraph Preprocessing
        B[StandardScaler]
    end
    subgraph Graph
        C[kNN Graph Construction]
        D[50 flows/window, k=5]
    end
    subgraph Sequence
        E[Sequence of 5 windows]
    end
    subgraph Models
        F[RF, MLP]
        G[Central GNN]
        H[Federated GNN]
    end
    subgraph Explainability
        I[Captum IG + GAT attention]
    end
    subgraph SIEM
        J[ECS JSON via FastAPI]
    end
    subgraph Evaluation
        K[Precision, Recall, F1, ROC-AUC, FAR]
    end
    A --> B --> C --> D --> E --> F
    E --> G
    E --> H
    F --> I
    G --> I
    H --> I
    I --> J --> K
\\
*Figure 1.* Research pipeline: CICIoT2023 flows are preprocessed (StandardScaler), converted to kNN graphs (50 flows/window, k=5), grouped into sequences of 5 windows, and fed to RF, MLP, central GNN, and federated GNN. Explainability (Captum IG and GAT attention) produces SIEM-style ECS JSON alerts via FastAPI. Evaluation uses Precision, Recall, F1, ROC-AUC, and FAR."""

    new_fig = """**Figure 1. Research pipeline flowchart.**

```mermaid
flowchart LR
    A[CICIoT2023] --> B[Preprocessing: StandardScaler]
    B --> C[kNN Graph: 50 flows/window, k=5]
    C --> D[Sequence of 5 windows]
    D --> E[Model Training: RF, MLP, Central GNN, Federated GNN]
    E --> F[Explainability: Captum IG + GAT attention]
    F --> G[SIEM Alerts: ECS JSON via FastAPI]
    G --> H[Evaluation: Precision, Recall, F1, ROC-AUC, FAR]
```

*Figure 1.* Research pipeline: CICIoT2023 flows are preprocessed (StandardScaler), converted to kNN graphs (50 flows/window, k=5), grouped into sequences of 5 windows, and fed to RF, MLP, central GNN, and federated GNN. Explainability (Captum IG and GAT attention) produces SIEM-style ECS JSON alerts via FastAPI. Evaluation uses Precision, Recall, F1, ROC-AUC, and FAR."""

    content = content.replace(old_fig, new_fig)
    print("Fix 3: Figure 1 flowchart - done")

    # Fix 4: Add design justification paragraph after 2.2 or in 2.3
    old_23 = """### 2.3 Models

Three models are used. **Random Forest** and **MLP** work on flat (tabular) features. The **dynamic GNN** uses GAT layers on each graph and a GRU over the sequence of graph embeddings, then a classifier. The same GNN architecture is used for both centralised and federated training so that performance can be compared fairly."""

    new_23 = """### 2.3 Models

Three models are used. **Random Forest** and **MLP** work on flat (tabular) features. The **dynamic GNN** uses GAT layers on each graph and a GRU over the sequence of graph embeddings, then a classifier. The same GNN architecture is used for both centralised and federated training so that performance can be compared fairly.

**Design justification.** The graph parameters are chosen to balance expressiveness and computational cost. A window size of 50 flows per graph allows enough structure for the GNN while keeping sequences manageable; Ngo et al. (2025) use similar windowing for attribute-based graphs. k=5 for kNN ensures each node has local neighbours without excessive connectivity. A sequence length of 5 windows gives the GRU sufficient temporal context for attack patterns. The Dirichlet alpha=0.5 for federated splits creates moderate non-IID (Albanbay et al., 2025 use similar settings); lower alpha would skew more severely. Binary classification (benign vs attack) is used because the project focuses on detection for SOC triage rather than fine-grained attack classification; multi-class labels could be added in future work."""

    content = content.replace(old_23, new_23)
    print("Fix 4: Design justification - done")

    # Fix 5: Add completion timeline table in 3.2
    old_32 = """### 3.2 How I Intend to Progress to Completion (In More Detail)

I will ensure all experiments (centralised baselines, centralised GNN, federated GNN) are run with the final configuration and that all metrics and figures are up to date."""

    new_32 = """### 3.2 How I Intend to Progress to Completion (In More Detail)

Table 1 gives a week-by-week plan from the current midpoint to final submission.

| Week | Tasks |
|------|-------|
| Week 1 | Run final experiments (centralised baselines, centralised GNN, federated GNN); collate metrics; generate ROC curves and confusion matrices |
| Week 2 | Sensitivity checks (if time); finalise code comments and README; begin writing results chapter |
| Week 3 | Complete results chapter; insert figures and tables; write discussion; link to research questions |
| Week 4 | Write conclusion; critical self-evaluation; incorporate interim report feedback |
| Week 5 | Finalise references (Harvard style); proofread; structure and word count check |
| Week 6 | Submission deadline; buffer for revisions and supervisor feedback |

*Table 1.* Completion timeline from midpoint to final submission.

I will ensure all experiments (centralised baselines, centralised GNN, federated GNN) are run with the final configuration and that all metrics and figures are up to date."""

    content = content.replace(old_32, new_32)
    print("Fix 5: Completion timeline - done")

    # Fix 4: Add design justification paragraph after 2.2 (alternative - in 2.2)
    old_22_end = """All settings are recorded in the project configuration file so that the work can be repeated."""

    # No change needed - already added in 2.3

    # Fix 6: Fix references - expand et al., fix italicisation
    refs = {
        "Albanbay, N., Tursynbek, Y., Graffi, K., et al. (2025)": "Albanbay, N., Tursynbek, Y., Graffi, K., Uskenbayeva, R., Kalpeyeva, Z., Abilkaiyr, Z. and Ayapov, Y. (2025)",
        "Kokhlikyan, N., Miglani, V., Martin, M., et al. (2020)": "Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., Melnikov, A., Kliushkina, N., Araya, C., Yan, S. and Reblitz-Richardson, O. (2020)",
        "Qu, Y., Gao, L., Luan, T.H., et al. (2019)": "Qu, Y., Gao, L., Luan, T.H., Xiang, Y., Yu, S., Li, B. and Zheng, B. (2019)",
    }
    for old, new in refs.items():
        content = content.replace(old, new)
    print("Fix 6: References (expand et al.) - done")

    # Fix 8: Format equations
    old_eq = """**Precision** is the proportion of predicted positives that are truly positive:

    Precision = TP / (TP + FP)

**Recall** is the proportion of true positives that are correctly predicted:

    Recall = TP / (TP + FN)

**F1-score** combines precision and recall (harmonic mean):

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

**False alarm rate (FAR)** is the proportion of true benign samples wrongly classified as attack:

    FAR = FP / (FP + TN)"""

    new_eq = """**Precision** (Eq. 1) is the proportion of predicted positives that are truly positive:

    $$\text{Precision} = \frac{TP}{TP + FP} \quad \text{(Eq. 1)}$$

**Recall** (Eq. 2) is the proportion of true positives that are correctly predicted:

    $$\text{Recall} = \frac{TP}{TP + FN} \quad \text{(Eq. 2)}$$

**F1-score** (Eq. 3) combines precision and recall (harmonic mean):

    $$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{(Eq. 3)}$$

**False alarm rate (FAR)** (Eq. 4) is the proportion of true benign samples wrongly classified as attack:

    $$\text{FAR} = \frac{FP}{FP + TN} \quad \text{(Eq. 4)}$$"""

    content = content.replace(old_eq, new_eq)
    print("Fix 8: Equations - done")

    PATH.write_text(content, encoding="utf-8")
    print("All fixes applied.")

if __name__ == "__main__":
    main()
