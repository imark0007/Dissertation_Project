"""
Generate Figure 1: Research Pipeline Diagram
Run: pip install matplotlib
Then: python scripts/generate_figure1.py
Output: figure1_pipeline.png in project root
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create_pipeline_figure():
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    colors = {
        'data': '#2E4057',
        'preprocess': '#048A81',
        'graph': '#8B5CF6',
        'model': '#DC2626',
        'explain': '#EA580C',
        'siem': '#0284C7',
        'eval': '#059669',
        'arrow': '#374151',
        'title': '#1E293B',
        'subtitle': '#64748B',
    }

    ax.text(8, 21.3, 'Figure 1: Research Pipeline',
            fontsize=18, fontweight='bold', ha='center', va='center',
            color=colors['title'], fontfamily='serif')
    ax.text(8, 20.8, 'From Raw IoT Flow Data to Explainable SIEM Alerts',
            fontsize=11, ha='center', va='center',
            color=colors['subtitle'], fontfamily='serif', style='italic')

    def draw_stage(x, y, w, h, color, stage_num, title, details, icon=''):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='white',
            linewidth=2, alpha=0.95
        )
        ax.add_patch(box)
        ax.text(x + 0.55, y + h - 0.35, str(stage_num),
                fontsize=10, fontweight='bold', ha='center', va='center',
                color='white', fontfamily='serif')
        ax.text(x + w / 2, y + h - 0.38, f'{icon}  {title}',
                fontsize=12, fontweight='bold', ha='center', va='center',
                color='white', fontfamily='serif')
        for i, detail in enumerate(details):
            ax.text(x + w / 2, y + h - 0.85 - (i * 0.3), detail,
                    fontsize=8.5, ha='center', va='center',
                    color='white', fontfamily='serif', alpha=0.95)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('',
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=colors['arrow'],
                        lw=2.5, connectionstyle='arc3,rad=0',
                        mutation_scale=18
                    ))

    def draw_curved_arrow(x1, y1, x2, y2, rad=0.3):
        ax.annotate('',
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='->', color=colors['arrow'],
                        lw=2, connectionstyle=f'arc3,rad={rad}',
                        mutation_scale=15
                    ))

    # Stage 1: Data Input
    draw_stage(1, 19, 6, 1.4, colors['data'], 1,
               'Data Input',
               ['CICIoT2023 Dataset',
                '46 numeric features · 35 labels · Train/Val/Test splits'])

    draw_arrow(4, 19, 4, 18.6)

    # Stage 2: Preprocessing
    draw_stage(1, 17, 6, 1.5, colors['preprocess'], 2,
               'Preprocessing',
               ['Clean rows · Remove NaN/Inf values',
                'Binary labels: Benign (0) / Attack (1)',
                'StandardScaler · Save fitted scaler'])

    draw_arrow(4, 17, 4, 16.6)

    # Stage 3: Graph Construction
    draw_stage(1, 14.8, 6, 1.7, colors['graph'], 3,
               'Graph Construction',
               ['kNN graphs (k=5, Euclidean distance)',
                '50 flows per window → 1 PyG graph',
                'Stratified windowing · minority_stride=25',
                'Sequences of 5 windows for temporal input'])

    draw_arrow(4, 14.8, 4, 14.4)

    # Stage 4: Model Training
    container = FancyBboxPatch(
        (0.5, 10.5), 15, 3.7,
        boxstyle="round,pad=0.2",
        facecolor='#FEF2F2', edgecolor=colors['model'],
        linewidth=2, alpha=0.3, linestyle='--'
    )
    ax.add_patch(container)

    ax.text(8, 13.85, '4    Model Training & Comparison',
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=colors['model'], fontfamily='serif')

    # RF
    rf_box = FancyBboxPatch(
        (1, 11.6), 3.2, 1.8,
        boxstyle="round,pad=0.12",
        facecolor=colors['model'], edgecolor='white',
        linewidth=1.5, alpha=0.85
    )
    ax.add_patch(rf_box)
    ax.text(2.6, 13, 'Random Forest', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', fontfamily='serif')
    ax.text(2.6, 12.6, 'Baseline 1', fontsize=8, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(2.6, 12.2, '200 trees · depth=20', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(2.6, 11.9, 'Flat 46-dim features', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)

    # MLP
    mlp_box = FancyBboxPatch(
        (4.6, 11.6), 3.2, 1.8,
        boxstyle="round,pad=0.12",
        facecolor=colors['model'], edgecolor='white',
        linewidth=1.5, alpha=0.85
    )
    ax.add_patch(mlp_box)
    ax.text(6.2, 13, 'MLP', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', fontfamily='serif')
    ax.text(6.2, 12.6, 'Baseline 2', fontsize=8, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(6.2, 12.2, 'Layers: 128→64→32', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(6.2, 11.9, 'Dropout=0.2 · ReLU', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)

    # Central GNN
    cgnn_box = FancyBboxPatch(
        (8.2, 11.6), 3.2, 1.8,
        boxstyle="round,pad=0.12",
        facecolor=colors['model'], edgecolor='white',
        linewidth=1.5, alpha=0.95
    )
    ax.add_patch(cgnn_box)
    ax.text(9.8, 13, 'Central GNN', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', fontfamily='serif')
    ax.text(9.8, 12.6, 'Main Model', fontsize=8, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(9.8, 12.2, 'GAT (2 layers, 4 heads)', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(9.8, 11.9, '→ Pool → GRU → Classifier', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)

    # Federated GNN
    fgnn_box = FancyBboxPatch(
        (11.8, 11.6), 3.2, 1.8,
        boxstyle="round,pad=0.12",
        facecolor=colors['model'], edgecolor='white',
        linewidth=1.5, alpha=0.95
    )
    ax.add_patch(fgnn_box)
    ax.text(13.4, 13, 'Federated GNN', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', fontfamily='serif')
    ax.text(13.4, 12.6, 'Privacy-Preserving', fontsize=8, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(13.4, 12.2, 'FedAvg · 3 clients', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)
    ax.text(13.4, 11.9, 'α=0.5 non-IID · 10 rounds', fontsize=7.5, ha='center',
            va='center', color='white', fontfamily='serif', alpha=0.8)

    ax.text(2.6, 11.2, '← Flat features →', fontsize=7, ha='center',
            va='center', color=colors['model'], fontfamily='serif',
            style='italic', alpha=0.7)
    ax.text(11.6, 11.2, '← Graph sequences →', fontsize=7, ha='center',
            va='center', color=colors['model'], fontfamily='serif',
            style='italic', alpha=0.7)

    # Arrows from stage 3 to model container
    draw_arrow(4, 14.4, 2.6, 13.5)
    draw_curved_arrow(4, 14.4, 6.2, 13.5, rad=0.15)
    draw_curved_arrow(4, 14.4, 9.8, 13.5, rad=0.25)
    draw_curved_arrow(4, 14.4, 13.4, 13.5, rad=0.35)

    # Arrows down from models
    draw_arrow(5, 11.0, 5, 10.1)
    draw_arrow(11, 11.0, 11, 10.1)

    # Stage 5: Explainability
    draw_stage(1, 8, 6.5, 1.9, colors['explain'], 5,
               'Explainability',
               ['Captum Integrated Gradients',
                'Top-5 feature attributions per alert',
                'GAT attention → influential neighbour flows',
                'Dual explanation: features + structure'])

    # Stage 6: SIEM
    draw_stage(8.5, 8, 6.5, 1.9, colors['siem'], 6,
               'SIEM Alert Generation',
               ['ECS (Elastic Common Schema) JSON',
                'FastAPI endpoint: POST /score',
                'Timestamp · Severity · Score · Label',
                'Top features + neighbour flows included'])

    draw_arrow(7.5, 9, 8.5, 9)
    draw_arrow(4.25, 8, 8, 7.2)
    draw_arrow(11.75, 8, 8, 7.2)

    # Stage 7: Evaluation
    draw_stage(3, 5, 10, 2.1, colors['eval'], 7,
               'Evaluation & Comparison',
               ['Precision · Recall · F1-Score · ROC-AUC',
                'False Alarm Rate (FAR) · Confusion Matrix',
                'CPU Inference Time (ms/sample)',
                'FL Communication Cost (bytes)',
                'Test against H1, H2, H3'])

    # Legend
    legend_y = 3.8
    ax.text(8, legend_y + 0.5, 'Pipeline Legend',
            fontsize=10, fontweight='bold', ha='center', va='center',
            color=colors['title'], fontfamily='serif')

    legend_items = [
        (colors['data'], 'Data Input'),
        (colors['preprocess'], 'Preprocessing'),
        (colors['graph'], 'Graph Construction'),
        (colors['model'], 'Model Training'),
        (colors['explain'], 'Explainability'),
        (colors['siem'], 'SIEM Output'),
        (colors['eval'], 'Evaluation'),
    ]

    start_x = 1.5
    for i, (color, label) in enumerate(legend_items):
        x = start_x + (i * 1.95)
        rect = FancyBboxPatch(
            (x, legend_y - 0.15), 0.35, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white',
            linewidth=1, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y, label,
                fontsize=7.5, ha='left', va='center',
                color=colors['title'], fontfamily='serif')

    # Footer
    ax.text(8, 3.0,
            'Arka Talukder · B01821011 · MSc Cyber Security · '
            'University of the West of Scotland',
            fontsize=8, ha='center', va='center',
            color=colors['subtitle'], fontfamily='serif', style='italic')

    plt.tight_layout(pad=1.0)
    output_path = ROOT / 'figure1_pipeline.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'\nFigure saved to: {output_path.resolve()}')
    print('Resolution: 300 DPI (print quality)')


if __name__ == '__main__':
    create_pipeline_figure()
