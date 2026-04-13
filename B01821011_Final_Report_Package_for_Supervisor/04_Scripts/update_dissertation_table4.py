"""
Update Table 4 (Ablation) in Dissertation_Arka_Talukder.md from results/metrics/ablation_table.csv.
Run after: python scripts/run_ablation.py --config config/experiment.yaml

Usage: python scripts/update_dissertation_table4.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "Dissertation_Arka_Talukder.md"
METRICS_DIR = ROOT / "results" / "metrics"
CSV_PATH = METRICS_DIR / "ablation_table.csv"
GAT_ONLY_JSON = METRICS_DIR / "ablation_gat_only.json"


def get_gat_only_row():
    """Get GAT-only row from ablation_table.csv or ablation_gat_only.json."""
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if "GAT only" in row.get("Variant", ""):
                    return row
    if GAT_ONLY_JSON.exists():
        with open(GAT_ONLY_JSON, encoding="utf-8") as f:
            m = json.load(f)
        return {
            "Variant": "GAT only (no GRU)",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "Inference (ms)": f"{m.get('inference_ms', 0):.2f}",
        }
    return None


def main():
    row = get_gat_only_row()
    if not row:
        print("No ablation results found. Run: python scripts/run_ablation.py --config config/experiment.yaml")
        return

    text = MD_PATH.read_text(encoding="utf-8")
    old_line = "| GAT only (no GRU) | — | — | — | — | — |"
    new_line = (
        f"| GAT only (no GRU) | {row['Precision']} | {row['Recall']} | {row['F1']} | {row['ROC-AUC']} | {row['Inference (ms)']} |"
    )
    if old_line not in text:
        print("Table 4 GAT-only row not found or already updated.")
        return
    text = text.replace(old_line, new_line)
    # Remove the "Fill the GAT only row" note if we've filled it
    note = '*Note: Fill the "GAT only" row from `results/metrics/ablation_table.csv` after running `scripts/run_ablation.py`.*'
    if note in text:
        text = text.replace(note, "*Note: Ablation results from `scripts/run_ablation.py`.*")
    MD_PATH.write_text(text, encoding="utf-8")
    print("Updated Table 4 in Dissertation_Arka_Talukder.md with GAT-only ablation results.")


if __name__ == "__main__":
    main()
