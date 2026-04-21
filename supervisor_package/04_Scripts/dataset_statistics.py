"""
Compute and save dataset statistics for the dissertation (§6.3) and reproducibility.
Reads from data/processed/ (parquets) and optionally data/graphs/ (sequence counts).

Usage: python scripts/dataset_statistics.py [--config config/experiment.yaml]
Output: results/metrics/dataset_stats.json (+ print summary)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="Dataset statistics for thesis §6.3")
    ap.add_argument("--config", default="config/experiment.yaml")
    args = ap.parse_args()

    from src.data.preprocess import load_config
    cfg = load_config(args.config)
    proc_dir = Path(cfg["data"]["processed_dir"])
    out_dir = Path(cfg["paths"]["metrics"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"splits": {}, "flow_level": {}, "graph_sequences": {}}

    for split in ["train", "validation", "test"]:
        pq = proc_dir / f"{split}.parquet"
        if not pq.exists():
            print(f"Skip {split}: {pq} not found")
            continue
        import pandas as pd
        df = pd.read_parquet(pq)
        n = len(df)
        if "binary_label" in df.columns:
            benign = (df["binary_label"] == 0).sum()
            attack = (df["binary_label"] == 1).sum()
            pct_benign = 100.0 * benign / n if n else 0
            pct_attack = 100.0 * attack / n if n else 0
        else:
            benign = attack = pct_benign = pct_attack = None
        stats["splits"][split] = {
            "rows": n,
            "benign_flows": int(benign) if benign is not None else None,
            "attack_flows": int(attack) if attack is not None else None,
            "pct_benign": round(pct_benign, 2) if pct_benign is not None else None,
            "pct_attack": round(pct_attack, 2) if pct_attack is not None else None,
        }
        stats["flow_level"][split] = f"{n} flows, {pct_benign:.1f}% benign / {pct_attack:.1f}% attack" if benign is not None else f"{n} flows"

    # Graph sequence counts (from saved .pt files)
    graph_dir = ROOT / "data" / "graphs"
    for split in ["train", "validation", "test"]:
        pt = graph_dir / f"{split}_graphs.pt"
        if pt.exists():
            import torch
            graphs = torch.load(pt, weights_only=False)
            seq_len = cfg.get("graph", {}).get("sequence_length", 5)
            n_seq = max(0, len(graphs) - seq_len + 1)
            stats["graph_sequences"][split] = {"graph_windows": len(graphs), "sequences": n_seq}
        else:
            stats["graph_sequences"][split] = None

    out_path = out_dir / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Dataset statistics saved to", out_path)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
