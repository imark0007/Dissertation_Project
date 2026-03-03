"""Apply project fixes: worktree, inference_timing, run_fl_simulation, requirements, federated export, README."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    # 1. worktrees.json - empty setup (Python project, no npm)
    wt = ROOT / ".cursor" / "worktrees.json"
    wt.write_text('{\n  "setup-worktree": []\n}\n', encoding="utf-8")
    print("1. worktrees.json updated")

    # 2. inference_timing.py - use config window_size and knn_k
    it = ROOT / "scripts" / "inference_timing.py"
    text = it.read_text(encoding="utf-8")
    old = """    seq_len = cfg.get("graph", {}).get("sequence_length", 5)
    graphs = []
    for _ in range(seq_len):
        feats = np.random.randn(200, 46).astype(np.float32)
        graphs.append(flows_to_knn_graph(feats, 0, k=8))"""
    new = """    graph_cfg = cfg.get("graph", {})
    seq_len = graph_cfg.get("sequence_length", 5)
    window_size = graph_cfg.get("window_size", 50)
    knn_k = graph_cfg.get("knn_k", 5)
    graphs = []
    for _ in range(seq_len):
        feats = np.random.randn(window_size, 46).astype(np.float32)
        graphs.append(flows_to_knn_graph(feats, 0, k=knn_k))"""
    text = text.replace(old, new)
    it.write_text(text, encoding="utf-8")
    print("2. inference_timing.py updated")

    # 3. run_fl_simulation.py - add argparse for --config
    rfl = ROOT / "scripts" / "run_fl_simulation.py"
    text = rfl.read_text(encoding="utf-8")
    text = text.replace('import json\nimport logging', 'import argparse\nimport json\nimport logging')
    old_main = '    cfg = load_config("config/experiment.yaml")'
    new_main = '    parser = argparse.ArgumentParser()\n    parser.add_argument("--config", default="config/experiment.yaml", help="Path to config YAML")\n    args = parser.parse_args()\n    cfg = load_config(args.config)'
    text = text.replace(old_main, new_main)
    rfl.write_text(text, encoding="utf-8")
    print("3. run_fl_simulation.py updated")

    # 4. requirements.txt - add python-docx, fpdf2
    req = ROOT / "requirements.txt"
    content = req.read_text(encoding="utf-8")
    if "python-docx" not in content:
        content += "\n# Document generation\npython-docx>=0.8.11\nfpdf2>=2.7.0\n"
        req.write_text(content, encoding="utf-8")
        print("4. requirements.txt updated")
    else:
        print("4. requirements.txt already has python-docx")

    # 5. federated __init__ - export split_and_save
    fed = ROOT / "src" / "federated" / "__init__.py"
    content = fed.read_text(encoding="utf-8")
    if "split_and_save" not in content:
        content = content.rstrip()
        if not content.endswith("split_and_save"):
            content += "\nfrom .data_split import split_and_save\n"
        fed.write_text(content, encoding="utf-8")
        print("5. federated __init__.py updated")
    else:
        print("5. federated __init__.py already exports split_and_save")

    # 6. README - fix window size 200 -> 50
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("windows of 200 flows", "windows of 50 flows")
    readme.write_text(text, encoding="utf-8")
    print("6. README.md updated")

    print("Done.")


if __name__ == "__main__":
    main()
