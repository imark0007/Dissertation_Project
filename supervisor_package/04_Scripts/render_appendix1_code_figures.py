"""
Render line-numbered code screenshots for Handbook Appendix 1 (project code figures).
Run from repo root: python scripts/render_appendix1_code_figures.py

Outputs: results/figures/appendix1/fig_a1_0*.png
Line ranges match Arka_Talukder_Dissertation_Final_DRAFT.md Appendix D at generation time.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "appendix1"

# (relative_path_from_ROOT, start_line_1based_inclusive, end_line_1based_inclusive, filename)
# Line numbers match files on disk; re-run this script if sources move.
SLICES: list[tuple[str, int, int, str]] = [
    ("src/models/dynamic_gnn.py", 12, 97, "fig_a1_01_dynamic_gnn.png"),
    ("src/data/graph_builder.py", 25, 58, "fig_a1_02_graph_builder_knn_graph.png"),
    ("src/data/graph_builder.py", 89, 129, "fig_a1_03_graph_builder_stratified.png"),
    ("src/explain/explainer.py", 53, 102, "fig_a1_04_explainer_integrated_gradients.png"),
    ("src/federated/run_federated.py", 28, 71, "fig_a1_05_federated_flower_client.png"),
    ("src/siem/api.py", 32, 89, "fig_a1_06_fastapi_score_alert.png"),
]


def _mono_family() -> str:
    for name in ("Consolas", "Cascadia Mono", "Courier New", "monospace"):
        try:
            plt.figure(figsize=(1, 1))
            plt.text(0, 0, "0", fontfamily=name)
            plt.close()
            return name
        except Exception:
            continue
    return "monospace"


def render_slice(rel: str, lo: int, hi: int, out_name: str, family: str) -> Path:
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[lo - 1 : hi]
    numbered = [f"{lo + i:4d}  {chunk[i]}" for i in range(len(chunk))]
    text = "\n".join(numbered)

    n = max(len(chunk), 1)
    fig_h = min(0.28 * n + 1.0, 26)
    fs = max(5.5, min(8.0, 9.5 - n / 55.0))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_axis_off()
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        fontsize=fs,
        fontfamily=family,
        va="top",
        ha="left",
        wrap=False,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    family = _mono_family()
    for rel, lo, hi, name in SLICES:
        p = render_slice(rel, lo, hi, name, family)
        print("Wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
