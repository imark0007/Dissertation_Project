"""
Render dark-theme, line-numbered code screenshots for Handbook Appendix 1.

Style target:
  - dark IDE-like theme
  - focused core snippets only (no full-module screenshots)
  - line numbers for examiner reference

Run from repo root:
    python scripts/render_appendix1_code_figures.py

Outputs:
    results/figures/appendix1/fig_a1_0*.png
"""
from __future__ import annotations

from pathlib import Path

from pygments import highlight
from pygments.formatters.img import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "appendix1"

# (relative_path, start_line, end_line, output_filename, font_size)
# Focused core excerpts only (no full file snapshots).
SLICES: list[tuple[str, int, int, str, int]] = [
    ("src/models/dynamic_gnn.py", 75, 97, "fig_a1_01_dynamic_gnn.png", 11),
    ("src/data/graph_builder.py", 37, 58, "fig_a1_02_graph_builder_knn_graph.png", 11),
    ("src/data/graph_builder.py", 106, 123, "fig_a1_03_graph_builder_stratified.png", 11),
    ("src/explain/explainer.py", 53, 95, "fig_a1_04_explainer_integrated_gradients.png", 11),
    ("src/federated/run_federated.py", 28, 71, "fig_a1_05_federated_flower_client.png", 11),
    ("src/siem/api.py", 67, 89, "fig_a1_06_fastapi_score_alert.png", 11),
]


def _font_candidates() -> list[str]:
    return ["Consolas", "Cascadia Mono", "Courier New", "Lucida Console", "monospace"]


def render_slice(rel: str, lo: int, hi: int, out_name: str, font_size: int) -> Path:
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = "\n".join(lines[lo - 1 : hi])
    last_err: Exception | None = None
    for font_name in _font_candidates():
        try:
            fmt = ImageFormatter(
                style="one-dark",
                font_name=font_name,
                font_size=font_size,
                line_numbers=True,
                line_number_start=lo,
                image_pad=12,
                line_number_pad=8,
            )
            png = highlight(chunk, PythonLexer(), fmt)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUT_DIR / out_name
            out.write_bytes(png)
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not render {rel} ({lo}-{hi}): {last_err}")


def main() -> None:
    for rel, lo, hi, name, fs in SLICES:
        p = render_slice(rel, lo, hi, name, fs)
        print("Wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
