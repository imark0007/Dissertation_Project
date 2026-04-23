"""
Render dark-theme, line-numbered code screenshots for Appendix A (core pipeline).

Run from repo root:
    python scripts/render_appendix1_code_figures.py

Outputs:
    results/figures/appendix1/fig_a1_*.png
"""
from __future__ import annotations

from pathlib import Path

from pygments import highlight
from pygments.formatters.img import ImageFormatter
from pygments.lexers import PythonLexer, YamlLexer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "appendix1"

# (relative_path, start_line, end_line inclusive, output filename, font_size, lexer name)
SLICES: list[tuple[str, int, int, str, int, str]] = [
    ("config/experiment.yaml", 60, 98, "fig_a1_01_experiment_yaml.png", 9, "yaml"),
    ("src/data/preprocess.py", 67, 98, "fig_a1_02_preprocess.png", 11, "python"),
    ("src/data/graph_builder.py", 37, 58, "fig_a1_03_graph_knn.png", 11, "python"),
    ("src/data/graph_builder.py", 106, 123, "fig_a1_04_graph_stratified.png", 11, "python"),
    ("src/data/dataset.py", 22, 39, "fig_a1_05_graph_sequence_dataset.png", 11, "python"),
    ("src/data/dataset.py", 49, 75, "fig_a1_06_dataloaders.png", 11, "python"),
    ("src/models/baselines.py", 31, 65, "fig_a1_07_baselines.png", 11, "python"),
    ("src/models/dynamic_gnn.py", 75, 97, "fig_a1_08_dynamic_gnn.png", 11, "python"),
    ("src/models/trainer.py", 39, 62, "fig_a1_09_train_one_epoch.png", 11, "python"),
    ("src/explain/explainer.py", 53, 102, "fig_a1_10_explain_sequence.png", 11, "python"),
    ("src/federated/run_federated.py", 28, 71, "fig_a1_11_run_federated.png", 11, "python"),
    ("src/federated/client.py", 24, 57, "fig_a1_12_flower_client.png", 11, "python"),
    ("src/federated/server.py", 52, 70, "fig_a1_13_flower_server.png", 11, "python"),
    ("src/siem/api.py", 67, 89, "fig_a1_14_fastapi_score.png", 11, "python"),
]


def _font_candidates() -> list[str]:
    return ["Consolas", "Cascadia Mono", "Courier New", "Lucida Console", "monospace"]


def _lexer(name: str):
    if name == "yaml":
        return YamlLexer()
    return PythonLexer()


def render_slice(
    rel: str, lo: int, hi: int, out_name: str, font_size: int, lexer_name: str
) -> Path:
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = "\n".join(lines[lo - 1 : hi])
    lexer = _lexer(lexer_name)
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
            png = highlight(chunk, lexer, fmt)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUT_DIR / out_name
            out.write_bytes(png)
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not render {rel} ({lo}-{hi}): {last_err}")


def main() -> None:
    for rel, lo, hi, name, fs, lex in SLICES:
        p = render_slice(rel, lo, hi, name, fs, lex)
        print("Wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
