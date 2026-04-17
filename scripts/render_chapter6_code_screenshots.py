"""
Render dark IDE-style code screenshots for Chapter 6 (implementation), matching
the presentation style used in high-quality dissertation samples (bold snippet
title + line-numbered syntax-highlighted code on a dark background).

Run from repo root:
    python scripts/render_chapter6_code_screenshots.py

Outputs under results/figures/chapter6/ — referenced from Dissertation_Arka_Talukder.md §6.10.
Each PNG is a **short excerpt** (function core, training step, or handler body), not a full module.
Requires: pygments, pillow (Pygments ImageFormatter).
"""
from __future__ import annotations

from pathlib import Path

from pygments import highlight
from pygments.formatters.img import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures" / "chapter6"

# (path relative to ROOT, start_line inclusive, end_line inclusive, output filename, font_size)
# Short “sample-style” excerpts only — not whole files (see Appendix D for wider views).
SLICES: list[tuple[str, int, int, str, int]] = [
    ("src/data/graph_builder.py", 37, 58, "fig_ch6_01_flows_to_knn_core.png", 11),
    ("src/data/graph_builder.py", 106, 123, "fig_ch6_02_stratified_split_core.png", 11),
    ("src/models/trainer.py", 39, 61, "fig_ch6_03_train_one_epoch.png", 11),
    ("src/models/dynamic_gnn.py", 84, 97, "fig_ch6_04_dynamic_gnn_forward.png", 11),
    ("src/explain/explainer.py", 34, 50, "fig_ch6_05_integrated_gradients_wrapper.png", 11),
    ("src/siem/api.py", 67, 89, "fig_ch6_06_fastapi_score_core.png", 11),
]


def _font_candidates() -> list[str]:
    return ["Consolas", "Cascadia Mono", "Courier New", "Lucida Console", "monospace"]


def render_one(rel: str, lo: int, hi: int, out_name: str, font_size: int) -> Path:
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
        except Exception as e:  # noqa: BLE001 — try next font
            last_err = e
            continue
    raise RuntimeError(f"Could not render {rel} ({lo}-{hi}): {last_err}")


def main() -> None:
    for rel, lo, hi, name, fs in SLICES:
        p = render_one(rel, lo, hi, name, fs)
        print("Wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
