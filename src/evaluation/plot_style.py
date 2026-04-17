"""
Matplotlib styling for thesis figures.

Uses built-in style sheets documented at:
https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
Falls back to manual rcParams if a style name is unavailable.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

_CANDIDATE_STYLES = (
    "seaborn-v0_8-whitegrid",
    "seaborn-whitegrid",
    "ggplot",
    "bmh",
    "tableau-colorblind10",
)


def apply_thesis_style() -> None:
    """Apply a consistent, publication-friendly Matplotlib style (no data changes)."""
    for name in _CANDIDATE_STYLES:
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    else:
        plt.rcParams.update(
            {
                "figure.figsize": (6.0, 4.0),
                "figure.dpi": 100,
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.grid": True,
                "grid.alpha": 0.3,
            }
        )
