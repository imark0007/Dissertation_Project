# Main report figures (packaged export)

This directory holds an additional **packaged set** of result figures and a small script used when assembling the **written report** or a slide-style summary. It is **not** the primary output of `scripts/run_all.py`.

## Primary thesis figures

The dissertation Markdown references figures under **`results/figures/`** at the repository root. Regenerate those with the main pipeline and `scripts/generate_alerts_and_plots.py` (and related scripts) as described in **`SETUP_AND_RUN.md`**.

## Contents here

- **`figures/`** — PNGs keyed for a “main report” layout (may duplicate or complement root `results/figures/`).
- **`scripts/`** — Helper to batch-generate or organise those PNGs; see **`scripts/README.md`** in this folder if present.

When in doubt, treat **`results/metrics/`** and **`results/figures/`** as the **evidence of record** for the MSc experiments; use this folder as a **convenience bundle** for sharing or archiving a fixed figure set.
