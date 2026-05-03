# Supervisor meeting bundle (single folder)

**Student:** Arka Talukder (B01821011)  
**Purpose:** Everything listed in **`OPEN_FILES_IN_ORDER.md`** is copied here with the **same relative paths** as the main repository (under this folder). Open this folder in an editor or zip it for Dr Ujjan.

**Canonical project:** The full runnable repo is the **parent repository**. This bundle is a **read-only snapshot** for walkthrough; it does **not** include `data/`, `venv/`, or all of `scripts/`.

---

## Start here

1. Read **`OPEN_FILES_IN_ORDER.md`** (same content as `docs/viva/supervisor_share/01_FILES_TO_OPEN_IN_ORDER.md`).  
2. Open files **top to bottom**; paths are **relative to this folder**.

---

## Path map (original repo → this bundle)

| Original path | In this bundle |
|-------------|----------------|
| `README.md` (repo root) | **`PROJECT_README.md`** |
| `config/experiment.yaml` | `config/experiment.yaml` |
| `src/data/preprocess.py` | `src/data/preprocess.py` |
| `src/data/graph_builder.py` | `src/data/graph_builder.py` |
| `src/data/dataset.py` | `src/data/dataset.py` |
| `src/models/dynamic_gnn.py` | `src/models/dynamic_gnn.py` |
| `src/models/baselines.py` | `src/models/baselines.py` |
| `scripts/run_all.py` | `scripts/run_all.py` |
| `src/federated/server.py` | `src/federated/server.py` |
| `src/federated/client.py` | `src/federated/client.py` |
| `src/federated/data_split.py` | `src/federated/data_split.py` |
| `src/explain/explainer.py` | `src/explain/explainer.py` |
| `src/siem/alert_formatter.py` | `src/siem/alert_formatter.py` |
| `src/siem/api.py` | `src/siem/api.py` |
| `results/metrics/results_table.csv` | `results/metrics/results_table.csv` |
| `results/metrics/rf_metrics.json` | `results/metrics/rf_metrics.json` |
| `results/metrics/mlp_metrics.json` | `results/metrics/mlp_metrics.json` |
| `results/metrics/fl_rounds.json` | `results/metrics/fl_rounds.json` |
| `results/alerts/example_alerts.json` | `results/alerts/example_alerts.json` |
| `results/figures/cm_gnn.png` | `results/figures/cm_gnn.png` |
| `results/figures/fl_convergence.png` | `results/figures/fl_convergence.png` |
| `Arka_Talukder_Dissertation_Final_DRAFT.md` | `thesis/Arka_Talukder_Dissertation_Final_DRAFT.md` |
| `submission/B01821011_Arka_Talukder_Dissertation_Final.docx` | `thesis/B01821011_Arka_Talukder_Dissertation_Final.docx` (if present when bundle was built) |

---

## Refresh this bundle from the main repo

From the **repository root** (parent of `docs/`):

```powershell
# Re-run the same Copy-Item block as in maintainers' notes, or use your IDE to sync.
# Easiest: delete docs/viva/supervisor_meeting_bundle and regenerate from a small script
# (ask to refresh in thesis_meeting prep before each supervisor call).
```

The **`GUIDE_supervisor_share_hub.md`** file links the rest of the viva pack in the **main** repo under `docs/viva/supervisor_share/`.

---

## Zip for email

Zip the whole **`supervisor_meeting_bundle`** folder. Size includes thesis **Markdown** and possibly **Word**; exclude from email if too large and send **Markdown + code + metrics** only.

---

*Generated as a snapshot for supervisor / viva screen share. Not a substitute for the full Git repository.*
