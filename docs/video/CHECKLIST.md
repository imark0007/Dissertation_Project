# Day-of checklist — preflight, API, record, export

Paths from **repository root** unless noted. This file is in `docs/video/`.

**aligned with final report:** What you show ([`results/metrics/results_table.csv`](../../results/metrics/results_table.csv), figures) should match **[`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx)** / [`Arka_Talukder_Dissertation_Final_DRAFT.md`](../../Arka_Talukder_Dissertation_Final_DRAFT.md). If you re-ran experiments after submitting, either refresh `results/` or say on camera that numbers are from the **logged** run cited in the thesis.

## Before you hit record

- [ ] Close AI chat / agent side panels — [`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md)
- [ ] Collapse folders you will not open; editor/browser zoom **125–150%**

### Have ready (Alt-Tab)

| # | File |
|---|------|
| 1 | [`results/metrics/results_table.csv`](../../results/metrics/results_table.csv) |
| 2 | [`results/figures/cm_gnn.png`](../../results/figures/cm_gnn.png) (or `cm_rf` / `cm_mlp`) |
| 3 | [`results/figures/roc_gnn.png`](../../results/figures/roc_gnn.png) (optional) |
| 4 | [`results/figures/fl_convergence.png`](../../results/figures/fl_convergence.png) |
| 5 | [`results/metrics/fl_rounds.json`](../../results/metrics/fl_rounds.json) (optional) |
| 6 | [`results/alerts/example_alerts.json`](../../results/alerts/example_alerts.json) |
| 7 | [`config/experiment.yaml`](../../config/experiment.yaml) |
| 8 | [`SETUP_AND_RUN.md`](../../SETUP_AND_RUN.md) |
| 9 | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) |

**Optional 5 s:** [`src/models/dynamic_gnn.py`](../../src/models/dynamic_gnn.py) (class name only)

### API (must work before recording)

```text
uvicorn src.siem.api:app --reload
```

- [ ] [GET /health](http://127.0.0.1:8000/health) returns `ok`
- [ ] [`results/checkpoints/dynamic_gnn_best.pt`](../../results/checkpoints/dynamic_gnn_best.pt) exists

1. [Open /docs](http://127.0.0.1:8000/docs) → **POST** `/score`
2. Body: **5** windows, each `features` = **50×46** (see [`tests/test_api.py`](../../tests/test_api.py))

**Generate `score_body.json` at repo root:**

```powershell
python -c "import json, numpy as np; w={'features': np.random.randn(50, 46).tolist()}; open('score_body.json','w',encoding='utf-8').write(json.dumps({'windows':[w]*5}))"
```

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" --data-binary "@score_body.json"
```

- [ ] Response has `prediction`, `score`, `inference_ms`, `alert` with `explanation`

**Sanity:** `python tests/test_api.py` (off camera)

**Do not** run a long cold `run_all` on video — use pre-generated `results/`.

**Script:** read [`SCRIPT.md`](SCRIPT.md)

---

## Recording

| Item | Suggestion |
|------|------------|
| Resolution | **1080p** minimum |
| Audio | Headset or quiet room |
| Pace | [`SCRIPT.md`](SCRIPT.md) — pause between blocks |
| Tools | OBS, Xbox Game Bar (Win+G), Clipchamp, or PowerPoint |
| Failsafe | If API fails, keep `example_alerts.json` walkthrough; offer live demo separately |

**Length:** **5:00–5:50** talking + 5–10 s title + 5–10 s end card → **≤ 6:00** total.

- **Over 6:30?** Shorten block 4 to one figure + one sentence, or use Swagger only (no curl).
- **Under 5:00?** Add ~15 s: [`ablation_bar.png`](../../results/figures/ablation_bar.png) or [`tests/test_api.py`](../../tests/test_api.py) on screen.

## After recording

- Trim start/end silence; normalise voice if needed; export **H.264 MP4**.

**Example name:** `B01821011_Talukder_IoT_GNN_SIEM_Demo_6min.mp4`

**More context:** [`GUIDE.md`](GUIDE.md) · viva: [`viva/PROJECT_VIVA_MASTER_BRIEF.md`](../viva/PROJECT_VIVA_MASTER_BRIEF.md)
