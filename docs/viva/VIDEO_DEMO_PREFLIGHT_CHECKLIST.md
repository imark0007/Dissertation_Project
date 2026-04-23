# 5–6 minute demo video — preflight checklist

Run through this **immediately before** you hit record. Paths are relative to the **repository root** (`attempt 2`).

## Screen and project hygiene

- [ ] Close AI chat / agent side panels (see [`viva_supervisor_materials/README.md`](../../viva_supervisor_materials/README.md)).
- [ ] Collapse folders you will not open; bump editor/browser zoom to **125–150%**.
- [ ] Work from a **clean** branch or stash unrelated edits if you will show the file tree.

## Assets to have open (or on fast Alt-Tab)

| # | File | Role |
|---|------|------|
| 1 | [`results/metrics/results_table.csv`](../../results/metrics/results_table.csv) | Block 3 — central + federated metrics |
| 2 | [`results/figures/cm_gnn.png`](../../results/figures/cm_gnn.png) (or `cm_rf.png` / `cm_mlp.png`) | Block 3 — one confusion matrix |
| 3 | [`results/figures/roc_gnn.png`](../../results/figures/roc_gnn.png) (optional second figure) | Block 3 — ROC |
| 4 | [`results/figures/fl_convergence.png`](../../results/figures/fl_convergence.png) | Block 4 — federated convergence |
| 5 | [`results/metrics/fl_rounds.json`](../../results/metrics/fl_rounds.json) (optional) | Block 4 — numeric rounds |
| 6 | [`results/alerts/example_alerts.json`](../../results/alerts/example_alerts.json) | Block 5 — scroll to **first object** (lines ~1–78): `explanation.top_nodes` / `top_features` |
| 7 | [`config/experiment.yaml`](../../config/experiment.yaml) | Block 2 — `graph:` window, `knn_k`, sequence length |
| 8 | [`SETUP_AND_RUN.md`](../../SETUP_AND_RUN.md) | Block 6 — “Full pipeline” / `run_all.py` |
| 9 | [`submission/`](../../submission/) — final `.docx` | Block 6 — written thesis (pick your current final file) |

**Optional code glance (5 seconds):** [`src/models/dynamic_gnn.py`](../../src/models/dynamic_gnn.py) (class header only).

## API: start server and verify before recording

From the project root, with `venv` activated:

```text
uvicorn src.siem.api:app --reload
```

- [ ] [`GET` `http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health) returns `{"status":"ok",...}`.
- [ ] Checkpoint exists: [`results/checkpoints/dynamic_gnn_best.pt`](../../results/checkpoints/dynamic_gnn_best.pt) (API loads it on startup).

### Option A — Swagger UI (easiest for screen capture)

1. Open [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).
2. Expand **POST** `/score`, **Try it out**.
3. Request body: JSON with `windows` = array of **5** **windows** (matches `graph.sequence_length` in [`config/experiment.yaml`](../../config/experiment.yaml)). Each window is `{ "features": <50×46 matrix> }` — fifty rows, forty-six feature columns, same as [`tests/test_api.py`](../../tests/test_api.py). **Do not** paste a tiny matrix by hand; generate it.

**Generate `score_body.json` from the repo root** (writes UTF-8, no BOM):

```powershell
python -c "import json, numpy as np; w={'features': np.random.randn(50, 46).tolist()}; open('score_body.json','w',encoding='utf-8').write(json.dumps({'windows':[w]*5}))"
```

Then POST (use **`curl.exe`** in PowerShell so you do not call `Invoke-WebRequest`):

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" --data-binary "@score_body.json"
```

Copy the same JSON from `score_body.json` into Swagger **Request body** if you prefer the **/docs** UI.

- [ ] Response includes `prediction`, `score`, `inference_ms`, and `alert` with `explanation`.

### Option B — pytest (off-camera sanity)

```text
python tests/test_api.py
```

- [ ] All tests pass (confirms **schema** without manual JSON).

## Do not do on video

- Do **not** run the full `run_all` pipeline on a **cold** run unless you use `--nrows 10000` and accept **wait** time—use **pre-generated** `results/` for the **table** and **figures**.

## Script

Read from [`VIDEO_DEMO_5MIN_ONE_PAGE_SCRIPT.md`](VIDEO_DEMO_5MIN_ONE_PAGE_SCRIPT.md) or print it.
