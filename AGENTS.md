# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

IoT Network Traffic Detection system using Dynamic GNN, Federated Learning, and Explainability. Single Python project (no monorepo, no containers, no databases). See `README.md` and `SETUP_AND_RUN.md` for full docs.

### Key services

| Service | Command | Notes |
|---|---|---|
| FastAPI SIEM API | `python3 -m uvicorn src.siem.api:app --host 0.0.0.0 --port 8000` | Main runnable service. Serves `/health` and `/score` endpoints. Requires `results/checkpoints/dynamic_gnn_best.pt` to exist. |

### Running tests

```bash
python3 -m pytest tests/test_api.py -v
```

**Known issue**: The tests use `TestClient(app)` without a context manager, so FastAPI's `on_event("startup")` handler does not fire. This means `_MODEL` stays `None` and `/score` tests fail. The tests pass when the API is tested directly via curl against a running server. The `/health` and `/memory_usage` tests pass regardless.

### Missing data module

The `src/data/` module (preprocess.py, graph_builder.py, dataset.py) was not originally committed to the repo. It has been recreated to match the interfaces expected by the rest of the codebase. If the module is missing after a fresh checkout, it must be restored for any code to work.

### Dataset

The CICIoT2023 dataset (~1.6 GB) is not included in the repo. Without it, the full ML pipeline (`scripts/run_all.py`) cannot run. However, the FastAPI API and its tests work with a randomly-initialised model checkpoint.

### Checkpoint compatibility

The model checkpoint at `results/checkpoints/dynamic_gnn_best.pt` must match the current `DynamicGNN` architecture (including the `no_gru_fc` layer). If you get a `Missing key(s)` error on startup, regenerate the checkpoint:

```python
import torch
from src.data.preprocess import load_config
from src.models.dynamic_gnn import DynamicGNN
cfg = load_config('config/experiment.yaml')
model = DynamicGNN.from_config(cfg)
torch.save(model.state_dict(), 'results/checkpoints/dynamic_gnn_best.pt')
```

### Lint / build

No dedicated linter or build step is configured. Use `python3 -m pytest --collect-only` to verify all imports resolve.

### Environment notes

- Python 3.12+ with pip. No virtualenv needed in Cloud Agent VMs.
- `python` is not aliased; always use `python3`.
- All dependencies from `requirements.txt` include PyTorch, PyG, Flower, FastAPI, etc.
