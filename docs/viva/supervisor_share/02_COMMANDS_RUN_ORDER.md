# Commands in run order (copy from repo after venv + data)

Assume: **`data/raw/train.csv`**, **`validation.csv`**, **`test.csv`** exist (CICIoT2023). Windows shell; adjust for Mac/Linux.

```text
cd <repo_root>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch Geometric for your CPU/CUDA (see PyG docs).
```

### Core pipeline (thesis main results path)

```text
python scripts/run_all.py --config config/experiment.yaml
```

### After checkpoints exist

```text
python scripts/generate_alerts_and_plots.py
```

### Chapter 8 extras

```text
python scripts/run_ablation.py --config config/experiment.yaml
python scripts/run_sensitivity_and_seeds.py --config config/experiment.yaml
```

### Federated (full commands in SETUP_AND_RUN.md)

```text
# 1) Split client graphs (once) — see SETUP_AND_RUN.md for exact module invocation
# 2) Start server + 3 clients — see SETUP_AND_RUN.md
```

### API demo

```text
uvicorn src.siem.api:app --reload
```

### Tests (quick check)

```text
python -m pytest tests/ -q
```

### Dissertation export (if discussing writing workflow)

```text
python scripts/dissertation_to_docx.py
# or
python scripts/sync_dissertation_and_docx.py
```

**Say in viva:** “The **authoritative** command list for my examiner is **`SETUP_AND_RUN.md`**; this file is the **short** path I use in the meeting.”
