# Master guide: 5–6 minute “how the whole project works” video

**Audience:** Dr Raja Ujjan (supervisor) and anyone who must see **end-to-end behaviour** and **artefacts**, not a chapter-by-chapter thesis defence.  
**Written thesis:** [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) — the video should **not contradict** the abstract (metrics stated there are on a **fixed** subset; limitations apply).

---

## 1. What the supervisor is really asking

Two questions, answered in this order:

1. **How does the whole system work?**  
   **Data (CICIoT2023 flows)** → **preprocess + scaler (train only)** → **kNN graphs in windows** → **sequences of windows** → **GAT+GRU** classifier, plus **RF/MLP** on the same flows as flat features → optional **Flower FedAvg** → **explanations (IG + attention)** → **ECS-style JSON** via **FastAPI** on **CPU**.

2. **What is the actual output?**  
   - **Research outputs:** metrics CSV/JSON, figures, checkpoints, FL logs, precomputed **example** alerts.  
   - **“Product-shaped” output:** the **JSON alert** structure + **live** `POST /score` response.  
   - **Academic output:** the **Word dissertation** in `submission/`.  
   See [`PROJECT_OUTPUTS.md`](PROJECT_OUTPUTS.md) for a copy-paste table.

---

## 2. Story arc (5–6 minutes total)

| Block | Time | Goal | On-screen |
|-------|------|------|-----------|
| **1** | 0:00–1:00 | **Problem + your contribution** in plain language | Title; optional README pipeline figure |
| **2** | 1:00–2:00 | **Data and representation** (no IP topology; kNN on attributes) | `config/experiment.yaml` |
| **3** | 2:00–3:10 | **Models + central results** (baselines + GNN) | `results/metrics/results_table.csv` + one of `cm_*.png` or `roc_*.png` |
| **4** | 3:10–3:50 | **Federated** training (why + evidence) | `fl_convergence.png` or `fl_rounds.json` |
| **5** | 3:50–5:10 | **Star segment:** **explainable alerts** + **live API** | `example_alerts.json` + browser `http://127.0.0.1:8000/docs` or `curl` |
| **6** | 5:10–5:50 | **Reproduce** + **limits** + thesis pointer | `SETUP_AND_RUN.md` + `submission/*.docx` |
| **Outro** | 5:50–6:00 | Thanks, open questions | Your name, ID |

If **long:** cut live FL terminals; keep one FL figure. If **short:** add ablation bar or `tests/test_api.py` (5–10 s).

---

## 3. How this maps to *your* dissertation (deep alignment)

- **Abstract** states the **research question**, **GAT+GRU**, **kNN flow graphs**, **Flower**, **Captum + attention**, **FastAPI**, **single** `config/experiment.yaml`, and **headline** subset metrics. The video is the **same story** in **operational** order—**not** a literature review.
- **Chapters 4–5 (design/implementation):** the video’s **Block 2** (graphs, sequences) and **Block 5** (alert formatting, API) match what you **built**; point to **files and config**, not UML, unless you have 10 seconds to spare.
- **Chapters 7–8 (evaluation):** **Block 3** (table + one figure) and **Block 4** (FL) match. Say **once** that **100%** metrics are **on your fixed split** and that the thesis discusses **reliability** (ablation, sensitivity, multi-seed)—**do not** sound defensive; **acknowledge and move on**.
- **Limitations (later chapters / discussion):** **Block 6** — CIC subset, **no** large-scale external test, **no** SOC user study, prototype not a production SIEM.

---

## 4. Credibility checklist (before you record)

- **Pre-generate** `results/`; do **not** run a full long training run on video unless using `--nrows` smoke settings.
- **Start** `uvicorn` and **one** successful `POST /score` **before** recording ([`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md)).
- **Screen hygiene:** close AI side panels; large font; see [`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md).

---

## 5. Files in this folder to use, in order

1. [`PROJECT_OUTPUTS.md`](PROJECT_OUTPUTS.md) — memorise the **output** table.  
2. [`DISSERTATION_TO_VIDEO_MAP.md`](DISSERTATION_TO_VIDEO_MAP.md) — if you forget what chapter backs a line.  
3. [`SPOKEN_SCRIPT.md`](SPOKEN_SCRIPT.md) — read or teleprompter.  
4. [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md) — day-of.  
5. [`RECORDING_GUIDE.md`](RECORDING_GUIDE.md) — export and length.

**Canonical copies** in [`viva/`](../viva/): `VIDEO_DEMO_*.md` (same content family). Prefer **`docs/video/`** (this tree) for one “presentation pack” folder.
