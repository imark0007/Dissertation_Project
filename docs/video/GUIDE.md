# 5–6 minute demo — guide (story + outputs + thesis map)

**Audience:** supervisor / anyone who needs **end-to-end behaviour**, not a full thesis read.

**Aligns with final report:** *Explainable Dynamic Graph Neural Network SIEM for Software-Defined IoT using Edge AI and Federated Learning* — **Arka Talukder**, **B01821011**, **MSc Cyber Security**, **University of the West of Scotland**, **Supervisor: Dr Raja Ujjan**. Written artefact: [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) (same narrative as root [`Arka_Talukder_Dissertation_Final_DRAFT.md`](../../Arka_Talukder_Dissertation_Final_DRAFT.md)). Do not contradict the **abstract** (fixed subset; limitations apply).

### Abstract ↔ video (from your final report)

- **Research question (headline):** how an **explainable GAT+GRU** on *k*NN **flow** graphs, with **Flower FedAvg** and **SIEM-shaped** output, can run on **CPU**.
- **Contribution (prototype):** **reproducible, integrated** system on **CICIoT2023** (Pinto *et al.*, 2023): public splits, **leakage-aware** preprocessing, **Captum** integrated gradients + **GAT** attention, **FastAPI** JSON, **one** [`config/experiment.yaml`](../../config/experiment.yaml).
- **Where the thesis develops this:** Chapters **2, 4–6** justify and build; **7–8** test; **9** answers the sub-questions with **Tables 7–12** (the video names **key files** and **one** results table, not every table).
- **Headline outcomes (abstract; say only if they match your on-screen `results_table.csv` and final Word):** e.g. central/federated GNN and baselines on the **fixed** subset; **~23 ms** mean CPU inference (five-window sequence); **~31 MB** federated weight traffic over **ten** rounds — otherwise **point to the table** and “see dissertation Chapter 8.”
- **Reliability studies named in the abstract:** ablation, **(window, *k*)** sensitivity, multi-seed — optional **15 s** in the video only if you are under time (see [CHECKLIST.md](CHECKLIST.md)).

---

## 1. Two questions to answer

1. **How does it work?** CICIoT2023 **flows** → preprocess (scaler on **train** only) → **kNN** graphs in **windows** → **sequences** → **GAT+GRU** + **RF/MLP** baselines → optional **Flower FedAvg** → **IG + GAT attention** → **ECS-style JSON** via **FastAPI** on **CPU**.

2. **What is the output?** See the table below; the **star** for non-coders is **JSON alerts** + live **`POST /score`**.

---

## 2. Tangible outputs (say this in the first 60–90 s)

| Output | Where |
|--------|--------|
| Written dissertation | [`submission/B01821011_Arka_Talukder_Dissertation_Final.docx`](../../submission/B01821011_Arka_Talukder_Dissertation_Final.docx) |
| Metrics (RF, MLP, central + federated GNN) | [`results/metrics/results_table.csv`](../../results/metrics/results_table.csv) |
| Figures (ROC, confusion, FL, etc.) | [`results/figures/`](../../results/figures/) |
| FL evidence | [`results/metrics/fl_rounds.json`](../../results/metrics/fl_rounds.json), [`fl_convergence.png`](../../results/figures/fl_convergence.png) |
| Example alerts | [`results/alerts/example_alerts.json`](../../results/alerts/example_alerts.json) |
| Live API | [`src/siem/api.py`](../../src/siem/api.py) — `uvicorn src.siem.api:app` |
| Reproduce | [`config/experiment.yaml`](../../config/experiment.yaml), [`scripts/run_all.py`](../../scripts/run_all.py) |

**One-line limit (end of video):** lab subset, no SOC user study; prototype not a product SIEM.

---

## 3. Story arc (timing)

| Block | Time | On-screen (summary) |
|-------|------|---------------------|
| 1 | 0:00–1:00 | Title; optional README pipeline; optional `submission/*.docx` |
| 2 | 1:00–2:00 | [`config/experiment.yaml`](../../config/experiment.yaml) → `graph:` |
| 3 | 2:00–3:10 | `results_table.csv` + one of `cm_gnn.png` / `roc_gnn.png` |
| 4 | 3:10–3:50 | `fl_convergence.png` or `fl_rounds.json` |
| 5 | 3:50–5:10 | `example_alerts.json` + live `/docs` or `curl` |
| 6 | 5:10–5:50 | `SETUP_AND_RUN.md` + submission docx + `run_all.py` line |
| Outro | 5:50–6:00 | Thanks, B01821011 |

**Too long?** Drop live Flower terminals; keep one FL figure. **Too short?** Add `ablation_bar.png` or `tests/test_api.py` (~15 s).

---

## 4. Dissertation ↔ video (one screen each)

This mirrors the **abstract’s evidence map**: **Ch 2, 4–6** build; **7–8** test; **9** with **Tables 7–12** for sub-question answers. The **6-minute** video is **not** a table read — it **shows** the pipeline and **one** metrics file.

| Thesis bit | Video |
|------------|--------|
| Aim / RQ (abstract, Ch 1) | Blocks 1–2 |
| Design / build (Ch 4–6) | Blocks 2, 5 |
| Evaluation (Ch 7–8) | Blocks 3–4 |
| Explain + API | Block 5 |
| Repro + limits | Block 6 |

---

## 5. Before you record

- Pre-generate `results/`; do not run a **long** cold `run_all` on camera.
- [`CHECKLIST.md`](CHECKLIST.md): `uvicorn` + one good `POST /score` first.
- Screen hygiene: [`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md).

---

## 6. Next

- Speak: [**SCRIPT.md**](SCRIPT.md)  
- Show: [**BLOCKS.md**](BLOCKS.md)  
- Day-of: [**CHECKLIST.md**](CHECKLIST.md)
