# 5–6 minute demo video — recording and editing

Use this after [`PREFLIGHT_CHECKLIST.md`](PREFLIGHT_CHECKLIST.md) passes. Target **5:00–5:50** of spoken content; **10–15 seconds** of title/end cards if your school wants them.

## Recording

| Item | Suggestion |
|------|------------|
| **Resolution** | **1920×1080** minimum; record **display** or **window** (not a tiny region). |
| **Audio** | Quiet room; **lapel** or **headset** mic; **clap** once at start to sync if you use separate audio later. |
| **Pace** | Follow [`SPOKEN_SCRIPT.md`](SPOKEN_SCRIPT.md); **pause** between blocks. |
| **Tooling** | **OBS Studio** (free), **Xbox Game Bar** (Win+G), **Clipchamp**, or **PowerPoint** screen record—use what you already know. |
| **Failsafe** | If the **API** misbehaves during a take, **keep** the **`example_alerts.json`** walkthrough and say you will show **live** `/score` in a follow-up. |

## Structure and length

| Segment | Target duration |
|---------|-----------------|
| Title card (optional) | 5–10 s |
| Blocks 1–6 (script) | ~5:10–5:40 |
| End card (name, ID: B01821011, thanks) | 5–10 s |
| **Total** | **≤ 6:00** |

**Over 6:30?** Shorten Block 4 to **one** figure + **one** sentence, or drop live **curl** and use **Swagger** or **static** JSON only.  
**Under 5:00?** Add **15–20 s** — [`ablation_bar.png`](../results/figures/ablation_bar.png) or [`results/metrics/ablation_table.csv`](../results/metrics/ablation_table.csv), or [`tests/test_api.py`](../tests/test_api.py).

## Editing

| Step | Action |
|------|--------|
| 1 | **Top and tail** dead air at start/end. |
| 2 | **Normalise** audio (rough target —12 to —6 LUFS for speech, or your editor’s auto-level). |
| 3 | **Add chapters** (YouTube) or a **one-page** **PDF** if the LMS wants a cover sheet. |
| 4 | **Export** **H.264** **MP4** for maximum compatibility. |

## Deliverable naming (example)

`B01821011_Talukder_IoT_GNN_SIEM_Demo_6min.mp4` — or follow your programme’s rules.

## Links

- [`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md) — screen-share etiquette
- [`docs/viva/PROJECT_VIVA_MASTER_BRIEF.md`](../docs/viva/PROJECT_VIVA_MASTER_BRIEF.md) — viva Q&A
- [`VIDEO_MASTER_GUIDE.md`](VIDEO_MASTER_GUIDE.md) — story and thesis alignment
