# 5–6 minute demo video — recording and editing

Use this after [`VIDEO_DEMO_PREFLIGHT_CHECKLIST.md`](VIDEO_DEMO_PREFLIGHT_CHECKLIST.md) passes. Target **5:00–5:50** of spoken content; **10–15 seconds** of title/end cards if your school wants them.

## Recording

| Item | Suggestion |
|------|------------|
| **Resolution** | **1920×1080** minimum; record **display** or **window** (not a tiny region). |
| **Audio** | Quiet room; **lapel** or **headset** mic; **clap** once at start to sync if you use separate audio later. |
| **Pace** | Follow [`VIDEO_DEMO_5MIN_ONE_PAGE_SCRIPT.md`](VIDEO_DEMO_5MIN_ONE_PAGE_SCRIPT.md); **pause** between blocks. |
| **Tooling** | **OBS Studio** (free), **Xbox Game Bar** (Win+G), **Clipchamp**, or **PowerPoint** screen record—use what you already know. |
| **Failsafe** | If the **API** misbehaves during a take, **keep** the **`example_alerts.json`** walkthrough and say you will show **live** `/score` in a follow-up—do not burn twenty minutes re-recording. |

## Structure and length

| Segment | Target duration |
|---------|-----------------|
| Title card (optional) | 5–10 s |
| Blocks 1–6 (script) | ~5:10–5:40 |
| End card (name, ID, thanks) | 5–10 s |
| **Total** | **≤ 6:00** |

If the first cut is **over 6:30**: trim **Block 4** (federated) to **one** figure + **one** sentence, or cut **live** curl and keep **JSON** file + **/docs** only.

If **under 5:00**: add **15–20 s** from the plan’s optional **ablation** (`results/metrics/ablation_table.csv` / `ablation_bar.png`) or show [`tests/test_api.py`](../../tests/test_api.py) as a **regression** check.

## Editing

| Step | Action |
|------|--------|
| 1 | **Top and tail** dead air at start/end. |
| 2 | **Normalise** audio (—12 to —6 LUFS-ish for speech is a common target; any auto-level in the editor helps). |
| 3 | **Add chapters** (YouTube) or a **single** slide **PDF** if uploading to Moodle. |
| 4 | **Export** **H.264** **MP4** for maximum compatibility. |

## Deliverable naming (example)

`B01821011_Talukder_IoT_GNN_SIEM_Demo_6min.mp4` — or follow your programme’s file-naming rules.

## Links

- Screen-share etiquette: [`viva_supervisor_materials/README.md`](../viva_supervisor_materials/README.md)
- Project narrative: [`PROJECT_VIVA_MASTER_BRIEF.md`](PROJECT_VIVA_MASTER_BRIEF.md)
