"""Internal: structural/style compare Final vs Humanized for Chapters 1-10 (no writes)."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "submission" / "Arka_Talukder_Dissertation_Final_Submission.docx"
HUM = ROOT / "submission" / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"

CHAP_LINE = re.compile(r"^Chapter\s+(\d+)\s+[–-]", re.I)


def norm_heading(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).replace("\u2013", "-").replace("\u2014", "-")


def iter_blocks(doc: Document):
    for el in doc.element.body:
        if el.tag == qn("w:p"):
            yield "p", Paragraph(el, doc)
        elif el.tag == qn("w:tbl"):
            yield "t", Table(el, doc)


def main_body_blocks(doc: Document) -> list[tuple[str, object]]:
    """Blocks from first 'Chapter 1' heading through end of 'Chapter 10' (inclusive)."""
    blocks = list(iter_blocks(doc))
    start = None
    end = None
    for i, (kind, obj) in enumerate(blocks):
        if kind != "p":
            continue
        t = norm_heading(obj.text or "")
        m = CHAP_LINE.match(t)
        if m and m.group(1) == "1":
            start = i
            break
    if start is None:
        return []

    for j in range(start + 1, len(blocks)):
        kind, obj = blocks[j]
        if kind != "p":
            continue
        t = norm_heading(obj.text or "")
        m = CHAP_LINE.match(t)
        if m and m.group(1) == "11":
            end = j
            break
    if end is None:
        end = len(blocks)
    return blocks[start:end]


def para_info(p: Paragraph) -> dict:
    st = (p.style.name if p.style else "") or ""
    t = (p.text or "").strip()
    hl = None
    if st.startswith("Heading"):
        try:
            hl = int(st.replace("Heading", "").strip())
        except ValueError:
            hl = None
    return {
        "style": st,
        "heading_level": hl,
        "text_len": len(t),
        "is_chapter": bool(CHAP_LINE.match(norm_heading(t))),
        "heading_key": norm_heading(t)[:120] if hl or st == "Figure Caption" else "",
    }


def analyze(path: Path) -> dict:
    doc = Document(str(path))
    blocks = main_body_blocks(doc)
    headings: list[dict] = []
    style_counts: dict[str, int] = {}
    captions = 0
    tables = 0
    figure_frames = 0

    for kind, obj in blocks:
        if kind == "p":
            info = para_info(obj)
            st = info["style"]
            style_counts[st] = style_counts.get(st, 0) + 1
            if info["heading_level"] is not None or info["is_chapter"]:
                headings.append(
                    {
                        "style": st,
                        "level": info["heading_level"],
                        "key": info["heading_key"],
                        "chapter": None,
                    }
                )
                m = CHAP_LINE.match(info["heading_key"])
                if m:
                    headings[-1]["chapter"] = int(m.group(1))
            if st == "Figure Caption":
                captions += 1
        else:
            tables += 1
            tbl = obj
            if len(tbl.rows) == 1 and len(tbl.rows[0].cells) == 1:
                figure_frames += 1

    return {
        "headings": headings,
        "style_counts": style_counts,
        "captions": captions,
        "tables": tables,
        "figure_frames": figure_frames,
        "block_count": len(blocks),
    }


def heading_sequence(hlist: list[dict]) -> list[str]:
    out = []
    for h in hlist:
        k = h["key"]
        if not k:
            continue
        ch = h["chapter"]
        tag = f"[H{h['level']}|ch{ch}] {k}"
        out.append(tag)
    return out


def compare_sequences(a: list[str], b: list[str]) -> tuple[int, list[tuple[int, str, str]]]:
    diffs = []
    n = max(len(a), len(b))
    mism = 0
    for i in range(n):
        sa = a[i] if i < len(a) else "<MISSING>"
        sb = b[i] if i < len(b) else "<MISSING>"
        if sa != sb:
            mism += 1
            if len(diffs) < 35:
                diffs.append((i, sa, sb))
    return mism, diffs


def main() -> None:
    af = analyze(FINAL)
    ah = analyze(HUM)
    seq_f = heading_sequence(af["headings"])
    seq_h = heading_sequence(ah["headings"])
    mism, diffs = compare_sequences(seq_f, seq_h)

    print("=== Scope: blocks from Chapter 1 through before Chapter 11 ===")
    print("Final blocks:", af["block_count"], "| Humanized blocks:", ah["block_count"])
    print("Final tables:", af["tables"], "| Hum tables:", ah["tables"])
    print("Final figure-frame tables (1x1):", af["figure_frames"], "| Hum:", ah["figure_frames"])
    print("Final Figure Caption paras:", af["captions"], "| Hum:", ah["captions"])
    print()
    print("=== Heading sequence mismatches (by position):", mism, "===")
    for i, sa, sb in diffs:
        print(f"  [{i}]")
        print(f"    F: {sa}")
        print(f"    H: {sb}")
    print()
    print("=== Style distribution (top 12, body region) ===")
    for label, sc in ("Final", af["style_counts"]), ("Humanized", ah["style_counts"]):
        top = sorted(sc.items(), key=lambda x: -x[1])[:12]
        print(label + ":", top)


if __name__ == "__main__":
    main()
