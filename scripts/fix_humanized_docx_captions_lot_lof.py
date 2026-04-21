"""
Post-process Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx:

1) Add Word TOC fields for List of Figures / List of Tables that collect entries by
   dedicated styles (FigureCaption / TableCaption), so static caption text works without SEQ.
2) Retarget existing Caption paragraphs: figures -> FigureCaption; table lead-ins -> TableCaption.
3) Light submission styling alignment (Normal / headings / Caption family / page margins)
   to match scripts/dissertation_to_docx.py defaults.

Does not change body wording — only styles, TOC field instructions, and new style definitions.

Usage:
  python scripts/fix_humanized_docx_captions_lot_lof.py [--in PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {"w": W_NS}

# ElementTree.tostring assigns ns0, ns1, … unless prefixes are registered; Word rejects that OOXML.
_DOCUMENT_XML_PREFIXES: list[tuple[str, str]] = [
    ("wpc", "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"),
    ("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006"),
    ("o", "urn:schemas-microsoft-com:office:office"),
    ("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("m", "http://schemas.openxmlformats.org/officeDocument/2006/math"),
    ("v", "urn:schemas-microsoft-com:vml"),
    ("wp14", "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"),
    ("wp", "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"),
    ("w10", "urn:schemas-microsoft-com:office:word"),
    ("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("w14", "http://schemas.microsoft.com/office/word/2010/wordml"),
    ("w15", "http://schemas.microsoft.com/office/word/2012/wordml"),
    ("wpg", "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"),
    ("wpi", "http://schemas.microsoft.com/office/word/2010/wordprocessingInk"),
    ("wne", "http://schemas.microsoft.com/office/word/2006/wordml"),
    ("wps", "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"),
    ("a", "http://schemas.openxmlformats.org/drawingml/2006/main"),
    ("pic", "http://schemas.openxmlformats.org/drawingml/2006/picture"),
    ("a14", "http://schemas.microsoft.com/office/drawing/2010/main"),
]

_STYLES_XML_PREFIXES: list[tuple[str, str]] = [
    ("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006"),
    ("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"),
    ("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main"),
    ("w14", "http://schemas.microsoft.com/office/word/2010/wordml"),
    ("w15", "http://schemas.microsoft.com/office/word/2012/wordml"),
]


def _qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _register_prefixes(pairs: list[tuple[str, str]]) -> None:
    for prefix, uri in pairs:
        ET.register_namespace(prefix, uri)


def _write_xml_document(root: ET.Element) -> bytes:
    _register_prefixes(_DOCUMENT_XML_PREFIXES)
    data = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    if data.startswith(b"<?xml"):
        _end = data.find(b"?>")
        rest = data[_end + 2 :].lstrip(b"\n\r")
        data = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n' + rest
    if re.search(rb"xmlns:ns\d+", data):
        raise RuntimeError(
            "document.xml would use ns0-style prefixes; add missing xmlns to _DOCUMENT_XML_PREFIXES"
        )
    return data


def _write_xml_styles(root: ET.Element) -> bytes:
    _register_prefixes(_STYLES_XML_PREFIXES)
    data = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    if data.startswith(b"<?xml"):
        _end = data.find(b"?>")
        rest = data[_end + 2 :].lstrip(b"\n\r")
        data = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n' + rest
    if re.search(rb"xmlns:ns\d+", data):
        raise RuntimeError(
            "styles.xml would use ns0-style prefixes; add missing xmlns to _STYLES_XML_PREFIXES"
        )
    return data


def _para_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NSMAP))


def _set_para_style(p: ET.Element, style_id: str) -> None:
    p_pr = p.find("w:pPr", NSMAP)
    if p_pr is None:
        p_pr = ET.Element(_qn("pPr"))
        p.insert(0, p_pr)
    ps = p_pr.find("w:pStyle", NSMAP)
    if ps is None:
        ps = ET.Element(_qn("pStyle"))
        p_pr.insert(0, ps)
    ps.set(_qn("val"), style_id)


def _get_para_style(p: ET.Element) -> str | None:
    p_pr = p.find("w:pPr", NSMAP)
    if p_pr is None:
        return None
    ps = p_pr.find("w:pStyle", NSMAP)
    if ps is None:
        return None
    return ps.get(_qn("val"))


def _instr_texts_in_p(p: ET.Element) -> list[str]:
    out = []
    for r in p.findall(".//w:instrText", NSMAP):
        if r.text:
            out.append(r.text.strip())
    return out


def _find_toc_field_paragraphs(body: ET.Element) -> dict[str, ET.Element]:
    """Map 'main' | 'lof' | 'lot' -> paragraph containing the TOC instrText."""
    found: dict[str, ET.Element | None] = {"main": None, "lof": None, "lot": None}
    for p in body.findall("w:p", NSMAP):
        texts = " ".join(_instr_texts_in_p(p))
        if "TOC" not in texts:
            continue
        if '\\c "Figure"' in texts or "\\c Figure" in texts:
            found["lof"] = p
        elif '\\c "Table"' in texts or "\\c Table" in texts:
            found["lot"] = p
        elif '\\o "' in texts and "1-3" in texts:
            if found["main"] is None:
                found["main"] = p
    return {k: v for k, v in found.items() if v is not None}  # type: ignore


def _replace_toc_instruction(p: ET.Element, new_instruction: str) -> None:
    for r in p.findall("w:r", NSMAP):
        for ins in list(r.findall("w:instrText", NSMAP)):
            if ins.text and "TOC" in ins.text:
                ins.text = f" {new_instruction} "


def _dedupe_main_toc(body: ET.Element) -> int:
    """Remove duplicate consecutive main TOC field paragraphs (same \\o 1-3). Kept: first only."""
    removed = 0
    kids = list(body)
    seen_main = False
    to_drop: list[ET.Element] = []
    for el in kids:
        if el.tag != _qn("p"):
            continue
        texts = " ".join(_instr_texts_in_p(el))
        if "TOC" in texts and '\\o "' in texts and "1-3" in texts:
            if seen_main:
                to_drop.append(el)
            else:
                seen_main = True
    for el in to_drop:
        body.remove(el)
        removed += 1
    return removed


def _add_caption_styles(styles_root: ET.Element) -> None:
    """Insert FigureCaption and TableCaption based on existing Caption style if present."""

    def _find_style_by_id(sid: str) -> ET.Element | None:
        for st in styles_root.findall("w:style", NSMAP):
            if st.get(_qn("styleId")) == sid:
                return st
        return None

    insert_at = max(
        (i for i, ch in enumerate(styles_root) if ch.tag == _qn("style")),
        default=-1,
    ) + 1

    base = _find_style_by_id("Caption")
    caption_based_on = "Caption" if base is not None else "Normal"
    for new_id, new_name in (
        ("FigureCaption", "Figure Caption"),
        ("TableCaption", "Table Caption"),
    ):
        if _find_style_by_id(new_id):
            continue
        if base is not None:
            st = deepcopy(base)
        else:
            st = ET.Element(_qn("style"))
            st.set(_qn("type"), "paragraph")
            st.set(_qn("customStyle"), "1")
            rpr = ET.SubElement(st, _qn("rPr"))
            ET.SubElement(rpr, _qn("rFonts")).set(_qn("ascii"), "Times New Roman")
            ET.SubElement(rpr, _qn("sz")).set(_qn("val"), "20")  # 10pt
            ET.SubElement(rpr, _qn("szCs")).set(_qn("val"), "20")
            ET.SubElement(rpr, _qn("i"))
        st.set(_qn("styleId"), new_id)
        name_el = st.find("w:name", NSMAP)
        if name_el is None:
            name_el = ET.SubElement(st, _qn("name"))
        name_el.set(_qn("val"), new_name)
        based_on = st.find("w:basedOn", NSMAP)
        if based_on is None:
            based_on = ET.SubElement(st, _qn("basedOn"))
        based_on.set(_qn("val"), caption_based_on)
        uip = st.find("w:uiPriority", NSMAP)
        if uip is not None:
            st.remove(uip)
        ppr = st.find("w:pPr", NSMAP)
        if ppr is None:
            ppr = ET.SubElement(st, _qn("pPr"))
        if ppr.find("w:keepNext", NSMAP) is None:
            ET.SubElement(ppr, _qn("keepNext"))
        styles_root.insert(insert_at, st)
        insert_at += 1


def _patch_caption_family_fonts(styles_root: ET.Element) -> None:
    """Ensure Caption + Figure/Table caption use Times New Roman 10pt (submission sample alignment)."""

    def _patch_one(st: ET.Element) -> None:
        rpr = st.find("w:rPr", NSMAP)
        if rpr is None:
            rpr = ET.SubElement(st, _qn("rPr"))
        rf = rpr.find("w:rFonts", NSMAP)
        if rf is None:
            rf = ET.SubElement(rpr, _qn("rFonts"))
        rf.set(_qn("ascii"), "Times New Roman")
        rf.set(_qn("hAnsi"), "Times New Roman")
        for tag, val in (("sz", "20"), ("szCs", "20")):
            el = rpr.find(f"w:{tag}", NSMAP)
            if el is None:
                el = ET.SubElement(rpr, _qn(tag))
            el.set(_qn("val"), val)

    for sid in ("Caption", "FigureCaption", "TableCaption"):
        for st in styles_root.findall("w:style", NSMAP):
            if st.get(_qn("styleId")) == sid:
                _patch_one(st)
                break


def _patch_normal_paragraph_layout(styles_root: ET.Element) -> None:
    """Body text: justified, 1.5 line spacing (handbook / sample alignment)."""
    for st in styles_root.findall("w:style", NSMAP):
        if st.get(_qn("styleId")) != "Normal":
            continue
        ppr = st.find("w:pPr", NSMAP)
        if ppr is None:
            ppr = ET.SubElement(st, _qn("pPr"))
        jc = ppr.find("w:jc", NSMAP)
        if jc is None:
            jc = ET.SubElement(ppr, _qn("jc"))
        jc.set(_qn("val"), "both")
        sp = ppr.find("w:spacing", NSMAP)
        if sp is None:
            sp = ET.SubElement(ppr, _qn("spacing"))
        sp.set(_qn("line"), "360")
        sp.set(_qn("lineRule"), "auto")
        break


def _tbl_size(tbl: ET.Element) -> tuple[int, int]:
    rows = tbl.findall("w:tr", NSMAP)
    nrows = len(rows)
    ncols = len(rows[0].findall("w:tc", NSMAP)) if rows else 0
    return nrows, ncols


def _patch_document_xml(root: ET.Element) -> tuple[int, int, int]:
    body = root.find("w:body", NSMAP)
    assert body is not None

    removed_dup = _dedupe_main_toc(body)

    toc_map = _find_toc_field_paragraphs(body)
    if "lof" in toc_map:
        # \\t uses the paragraph style *name* (w:name), not the styleId.
        _replace_toc_instruction(toc_map["lof"], 'TOC \\h \\z \\t "Figure Caption,1"')
    if "lot" in toc_map:
        _replace_toc_instruction(toc_map["lot"], 'TOC \\h \\z \\t "Table Caption,1"')

    fig_re = re.compile(r"^Figure\s+(\d+|A1-\d+)\s*[:\-–—]\s*", re.I)
    table_re = re.compile(r"^Table\s+\d+\s*:", re.I)

    fig_fixed = 0
    tbl_fixed = 0

    for p in body.findall("w:p", NSMAP):
        sid = _get_para_style(p)
        txt = _para_text(p).strip()
        if not txt:
            continue
        if sid == "Caption" and fig_re.match(txt):
            _set_para_style(p, "FigureCaption")
            fig_fixed += 1

    kids = list(body)
    for i, el in enumerate(kids):
        if el.tag != _qn("p"):
            continue
        txt = _para_text(el).strip()
        if not txt or not table_re.match(txt):
            continue
        nxt = kids[i + 1] if i + 1 < len(kids) else None
        if nxt is None or nxt.tag != _qn("tbl"):
            continue
        nrows, ncols = _tbl_size(nxt)
        if (nrows, ncols) == (1, 1):
            continue
        sid = _get_para_style(el)
        if sid not in (None, "Normal", "BodyText", "Caption"):
            continue
        _set_para_style(el, "TableCaption")
        tbl_fixed += 1

    return fig_fixed, tbl_fixed, removed_dup


def _patch_settings_margins(doc_root: ET.Element) -> None:
    """Set sectPr margins on first section (binding-friendly), if sectPr present."""
    body = doc_root.find("w:body", NSMAP)
    if body is None:
        return
    sect = body.find("w:sectPr", NSMAP)
    if sect is None:
        return
    # twips: 1 inch = 1440. Left 1.25 in = 1800, others 1 in = 1440
    pg_sz = sect.find("w:pgSz", NSMAP)
    pg_mar = sect.find("w:pgMar", NSMAP)
    if pg_mar is None:
        pg_mar = ET.SubElement(sect, _qn("pgMar"))
    pg_mar.set(_qn("top"), "1440")
    pg_mar.set(_qn("bottom"), "1440")
    pg_mar.set(_qn("right"), "1440")
    pg_mar.set(_qn("left"), "1800")
    pg_mar.set(_qn("header"), "708")
    pg_mar.set(_qn("footer"), "708")
    pg_mar.set(_qn("gutter"), "0")


def _patch_styles_defaults(styles_root: ET.Element) -> None:
    """Normal / headings: Times New Roman, spacing similar to dissertation_to_docx.py."""
    for sid, sz_half_pts, bold, italic in (
        ("Normal", 24, None, False),  # 12pt
        ("Heading1", 28, True, False),  # 14pt
        ("Heading2", 24, True, False),  # 12pt bold
        ("Heading3", 24, False, True),  # 12pt italic
        ("Heading4", 22, False, True),  # 11pt italic (sample-style sub-head)
    ):
        for st in styles_root.findall("w:style", NSMAP):
            if st.get(_qn("styleId")) != sid:
                continue
            rpr = st.find("w:rPr", NSMAP)
            if rpr is None:
                rpr = ET.SubElement(st, _qn("rPr"))
            rf = rpr.find("w:rFonts", NSMAP)
            if rf is None:
                rf = ET.SubElement(rpr, _qn("rFonts"))
            rf.set(_qn("ascii"), "Times New Roman")
            rf.set(_qn("hAnsi"), "Times New Roman")
            sz = rpr.find("w:sz", NSMAP)
            if sz is None:
                sz = ET.SubElement(rpr, _qn("sz"))
            sz.set(_qn("val"), str(sz_half_pts))
            sz_cs = rpr.find("w:szCs", NSMAP)
            if sz_cs is None:
                sz_cs = ET.SubElement(rpr, _qn("szCs"))
            sz_cs.set(_qn("val"), str(sz_half_pts))
            if bold is True:
                if rpr.find("w:b", NSMAP) is None:
                    ET.SubElement(rpr, _qn("b"))
                if rpr.find("w:bCs", NSMAP) is None:
                    ET.SubElement(rpr, _qn("bCs"))
            elif bold is False:
                for tag in ("b", "bCs"):
                    el = rpr.find(f"w:{tag}", NSMAP)
                    if el is not None:
                        rpr.remove(el)
            if italic:
                if rpr.find("w:i", NSMAP) is None:
                    ET.SubElement(rpr, _qn("i"))
            else:
                for tag in ("i", "iCs"):
                    el = rpr.find(f"w:{tag}", NSMAP)
                    if el is not None:
                        rpr.remove(el)
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "submission"
        / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Defaults to overwriting --in",
    )
    args = ap.parse_args()
    src = args.in_path.resolve()
    dst = (args.out_path or src).resolve()

    if dst == src:
        backup = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, backup)
        print(f"Backup: {backup}")

    with zipfile.ZipFile(src, "r") as zin:
        parts = {n: zin.read(n) for n in zin.namelist()}

    styles = _load_xml_from_bytes(parts["word/styles.xml"])
    _add_caption_styles(styles)
    _patch_caption_family_fonts(styles)
    _patch_styles_defaults(styles)
    _patch_normal_paragraph_layout(styles)
    parts["word/styles.xml"] = _write_xml_styles(styles)

    doc = _load_xml_from_bytes(parts["word/document.xml"])
    ff, tt, rd = _patch_document_xml(doc)
    _patch_settings_margins(doc)
    parts["word/document.xml"] = _write_xml_document(doc)

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in parts.items():
            zout.writestr(name, data)

    print(f"Saved: {dst}")
    print(f"Figure captions -> FigureCaption: {ff}")
    print(f"Table lead-ins -> TableCaption: {tt}")
    print(f"Duplicate main TOC fields removed: {rd}")
    print("In Word: place cursor in each TOC/LOF/LOT, press F9, choose Update entire field.")


def _load_xml_from_bytes(data: bytes) -> ET.Element:
    return ET.fromstring(data)


if __name__ == "__main__":
    main()
