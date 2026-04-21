"""Audit OOXML for issues that make Word show repair dialog."""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


def main() -> None:
    import sys

    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent
        / "submission"
        / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
    )
    z = zipfile.ZipFile(path)
    styles = z.read("word/styles.xml").decode("utf-8")
    ids = re.findall(r'w:styleId="([^"]+)"', styles)
    c = Counter(ids)
    dups = [k for k, v in c.items() if v > 1]
    print("Duplicate w:styleId:", dups if dups else "none")

    # Parse with ET and check style elements
    root = ET.fromstring(z.read("word/styles.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    seen: dict[str, int] = {}
    bad = []
    for st in root.findall("w:style", ns):
        sid = st.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}styleId")
        if not sid:
            continue
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            bad.append(sid)
    print("Duplicate from ET:", bad if bad else "none")

    doc = z.read("word/document.xml").decode("utf-8")
    # Illegal XML 1.0 chars (Word may choke on some)
    bad_chars = []
    for i, ch in enumerate(doc):
        o = ord(ch)
        if ch in "\x00":
            bad_chars.append((i, o))
        elif 0 < o < 32 and ch not in "\t\n\r":
            bad_chars.append((i, o))
    print("Suspicious ctrl chars in document.xml:", len(bad_chars))

    # First bytes of saved XML (BOM / declaration)
    raw_doc = z.read("word/document.xml")
    print("document.xml starts with:", raw_doc[:80])
    if re.search(rb"xmlns:ns\d+", raw_doc):
        print("WARNING: generic ns0/ns1 prefixes — Word may show a repair dialog")
    elif raw_doc.startswith(b"<?xml") and b"<w:document" in raw_doc[:400]:
        print("document root uses w: prefix (OK for Word)")


if __name__ == "__main__":
    main()
