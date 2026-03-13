"""
Convert Dissertation_Arka_Talukder.md to Word for final submission.
Follows MSc Project Handbook: 1.5 line spacing, 11pt+ font.
Run: python scripts/dissertation_to_docx.py
Output: Arka_Talukder_Dissertation_Final.docx
"""
from pathlib import Path
import re

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_LINE_SPACING
    from docx.enum.style import WD_STYLE_TYPE
except ImportError:
    print("Install: pip install python-docx")
    raise

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "Dissertation_Arka_Talukder.md"
OUT_PATH = ROOT / "Arka_Talukder_Dissertation_Final.docx"
ASSETS = ROOT / "assets" / "figure1_pipeline.png"


def add_paragraph_with_style(doc, text, style_name="Normal"):
    p = doc.add_paragraph(text, style=style_name)
    return p


def main():
    content = MD_PATH.read_text(encoding="utf-8")
    doc = Document()

    # Set default font and line spacing
    style = doc.styles["Normal"]
    font = style.font
    font.size = Pt(11)
    font.name = "Times New Roman"
    paragraph_format = style.paragraph_format
    paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    # Process markdown
    lines = content.split("\n")
    i = 0
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i]

        # Skip front matter placeholder
        if "## Front Matter" in line and "complete before" in lines[i + 2] if i + 2 < len(lines) else False:
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue

        # Headers
        if line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("#### "):
            doc.add_heading(line[5:], level=3)

        # Horizontal rule - skip
        elif line.strip() == "---":
            pass

        # Table
        elif "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        else:
            if in_table and table_lines:
                # Parse and add table
                headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                rows = []
                for tl in table_lines[2:]:  # Skip header and separator
                    if "|" in tl:
                        cells = [c.strip() for c in tl.split("|")[1:-1]]
                        rows.append(cells)
                if headers and rows:
                    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
                    for j, h in enumerate(headers):
                        t.rows[0].cells[j].text = h
                    for ri, row in enumerate(rows):
                        for ci, cell in enumerate(row):
                            if ci < len(t.rows[ri + 1].cells):
                                t.rows[ri + 1].cells[ci].text = cell
                in_table = False
                table_lines = []

            # Image
            if "![" in line and ("](assets/" in line or "](../../assets/" in line or ".png)" in line):
                match = re.search(r'!\[([^\]]*)\]\(([^)]+)\)', line)
                if match and ASSETS.exists():
                    doc.add_picture(str(ASSETS), width=Inches(5.5))
                    p = doc.add_paragraph(match.group(1), style="Caption")
                else:
                    doc.add_paragraph(line)
            # Bold/italic
            elif line.strip():
                # Simple markdown cleanup
                text = line
                text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                text = re.sub(r"\*([^*]+)\*", r"\1", text)
                text = re.sub(r"`([^`]+)`", r"\1", text)
                if text.strip():
                    doc.add_paragraph(text)

        i += 1

    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print("Apply handbook formatting: 1.5 line spacing, 11pt font, add front sheet, declaration, library form.")


if __name__ == "__main__":
    main()
