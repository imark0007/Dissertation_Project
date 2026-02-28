"""
Convert Interim_Report_Arka_Talukder.md to Word (B01821011_Interim_Report_Final.docx).
Run from project root: python scripts/md_to_docx.py
Requires: pip install python-docx
"""
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Pt
    from docx.enum.style import WD_STYLE_TYPE
except ImportError:
    print("Please install python-docx: pip install python-docx")
    raise

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "Interim_Report_Arka_Talukder.md"
OUT_PATH = PROJECT_ROOT / "B01821011_Interim_Report_Final.docx"


def add_run_with_format(p, text):
    """Add text to paragraph, handling **bold** and *italic*."""
    import re
    remaining = text
    while remaining:
        b = remaining.find("**")
        i = remaining.find("*") if remaining.find("*") >= 0 else len(remaining)
        if b >= 0 and (b < i or i < 0):
            # bold
            end = remaining.find("**", b + 2)
            if end >= 0:
                if b > 0:
                    p.add_run(remaining[:b])
                run = p.add_run(remaining[b + 2:end])
                run.bold = True
                remaining = remaining[end + 2:]
                continue
        if i >= 0 and not (i > 0 and remaining[i - 1] == "*"):
            end = remaining.find("*", i + 1)
            if end >= 0:
                if i > 0:
                    p.add_run(remaining[:i])
                run = p.add_run(remaining[i + 1:end])
                run.italic = True
                remaining = remaining[end + 1:]
                continue
        p.add_run(remaining)
        break


def parse_md_to_docx(md_path: Path, out_path: Path) -> None:
    doc = Document()
    for style in doc.styles:
        if style.type == WD_STYLE_TYPE.PARAGRAPH and style.name == "Normal":
            style.font.size = Pt(11)
            style.font.name = "Calibri"
            break

    content = md_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("# ") and not stripped.startswith("## "):
            doc.add_heading(stripped[2:].strip(), level=0)
            i += 1
            continue

        if stripped == "## References":
            doc.add_heading("References", level=1)
            i += 1
            while i < len(lines):
                ref_line = lines[i].strip()
                if ref_line:
                    doc.add_paragraph(ref_line)
                i += 1
            break

        if stripped.startswith("## "):
            doc.add_heading(stripped[3:].strip(), level=1)
            i += 1
            continue

        if stripped.startswith("### "):
            doc.add_heading(stripped[4:].strip(), level=2)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # Equation block: line starts with 4 spaces
        if line.startswith("    ") and stripped:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(24)
            run = p.add_run(stripped)
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            i += 1
            continue

        # Normal paragraph
        p = doc.add_paragraph()
        add_run_with_format(p, stripped)
        i += 1

    doc.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parse_md_to_docx(MD_PATH, OUT_PATH)
