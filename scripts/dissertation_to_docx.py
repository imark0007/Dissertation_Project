"""
Convert Dissertation_Arka_Talukder.md to Word for final submission.
Follows MSc Project Handbook: 1.5 line spacing, 11pt+ font, page numbers.
Embeds appendices (process docs, project spec) if available.
Run: python scripts/dissertation_to_docx.py
Output: Arka_Talukder_Dissertation_Final.docx
"""
from pathlib import Path
import re

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_LINE_SPACING
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
except ImportError:
    print("Install: pip install python-docx")
    raise

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "Dissertation_Arka_Talukder.md"
OUT_PATH = ROOT / "Arka_Talukder_Dissertation_Final.docx"

# Figure paths (relative to ROOT)
FIGURE_PATHS = {
    "assets/figure1_pipeline.png": ROOT / "assets" / "figure1_pipeline.png",
    "assets/literature_positioning.png": ROOT / "assets" / "literature_positioning.png",
    "assets/literature_ids_taxonomy.png": ROOT / "assets" / "literature_ids_taxonomy.png",
    "assets/literature_dynamic_gnn_concept.png": ROOT / "assets" / "literature_dynamic_gnn_concept.png",
    "assets/literature_fedavg_flow.png": ROOT / "assets" / "literature_fedavg_flow.png",
    "assets/literature_explainability.png": ROOT / "assets" / "literature_explainability.png",
    "results/figures/cm_gnn.png": ROOT / "results" / "figures" / "cm_gnn.png",
    "results/figures/roc_gnn.png": ROOT / "results" / "figures" / "roc_gnn.png",
    "results/figures/fl_convergence.png": ROOT / "results" / "figures" / "fl_convergence.png",
    "results/figures/model_comparison_bar.png": ROOT / "results" / "figures" / "model_comparison_bar.png",
    "results/figures/model_comparison.png": ROOT / "results" / "figures" / "model_comparison.png",
    "results/figures/sensitivity.png": ROOT / "results" / "figures" / "sensitivity.png",
    "results/figures/ablation_bar.png": ROOT / "results" / "figures" / "ablation_bar.png",
    "results/figures/cm_rf.png": ROOT / "results" / "figures" / "cm_rf.png",
    "results/figures/cm_mlp.png": ROOT / "results" / "figures" / "cm_mlp.png",
    "results/figures/roc_rf.png": ROOT / "results" / "figures" / "roc_rf.png",
    "results/figures/roc_mlp.png": ROOT / "results" / "figures" / "roc_mlp.png",
}

# Appendix paths
PROCESS_DOC = ROOT / "archive" / "process_attendance" / "Arka_Talukder_Process_Documentation_B01821011.docx"
ATTENDANCE_DOC = ROOT / "archive" / "process_attendance" / "Arka Talukder_Attendance_Jan-Feb_B01821011.docx"
_SPEC_NAME = "Arka-B01821011_MSc Cyber Security Project specification_form_2025-26.docx"
_PROJECT_SPEC_CANDIDATES = [
    ROOT / _SPEC_NAME,
    ROOT / "B01821011_Final_Report_Package_for_Supervisor" / "05_Appendix_documents" / _SPEC_NAME,
    ROOT / "docs" / "reference" / _SPEC_NAME,
]


def _resolve_project_spec():
    for p in _PROJECT_SPEC_CANDIDATES:
        if p.exists():
            return p
    return _PROJECT_SPEC_CANDIDATES[0]


PROJECT_SPEC = _resolve_project_spec()


def add_page_number_footer(section):
    """Add page number to section footer using XML."""
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = 1  # CENTER
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def add_placeholder_page(doc, title, instructions):
    """Add a placeholder page for Moodle forms."""
    doc.add_heading(title, level=1)
    doc.add_paragraph(instructions)
    doc.add_paragraph("[Insert downloaded form from Moodle here]")
    doc.add_page_break()


def resolve_image_path(md_path):
    """Resolve image path from markdown to filesystem."""
    for pattern, full_path in FIGURE_PATHS.items():
        if pattern in md_path and full_path.exists():
            return full_path
    # Fallback: try assets or results/figures
    if "assets/" in md_path:
        p = ROOT / md_path.replace("assets/", "assets/")
        if p.exists():
            return p
    if "results/figures/" in md_path:
        name = md_path.split("/")[-1]
        p = ROOT / "results" / "figures" / name
        if p.exists():
            return p
    return None


def process_markdown(doc, content):
    """Parse markdown and add to document."""
    lines = content.split("\n")
    i = 0
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i]

        # Skip front matter placeholder
        if "## Front Matter" in line:
            i += 1
            while i < len(lines) and not lines[i].startswith("## ") and not (lines[i].startswith("# ") and "Abstract" not in lines[i]):
                i += 1
            if i < len(lines) and "Table of Contents" in lines[i]:
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
                headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                rows = []
                for tl in table_lines[2:]:
                    if "|" in tl:
                        cells = [c.strip() for c in tl.split("|")[1:-1]]
                        rows.append(cells)
                if headers and rows:
                    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
                    for j, h in enumerate(headers):
                        t.rows[0].cells[j].text = h
                    # Estimated page numbers for Table of Figures and Tables
                    fig_pages = {1: 28, 2: 42, 3: 43, 4: 45, 5: 46, 6: 47, 7: 48, 8: 49, 9: 50}
                    table_pages = {1: 42, 2: 45, 3: 46}
                    for ri, row in enumerate(rows):
                        for ci, cell in enumerate(row):
                            if ci < len(t.rows[ri + 1].cells):
                                val = cell
                                if (val.strip() in ("—", "–", "-") or val == "—") and ci == len(headers) - 1 and "Page" in str(headers):
                                    if "Figure" in str(headers):
                                        val = str(fig_pages.get(ri + 1, "—"))
                                    elif "Table" in str(headers):
                                        val = str(table_pages.get(ri + 1, "—"))
                                t.rows[ri + 1].cells[ci].text = val
                in_table = False
                table_lines = []

            # Image
            if "![" in line and "]" in line and "(" in line:
                match = re.search(r'!\[([^\]]*)\]\(([^)]+)\)', line)
                if match:
                    img_path = resolve_image_path(match.group(2))
                    if img_path and img_path.exists():
                        doc.add_picture(str(img_path), width=Inches(5.5))
                        doc.add_paragraph(match.group(1), style="Caption")
                    else:
                        doc.add_paragraph(f"[Figure: {match.group(1)} — image not found at {match.group(2)}]")
            elif line.strip() and not line.strip().startswith("*Note:"):
                text = line
                text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
                text = re.sub(r"\*([^*]+)\*", r"\1", text)
                text = re.sub(r"`([^`]+)`", r"\1", text)
                if text.strip():
                    doc.add_paragraph(text)

        i += 1


def append_document(master_doc, other_path):
    """Append another docx to master. Uses docxcompose if available."""
    try:
        from docxcompose.composer import Composer
        composer = Composer(master_doc)
        other = Document(str(other_path))
        composer.append(other)
        return True
    except ImportError:
        try:
            other = Document(str(other_path))
            for element in other.element.body:
                master_doc.element.body.append(element)
            return True
        except Exception as e:
            print(f"Warning: Could not append {other_path}: {e}")
            return False
    except Exception as e:
        print(f"Warning: Could not append {other_path}: {e}")
        return False


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

    # Heading styles (sans-serif per handbook; #### in MD uses Heading 4)
    for h in ["Heading 1", "Heading 2", "Heading 3", "Heading 4"]:
        if h in doc.styles:
            doc.styles[h].font.size = Pt(12)
            doc.styles[h].font.name = "Arial"

    # Placeholder pages for Moodle forms (user replaces with downloaded forms)
    doc.add_heading("Front Sheet", level=1)
    doc.add_paragraph("Download from Moodle: Final Submission of Project Report / MSc Project - Front sheet for final report")
    doc.add_paragraph("[Insert front sheet here]")
    doc.add_page_break()

    doc.add_heading("Declaration of Originality", level=1)
    doc.add_paragraph("Download from Moodle: MSc Project - declaration of originality form (signed)")
    doc.add_paragraph("[Insert signed declaration here]")
    doc.add_page_break()

    doc.add_heading("Library Release Form", level=1)
    doc.add_paragraph("Download from Moodle: MSc Project - Library Release Form (signed)")
    doc.add_paragraph("[Insert signed library form here]")
    doc.add_page_break()

    # Main content
    process_markdown(doc, content)

    # Page numbers in footer (all sections)
    for section in doc.sections:
        add_page_number_footer(section)

    # Append appendices if they exist
    if PROCESS_DOC.exists():
        doc.add_page_break()
        doc.add_heading("Appendix A: Project Process Documentation", level=1)
        append_document(doc, PROCESS_DOC)

    if ATTENDANCE_DOC.exists():
        doc.add_page_break()
        doc.add_heading("Appendix B: Attendance Log", level=1)
        append_document(doc, ATTENDANCE_DOC)

    if PROJECT_SPEC.exists():
        doc.add_page_break()
        doc.add_heading("Appendix C: Project Specification", level=1)
        append_document(doc, PROJECT_SPEC)

    doc.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    if not PROJECT_SPEC.exists():
        print(
            "WARNING: Agreed specification .docx not found. Embed manually from:",
            [str(p.relative_to(ROOT)) for p in _PROJECT_SPEC_CANDIDATES],
        )
    print("Next steps:")
    print("  1. Replace placeholder pages with downloaded forms from Moodle")
    print("  2. Add page numbers to Table of Figures/Tables manually if needed")
    print("  3. Verify 1.5 line spacing and 11pt font")
    print("  4. Submit to Turnitin and email to module co-ordinator")


if __name__ == "__main__":
    main()
