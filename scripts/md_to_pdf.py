"""Convert Project_Overview_Reference.md to PDF using fpdf2."""
from pathlib import Path
import re

try:
    from fpdf import FPDF
except ImportError:
    print("Install: pip install fpdf2")
    raise

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "Project_Overview_Reference.md"
OUT_PATH = ROOT / "Project_Overview_Reference.pdf"


def clean_for_pdf(text):
    """Remove markdown syntax for plain text PDF."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)
    return text


def main():
    content = MD_PATH.read_text(encoding="utf-8")
    content = clean_for_pdf(content)
    content = content.replace("\u2014", "-").replace("\u2013", "-")
    content = content.replace("\u2018", "'").replace("\u2019", "'")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in content.split("\n"):
        line = line.rstrip()
        if not line:
            pdf.ln(4)
            continue
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        for chunk in [safe[i:i+100] for i in range(0, len(safe), 100)]:
            if chunk.strip():
                pdf.multi_cell(190, 5, chunk)
    pdf.output(str(OUT_PATH))
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
