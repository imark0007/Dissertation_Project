"""Final dissertation QA report for submission readiness."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "Dissertation_Arka_Talukder.md"
DOCX = ROOT / "Arka_Talukder_Dissertation_Final_Submission.docx"


def md_qa(text: str) -> list[str]:
    out: list[str] = []
    out.append(f"MD em-dash count: {text.count('—')}")

    # Chapter headers 1..13
    for n in range(1, 14):
        ok = re.search(rf"^## Chapter {n}\b", text, flags=re.M) is not None
        out.append(f"MD chapter {n} heading: {'OK' if ok else 'MISSING'}")

    # Figure numbering 1..24 and A1-1..A1-6
    missing_fig = [n for n in range(1, 25) if f"Figure {n}:" not in text]
    missing_a1 = [n for n in range(1, 7) if f"Figure A1-{n}" not in text]
    out.append(f"MD body figures 1..24: {'OK' if not missing_fig else 'MISSING ' + str(missing_fig)}")
    out.append(f"MD appendix figures A1-1..A1-6: {'OK' if not missing_a1 else 'MISSING ' + str(missing_a1)}")

    # Table numbering 1..7
    missing_tab = [n for n in range(1, 8) if f"Table {n}:" not in text and f"| {n} |" not in text]
    out.append(f"MD tables 1..7 presence: {'OK' if not missing_tab else 'CHECK ' + str(missing_tab)}")

    # Required constants
    required_literals = [
        "F1 = 100%",
        "ROC-AUC = 100%",
        "23 ms",
        "31 MB",
        "10 rounds",
        "3 clients",
        "alpha = 0.5",
        "128,002",
        "500,000 flows",
        "920 training sequences",
        "928 validation sequences",
        "934 test sequences",
    ]
    missing_literals = [x for x in required_literals if x not in text]
    out.append(
        "MD required key numbers: "
        + ("OK" if not missing_literals else "MISSING " + "; ".join(missing_literals))
    )

    # DOI check in Chapter 11
    if "## Chapter 11" in text and "## Chapter 12" in text:
        refs = text.split("## Chapter 11", 1)[1].split("## Chapter 12", 1)[0]
        entries = [ln.strip() for ln in refs.splitlines() if re.match(r"^[A-Z].*\(\d{4}\)", ln.strip())]
        non_doi = [e for e in entries if "Available at:" in e and "doi.org" not in e.lower()]
        out.append(f"MD references entries: {len(entries)}")
        out.append(f"MD non-DOI references: {len(non_doi)}")
    return out


def docx_qa(doc: Document) -> list[str]:
    out: list[str] = []
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full = "\n".join(paras)
    out.append(f"DOCX em-dash count: {full.count('—')}")

    required_front = [
        "DECLARATION OF ORIGINALITY",
        "Library Release Form",
        "Table of Contents",
        "List of Figures",
        "List of Tables",
    ]
    for x in required_front:
        out.append(f"DOCX contains '{x}': {'OK' if x in full else 'MISSING'}")

    # Chapter heading presence in docx body
    for n in range(1, 14):
        ok = any(re.match(rf"^Chapter {n}\b", p, flags=re.I) for p in paras)
        out.append(f"DOCX chapter {n} heading: {'OK' if ok else 'MISSING'}")

    # Simple scan for unresolved placeholders
    placeholders = [
        "[Insert",
        "UWS LOGO (insert official logo here)",
    ]
    for ph in placeholders:
        out.append(f"DOCX placeholder '{ph}': {'PRESENT' if ph in full else 'not found'}")

    # Ensure appendix headline exists
    out.append(f"DOCX Appendix D section: {'OK' if 'Appendix D' in full else 'CHECK'}")
    return out


def main() -> None:
    md_text = MD.read_text(encoding="utf-8")
    doc = Document(str(DOCX))

    print("=== FINAL SUBMISSION QA REPORT ===")
    print(f"Markdown source: {MD.name}")
    print(f"DOCX target: {DOCX.name}")
    print("\n[Markdown QA]")
    for ln in md_qa(md_text):
        print("-", ln)
    print("\n[DOCX QA]")
    for ln in docx_qa(doc):
        print("-", ln)
    print("\n=== END OF REPORT ===")


if __name__ == "__main__":
    main()
