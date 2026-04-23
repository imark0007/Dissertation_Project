"""Final dissertation QA report for submission readiness."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "Arka_Talukder_Dissertation_Final_DRAFT.md"
DOCX = ROOT / "submission" / "B01821011_Arka_Talukder_Dissertation_Final.docx"


def md_qa(text: str) -> list[str]:
    out: list[str] = []
    out.append(f"MD em-dash count: {text.count('—')}")

    # Chapter headers 1..13
    for n in range(1, 14):
        ok = re.search(rf"^## Chapter {n}\b", text, flags=re.M) is not None
        out.append(f"MD chapter {n} heading: {'OK' if ok else 'MISSING'}")

    # Figure numbering 1..24 and appendix code figures (A1-1..A1-14 via paths or explicit labels)
    missing_fig = [n for n in range(1, 25) if f"Figure {n}:" not in text]
    a1_hits = text.count("results/figures/appendix1/fig_a1_")
    has_a1 = a1_hits == 14
    missing_a1 = [] if has_a1 else [f"expected 14 appendix1 images, found {a1_hits}"]
    out.append(f"MD body figures 1..24: {'OK' if not missing_fig else 'MISSING ' + str(missing_fig)}")
    out.append(f"MD appendix A1 code images (A1-1..A1-14 paths): {'OK' if not missing_a1 else 'MISSING ' + str(missing_a1)}")

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

    front_checks = [
        ("DECLARATION OF ORIGINALITY", r"declaration\s+of\s+originality"),
        ("Library Release Form", r"(library\s+release\s+form|dissertation\s+library\s+form)"),
        ("Table of Contents", r"table\s+of\s+contents"),
        ("List of Figures", r"list\s+of\s+figures"),
        ("List of Tables", r"list\s+of\s+tables"),
    ]
    full_l = full.lower()
    for label, pat in front_checks:
        ok = re.search(pat, full_l) is not None
        out.append(f"DOCX contains '{label}': {'OK' if ok else 'MISSING'}")

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
