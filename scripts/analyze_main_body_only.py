"""Structural and integrity audit: Chapters 1-10 only (main submission .docx)."""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[1]
PATH_MAIN = ROOT / "submission" / "Arka_Talukder_Dissertation_Final_Submission.docx"

# Chapter 11 reference surnames + year (canonical for in-text check)
REF_PATTERN = re.compile(
    r"^([A-Za-z\-\']+).*?\((\d{4})\)",
    re.M,
)


def load_paras(path: Path) -> list[str]:
    return [p.text.strip() for p in docx.Document(str(path)).paragraphs if p.text.strip()]


def chapter_indices(paras: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, t in enumerate(paras):
        m = re.match(r"^Chapter\s+(\d+)\s+[–-]", t)
        if m:
            out[int(m.group(1))] = i
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit dissertation main body (Ch 1-10) only.")
    ap.add_argument(
        "docx",
        nargs="?",
        type=Path,
        default=PATH_MAIN,
        help=f"Path to .docx (default: {PATH_MAIN.name})",
    )
    args = ap.parse_args()
    path: Path = Path(args.docx)
    if not path.is_file():
        print("Missing", path, file=sys.stderr)
        return 1

    paras = load_paras(path)
    idx = chapter_indices(paras)
    if 1 not in idx or 11 not in idx:
        print("Could not find Chapter 1 and/or Chapter 11", file=sys.stderr)
        return 1

    start, end = idx[1], idx[11]
    main_paras = paras[start:end]
    main_text = "\n".join(main_paras)

    # --- Section headings (only lines in main body that look like X.Y ...)
    sec_pat = re.compile(r"^(\d+(?:\.\d+)+)\s+")
    sections: list[tuple[str, str]] = []
    for t in main_paras:
        m = sec_pat.match(t)
        if m:
            sections.append((m.group(1), t[:110]))

    # Gap detection at depth 2 (X.Y)
    by_major: dict[str, list[tuple[int, ...]]] = collections.defaultdict(list)
    for num, _ in sections:
        parts = tuple(int(x) for x in num.split("."))
        if len(parts) >= 1:
            by_major[parts[0]].append(parts)

    gap_report: list[str] = []
    for major in sorted(by_major, key=int):
        level2 = sorted({p[:2] for p in by_major[major] if len(p) >= 2})
        for j in range(len(level2) - 1):
            a, b = level2[j], level2[j + 1]
            if a[0] == b[0] and b[1] - a[1] > 1:
                gap_report.append(
                    f"Chapter {major}: after §{a[0]}.{a[1]} next is §{b[0]}.{b[1]} "
                    f"(missing §{a[0]}.{a[1] + 1}…§{b[0]}.{b[1] - 1})"
                )

    # --- Figures / tables (caption-style lines in main body)
    fig_lines = [(i, t) for i, t in enumerate(main_paras) if re.match(r"^Figure\s+", t, re.I)]
    tbl_lines = [(i, t) for i, t in enumerate(main_paras) if re.match(r"^Table\s+\d+", t, re.I)]

    def fig_id(line: str) -> str | None:
        m = re.match(r"^Figure\s+(\S+)", line, re.I)
        return m.group(1).rstrip(":") if m else None

    def tbl_id(line: str) -> int | None:
        m = re.match(r"^Table\s+(\d+)", line, re.I)
        return int(m.group(1)) if m else None

    fig_nums = [fig_id(t) for _, t in fig_lines if fig_id(t)]
    tbl_nums = [tbl_id(t) for _, t in tbl_lines if tbl_id(t)]
    from collections import Counter

    fdup = [k for k, v in Counter(fig_nums).items() if v > 1]
    tdup = [k for k, v in Counter(tbl_nums).items() if v > 1]

    nums_int = [int(x) for x in fig_nums if str(x).isdigit()]
    fig_gaps: list[str] = []
    if nums_int:
        u = sorted(set(nums_int))
        for a, b in zip(u, u[1:]):
            if b - a > 1:
                fig_gaps.append(f"Figure numbers jump {a} → {b}")

    # --- Cross-references in main body vs full doc section/figure/table inventory
    all_paras_text = "\n".join(paras)
    existing_secs = {num for num, _ in sections}
    # also collect section numbers from whole doc for targets referenced from ch1-10
    all_sections = set()
    for t in paras:
        m = sec_pat.match(t)
        if m:
            all_sections.add(m.group(1))

    all_fig_caption_nums = set()
    all_tbl_caption_nums = set()
    for t in paras:
        m = re.match(r"^Figure\s+(\d+)\b", t, re.I)
        if m:
            all_fig_caption_nums.add(int(m.group(1)))
        m = re.match(r"^Table\s+(\d+)", t, re.I)
        if m:
            all_tbl_caption_nums.add(int(m.group(1)))

    xref_issues: list[str] = []
    for t in main_paras:
        for m in re.finditer(r"\bSection\s+(\d+(?:\.\d+)+)\b", t):
            num = m.group(1)
            if num not in all_sections:
                xref_issues.append(f"Section {num} → not found as heading in document")
        for m in re.finditer(r"\b(?:Figure|Fig\.)\s+(\d+)\b", t, re.I):
            n = int(m.group(1))
            if n not in all_fig_caption_nums:
                xref_issues.append(f"Figure {n} → no caption line 'Figure {n}:' in doc")
        for m in re.finditer(r"\bTable\s+(\d+)\b", t):
            n = int(m.group(1))
            if n not in all_tbl_caption_nums:
                xref_issues.append(f"Table {n} → no caption line 'Table {n}' in doc")

    # --- Chapter 11 references (full list from doc)
    ch11_start, ch12_start = idx[11], idx[12]
    ref_block = "\n".join(paras[ch11_start:ch12_start])
    ref_pairs: set[tuple[str, str]] = set()
    for line in paras[ch11_start + 2 : ch12_start]:
        if len(line) < 30:
            continue
        ym = re.search(r"\((\d{4})\)", line)
        if not ym:
            continue
        sur_m = re.match(r"^([A-Za-z\-\']+)", line)
        if sur_m:
            ref_pairs.add((sur_m.group(1), ym.group(1)))

    approved_surnames = {p[0] for p in ref_pairs}

    # In-text (main body): rough Harvard paren form
    cite_paren = re.findall(r"\(([A-Z][^)]{3,120}?\d{4}[a-z]?)\)", main_text)
    body_cite_flags: list[str] = []
    for c in cite_paren:
        if "Figure" in c or "Table" in c or "Section" in c:
            continue
        low = c.lower()
        if "february" in low or "april" in low:
            continue
        if "ciciot" in low and "2023" in c:
            body_cite_flags.append(f"Non-author paren cite: ({c[:80]}…)")
            continue
        year_m = re.search(r"(\d{4})", c)
        if not year_m:
            continue
        y = year_m.group(1)
        # surname first token
        sm = re.match(r"^([A-Za-z\-\']+)", c)
        if not sm:
            continue
        sur = sm.group(1)
        if sur == "e":
            continue
        ok = False
        for rs, ry in ref_pairs:
            if ry != y:
                continue
            if rs.lower() in low or sur.lower() == rs.lower()[: len(sur)]:
                ok = True
                break
        if "Miège" in c or "Cuppens" in c:
            ok = any(rs == "Cuppens" and ry == y for rs, ry in ref_pairs)
        if "Lundberg" in c or (sur == "Lee" and y == "2017"):
            ok = any(rs == "Lundberg" and ry == y for rs, ry in ref_pairs)
        if "Alabbadi" in c or "Bajaber" in c:
            ok = any(rs == "Alabbadi" and ry == y for rs, ry in ref_pairs)
        if not ok:
            body_cite_flags.append(f"Possibly not in Chapter 11: ({c[:90]}…)")

    # Refs never cited in main body (surname + year scan)
    uncited_refs: list[str] = []
    for line in paras[ch11_start + 2 : ch12_start]:
        if len(line) < 30:
            continue
        ym = re.search(r"\((\d{4})\)", line)
        if not ym:
            continue
        y = ym.group(1)
        sm = re.match(r"^([A-Za-z\-\']+)", line)
        if not sm:
            continue
        sur = sm.group(1)
        if re.search(rf"{re.escape(sur)}[^\n]{{0,40}}{y}", main_text, re.I):
            continue
        if sur == "Lundberg" and "Lee" in main_text and y == "2017":
            continue
        uncited_refs.append(f"{sur} ({y})")

    # --- Word counts per chapter 1-10
    wc: dict[int, int] = {}
    for n in range(1, 11):
        s, e = idx[n], idx[n + 1]
        wc[n] = len(" ".join(paras[s:e]).split())

    # --- Locked metric substrings (spot check in main body)
    locks = [
        "0.9986",
        "920",
        "934",
        "128,002",
        "3.07",
        "31 MB",
        "Dirichlet alpha = 0.5",
    ]
    missing_locks = [s for s in locks if s.replace(",", "") not in main_text.replace(",", "")]

    print("=" * 72)
    print("MAIN BODY AUDIT ONLY (Chapters 1-10)")
    print("File:", path.name)
    print("Excluded: front matter, declaration, Ch 11-13")
    print("=" * 72)
    print("\n## Word counts (Ch 1-10, paragraph text only)")
    print("Total:", sum(wc.values()))
    for n in range(1, 11):
        print(f"  Chapter {n}: {wc[n]}")
    print("\n## Section heading gaps (second-level, within numbered sections in main body)")
    if gap_report:
        for g in gap_report:
            print("  FLAG:", g)
    else:
        print("  None detected at X.Y level.")

    print("\n## All section headings in main body (in order) —", len(sections), "lines")
    for num, title in sections:
        print(f"  {num}  {title}")

    print("\n## Figure caption lines in main body —", len(fig_lines))
    if fdup:
        print("  Duplicate figure IDs (often narrative + formal caption):", fdup)
    if fig_gaps:
        for g in fig_gaps:
            print("  FLAG:", g)
    for _, t in fig_lines[:5]:
        print("  ", t[:100])
    if len(fig_lines) > 5:
        print("  …")
    for _, t in fig_lines[-3:]:
        print("  ", t[:100])

    print("\n## Table caption lines in main body —", len(tbl_lines))
    if tdup:
        print("  Duplicate table numbers:", tdup)
    for _, t in tbl_lines:
        print("  ", t[:100])

    print("\n## Cross-reference check (targets must exist anywhere in full .docx)")
    if xref_issues:
        for x in sorted(set(xref_issues)):
            print("  FLAG:", x)
    else:
        print("  No broken Section/Figure/Table references flagged by this scan.")

    print("\n## Citations: main body vs Chapter 11")
    print("  Chapter 11 reference entries (surname+year):", len(ref_pairs))
    if body_cite_flags:
        print("  FLAGS (manual review):")
        for f in body_cite_flags[:25]:
            print("   ", f)
        if len(body_cite_flags) > 25:
            print("   …", len(body_cite_flags) - 25, "more")
    else:
        print("  No obvious orphan paren-cites flagged.")
    if uncited_refs:
        print("  Possible Chapter 11 entries with no surname+year hit in Ch 1-10:")
        for u in uncited_refs:
            print("   ", u)
    else:
        print("  All Chapter 11 surnames appear referenced in main body (rough scan).")

    print("\n## Locked metrics spot-check (substring in Ch 1-10)")
    if missing_locks:
        print("  Missing in main body:", missing_locks)
    else:
        print("  All sample lock strings present in main body text.")

    print("\n## Pass 4 / interpretation note")
    print(
        "  Present:",
        "A note on interpretation" in main_text,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
