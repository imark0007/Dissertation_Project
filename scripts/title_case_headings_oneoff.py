"""
One-off Title-Case pass on '###' and '####' chapter sub-section headings.

Rules:
- Only touches lines that start with '### N.N ' or '#### N.N.N '.
- Leaves the heading number untouched.
- Title-cases the words after the number, but keeps short function words
  (a, an, and, as, at, but, by, for, in, of, on, or, the, to, vs., with)
  in lowercase except when they are the first word of the heading.
- Acronyms (CPU, GNN, GAT, GRU, FL, kNN, IoT, MSc, etc.) and code-style
  tokens are preserved as-is.
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "Dissertation_Arka_Talukder.md"

SMALL = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "of",
    "on", "or", "the", "to", "vs.", "vs", "with", "via", "per",
}

# Tokens that should keep their original casing (acronyms, code-y tokens).
KEEP = {
    "CPU", "GPU", "GNN", "GAT", "GRU", "FL", "kNN", "IoT", "IDS", "IDS,",
    "MSc", "RF", "MLP", "ROC-AUC", "F1", "API", "SIEM", "SOC", "XAI",
    "FedAvg", "PyTorch", "FastAPI", "Captum", "Flower", "Dirichlet",
    "CICIoT2023", "ECS", "JSON", "YAML", "URL", "HTTP", "HTTPS",
    "100%", "(Sub-Question", "(Sub-Question 1)", "(Sub-Question 2)",
    "(Sub-Question 3)", "(Sub-Question 2 and Deployment)",
    "(Stability of Design Choices)", "(Priority 1: Evidence)",
    "(Programme Format)", "(Author's Codebase)", "(Author\u2019s Codebase)",
    "(Conceptual)", "(Cyber Security Body of Knowledge)",
    "(Fifteen to Twenty Core Sources)",
}


def title_case_token(tok: str, is_first: bool) -> str:
    if tok in KEEP:
        return tok
    bare = tok.strip("().,:;\"'\u2018\u2019\u201c\u201d")
    if bare in KEEP:
        return tok  # leave punctuation around acronym alone
    low = tok.lower()
    if not is_first and low in SMALL:
        return low
    # Default: capitalise first letter, keep the rest as-is (preserves
    # mid-token caps like 'kNN' inside hyphens)
    if not tok:
        return tok
    return tok[0].upper() + tok[1:]


def title_case_phrase(phrase: str) -> str:
    out = []
    for i, tok in enumerate(phrase.split(" ")):
        out.append(title_case_token(tok, is_first=(i == 0)))
    return " ".join(out)


def transform_line(line: str) -> str:
    m = re.match(r"^(#{3,4})\s+(\d+(?:\.\d+){1,2})\s+(.+?)\s*$", line)
    if not m:
        return line
    hashes, number, rest = m.group(1), m.group(2), m.group(3)
    return f"{hashes} {number} {title_case_phrase(rest)}\n"


text = PATH.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)
new_lines = [transform_line(ln) for ln in lines]
new = "".join(new_lines)

if new == text:
    raise SystemExit("No heading changes made.")

PATH.write_text(new, encoding="utf-8")
print("Title-cased headings updated.")
