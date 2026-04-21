"""
Normalize in-text Harvard citations to parenthetical form (Author et al. Year) with no
comma before the year — e.g. Wang et al. (2025) -> (Wang et al. 2025).

Only edits the body before '## Chapter 11 – References' so reference/bibliography lists stay
in UWS Harvard list format (Author, A. (Year) 'Title' ...).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SPLIT_MARKER = "## Chapter 11 – References"
# Surname token: Latin letters, hyphen, apostrophe; allow Miège-style accents common in file
SUR = r"[A-Z][A-Za-zÀ-ÿ'\-]*"
ETAL = SUR + r" et al\."
TWO = SUR + r" and " + SUR


def _transform_body(text: str) -> str:
    # 1) Drop Harvard comma before year: "Author et al., 2023" / "Author and Author, 2002"
    text = re.sub(rf"\b({ETAL}), (\d{{4}})\b", r"\1 \2", text)
    text = re.sub(rf"\b({TWO}), (\d{{4}})\b", r"\1 \2", text)

    # 2) Narrative with year in parentheses -> full parenthetical
    text = re.sub(rf"\b({ETAL}) \((\d{{4}})\)", r"(\1 \2)", text)
    text = re.sub(rf"\b({TWO}) \((\d{{4}})\)", r"(\1 \2)", text)

    # 3) Bare "Author et al. YYYY" / two-author not already wrapped in ( ... )
    def wrap_et_al(m: re.Match) -> str:
        return f"({m.group(1)} {m.group(2)})"

    text = re.sub(
        rf"(?<!\()(?<![\w/])({ETAL}) (\d{{4}})(?!\))",
        wrap_et_al,
        text,
    )

    def wrap_two(m: re.Match) -> str:
        return f"({m.group(1)} {m.group(2)})"

    text = re.sub(
        rf"(?<!\()(?<![\w/])({TWO}) (\d{{4}})(?!\))",
        wrap_two,
        text,
    )

    return text


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    md_path = root / "Dissertation_Arka_Talukder.md"
    if len(sys.argv) > 1:
        md_path = Path(sys.argv[1])

    full = md_path.read_text(encoding="utf-8")
    if SPLIT_MARKER not in full:
        print("Marker not found:", SPLIT_MARKER, file=sys.stderr)
        sys.exit(1)

    head, tail = full.split(SPLIT_MARKER, 1)
    new_head = _transform_body(head)
    out = new_head + SPLIT_MARKER + tail
    md_path.write_text(out, encoding="utf-8")
    print(f"Updated: {md_path}")


if __name__ == "__main__":
    main()
