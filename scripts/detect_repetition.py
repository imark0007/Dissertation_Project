"""
Detect repeated sentences/paragraphs in Dissertation_Arka_Talukder.md.

This is a heuristic tool to find obvious duplication that harms readability.

Usage:
    python scripts/detect_repetition.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'")
    return s.lower()


def split_sentences(text: str) -> list[str]:
    # Simple splitter: break on period/question/exclamation followed by space/newline.
    # Keep it conservative to avoid over-splitting abbreviations.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    md = Path("Dissertation_Arka_Talukder.md").read_text(encoding="utf-8")

    # Remove code fences, tables, and reference list to focus on narrative repetition
    md_wo_code = re.sub(r"```[\s\S]*?```", "", md)
    md_wo_tables = re.sub(r"^\|.*\|\s*$", "", md_wo_code, flags=re.M)
    # Drop Chapter 11 references block
    md_wo_refs = re.split(r"^## Chapter 11\s+–\s+References\s*$", md_wo_tables, flags=re.M)[0]

    # Paragraph repetition
    paras = [p.strip() for p in re.split(r"\n\s*\n", md_wo_refs) if p.strip()]
    para_map: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(paras):
        n = normalize(p)
        # Skip very short paras
        if len(n) < 120:
            continue
        para_map[n].append(i)

    repeated_paras = {k: v for k, v in para_map.items() if len(v) >= 2}

    # Sentence repetition
    sentences = split_sentences(md_wo_refs)
    sent_map: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sentences):
        n = normalize(s)
        if len(n) < 90:
            continue
        sent_map[n].append(i)
    repeated_sents = {k: v for k, v in sent_map.items() if len(v) >= 2}

    print(f"Paragraphs scanned: {len(paras)}")
    print(f"Repeated paragraphs (>=120 chars): {len(repeated_paras)}")
    for k, idxs in sorted(repeated_paras.items(), key=lambda kv: (-len(kv[1]), -len(kv[0])))[:25]:
        print("\n--- Repeated paragraph (occurrences: %d) ---" % len(idxs))
        print(k[:500] + ("..." if len(k) > 500 else ""))

    print("\nSentences scanned:", len(sentences))
    print("Repeated sentences (>=90 chars):", len(repeated_sents))
    for k, idxs in sorted(repeated_sents.items(), key=lambda kv: (-len(kv[1]), -len(kv[0])))[:40]:
        print("\n--- Repeated sentence (occurrences: %d) ---" % len(idxs))
        print(k)


if __name__ == "__main__":
    main()

