"""
Detect near-duplicate sentences in Dissertation_Arka_Talukder.md using TF-IDF cosine similarity.

Usage:
    python scripts/detect_near_duplicate_sentences.py
"""
from __future__ import annotations

import re
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    md = Path("Dissertation_Arka_Talukder.md").read_text(encoding="utf-8")
    md = re.sub(r"```[\s\S]*?```", "", md)
    md = re.sub(r"^\|.*\|\s*$", "", md, flags=re.M)
    md = re.split(r"^## Chapter 11\s+–\s+References\s*$", md, flags=re.M)[0]

    sents = split_sentences(md)
    # Filter out very short sentences
    keep = [s for s in sents if len(s) >= 120]
    if not keep:
        print("No sentences long enough to compare.")
        return

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
    except Exception as e:
        raise SystemExit("scikit-learn is required for this script") from e

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    X = vec.fit_transform(keep)
    K = linear_kernel(X, X)

    threshold = 0.86
    pairs: list[tuple[float, int, int]] = []
    n = K.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(K[i, j])
            if sim >= threshold:
                pairs.append((sim, i, j))

    pairs.sort(reverse=True)
    print(f"Sentences compared: {n}")
    print(f"Near-duplicate pairs (sim >= {threshold}): {len(pairs)}")
    for sim, i, j in pairs[:40]:
        print("\n--- sim=%.3f ---" % sim)
        print("A:", keep[i])
        print("B:", keep[j])


if __name__ == "__main__":
    main()

