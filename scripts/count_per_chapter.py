"""Count words per chapter (Ch 1..10) in Dissertation_Arka_Talukder.md."""
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "Dissertation_Arka_Talukder.md"
text = PATH.read_text(encoding="utf-8")

# Split into chapters by '## Chapter N' heading.
chapters = re.split(r"(?m)^## Chapter (\d+) ", text)
# chapters = [pre, '1', body1, '2', body2, ...]
def count_words(s: str) -> int:
    s = re.sub(r"```.*?```", " ", s, flags=re.DOTALL)        # drop code fences
    s = re.sub(r"!\[.*?\]\(.*?\)", " ", s)                   # drop image syntax
    s = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", s)                # link text only
    s = re.sub(r"[#*_`>|]", " ", s)                          # markdown punctuation
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s))

total = 0
for i in range(1, len(chapters), 2):
    num = chapters[i]
    body = chapters[i + 1]
    # Stop at next chapter (already handled by split) — body now ends at next '## ' or EOF
    body = re.split(r"(?m)^## (?:Chapter|Front|Acknowledgements|List|Table)", body, maxsplit=1)[0]
    if int(num) > 10:
        continue
    w = count_words(body)
    total += w
    print(f"Chapter {num}: {w:,} words")
print(f"---")
print(f"Total Ch 1-10: {total:,} words")
