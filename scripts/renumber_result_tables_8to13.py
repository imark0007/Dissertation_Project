"""Renumber Ch8 result tables: old Table 2-8 -> 7-13 (frees 2-6 for method chapters)."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "Arka_Talukder_Dissertation_Final_DRAFT.md"
s = p.read_text(encoding="utf-8")
for old, new in [(8, 13), (7, 12), (6, 11), (5, 10), (4, 9), (3, 8), (2, 7)]:
    s = s.replace("**Table %d:**" % old, "**Table %d:**" % new)
    s = s.replace("**Table %d**" % old, "**Table %d**" % new)
# Fix any dash ranges in prose
s = s.replace("Tables 2–8", "Tables 7–13")
p.write_text(s, encoding="utf-8", newline="\n")
print("OK")
