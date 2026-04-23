"""One-off: copy Chapter 2 + Ch2 TOC from Arka_Talukder_Dissertation_Final_DRAFT.md into Dissertation_Arka_Talukder_Humanized.md; fix ENISA in Ch11/12."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAN = ROOT / "Arka_Talukder_Dissertation_Final_DRAFT.md"
HUM = ROOT / "Dissertation_Arka_Talukder_Humanized.md"


def main() -> None:
    can = CAN.read_text(encoding="utf-8")
    hum = HUM.read_text(encoding="utf-8")

    # --- Chapter 2 body (canonical) ---
    m2 = re.search(
        r"## Chapter 2 . Literature Review.*?(?=\n## Chapter 3 )",
        can,
        flags=re.DOTALL,
    )
    if not m2:
        raise SystemExit("Could not find Chapter 2 in canonical file")
    ch2 = m2.group(0).rstrip() + "\n"

    mh = re.search(
        r"## Chapter 2 . Literature Review.*?(?=\n## Chapter 3 )",
        hum,
        flags=re.DOTALL,
    )
    if not mh:
        raise SystemExit("Could not find Chapter 2 in humanized file")
    hum = hum[: mh.start()] + ch2 + hum[mh.end() :]

    # --- TOC: Chapter 2 block only (keep humanized List of Figures lines at top of TOC) ---
    mt_can = re.search(
        r"(\*\*Chapter 2 – Literature Review\*\*  \n)(.+?)(\n\*\*Chapter 3 – Project Management\*\*)",
        can,
        flags=re.DOTALL,
    )
    mt_hum = re.search(
        r"(\*\*Chapter 2 – Literature Review\*\*  \n)(.+?)(\n\*\*Chapter 3 – Project Management\*\*)",
        hum,
        flags=re.DOTALL,
    )
    if mt_can and mt_hum:
        hum = (
            hum[: mt_hum.start()]
            + mt_can.group(1)
            + mt_can.group(2)
            + mt_hum.group(3)
            + hum[mt_hum.end() :]
        )

    # --- ENISA: cited in Ch2; belong in Ch11 only (match canonical) ---
    enisa_line = (
        "ENISA (2017) *Baseline security recommendations for IoT in the context of critical information infrastructures*. "
        "Heraklion: European Union Agency for Network and Information Security. "
        "Available at: https://doi.org/10.2824/03228\n\n"
    )
    ch11_start = hum.find("## Chapter 11 – References")
    ch12_start = hum.find("## Chapter 12 – Bibliography")
    if ch11_start == -1 or ch12_start == -1:
        raise SystemExit("Missing Ch11 or Ch12")

    ch11 = hum[ch11_start:ch12_start]
    if "ENISA (2017)" not in ch11:
        marker = "https://doi.org/10.1109/secpri.2002.1004372"
        p = ch11.find(marker)
        if p == -1:
            raise SystemExit("Cuppens ref URL not found in Ch11")
        line_end = ch11.find("\n", p)
        if line_end == -1:
            line_end = len(ch11)
        else:
            line_end += 1
        ch11_new = ch11[:line_end] + enisa_line + ch11[line_end:]
        hum = hum[:ch11_start] + ch11_new + hum[ch12_start:]

    # Remove ENISA from Ch12 if present under Standards
    ch12_match = re.search(r"## Chapter 12 – Bibliography.*?(?=## Chapter 13)", hum, re.DOTALL)
    if ch12_match:
        ch12 = ch12_match.group(0)
        ch12_new = re.sub(
            r"\nENISA \(2017\)[^\n]*https://doi\.org/10\.2824/03228\n",
            "\n",
            ch12,
            count=1,
        )
        if ch12_new != ch12:
            hum = hum.replace(ch12, ch12_new, 1)

    HUM.write_text(hum, encoding="utf-8", newline="\n")
    print("Updated:", HUM)


if __name__ == "__main__":
    main()
