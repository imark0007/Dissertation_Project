"""Repair Word OOXML: styles.xml root must declare w14/w15 if mc:Ignorable lists them."""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OLD = (
    '<w:styles xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" mc:Ignorable="w14 w15">'
)
NEW = (
    '<w:styles xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
    'mc:Ignorable="w14 w15">'
)


def main() -> int:
    src = ROOT / "submission" / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if not src.is_file():
        print("Missing", src, file=sys.stderr)
        return 1

    backup = src.with_suffix(".docx.pre_styles_fix_backup")
    shutil.copy2(src, backup)

    out = src.with_suffix(".docx._rebuild")
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(src, "r") as zin:
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "word/styles.xml":
                    text = data.decode("utf-8")
                    if OLD not in text:
                        print(
                            "Expected broken root tag not found (already fixed or different file).",
                            file=sys.stderr,
                        )
                        return 1
                    text = text.replace(OLD, NEW, 1)
                    data = text.encode("utf-8")
                zout.writestr(info, data)

    out.replace(src)
    print("Patched:", src.name)
    print("Backup:", backup.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
