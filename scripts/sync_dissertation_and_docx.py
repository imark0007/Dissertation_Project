"""
Keep the final dissertation Markdown and Word export in sync.

After editing repo-root `Dissertation_Arka_Talukder.md`, run:

    python scripts/sync_dissertation_and_docx.py

This will:
  1. Copy that Markdown into `B01821011_Final_Report_Package_for_Supervisor/01_Dissertation/`
     when that folder exists (supervisor package stays aligned).
  2. Run `dissertation_to_docx.py` to rebuild `Arka_Talukder_Dissertation_Final.docx` at repo root.
  3. Copy the new `.docx` into the same `01_Dissertation/` folder when present.

Use this instead of running `dissertation_to_docx.py` alone whenever you change the thesis MD.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_SRC = ROOT / "Dissertation_Arka_Talukder.md"
DOCX_OUT = ROOT / "Arka_Talukder_Dissertation_Final.docx"
PACKAGE_DIR = ROOT / "B01821011_Final_Report_Package_for_Supervisor" / "01_Dissertation"
PACKAGE_MD = PACKAGE_DIR / "Dissertation_Arka_Talukder.md"
PACKAGE_DOCX = PACKAGE_DIR / "Arka_Talukder_Dissertation_Final.docx"
CONVERTER = ROOT / "scripts" / "dissertation_to_docx.py"


def main() -> None:
    if not MD_SRC.is_file():
        print(f"ERROR: missing {MD_SRC}", file=sys.stderr)
        sys.exit(1)
    if not CONVERTER.is_file():
        print(f"ERROR: missing {CONVERTER}", file=sys.stderr)
        sys.exit(1)

    if PACKAGE_DIR.is_dir():
        shutil.copy2(MD_SRC, PACKAGE_MD)
        print(f"Copied MD -> {PACKAGE_MD.relative_to(ROOT)}")
    else:
        print(f"Skip package MD (folder missing): {PACKAGE_DIR.relative_to(ROOT)}")

    r = subprocess.run([sys.executable, str(CONVERTER)], cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)

    if not DOCX_OUT.is_file() and not PACKAGE_DOCX.is_file():
        print(f"ERROR: converter did not produce {DOCX_OUT} (or package fallback)", file=sys.stderr)
        sys.exit(1)

    def _newer(a: Path, b: Path) -> Path:
        if not a.is_file():
            return b
        if not b.is_file():
            return a
        return a if a.stat().st_mtime >= b.stat().st_mtime else b

    built = _newer(DOCX_OUT, PACKAGE_DOCX)

    if PACKAGE_DIR.is_dir() and built.resolve() != PACKAGE_DOCX.resolve():
        shutil.copy2(built, PACKAGE_DOCX)
        print(f"Copied DOCX -> {PACKAGE_DOCX.relative_to(ROOT)}")

    print(f"OK: {built.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
