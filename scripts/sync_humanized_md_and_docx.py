"""
Keep Humanized dissertation Markdown and Word export in sync.

Source: repo-root `Dissertation_Arka_Talukder_Humanized.md`
Output: `submission/Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx`

After editing the Humanized Markdown, run:

    python scripts/sync_humanized_md_and_docx.py

This also copies `.md` + `.docx` into `supervisor_package/01_Dissertation/`
when that folder exists (same pattern as the Final sync).

Full pipeline (Final + Humanized when Humanized.md exists):

    python scripts/sync_dissertation_and_docx.py

Regenerating from Markdown overwrites prior Word-only formatting in the output file;
re-apply any Word-only fixes (or keep those edits in the Humanized `.md` instead).

To rebuild the Markdown from the current Humanized Word file (docx → md; keeps the
manual Table of Contents block and aligns body text to the .docx):

    python scripts/export_humanized_docx_to_md.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "submission"
HUMANIZED_MD = ROOT / "Dissertation_Arka_Talukder_Humanized.md"
HUMANIZED_DOCX = SUBMISSION_DIR / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
PACKAGE_DIR = ROOT / "supervisor_package" / "01_Dissertation"
PACKAGE_MD = PACKAGE_DIR / "Dissertation_Arka_Talukder_Humanized.md"
PACKAGE_DOCX = PACKAGE_DIR / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
CONVERTER = ROOT / "scripts" / "dissertation_to_docx.py"


def _newer(a: Path, b: Path) -> Path:
    if not a.is_file():
        return b
    if not b.is_file():
        return a
    return a if a.stat().st_mtime >= b.stat().st_mtime else b


def main() -> None:
    p = argparse.ArgumentParser(description="Sync Humanized dissertation .md -> .docx")
    p.add_argument(
        "--no-package",
        action="store_true",
        help="Do not copy into 01_Dissertation/ (only rebuild root Humanized .docx)",
    )
    args = p.parse_args()
    copy_pkg = not args.no_package

    if not HUMANIZED_MD.is_file():
        print(f"ERROR: missing {HUMANIZED_MD}", file=sys.stderr)
        sys.exit(1)
    if not CONVERTER.is_file():
        print(f"ERROR: missing {CONVERTER}", file=sys.stderr)
        sys.exit(1)

    if copy_pkg and PACKAGE_DIR.is_dir():
        shutil.copy2(HUMANIZED_MD, PACKAGE_MD)
        print(f"Copied MD -> {PACKAGE_MD.relative_to(ROOT)}")
    elif copy_pkg:
        print(f"Skip package MD (folder missing): {PACKAGE_DIR.relative_to(ROOT)}")

    r = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--md",
            str(HUMANIZED_MD),
            "--out",
            str(HUMANIZED_DOCX),
        ],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    if not HUMANIZED_DOCX.is_file() and not PACKAGE_DOCX.is_file():
        print(
            f"ERROR: converter did not produce {HUMANIZED_DOCX} (or package fallback)",
            file=sys.stderr,
        )
        sys.exit(1)

    built = _newer(HUMANIZED_DOCX, PACKAGE_DOCX)

    if copy_pkg and PACKAGE_DIR.is_dir() and built.resolve() != PACKAGE_DOCX.resolve():
        shutil.copy2(built, PACKAGE_DOCX)
        print(f"Copied DOCX -> {PACKAGE_DOCX.relative_to(ROOT)}")

    print(f"OK: {built.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
