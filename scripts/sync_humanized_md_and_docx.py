"""
Keep Humanized dissertation Markdown and Word export in sync.

Source: repo-root `Dissertation_Arka_Talukder_Humanized.md`

Primary output (paired Word export; keeps repo root to sources only):

    thesis_artifacts/01_Humanized_Updated.docx

A copy is also written for submission / handback naming:

    submission/Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx

After editing the Humanized Markdown, run:

    python scripts/sync_humanized_md_and_docx.py

This also copies `.md` + both `.docx` targets into `supervisor_package/01_Dissertation/`
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import thesis_artifact_paths as tap

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "submission"
HUMANIZED_MD = ROOT / "Dissertation_Arka_Talukder_Humanized.md"
# Humanized Word: thesis_artifacts/ (see thesis_artifact_paths.py)
HUMANIZED_DOCX_SIBLING = tap.HUMANIZED_DOCX
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
    tap.THESIS_ARTIFACTS.mkdir(parents=True, exist_ok=True)

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
            str(HUMANIZED_DOCX_SIBLING),
        ],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    if not HUMANIZED_DOCX_SIBLING.is_file():
        print(
            f"ERROR: converter did not produce {HUMANIZED_DOCX_SIBLING}",
            file=sys.stderr,
        )
        sys.exit(1)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HUMANIZED_DOCX_SIBLING, HUMANIZED_DOCX)
    print(
        f"Paired: {HUMANIZED_MD.name} + {HUMANIZED_DOCX_SIBLING.relative_to(ROOT)}",
    )
    print(f"Copied  -> {HUMANIZED_DOCX.relative_to(ROOT)}")

    built = _newer(HUMANIZED_DOCX, PACKAGE_DOCX)

    if copy_pkg and PACKAGE_DIR.is_dir() and built.resolve() != PACKAGE_DOCX.resolve():
        shutil.copy2(built, PACKAGE_DOCX)
        # Supervisor bundle: keep the “together” name next to the .md for clarity
        if PACKAGE_MD.is_file():
            pkg_sibling = PACKAGE_MD.with_suffix(".docx")
            shutil.copy2(HUMANIZED_DOCX_SIBLING, pkg_sibling)
        print(f"Copied DOCX -> {PACKAGE_DOCX.relative_to(ROOT)}")

    print(f"OK: {built.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
