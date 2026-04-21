"""
Keep dissertation Markdown and Word exports in sync (Final and Humanized).

After editing repo-root Markdown, run:

    python scripts/sync_dissertation_and_docx.py

This will:
  1. Final track — `Dissertation_Arka_Talukder.md` → `submission/Arka_Talukder_Dissertation_Final.docx`
  2. Humanized track — when `Dissertation_Arka_Talukder_Humanized.md` exists:
     → `submission/Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx`
  3. When `supervisor_package/01_Dissertation/` exists,
     copy each updated `.md` and `.docx` there (supervisor package stays aligned).

Options:
  --final-only       Only sync the Final Markdown / Word pair
  --humanized-only   Only sync the Humanized pair (requires Humanized .md)

Humanized-only shortcut:

    python scripts/sync_humanized_md_and_docx.py

Regenerating from Markdown overwrites prior Word-only edits in the target `.docx`;
prefer capturing prose in the matching `.md` when possible.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "submission"
MD_SRC = ROOT / "Dissertation_Arka_Talukder.md"
DOCX_OUT = SUBMISSION_DIR / "Arka_Talukder_Dissertation_Final.docx"
HUMANIZED_MD = ROOT / "Dissertation_Arka_Talukder_Humanized.md"
HUMANIZED_DOCX = SUBMISSION_DIR / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
PACKAGE_DIR = ROOT / "supervisor_package" / "01_Dissertation"
PACKAGE_MD = PACKAGE_DIR / "Dissertation_Arka_Talukder.md"
PACKAGE_DOCX = PACKAGE_DIR / "Arka_Talukder_Dissertation_Final.docx"
PACKAGE_HUMANIZED_MD = PACKAGE_DIR / "Dissertation_Arka_Talukder_Humanized.md"
PACKAGE_HUMANIZED_DOCX = PACKAGE_DIR / "Arka_Talukder_Dissertation_Final_Submission_Humanized_version.docx"
CONVERTER = ROOT / "scripts" / "dissertation_to_docx.py"


def _newer(a: Path, b: Path) -> Path:
    if not a.is_file():
        return b
    if not b.is_file():
        return a
    return a if a.stat().st_mtime >= b.stat().st_mtime else b


def _run_convert(md: Path, docx: Path) -> int:
    return subprocess.run(
        [sys.executable, str(CONVERTER), "--md", str(md), "--out", str(docx)],
        cwd=str(ROOT),
    ).returncode


def _sync_one_track(
    *,
    md_src: Path,
    docx_out: Path,
    package_md: Path,
    package_docx: Path,
    label: str,
    copy_package: bool,
) -> None:
    if not md_src.is_file():
        print(f"ERROR: missing {md_src}", file=sys.stderr)
        sys.exit(1)

    if copy_package and PACKAGE_DIR.is_dir():
        shutil.copy2(md_src, package_md)
        print(f"[{label}] Copied MD -> {package_md.relative_to(ROOT)}")
    elif copy_package:
        print(
            f"[{label}] Skip package MD (folder missing): {PACKAGE_DIR.relative_to(ROOT)}",
        )

    r = _run_convert(md_src, docx_out)
    if r != 0:
        sys.exit(r)

    if not docx_out.is_file() and not package_docx.is_file():
        print(
            f"ERROR: [{label}] converter did not produce {docx_out} (or package fallback)",
            file=sys.stderr,
        )
        sys.exit(1)

    built = _newer(docx_out, package_docx)

    if copy_package and PACKAGE_DIR.is_dir() and built.resolve() != package_docx.resolve():
        shutil.copy2(built, package_docx)
        print(f"[{label}] Copied DOCX -> {package_docx.relative_to(ROOT)}")

    print(f"OK [{label}]: {built.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Sync dissertation .md -> .docx (Final and/or Humanized)")
    p.add_argument(
        "--final-only",
        action="store_true",
        help="Only sync Dissertation_Arka_Talukder.md -> Final .docx",
    )
    p.add_argument(
        "--humanized-only",
        action="store_true",
        help="Only sync Humanized .md -> Humanized .docx",
    )
    p.add_argument(
        "--no-package",
        action="store_true",
        help="Do not copy into 01_Dissertation/",
    )
    args = p.parse_args()
    copy_pkg = not args.no_package

    if args.final_only and args.humanized_only:
        print("ERROR: choose at most one of --final-only / --humanized-only", file=sys.stderr)
        sys.exit(2)

    if not CONVERTER.is_file():
        print(f"ERROR: missing {CONVERTER}", file=sys.stderr)
        sys.exit(1)

    if args.humanized_only:
        _sync_one_track(
            md_src=HUMANIZED_MD,
            docx_out=HUMANIZED_DOCX,
            package_md=PACKAGE_HUMANIZED_MD,
            package_docx=PACKAGE_HUMANIZED_DOCX,
            label="humanized",
            copy_package=copy_pkg,
        )
        return

    if not args.final_only:
        _sync_one_track(
            md_src=MD_SRC,
            docx_out=DOCX_OUT,
            package_md=PACKAGE_MD,
            package_docx=PACKAGE_DOCX,
            label="final",
            copy_package=copy_pkg,
        )
        if HUMANIZED_MD.is_file():
            _sync_one_track(
                md_src=HUMANIZED_MD,
                docx_out=HUMANIZED_DOCX,
                package_md=PACKAGE_HUMANIZED_MD,
                package_docx=PACKAGE_HUMANIZED_DOCX,
                label="humanized",
                copy_package=copy_pkg,
            )
        else:
            print("Skip humanized track (no Dissertation_Arka_Talukder_Humanized.md)")
        return

    _sync_one_track(
        md_src=MD_SRC,
        docx_out=DOCX_OUT,
        package_md=PACKAGE_MD,
        package_docx=PACKAGE_DOCX,
        label="final",
        copy_package=copy_pkg,
    )


if __name__ == "__main__":
    main()
