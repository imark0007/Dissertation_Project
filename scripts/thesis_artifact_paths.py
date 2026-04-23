"""
Paths for thesis_artifacts/ — one folder for the main deliverable exports.

Edit sources remain at repo root; sync / build scripts copy or regenerate here.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS_ARTIFACTS = ROOT / "thesis_artifacts"
# Single canonical thesis Markdown at repo root (edit this; sync copies elsewhere)
CANONICAL_THESIS_MD = ROOT / "Arka_Talukder_Dissertation_Final_DRAFT.md"
# Regenerated from Dissertation_Arka_Talukder_Humanized.md (parallel track to the DRAFT)
HUMANIZED_DOCX = THESIS_ARTIFACTS / "01_Humanized_Updated.docx"
# Snapshot copy of canonical thesis (refreshed when you run sync_dissertation_and_docx.py)
FINAL_MD = THESIS_ARTIFACTS / "Arka_Talukder_Dissertation_Final_DRAFT.md"
# Optional draft-prose merge experiment (see build_draftprose_merged_md.py)
DRAFT_PROSE_MD = THESIS_ARTIFACTS / "Dissertation_Arka_Talukder_DraftProse_Merged.md"
DRAFT_PROSE_DOCX = THESIS_ARTIFACTS / "02_Draft_Prose_Merged.docx"
# Older / superseded thesis files
ARCHIVE_PREVIOUS = ROOT / "archive" / "previous_thesis_files"
