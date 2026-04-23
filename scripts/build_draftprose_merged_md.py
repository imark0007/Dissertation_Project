"""
Build Dissertation_Arka_Talukder_DraftProse_Merged.md:
- Canonical Arka_Talukder_Dissertation_Final_DRAFT.md for all chapters except
- Acknowledgements + Section 1.1–1.2 from the previous humanized Word
  (Arka_Talukder_Dissertation_Final_DRAFT_old.docx) — stored inline below
  to avoid a runtime docx dependency in CI.

Regenerate Word (or use --docx on this script):
  python scripts/dissertation_to_docx.py --md thesis_artifacts/Dissertation_Arka_Talukder_DraftProse_Merged.md \\
      --out thesis_artifacts/02_Draft_Prose_Merged.docx
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import thesis_artifact_paths as tap

ROOT = Path(__file__).resolve().parent.parent

# From Arka_Talukder_Dissertation_Final_DRAFT_old.docx; moderator line uses "who" (not "as she")
ACK = """
I would like to thank my supervisor, Dr. Raja Ujjan, who provided technical support, feedback on design and evaluation, and support during the project. I also want to thank my moderator, Muhsin Hassanu, who reviewed the interim work and helped to refine the final report.

I also owe a debt of gratitude to Dr. Daune West who helped me and gave me academic support during the submission period. I would like to thank School and programme staff in terms of module materials and guidance on submissions, and the MSc Project co-ordinator in terms of administrative communication regarding the milestones and ethics.

Lastly, I would like to thank friends and family that tolerate me during intensive writing and experiment runs.
""".strip()

S1_1 = """This chapter presents the project background, IoT flow telemetry, SOC alert volume and the practical limitation at the edge node of a CPU. It describes the rationale behind the inclusion of dynamic graph learning, federated training, and explainable alerts in the same pipeline, followed by the research aim, sub-questions, scope, and structure of the chapters to ensure that subsequent chapters in the work are focused on method, implementation, and evidence."""

S1_2_PARAS = [
    "The Internet of Things (IoT) has grown at a very rapid pace in residential, workplace, campus and factory locations. There are now smart plugs, cameras, sensors and controllers everywhere. Most of these devices are small, low priced and limited and in most cases, security is not given much consideration during design. This has left weak credentials, patching, and misconfiguration as a common occurrence in actual deployments.",
    "This implies that there is a high exposure to attack. A compromised IoT device may be used in botnet activities, service denial, credential theft or subsequent lateral movement into the broader infrastructure. These are not uncommon situations in security reports. Due to this, constant surveillance of network behaviour is not a choice, particularly where there are mixed old and new devices.",
    "Software-defined techniques (such as SDN and software-defined IoT) enable switch and router operators to monitor switch flow statistics and router flow statistics centrally. Intrusion detection can be achieved with flow summaries, without full packet capture, often enough, reducing the storage pressure and alleviating some of the privacy concerns. The remaining question is to accurately analyse such flows, to explain decisions in a way that can be used by analysts, and in a manner which can be done on hardware which is realistic at the edge (which is often CPU-only).",
    "The networks are monitored, alerts investigated, and incident response coordinated by Security Operations Centres (SOCs) typically in connection with SIEM platforms that consolidate logs and events derived by flow. Another real-life challenge is the problem of alert fatigue: large numbers of alerts and a high number of false positives decrease the time that analysts have to devote to real incidents. Alerts which are simply a label with no information as to why the model fired take even longer to triage. This necessitates accurate and explainable detectors hence staff can have confidence in outputs and can take action as time runs out.",
    "Random Forests or feed-forward networks can be used to apply traditional models to tabular flow features, but traffic is relational: flows are like neighbours in feature space, have common endpoints in the presence of identifiers, or occur in bursts of traffic, which are meaningful as a whole. Graph models consider flows to be nodes and the adjacent related flows are connected using edges (in this case, kNN links in feature space since the public CICIoT2023 release does not support device-level topology). Dynamic GNNs build upon this concept by training on how the snapshots of the graph change within short periods of time. Individually, IoT data can also be spread out across locations where it is legally or politically challenging to pool raw flows; federated learning trains a single shared model, but maintains raw data locally. Most IoT gateways cannot also take on a GPU in each segment and a CPU pipeline able to emit SIEM friendly JSON is more of a deployable result than a GPU only laboratory result.",
]
S1_2 = "\n\n".join(S1_2_PARAS)

NOTE = (
    "\n*Merged draft: **Acknowledgements** and **Sections 1.1–1.2** follow the previous humanized "
    "`Arka_Talukder_Dissertation_Final_DRAFT_old.docx` (minor edit: moderator line uses *who*). "
    "**Sections 1.3 onward and all other chapters** follow the current canonical "
    "`Arka_Talukder_Dissertation_Final_DRAFT.md` (methods, table numbering, data handling, and Harvard references). "
    "Export with `python scripts/dissertation_to_docx.py`.*\n"
)


def main() -> Path:
    canon = (ROOT / "Arka_Talukder_Dissertation_Final_DRAFT.md").read_text(encoding="utf-8")

    canon = re.sub(
        r"## Acknowledgements\n\n.*?(?=\n## List of Abbreviations\n)",
        "## Acknowledgements\n\n" + ACK + "\n\n",
        canon,
        count=1,
        flags=re.DOTALL,
    )

    canon = re.sub(
        r"(### 1.1 Chapter Overview\n\n).*(?=\n### 1.2 Background and Motivation\n)",
        r"\1" + S1_1 + "\n",
        canon,
        count=1,
        flags=re.DOTALL,
    )

    canon = re.sub(
        r"(### 1.2 Background and Motivation\n\n).*(?=\n### 1.3 Research Aim and Questions\n)",
        r"\1" + S1_2 + "\n",
        canon,
        count=1,
        flags=re.DOTALL,
    )

    canon = canon.replace(
        "**Supervisor: Dr. Raja Ujjan**\n\n---",
        "**Supervisor: Dr. Raja Ujjan**" + NOTE + "\n---",
        1,
    )

    tap.THESIS_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = tap.DRAFT_PROSE_MD
    out.write_text(canon, encoding="utf-8")
    print("OK:", out)
    return out


def _run_docx(md: Path) -> int:
    conv = ROOT / "scripts" / "dissertation_to_docx.py"
    if not conv.is_file():
        print(f"ERROR: missing {conv}", file=sys.stderr)
        return 1
    r = subprocess.run(
        [sys.executable, str(conv), "--md", str(md), "--out", str(tap.DRAFT_PROSE_DOCX)],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        return r.returncode
    sub = ROOT / "submission" / "Arka_Talukder_Dissertation_Final_DraftProse_Merged.docx"
    sub.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tap.DRAFT_PROSE_DOCX, sub)
    print("OK:", tap.DRAFT_PROSE_DOCX)
    print("Copied ->", sub)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__ or "Build DraftProse merge Markdown")
    ap.add_argument(
        "--docx",
        action="store_true",
        help="Also run dissertation_to_docx.py to thesis_artifacts/02 and copy to submission/",
    )
    args = ap.parse_args()
    md = main()
    if args.docx and md is not None:
        sys.exit(_run_docx(md))
