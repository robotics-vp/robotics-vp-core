#!/usr/bin/env python3
"""Build first Economic WM scaffold artifacts from entry-preflight evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.economic_wm_entry_preflight import (  # noqa: E402
    run_economic_wm_entry_preflight,
)
from src.economics.economic_wm_entry import (  # noqa: E402
    load_economic_wm_entry_preflight_report,
)
from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMScaffoldReport,
    build_economic_wm_scaffold_report,
    save_economic_wm_scaffold_report,
)


def _write_markdown(path: Path, report: EconomicWMScaffoldReport) -> None:
    payload = report.to_dict()
    state = payload["economic_state"]
    envelope = payload["allocation_envelope"]
    lines = [
        "# Economic WM Scaffold",
        "",
        f"- Scaffold ID: `{payload['scaffold_id']}`",
        f"- State ID: `{state['state_id']}`",
        f"- Allocation envelope ID: `{envelope['envelope_id']}`",
        f"- Readiness class: `{state['regime']}`",
        f"- Ready for scaffold: `{str(payload['ready_for_scaffold']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Authority class: `{envelope['authority_class']}`",
        f"- Reward math mutation: `{str(envelope['reward_math_mutation']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Resource reservoirs",
    ]
    for key, value in state["resource_reservoirs"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Flow fields"])
    for key, value in state["flow_fields"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Dissipation fields"])
    for key, value in state["dissipation_fields"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Denied actions"])
    for action in envelope["denied_actions"]:
        lines.append(f"- `{action}`")
    lines.extend(["", "## Training blockers"])
    blockers = payload["training_blockers"]
    lines.append("- none" if not blockers else "")
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    path.write_text("\n".join(lines).replace("\n\n\n", "\n\n") + "\n", encoding="utf-8")


def run_build_economic_wm_scaffold(
    *,
    output_dir: str | Path,
    entry_preflight_report_path: Optional[str | Path] = None,
    run_preflight_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if entry_preflight_report_path is None:
        expected_path = (
            output_root / "entry_preflight" / "economic_wm_entry_preflight_report.json"
        )
        if run_preflight_if_missing:
            preflight_payload = run_economic_wm_entry_preflight(
                output_dir=output_root / "entry_preflight"
            )
            preflight_path = Path(preflight_payload["report_path"])
        else:
            preflight_path = expected_path
        preflight_report = load_economic_wm_entry_preflight_report(preflight_path)
    else:
        preflight_report = load_economic_wm_entry_preflight_report(
            entry_preflight_report_path
        )

    report = build_economic_wm_scaffold_report(preflight_report)
    scaffold_path = output_root / "economic_wm_scaffold_report_v1.json"
    state_path = output_root / "economic_state_v1.json"
    envelope_path = output_root / "allocation_envelope_v1.json"
    markdown_path = output_root / "economic_wm_scaffold_report_v1.md"
    save_economic_wm_scaffold_report(scaffold_path, report)
    state_path.write_text(
        json.dumps(report.economic_state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    envelope_path.write_text(
        json.dumps(report.allocation_envelope.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "scaffold_report_path": str(scaffold_path),
        "economic_state_path": str(state_path),
        "allocation_envelope_path": str(envelope_path),
        "markdown_path": str(markdown_path),
    }
    scaffold_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, report)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_scaffold",
        help="Directory for scaffold artifacts.",
    )
    parser.add_argument(
        "--entry-preflight-report",
        default=None,
        help="Optional existing economic_wm_entry_preflight_report.json.",
    )
    parser.add_argument(
        "--no-run-preflight",
        action="store_true",
        help="Do not run the entry preflight when no report is provided.",
    )
    args = parser.parse_args()
    payload = run_build_economic_wm_scaffold(
        output_dir=args.output_dir,
        entry_preflight_report_path=args.entry_preflight_report,
        run_preflight_if_missing=not args.no_run_preflight,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["entry_preflight"].get("ready_for_scaffold") else 1


if __name__ == "__main__":
    raise SystemExit(main())
