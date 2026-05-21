#!/usr/bin/env python3
"""Run the Economic WM entry preflight over local Stage-1 readiness evidence."""

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

from scripts.economic_world_model.sweep_stage1_bridge_readiness import (  # noqa: E402
    run_stage1_bridge_readiness_sweep,
)
from src.economics.economic_wm_entry import (  # noqa: E402
    EconomicWMEntryPreflightReport,
    evaluate_economic_wm_entry_preflight,
    save_economic_wm_entry_preflight_report,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_markdown(path: Path, report: EconomicWMEntryPreflightReport) -> None:
    payload = report.to_dict()
    lines = [
        "# Economic WM Entry Preflight",
        "",
        f"- Report ID: `{payload['report_id']}`",
        f"- Readiness class: `{payload['readiness_class']}`",
        f"- Ready for scaffold: `{str(payload['ready_for_scaffold']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        "",
        "## Counts",
    ]
    for key, value in payload["counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Required Surfaces"])
    for key, value in payload["required_surfaces"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Scaffold Blockers"])
    blockers = payload["scaffold_blockers"]
    lines.append("- none" if not blockers else "")
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Training Blockers"])
    training_blockers = payload["training_blockers"]
    lines.append("- none" if not training_blockers else "")
    for blocker in training_blockers:
        lines.append(f"- `{blocker}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_economic_wm_entry_preflight(
    *,
    output_dir: str | Path,
    stage1_sweep_report_path: Optional[str | Path] = None,
    run_sweep_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if stage1_sweep_report_path is not None:
        sweep_path = Path(stage1_sweep_report_path)
        sweep_report = _load_json(sweep_path)
    elif run_sweep_if_missing:
        sweep_report = run_stage1_bridge_readiness_sweep(
            output_dir=output_root / "stage1_bridge_readiness_sweep",
            quiet=True,
        )
        sweep_path = Path(sweep_report["report_path"])
    else:
        sweep_path = (
            output_root
            / "stage1_bridge_readiness_sweep"
            / "stage1_bridge_readiness_report.json"
        )
        sweep_report = _load_json(sweep_path)

    report = evaluate_economic_wm_entry_preflight(
        stage1_sweep_report=sweep_report,
        artifact_refs={
            "stage1_sweep_report_path": str(sweep_path),
            "stage1_manifest_path": sweep_report.get("manifest_path", ""),
            "bridge_manifest": sweep_report.get("bridge_manifest", {}),
        },
        metadata={"source": "economic_wm_entry_preflight_script"},
    )
    json_path = output_root / "economic_wm_entry_preflight_report.json"
    md_path = output_root / "economic_wm_entry_preflight_report.md"
    save_economic_wm_entry_preflight_report(json_path, report)
    _write_markdown(md_path, report)
    payload = report.to_dict()
    payload["report_path"] = str(json_path)
    payload["markdown_path"] = str(md_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_entry_preflight",
        help="Directory for preflight outputs.",
    )
    parser.add_argument(
        "--stage1-sweep-report",
        default=None,
        help="Optional existing stage1_bridge_readiness_report.json.",
    )
    parser.add_argument(
        "--no-run-sweep",
        action="store_true",
        help="Do not run the Stage-1 sweep when no report is provided.",
    )
    args = parser.parse_args()
    payload = run_economic_wm_entry_preflight(
        output_dir=args.output_dir,
        stage1_sweep_report_path=args.stage1_sweep_report,
        run_sweep_if_missing=not args.no_run_sweep,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ready_for_scaffold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
