#!/usr/bin/env python3
"""Run the Economic WM local shadow outcome loop."""

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

from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMShadowOutcomeLoopReport,
    build_economic_wm_shadow_outcome_loop_from_paths,
)


def _write_markdown(path: Path, report: EconomicWMShadowOutcomeLoopReport) -> None:
    payload = report.to_dict()
    lines = [
        "# Economic WM Shadow Outcome Loop",
        "",
        f"- Report ID: `{payload['report_id']}`",
        f"- Shadow execution report ID: `{payload['shadow_execution_report_id']}`",
        f"- Supervision manifest ID: `{payload['supervision_manifest_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Outcome receipts: `{payload['outcome_receipt_count']}`",
        f"- Completed comparisons: `{payload['completed_comparison_count']}`",
        f"- Local structural loop closed: `{str(payload['local_structural_loop_closed']).lower()}`",
        f"- Hardware executed: `{str(payload['hardware_executed']).lower()}`",
        f"- Provider executed: `{str(payload['provider_executed']).lower()}`",
        f"- Live policy control: `{str(payload['live_policy_control']).lower()}`",
        f"- Reward math mutation: `{str(payload['reward_math_mutation']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in payload["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This loop joins advisory shadow work orders to local structural supervision receipts. It does not execute hardware, providers, live policy, GPU training, promotion, or reward-math mutation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_shadow_execution(
    *,
    output_root: Path,
    shadow_execution_report_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    path = Path(
        shadow_execution_report_path
        or "artifacts/economic_world_model/economic_wm_shadow_execution/economic_wm_shadow_execution_report_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.run_economic_wm_shadow_execution import (
        run_economic_wm_shadow_execution,
    )

    payload = run_economic_wm_shadow_execution(
        output_dir=output_root / "shadow_execution"
    )
    return Path(payload["artifact_refs"]["report_path"])


def _resolve_supervision(
    *,
    output_root: Path,
    supervision_manifest_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    path = Path(
        supervision_manifest_path
        or "artifacts/economic_world_model/economic_wm_supervision_substrate/economic_wm_supervision_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.prepare_economic_wm_supervision_substrate import (
        run_prepare_economic_wm_supervision_substrate,
    )

    payload = run_prepare_economic_wm_supervision_substrate(
        output_dir=output_root / "supervision_substrate"
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def run_economic_wm_shadow_outcome_loop(
    *,
    output_dir: str | Path,
    shadow_execution_report_path: Optional[str | Path] = None,
    supervision_manifest_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_shadow = _resolve_shadow_execution(
        output_root=output_root,
        shadow_execution_report_path=shadow_execution_report_path,
        run_if_missing=run_dependencies_if_missing,
    )
    resolved_supervision = _resolve_supervision(
        output_root=output_root,
        supervision_manifest_path=supervision_manifest_path,
        run_if_missing=run_dependencies_if_missing,
    )
    report_path = output_root / "economic_wm_shadow_outcome_loop_report_v1.json"
    receipts_path = output_root / "economic_wm_shadow_outcome_receipts_v1.jsonl"
    comparisons_path = (
        output_root / "economic_wm_shadow_outcome_comparisons_joined_v1.jsonl"
    )
    markdown_path = output_root / "economic_wm_shadow_outcome_loop_v1.md"
    report = build_economic_wm_shadow_outcome_loop_from_paths(
        shadow_execution_report_path=resolved_shadow,
        supervision_manifest_path=resolved_supervision,
        report_path=report_path,
        outcome_receipts_path=receipts_path,
        updated_comparisons_path=comparisons_path,
        metadata={"source": "run_economic_wm_shadow_outcome_loop_script"},
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "report_path": str(report_path),
        "outcome_receipts_path": str(receipts_path),
        "updated_comparisons_path": str(comparisons_path),
        "markdown_path": str(markdown_path),
        "shadow_execution_report_path": str(resolved_shadow),
        "supervision_manifest_path": str(resolved_supervision),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMShadowOutcomeLoopReport.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_shadow_outcome_loop",
    )
    parser.add_argument("--shadow-execution", default=None)
    parser.add_argument("--supervision-manifest", default=None)
    parser.add_argument("--no-run-dependencies", action="store_true")
    args = parser.parse_args()
    payload = run_economic_wm_shadow_outcome_loop(
        output_dir=args.output_dir,
        shadow_execution_report_path=args.shadow_execution,
        supervision_manifest_path=args.supervision_manifest,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["local_structural_loop_closed"]
        and not payload["hardware_executed"]
        and not payload["provider_executed"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
