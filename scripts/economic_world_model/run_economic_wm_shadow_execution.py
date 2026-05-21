#!/usr/bin/env python3
"""Emit Economic WM shadow work orders and outcome-comparison slots."""

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
    EconomicWMShadowExecutionReport,
    build_economic_wm_shadow_execution_report_from_paths,
)


def _write_markdown(path: Path, report: EconomicWMShadowExecutionReport) -> None:
    payload = report.to_dict()
    lines = [
        "# Economic WM Shadow Execution",
        "",
        f"- Report ID: `{payload['report_id']}`",
        f"- Phase-5 manifest ID: `{payload['phase5_manifest_id']}`",
        f"- Allocation eval ID: `{payload['allocation_eval_id']}`",
        f"- Trainer scaffold ID: `{payload['trainer_scaffold_id']}`",
        f"- Recommended candidate: `{payload['recommended_candidate']}`",
        f"- Work orders: `{payload['work_order_count']}`",
        f"- Outcome comparisons: `{payload['outcome_comparison_count']}`",
        f"- Authority class: `{payload['authority_class']}`",
        f"- Ready for shadow comparison: `{str(payload['ready_for_shadow_comparison']).lower()}`",
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
            "Shadow execution emits advisory work orders and future outcome-comparison slots. It does not control live policy, mutate reward math, run providers, run GPU training, or promote the Economic WM.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_phase5(
    *, output_root: Path, phase5_prep_path: Optional[str | Path], run_if_missing: bool
) -> Path:
    path = Path(
        phase5_prep_path
        or "artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.prepare_economic_wm_phase5_local_prep import (
        run_prepare_economic_wm_phase5_local_prep,
    )

    payload = run_prepare_economic_wm_phase5_local_prep(
        output_dir=output_root / "phase5_local_prep"
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def _resolve_allocation_eval(
    *,
    output_root: Path,
    allocation_eval_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    path = Path(
        allocation_eval_path
        or "artifacts/economic_world_model/economic_wm_shadow_allocation_eval/economic_wm_shadow_allocation_eval_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.evaluate_economic_wm_shadow_allocations import (
        run_evaluate_economic_wm_shadow_allocations,
    )

    payload = run_evaluate_economic_wm_shadow_allocations(
        output_dir=output_root / "shadow_allocation_eval"
    )
    return Path(payload["artifact_refs"]["eval_path"])


def _resolve_trainer(
    *,
    output_root: Path,
    trainer_scaffold_path: Optional[str | Path],
    phase5_prep_path: Path,
    run_if_missing: bool,
) -> Path:
    path = Path(
        trainer_scaffold_path
        or "artifacts/economic_world_model/economic_wm_trainer_scaffold/economic_wm_trainer_scaffold_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.train_economic_world_model_v0 import (
        run_train_economic_world_model_v0_scaffold,
    )

    payload = run_train_economic_world_model_v0_scaffold(
        output_dir=output_root / "trainer_scaffold",
        phase5_prep_path=phase5_prep_path,
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def run_economic_wm_shadow_execution(
    *,
    output_dir: str | Path,
    phase5_prep_path: Optional[str | Path] = None,
    allocation_eval_path: Optional[str | Path] = None,
    trainer_scaffold_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_phase5 = _resolve_phase5(
        output_root=output_root,
        phase5_prep_path=phase5_prep_path,
        run_if_missing=run_dependencies_if_missing,
    )
    resolved_allocation = _resolve_allocation_eval(
        output_root=output_root,
        allocation_eval_path=allocation_eval_path,
        run_if_missing=run_dependencies_if_missing,
    )
    resolved_trainer = _resolve_trainer(
        output_root=output_root,
        trainer_scaffold_path=trainer_scaffold_path,
        phase5_prep_path=resolved_phase5,
        run_if_missing=run_dependencies_if_missing,
    )
    report_path = output_root / "economic_wm_shadow_execution_report_v1.json"
    work_orders_path = output_root / "economic_wm_shadow_work_orders_v1.jsonl"
    comparisons_path = output_root / "economic_wm_shadow_outcome_comparisons_v1.jsonl"
    markdown_path = output_root / "economic_wm_shadow_execution_v1.md"
    report = build_economic_wm_shadow_execution_report_from_paths(
        phase5_prep_path=resolved_phase5,
        allocation_eval_path=resolved_allocation,
        trainer_scaffold_path=resolved_trainer,
        report_path=report_path,
        work_orders_path=work_orders_path,
        outcome_comparisons_path=comparisons_path,
        metadata={"source": "run_economic_wm_shadow_execution_script"},
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {
        **dict(payload.get("artifact_refs", {}) or {}),
        "report_path": str(report_path),
        "work_orders_path": str(work_orders_path),
        "outcome_comparisons_path": str(comparisons_path),
        "markdown_path": str(markdown_path),
        "phase5_prep_path": str(resolved_phase5),
        "allocation_eval_path": str(resolved_allocation),
        "trainer_scaffold_path": str(resolved_trainer),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(markdown_path, EconomicWMShadowExecutionReport.from_dict(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_shadow_execution",
        help="Directory for Economic WM shadow execution artifacts.",
    )
    parser.add_argument("--phase5-prep", default=None)
    parser.add_argument("--allocation-eval", default=None)
    parser.add_argument("--trainer-scaffold", default=None)
    parser.add_argument(
        "--no-run-dependencies",
        action="store_true",
        help="Do not run missing Phase-5 prep, allocation eval, or trainer scaffold.",
    )
    args = parser.parse_args()
    payload = run_economic_wm_shadow_execution(
        output_dir=args.output_dir,
        phase5_prep_path=args.phase5_prep,
        allocation_eval_path=args.allocation_eval,
        trainer_scaffold_path=args.trainer_scaffold,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_shadow_comparison"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
