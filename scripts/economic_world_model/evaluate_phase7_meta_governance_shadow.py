#!/usr/bin/env python3
"""Evaluate Phase 7 shadow Meta-Regal runtime events and outcome joins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_RUNTIME_DIR,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (  # noqa: E402
    run_wire_phase7_meta_regal_runtime_shadow,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase7_meta_governance_evaluation,
    load_phase7_runtime_eval_inputs,
    save_phase7_meta_governance_evaluation,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase7_meta_governance_eval"
)


def _required_runtime_paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "field_receipts": runtime_dir / "phase7_control_field_runtime_receipts.jsonl",
        "conflict_receipts": runtime_dir
        / "phase7_conflict_runtime_join_receipts.jsonl",
        "event_spine": runtime_dir / "event_spine.json",
        "decision_ledger": runtime_dir / "decision_ledger.json",
        "summary": runtime_dir / "summary.json",
    }


def _paths(output: Path, runtime_dir: Path) -> dict[str, Path]:
    return {
        **_required_runtime_paths(runtime_dir),
        "report": output / "phase7_meta_governance_evaluation_report_v1.json",
        "control_field_evals": output
        / "phase7_control_field_eval_reports_v1.jsonl",
        "conflict_join_evals": output
        / "phase7_conflict_join_eval_reports_v1.jsonl",
        "pareto_regime_evals": output
        / "phase7_pareto_regime_eval_reports_v1.jsonl",
        "outcome_join_rows": output / "phase7_outcome_join_rows_v1.jsonl",
        "markdown": output / "phase7_meta_governance_evaluation_report_v1.md",
    }


def _resolve_runtime_inputs(
    *,
    runtime_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _required_runtime_paths(runtime_dir)
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase 7 runtime eval inputs: " + ", ".join(missing)
        )
    run_wire_phase7_meta_regal_runtime_shadow(
        output_dir=runtime_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 7 runtime wiring did not materialize: " + ", ".join(missing)
        )
    return paths


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Meta-Governance Shadow Evaluation",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Run ID: `{payload['run_id']}`",
        "- Local meta-governance eval complete: "
        f"`{str(payload['local_meta_governance_eval_complete']).lower()}`",
        "- Control-field-only eval complete: "
        f"`{str(payload['control_field_only_eval_complete']).lower()}`",
        "- Conflict-join eval complete: "
        f"`{str(payload['conflict_join_eval_complete']).lower()}`",
        "- Pareto/regime eval complete: "
        f"`{str(payload['pareto_regime_eval_complete']).lower()}`",
        "- Outcome join slots complete: "
        f"`{str(payload['outcome_join_slots_complete']).lower()}`",
        f"- Control-field evals: `{payload['control_field_eval_count']}`",
        f"- Conflict-join evals: `{payload['conflict_join_eval_count']}`",
        f"- Pareto/regime evals: `{payload['pareto_regime_eval_count']}`",
        f"- Outcome join rows: `{payload['outcome_join_row_count']}`",
        f"- Phase 7 events: `{payload['phase7_event_count']}`",
        f"- Phase 7 decisions: `{payload['phase7_decision_count']}`",
        "",
        "## Boundary",
        "",
        "This is an evaluation and replay-export harness only. It verifies",
        "shadow event/decision joins, decomposes control-field and conflict",
        "receipts, creates Pareto/regime labels, and materializes outcome",
        "join rows. It does not train, write weights, dispatch live actions,",
        "execute hard vetoes, mutate reward math, promote outputs, or claim",
        "provider/hardware/Unitree runtime authority.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluate_phase7_meta_governance_shadow(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase7_runtime_dir: str | Path = DEFAULT_PHASE7_RUNTIME_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(phase7_runtime_dir)
    _resolve_runtime_inputs(
        runtime_dir=runtime_dir,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    (
        field_receipts,
        conflict_receipts,
        runtime_events,
        decisions,
        summary_payload,
    ) = load_phase7_runtime_eval_inputs(runtime_dir)
    paths = _paths(output, runtime_dir)
    refs = {f"{key}_path": str(path) for key, path in paths.items()}
    report, field_evals, conflict_evals, regime_evals, outcome_rows = (
        build_phase7_meta_governance_evaluation(
            run_id=str(summary_payload.get("run_id", "")),
            field_receipts=field_receipts,
            conflict_receipts=conflict_receipts,
            runtime_events=runtime_events,
            decision_entries=decisions,
            summary_payload=summary_payload,
            artifact_refs=refs,
        )
    )
    saved_refs = save_phase7_meta_governance_evaluation(
        output,
        report,
        field_evals,
        conflict_evals,
        regime_evals,
        outcome_rows,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(paths["markdown"], payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase7-runtime-dir", default=str(DEFAULT_PHASE7_RUNTIME_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_evaluate_phase7_meta_governance_shadow(
        output_dir=args.output_dir,
        phase7_runtime_dir=args.phase7_runtime_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_meta_governance_eval_complete"]
        and payload["replay_export_ready"]
        and not payload["phase7_authority_granted"]
        and not payload["live_dispatch_allowed"]
        and not payload["hard_veto_dispatch"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
