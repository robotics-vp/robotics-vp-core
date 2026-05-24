#!/usr/bin/env python3
"""Audit Phase 3.5 bipedal readiness before GPU, sim, or hardware exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.prepare_phase35_bipedal_chassis_scaffold import (  # noqa: E402
    run_prepare_phase35_bipedal_chassis_scaffold,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    build_phase35_bipedal_readiness_audit,
    load_balance_envelope_receipts,
    load_bipedal_chassis_scaffold_report,
    load_bipedal_support_states,
    load_humanoid_chassis_profile,
    load_humanoid_frame_tree,
    load_joint_limit_envelopes,
    load_whole_body_action_schema,
    load_whole_body_observation_schema,
    save_phase35_bipedal_readiness_audit,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase35_bipedal_readiness_audit"
)
DEFAULT_BIPEDAL_CHASSIS_DIR = Path(
    "artifacts/economic_world_model/phase35_bipedal_chassis_scaffold"
)


def _readiness_refs(output: Path) -> dict[str, str]:
    return {
        "audit_path": str(output / "phase35_bipedal_readiness_audit_v1.json"),
        "asset_contract_path": str(output / "humanoid_robot_asset_contract_v1.json"),
        "asset_parse_receipts_path": str(
            output / "robot_asset_parse_receipts_v1.jsonl"
        ),
        "kinematic_report_path": str(output / "kinematic_consistency_report_v1.json"),
        "joint_vector_receipts_path": str(
            output / "joint_vector_validation_receipts_v1.jsonl"
        ),
        "balance_geometry_reports_path": str(
            output / "balance_geometry_reports_v1.jsonl"
        ),
        "whole_body_replay_rows_path": str(output / "whole_body_replay_rows_v1.jsonl"),
        "markdown_path": str(output / "phase35_bipedal_readiness_audit_v1.md"),
    }


def _bipedal_chassis_refs(output: Path) -> dict[str, Path]:
    return {
        "report_path": output / "bipedal_chassis_scaffold_report_v1.json",
        "chassis_profile_path": output / "humanoid_chassis_profile_v1.json",
        "frame_tree_path": output / "humanoid_frame_tree_v1.json",
        "joint_limits_path": output / "joint_limit_envelopes_v1.jsonl",
        "observation_schema_path": output / "whole_body_observation_schema_v1.json",
        "action_schema_path": output / "whole_body_action_schema_v1.json",
        "support_states_path": output / "bipedal_support_states_v1.jsonl",
        "balance_receipts_path": output / "balance_envelope_receipts_v1.jsonl",
    }


def _missing_artifacts(refs: Mapping[str, Path]) -> list[str]:
    return [str(path) for path in refs.values() if not path.exists()]


def _ensure_bipedal_chassis_artifacts(
    *,
    bipedal_chassis_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    refs = _bipedal_chassis_refs(bipedal_chassis_dir)
    missing = _missing_artifacts(refs)
    if missing and run_dependencies_if_missing:
        run_prepare_phase35_bipedal_chassis_scaffold(output_dir=bipedal_chassis_dir)
        refs = _bipedal_chassis_refs(bipedal_chassis_dir)
        missing = _missing_artifacts(refs)
    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Missing Phase 3.5 bipedal chassis artifacts:\n" + missing_lines
        )
    return refs


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3.5 Bipedal Readiness Audit",
        "",
        f"- Audit: `{payload['audit_id']}`",
        f"- Status: `{payload['status']}`",
        "- No-GPU/no-hardware preparation complete: "
        f"`{str(payload['phase35_no_gpu_no_hardware_prepared']).lower()}`",
        "- Local asset-ingestion contract present: "
        f"`{str(payload['local_asset_ingestion_contract_present']).lower()}`",
        f"- Asset parse receipts: `{payload['asset_parse_receipt_count']}`",
        f"- Real asset parsed: `{str(payload['real_asset_parsed']).lower()}`",
        "- Kinematic validators present: "
        f"`{str(payload['kinematic_validators_present']).lower()}`",
        "- Joint-vector validation receipts: "
        f"`{payload['joint_vector_validation_receipt_count']}`",
        f"- Balance geometry reports: `{payload['balance_geometry_report_count']}`",
        f"- Whole-body replay rows: `{payload['whole_body_replay_row_count']}`",
        "",
        "## Boundary",
        "",
        "This audit closes local scaffolding only. It does not run Unitree sim,",
        "hardware, providers, GPU training, weight writes, live policy control,",
        "reward-math mutation, or promotion.",
        "",
        "## Closed Local Surfaces",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["closed_local_surfaces"])
    lines.extend(["", "## Remaining Blockers", ""])
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit_phase35_bipedal_readiness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    asset_paths: Iterable[str | Path] | None = None,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    readiness_refs = _readiness_refs(output)
    chassis_refs = _ensure_bipedal_chassis_artifacts(
        bipedal_chassis_dir=Path(bipedal_chassis_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )

    chassis_report = load_bipedal_chassis_scaffold_report(chassis_refs["report_path"])
    chassis = load_humanoid_chassis_profile(chassis_refs["chassis_profile_path"])
    frame_tree = load_humanoid_frame_tree(chassis_refs["frame_tree_path"])
    joint_limits = load_joint_limit_envelopes(chassis_refs["joint_limits_path"])
    observation_schema = load_whole_body_observation_schema(
        chassis_refs["observation_schema_path"]
    )
    action_schema = load_whole_body_action_schema(chassis_refs["action_schema_path"])
    support_states = load_bipedal_support_states(chassis_refs["support_states_path"])
    balance_receipts = load_balance_envelope_receipts(
        chassis_refs["balance_receipts_path"]
    )

    artifact_refs = {
        **readiness_refs,
        **{f"bipedal_chassis_{key}": str(path) for key, path in chassis_refs.items()},
    }
    (
        audit,
        asset_contract,
        parse_receipts,
        kinematic_report,
        joint_vector_receipts,
        balance_geometry_reports,
        replay_rows,
    ) = build_phase35_bipedal_readiness_audit(
        chassis_report=chassis_report,
        chassis=chassis,
        frame_tree=frame_tree,
        joint_limits=joint_limits,
        observation_schema=observation_schema,
        action_schema=action_schema,
        support_states=support_states,
        balance_receipts=balance_receipts,
        asset_paths=asset_paths,
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase35_bipedal_readiness_audit(
        output,
        audit=audit,
        asset_contract=asset_contract,
        parse_receipts=parse_receipts,
        kinematic_report=kinematic_report,
        joint_vector_receipts=joint_vector_receipts,
        balance_geometry_reports=balance_geometry_reports,
        replay_rows=replay_rows,
    )
    payload = audit.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(readiness_refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--bipedal-chassis-dir", default=str(DEFAULT_BIPEDAL_CHASSIS_DIR)
    )
    parser.add_argument("--asset-path", action="append", default=[])
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_audit_phase35_bipedal_readiness(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        asset_paths=args.asset_path,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["phase35_no_gpu_no_hardware_prepared"]
        and payload["local_asset_ingestion_contract_present"]
        and payload["kinematic_validators_present"]
        and not payload["ready_for_unitree_runtime"]
        and not payload["ready_for_training"]
        and not payload["hardware_calibrated_limits"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
