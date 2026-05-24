#!/usr/bin/env python3
"""Materialize the Phase 4 downstream controller dry-run scaffold."""

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

from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    DEFAULT_BIPEDAL_CHASSIS_DIR,
)
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE35_BIPEDAL_READINESS_DIR,
)
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (  # noqa: E402
    run_audit_phase35_bipedal_readiness,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_DIR,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (  # noqa: E402
    run_prepare_phase4_deployment_enabler_sweep,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    load_humanoid_chassis_profile,
    load_joint_limit_envelopes,
    load_phase35_bipedal_readiness_audit,
    load_whole_body_replay_rows,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase4_downstream_controller_scaffold,
    load_phase4_deployment_enabler_sweep_report,
    save_phase4_downstream_controller_scaffold,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_downstream_controller_scaffold"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "phase4_downstream_controller_scaffold_report_v1.json"),
        "bridge_targets_path": str(output / "controller_bridge_targets_v1.jsonl"),
        "modes_path": str(output / "controller_mode_specs_v1.jsonl"),
        "proposals_path": str(output / "downstream_controller_proposals_v1.jsonl"),
        "command_frames_path": str(output / "low_level_command_frames_v1.jsonl"),
        "safety_receipts_path": str(output / "controller_safety_receipts_v1.jsonl"),
        "invocations_path": str(output / "controller_invocations_v1.jsonl"),
        "controller_receipts_path": str(output / "controller_receipts_v1.jsonl"),
        "markdown_path": str(output / "phase4_downstream_controller_scaffold_v1.md"),
    }


def _input_paths(
    *,
    phase4_dir: Path,
    bipedal_chassis_dir: Path,
    phase35_bipedal_readiness_dir: Path,
) -> dict[str, Path]:
    return {
        "phase4_report": phase4_dir
        / "humanoid_phase4_deployment_enabler_sweep_report_v1.json",
        "chassis_profile": bipedal_chassis_dir / "humanoid_chassis_profile_v1.json",
        "joint_limits": bipedal_chassis_dir / "joint_limit_envelopes_v1.jsonl",
        "phase35_bipedal_readiness_audit": phase35_bipedal_readiness_dir
        / "phase35_bipedal_readiness_audit_v1.json",
        "whole_body_replay_rows": phase35_bipedal_readiness_dir
        / "whole_body_replay_rows_v1.jsonl",
    }


def _ensure_inputs(
    *,
    phase4_dir: Path,
    bipedal_chassis_dir: Path,
    phase35_bipedal_readiness_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _input_paths(
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
    )
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError("Missing downstream controller inputs: " + ", ".join(missing))
    run_prepare_phase4_deployment_enabler_sweep(
        output_dir=phase4_dir,
        run_dependencies_if_missing=True,
    )
    run_audit_phase35_bipedal_readiness(
        output_dir=phase35_bipedal_readiness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        run_dependencies_if_missing=True,
    )
    paths = _input_paths(
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Downstream controller dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return paths


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Downstream Controller Scaffold",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local downstream controller scaffold complete: "
        f"`{str(payload['local_downstream_controller_scaffold_complete']).lower()}`",
        f"- Bridge targets: `{payload['bridge_target_count']}`",
        f"- Controller modes: `{payload['mode_count']}`",
        f"- Proposals: `{payload['proposal_count']}`",
        f"- Command frames: `{payload['command_frame_count']}`",
        f"- Safety receipts: `{payload['safety_receipt_count']}`",
        f"- Invocations: `{payload['invocation_count']}`",
        f"- Controller receipts: `{payload['controller_receipt_count']}`",
        "- Unitree bridge contract present: "
        f"`{str(payload['unitree_bridge_contract_present']).lower()}`",
        "- G1Pilot fallback contract present: "
        f"`{str(payload['g1pilot_fallback_contract_present']).lower()}`",
        "- Dry-run controller present: "
        f"`{str(payload['dry_run_controller_present']).lower()}`",
        "",
        "## Boundary",
        "",
        "This scaffold emits dry-run command frames and receipts only. It does",
        "not publish ROS2/DDS messages, write Unitree SDK2 commands, invoke",
        "G1Pilot, execute sim or hardware, train weights, mutate reward math, or",
        "promote a controller.",
        "",
        "## Key Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["key_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase4_downstream_controller_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase4_dir: str | Path = DEFAULT_PHASE4_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    phase35_bipedal_readiness_dir: str | Path = (
        DEFAULT_PHASE35_BIPEDAL_READINESS_DIR
    ),
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    input_paths = _ensure_inputs(
        phase4_dir=Path(phase4_dir),
        bipedal_chassis_dir=Path(bipedal_chassis_dir),
        phase35_bipedal_readiness_dir=Path(phase35_bipedal_readiness_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    artifact_refs = {
        **refs,
        **{f"input_{key}_path": str(path) for key, path in input_paths.items()},
    }
    (
        report,
        bridge_targets,
        modes,
        proposals,
        command_frames,
        safety_receipts,
        invocations,
        controller_receipts,
    ) = build_phase4_downstream_controller_scaffold(
        phase4_report=load_phase4_deployment_enabler_sweep_report(
            input_paths["phase4_report"]
        ),
        phase35_readiness_audit=load_phase35_bipedal_readiness_audit(
            input_paths["phase35_bipedal_readiness_audit"]
        ),
        chassis=load_humanoid_chassis_profile(input_paths["chassis_profile"]),
        joint_limits=load_joint_limit_envelopes(input_paths["joint_limits"]),
        replay_rows=load_whole_body_replay_rows(input_paths["whole_body_replay_rows"]),
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase4_downstream_controller_scaffold(
        output,
        report=report,
        bridge_targets=bridge_targets,
        modes=modes,
        proposals=proposals,
        command_frames=command_frames,
        safety_receipts=safety_receipts,
        invocations=invocations,
        controller_receipts=controller_receipts,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase4-dir", default=str(DEFAULT_PHASE4_DIR))
    parser.add_argument("--bipedal-chassis-dir", default=str(DEFAULT_BIPEDAL_CHASSIS_DIR))
    parser.add_argument(
        "--phase35-bipedal-readiness-dir",
        default=str(DEFAULT_PHASE35_BIPEDAL_READINESS_DIR),
    )
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase4_downstream_controller_scaffold(
        output_dir=args.output_dir,
        phase4_dir=args.phase4_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_downstream_controller_scaffold_complete"]
        and payload["unitree_bridge_contract_present"]
        and payload["g1pilot_fallback_contract_present"]
        and payload["dry_run_controller_present"]
        and not payload["hardware_dispatch_enabled"]
        and not payload["ros2_publish_attempted"]
        and not payload["unitree_sdk2_write_enabled"]
        and not payload["g1pilot_runtime_invoked"]
        and not payload["live_policy_control"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
