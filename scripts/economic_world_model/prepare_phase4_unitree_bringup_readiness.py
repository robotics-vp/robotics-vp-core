#!/usr/bin/env python3
"""Materialize Phase 4 Unitree/G1 bring-up readiness receipts."""

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
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR,
)
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (  # noqa: E402
    run_prepare_phase4_downstream_controller_scaffold,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    load_humanoid_chassis_profile,
    load_joint_limit_envelopes,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase4_unitree_bringup_readiness,
    default_unitree_local_roots,
    load_low_level_command_frames,
    load_phase4_downstream_controller_scaffold_report,
    save_phase4_unitree_bringup_readiness,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_unitree_bringup_readiness"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "phase4_unitree_bringup_readiness_report_v1.json"),
        "block_receipts_path": str(output / "unitree_bringup_block_receipts_v1.jsonl"),
        "dependency_targets_path": str(output / "unitree_dependency_targets_v1.jsonl"),
        "asset_receipts_path": str(
            output / "unitree_asset_calibration_receipts_v1.jsonl"
        ),
        "stream_contracts_path": str(output / "unitree_stream_contracts_v1.jsonl"),
        "command_receipts_path": str(
            output / "unitree_command_conformance_receipts_v1.jsonl"
        ),
        "timing_receipts_path": str(
            output / "unitree_timing_jitter_probe_receipts_v1.jsonl"
        ),
        "safety_receipts_path": str(
            output / "unitree_safety_preflight_receipts_v1.jsonl"
        ),
        "operator_runbooks_path": str(
            output / "unitree_operator_recovery_runbooks_v1.jsonl"
        ),
        "evidence_ledgers_path": str(
            output / "unitree_sim_hardware_evidence_ledgers_v1.jsonl"
        ),
        "markdown_path": str(output / "phase4_unitree_bringup_readiness_v1.md"),
    }


def _input_paths(
    *,
    bipedal_chassis_dir: Path,
    phase4_downstream_controller_dir: Path,
) -> dict[str, Path]:
    return {
        "chassis_profile": bipedal_chassis_dir / "humanoid_chassis_profile_v1.json",
        "joint_limits": bipedal_chassis_dir / "joint_limit_envelopes_v1.jsonl",
        "phase4_downstream_controller_report": phase4_downstream_controller_dir
        / "phase4_downstream_controller_scaffold_report_v1.json",
        "command_frames": phase4_downstream_controller_dir
        / "low_level_command_frames_v1.jsonl",
    }


def _ensure_inputs(
    *,
    bipedal_chassis_dir: Path,
    phase35_bipedal_readiness_dir: Path,
    phase4_downstream_controller_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _input_paths(
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
    )
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Unitree bring-up readiness inputs: " + ", ".join(missing)
        )
    run_prepare_phase4_downstream_controller_scaffold(
        output_dir=phase4_downstream_controller_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        run_dependencies_if_missing=True,
    )
    paths = _input_paths(
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Unitree bring-up dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return paths


def _parse_local_root_overrides(values: list[str] | None) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected --local-root KEY=PATH, got {value!r}")
        key, path = value.split("=", 1)
        roots[key.strip()] = path.strip()
    return roots


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Unitree Bring-Up Readiness",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local pre-purchase prepared: "
        f"`{str(payload['local_pre_purchase_prepared']).lower()}`",
        f"- Blocks emitted: `{payload['block_count']}`",
        f"- Dependency targets: `{payload['dependency_target_count']}`",
        f"- Verified local dependency layouts: `{payload['dependency_verified_count']}`",
        "- Asset joint subset aligned: "
        f"`{str(payload['asset_joint_subset_aligned']).lower()}`",
        "- Command conformance dry-run ready: "
        f"`{str(payload['command_conformance_dry_run_ready']).lower()}`",
        "- Honest sim or hardware evidence present: "
        f"`{str(payload['honest_sim_or_hardware_evidence_present']).lower()}`",
        "",
        "## Blocks",
        "",
        "- `runtime_dependency_manifest`",
        "- `g1pilot_or_fallback_review`",
        "- `robot_asset_calibration_intake`",
        "- `live_stream_interface_contracts`",
        "- `command_interface_conformance`",
        "- `timing_jitter_probe`",
        "- `physical_safety_preflight`",
        "- `operator_estop_recovery_runbook`",
        "- `sim_hardware_evidence_ledger`",
        "",
        "## Remaining Evidence Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_key_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This artifact goes as far as local pre-purchase preparation allows.",
            "It inventories local roots and emits contracts/receipts only. It",
            "does not publish ROS2/DDS, write Unitree SDK2 commands, invoke",
            "G1Pilot, execute sim or hardware, train weights, mutate reward math,",
            "or promote controller authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase4_unitree_bringup_readiness(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    phase35_bipedal_readiness_dir: str | Path = (
        DEFAULT_PHASE35_BIPEDAL_READINESS_DIR
    ),
    phase4_downstream_controller_dir: str | Path = (
        DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR
    ),
    local_roots: Mapping[str, str | Path] | None = None,
    asset_paths: list[str | Path] | None = None,
    timing_iterations: int = 200,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    input_paths = _ensure_inputs(
        bipedal_chassis_dir=Path(bipedal_chassis_dir),
        phase35_bipedal_readiness_dir=Path(phase35_bipedal_readiness_dir),
        phase4_downstream_controller_dir=Path(phase4_downstream_controller_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    root_map = {**default_unitree_local_roots(), **dict(local_roots or {})}
    artifact_refs = {
        **refs,
        **{f"input_{key}_path": str(path) for key, path in input_paths.items()},
        "local_roots": {key: str(value) for key, value in root_map.items()},
    }
    (
        report,
        block_receipts,
        dependency_targets,
        asset_receipts,
        stream_contracts,
        command_receipts,
        timing_receipts,
        safety_receipts,
        operator_runbooks,
        evidence_ledgers,
    ) = build_phase4_unitree_bringup_readiness(
        phase4_downstream_controller_report=(
            load_phase4_downstream_controller_scaffold_report(
                input_paths["phase4_downstream_controller_report"]
            )
        ),
        chassis=load_humanoid_chassis_profile(input_paths["chassis_profile"]),
        joint_limits=load_joint_limit_envelopes(input_paths["joint_limits"]),
        command_frames=load_low_level_command_frames(input_paths["command_frames"]),
        local_roots=root_map,
        asset_paths=asset_paths,
        timing_iterations=timing_iterations,
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase4_unitree_bringup_readiness(
        output,
        report=report,
        block_receipts=block_receipts,
        dependency_targets=dependency_targets,
        asset_receipts=asset_receipts,
        stream_contracts=stream_contracts,
        command_receipts=command_receipts,
        timing_receipts=timing_receipts,
        safety_receipts=safety_receipts,
        operator_runbooks=operator_runbooks,
        evidence_ledgers=evidence_ledgers,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bipedal-chassis-dir", default=str(DEFAULT_BIPEDAL_CHASSIS_DIR))
    parser.add_argument(
        "--phase35-bipedal-readiness-dir",
        default=str(DEFAULT_PHASE35_BIPEDAL_READINESS_DIR),
    )
    parser.add_argument(
        "--phase4-downstream-controller-dir",
        default=str(DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR),
    )
    parser.add_argument(
        "--local-root",
        action="append",
        help="Override a Unitree dependency root as KEY=PATH. May be repeated.",
    )
    parser.add_argument(
        "--asset-path",
        action="append",
        help="Additional G1 asset path to parse before default local discovery.",
    )
    parser.add_argument("--timing-iterations", type=int, default=200)
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase4_unitree_bringup_readiness(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=args.phase4_downstream_controller_dir,
        local_roots=_parse_local_root_overrides(args.local_root),
        asset_paths=args.asset_path,
        timing_iterations=args.timing_iterations,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_pre_purchase_prepared"]
        and payload["all_block_receipts_emitted"]
        and payload["asset_joint_subset_aligned"]
        and payload["command_conformance_dry_run_ready"]
        and not payload["honest_sim_or_hardware_evidence_present"]
        and not payload["hardware_dispatch_enabled"]
        and not payload["ros2_publish_attempted"]
        and not payload["unitree_sdk2_write_enabled"]
        and not payload["g1pilot_runtime_invoked"]
        and not payload["honest_sim_executed"]
        and not payload["hardware_executed"]
        and not payload["live_policy_control"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
