#!/usr/bin/env python3
"""Materialize runnable local Unitree/G1 harness artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
    build_phase4_unitree_local_harnesses,
    build_trace_harness_receipts,
    default_unitree_local_roots,
    load_low_level_command_frames,
    save_phase4_unitree_local_harnesses,
)
from src.world_model.humanoid_readiness.common import write_jsonl  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_unitree_local_harnesses"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "phase4_unitree_local_harness_report_v1.json"),
        "low_state_traces_path": str(output / "unitree_low_state_traces_v1.jsonl"),
        "imu_traces_path": str(output / "unitree_imu_traces_v1.jsonl"),
        "wireless_estop_traces_path": str(
            output / "unitree_wireless_estop_traces_v1.jsonl"
        ),
        "contact_traces_path": str(output / "unitree_contact_traces_v1.jsonl"),
        "trace_replay_receipts_path": str(
            output / "unitree_trace_replay_receipts_v1.jsonl"
        ),
        "mock_receiver_receipts_path": str(
            output / "unitree_mock_receiver_receipts_v1.jsonl"
        ),
        "stale_validation_receipts_path": str(
            output / "unitree_stale_data_validation_receipts_v1.jsonl"
        ),
        "ros_message_definitions_path": str(
            output / "unitree_ros_message_definitions_v1.jsonl"
        ),
        "command_shape_receipts_path": str(
            output / "unitree_command_shape_validation_receipts_v1.jsonl"
        ),
        "mock_timing_receipts_path": str(
            output / "unitree_mock_timing_run_receipts_v1.jsonl"
        ),
        "watchdog_demotion_receipts_path": str(
            output / "unitree_watchdog_demotion_receipts_v1.jsonl"
        ),
        "safety_transitions_path": str(
            output / "unitree_safety_state_transitions_v1.jsonl"
        ),
        "synthetic_safety_drills_path": str(
            output / "unitree_synthetic_safety_drill_receipts_v1.jsonl"
        ),
        "runtime_preflight_receipts_path": str(
            output / "unitree_runtime_preflight_receipts_v1.jsonl"
        ),
        "markdown_path": str(output / "phase4_unitree_local_harnesses_v1.md"),
    }


def _input_paths(
    *,
    bipedal_chassis_dir: Path,
    phase4_downstream_controller_dir: Path,
) -> dict[str, Path]:
    return {
        "chassis_profile": bipedal_chassis_dir / "humanoid_chassis_profile_v1.json",
        "joint_limits": bipedal_chassis_dir / "joint_limit_envelopes_v1.jsonl",
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
        raise FileNotFoundError("Missing Unitree harness inputs: " + ", ".join(missing))
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
            "Unitree harness dependency builders did not materialize: "
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


def _write_trace_rows(paths: Mapping[str, str], rows: Mapping[str, list[Any]]) -> None:
    write_jsonl(
        paths["low_state_traces_path"],
        [row.to_dict() for row in rows["low_state"]],
    )
    write_jsonl(paths["imu_traces_path"], [row.to_dict() for row in rows["imu"]])
    write_jsonl(
        paths["wireless_estop_traces_path"],
        [row.to_dict() for row in rows["wireless_estop"]],
    )
    write_jsonl(
        paths["contact_traces_path"],
        [row.to_dict() for row in rows["contact"]],
    )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Unitree Local Harnesses",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local harnesses complete: "
        f"`{str(payload['local_harnesses_complete']).lower()}`",
        "- Trace/stream harness complete: "
        f"`{str(payload['trace_stream_harness_complete']).lower()}`",
        "- Command shape harness complete: "
        f"`{str(payload['command_shape_harness_complete']).lower()}`",
        "- Mock timing/watchdog harness complete: "
        f"`{str(payload['mock_timing_watchdog_harness_complete']).lower()}`",
        "- Safety/recovery harness complete: "
        f"`{str(payload['safety_recovery_harness_complete']).lower()}`",
        "- Runtime preflight harness complete: "
        f"`{str(payload['runtime_preflight_harness_complete']).lower()}`",
        f"- Low-state traces: `{payload['low_state_trace_count']}`",
        f"- IMU traces: `{payload['imu_trace_count']}`",
        f"- Wireless/e-stop traces: `{payload['wireless_estop_trace_count']}`",
        f"- Contact traces: `{payload['contact_trace_count']}`",
        f"- Command shape receipts: "
        f"`{payload['command_shape_validation_receipt_count']}`",
        f"- Runtime preflight receipts: "
        f"`{payload['runtime_preflight_receipt_count']}`",
        "",
        "## Remaining Evidence Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_evidence_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These harnesses run local synthetic traces, no-publish validators,",
            "mock timing loops, synthetic safety drills, and preflight checks.",
            "They do not publish ROS2/DDS messages, write Unitree SDK2 commands,",
            "invoke G1Pilot, launch MuJoCo, execute hardware, train weights,",
            "mutate reward math, or promote authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase4_unitree_local_harnesses(
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
    sample_count: int = 12,
    timing_iterations: int = 32,
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
        low_state,
        imu,
        wireless,
        contacts,
        replay_receipts,
        receiver_receipts,
        stale_receipts,
        definitions,
        command_shape,
        timing_receipts,
        watchdog_receipts,
        transitions,
        drills,
        preflights,
    ) = build_phase4_unitree_local_harnesses(
        chassis=load_humanoid_chassis_profile(input_paths["chassis_profile"]),
        joint_limits=load_joint_limit_envelopes(input_paths["joint_limits"]),
        command_frames=load_low_level_command_frames(input_paths["command_frames"]),
        local_roots=root_map,
        sample_count=sample_count,
        timing_iterations=timing_iterations,
        artifact_refs=artifact_refs,
    )
    _write_trace_rows(
        refs,
        {
            "low_state": low_state,
            "imu": imu,
            "wireless_estop": wireless,
            "contact": contacts,
        },
    )
    replay_receipts, receiver_receipts, stale_receipts = build_trace_harness_receipts(
        low_state=low_state,
        imu=imu,
        wireless=wireless,
        contacts=contacts,
        trace_paths={
            "low_state": refs["low_state_traces_path"],
            "imu": refs["imu_traces_path"],
            "wireless_estop": refs["wireless_estop_traces_path"],
            "contact": refs["contact_traces_path"],
        },
    )
    report = replace(
        report,
        trace_replay_receipt_count=len(replay_receipts),
        mock_receiver_receipt_count=len(receiver_receipts),
        stale_validation_receipt_count=len(stale_receipts),
        trace_stream_harness_complete=(
            report.trace_stream_harness_complete
            and all(receipt.jsonl_import_verified for receipt in replay_receipts)
            and all(receipt.receiver_executed for receipt in receiver_receipts)
        ),
        local_harnesses_complete=(
            report.command_shape_harness_complete
            and report.mock_timing_watchdog_harness_complete
            and report.safety_recovery_harness_complete
            and report.runtime_preflight_harness_complete
            and all(receipt.jsonl_import_verified for receipt in replay_receipts)
            and all(receipt.receiver_executed for receipt in receiver_receipts)
        ),
        status="ok"
        if (
            report.command_shape_harness_complete
            and report.mock_timing_watchdog_harness_complete
            and report.safety_recovery_harness_complete
            and report.runtime_preflight_harness_complete
            and all(receipt.jsonl_import_verified for receipt in replay_receipts)
            and all(receipt.receiver_executed for receipt in receiver_receipts)
        )
        else "blocked",
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase4_unitree_local_harnesses(
        output,
        report=report,
        low_state=low_state,
        imu=imu,
        wireless=wireless,
        contacts=contacts,
        replay_receipts=replay_receipts,
        receiver_receipts=receiver_receipts,
        stale_receipts=stale_receipts,
        definitions=definitions,
        command_shape=command_shape,
        timing_receipts=timing_receipts,
        watchdog_receipts=watchdog_receipts,
        transitions=transitions,
        drills=drills,
        preflights=preflights,
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
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--timing-iterations", type=int, default=32)
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase4_unitree_local_harnesses(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=args.phase4_downstream_controller_dir,
        local_roots=_parse_local_root_overrides(args.local_root),
        sample_count=args.sample_count,
        timing_iterations=args.timing_iterations,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_harnesses_complete"]
        and payload["trace_stream_harness_complete"]
        and payload["command_shape_harness_complete"]
        and payload["mock_timing_watchdog_harness_complete"]
        and payload["safety_recovery_harness_complete"]
        and payload["runtime_preflight_harness_complete"]
        and not payload["live_stream_observed"]
        and not payload["ros2_publish_attempted"]
        and not payload["unitree_sdk2_write_enabled"]
        and not payload["g1pilot_runtime_invoked"]
        and not payload["mujoco_launch_executed"]
        and not payload["ros2_launch_executed"]
        and not payload["hardware_executed"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
