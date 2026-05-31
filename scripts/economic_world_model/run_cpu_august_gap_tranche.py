#!/usr/bin/env python3
"""Run the CPU/non-GPU August-gap Unitree tranche and emit joined receipts."""

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
from scripts.economic_world_model.prepare_phase4_unitree_local_harnesses import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR,
)
from scripts.economic_world_model.prepare_phase4_unitree_local_harnesses import (  # noqa: E402
    run_prepare_phase4_unitree_local_harnesses,
)
from scripts.economic_world_model.prepare_phase4_unitree_runtime_evidence_bridge import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_UNITREE_RUNTIME_BRIDGE_DIR,
)
from scripts.economic_world_model.prepare_phase4_unitree_runtime_evidence_bridge import (  # noqa: E402
    run_prepare_phase4_unitree_runtime_evidence_bridge,
)
from scripts.economic_world_model.probe_phase4_unitree_blockers import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE4_UNITREE_BLOCKER_STRESS_PROBE_DIR,
)
from scripts.economic_world_model.probe_phase4_unitree_blockers import (  # noqa: E402
    run_probe_phase4_unitree_blockers,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_cpu_august_gap_event_replay_lower_wm_surfaces,
    build_cpu_august_gap_execution_report,
    build_unitree_ros2_sdk2_build_message_validation_receipts,
    default_unitree_local_roots,
    save_cpu_august_gap_execution,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/economic_world_model/cpu_august_gap_execution")


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "cpu_august_gap_execution_report_v1.json"),
        "validation_receipts_path": str(
            output / "unitree_ros2_sdk2_build_message_validation_receipts_v1.jsonl"
        ),
        "event_spine_path": str(output / "event_spine.json"),
        "decision_ledger_path": str(output / "decision_ledger.json"),
        "replay_episodes_path": str(output / "unitree_replay_episodes_v1.jsonl"),
        "replay_steps_path": str(output / "unitree_replay_steps_v1.jsonl"),
        "replay_windows_path": str(output / "unitree_replay_windows_v1.jsonl"),
        "event_replay_join_rows_path": str(
            output / "unitree_event_replay_join_rows_v1.jsonl"
        ),
        "lower_wm_ingestion_rows_path": str(
            output / "unitree_lower_wm_ingestion_rows_v1.jsonl"
        ),
        "markdown_path": str(output / "cpu_august_gap_execution_v1.md"),
    }


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
        "# CPU August-Gap Execution Tranche",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Tranche complete: "
        f"`{str(payload['cpu_august_gap_tranche_complete']).lower()}`",
        "- ROS2 / SDK2 build-message validation complete: "
        f"`{str(payload['ros2_sdk2_build_message_validation_complete']).lower()}`",
        f"- Trace import complete: `{str(payload['trace_import_complete']).lower()}`",
        "- Command dry-run complete: "
        f"`{str(payload['command_dry_run_complete']).lower()}`",
        "- Timing/watchdog complete: "
        f"`{str(payload['timing_watchdog_complete']).lower()}`",
        "- Safety/recovery complete: "
        f"`{str(payload['safety_recovery_complete']).lower()}`",
        "- CPU MuJoCo probe complete: "
        f"`{str(payload['cpu_mujoco_probe_complete']).lower()}`",
        "- Event-spine/replay joins complete: "
        f"`{str(payload['event_spine_replay_joins_complete']).lower()}`",
        "- Lower-WM ingestion complete: "
        f"`{str(payload['lower_wm_ingestion_complete']).lower()}`",
        f"- Validation receipts: `{payload['validation_receipt_count']}`",
        f"- Events: `{payload['event_count']}`",
        f"- Decisions: `{payload['decision_count']}`",
        f"- Replay steps: `{payload['replay_step_count']}`",
        f"- Lower-WM ingestion rows: `{payload['lower_wm_ingestion_row_count']}`",
        "",
        "## Runtime Truth",
        "",
        f"- ROS2 build attempted: `{str(payload['ros2_build_attempted']).lower()}`",
        f"- ROS2 build succeeded: `{str(payload['ros2_build_succeeded']).lower()}`",
        "- Generated message import succeeded: "
        f"`{str(payload['generated_message_import_succeeded']).lower()}`",
        "- SDK2 header compile succeeded: "
        f"`{str(payload['sdk2_header_compile_succeeded']).lower()}`",
        "- SDK2 CMake build attempted: "
        f"`{str(payload['sdk2_cmake_build_attempted']).lower()}`",
        "- SDK2 CMake build succeeded: "
        f"`{str(payload['sdk2_cmake_build_succeeded']).lower()}`",
        "- Minimal MuJoCo headless step executed: "
        f"`{str(payload['minimal_mujoco_headless_step_executed']).lower()}`",
        "- G1 MuJoCo model stress succeeded: "
        f"`{str(payload['g1_mujoco_model_stress_succeeded']).lower()}`",
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
            "This tranche executes local CPU-safe validation and joins only. It",
            "does not publish ROS2/DDS messages, write Unitree SDK2 commands,",
            "invoke G1Pilot, execute hardware, grant live policy control, train",
            "weights, mutate reward math, expand Phase 7 authority, or promote",
            "any model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cpu_august_gap_tranche(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    phase35_bipedal_readiness_dir: str | Path = (
        DEFAULT_PHASE35_BIPEDAL_READINESS_DIR
    ),
    phase4_downstream_controller_dir: str | Path = (
        DEFAULT_PHASE4_DOWNSTREAM_CONTROLLER_DIR
    ),
    phase4_unitree_local_harness_dir: str | Path = (
        DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR
    ),
    phase4_unitree_runtime_bridge_dir: str | Path = (
        DEFAULT_PHASE4_UNITREE_RUNTIME_BRIDGE_DIR
    ),
    phase4_unitree_blocker_stress_probe_dir: str | Path = (
        DEFAULT_PHASE4_UNITREE_BLOCKER_STRESS_PROBE_DIR
    ),
    local_roots: Mapping[str, str | Path] | None = None,
    sample_count: int = 12,
    timing_iterations: int = 32,
    mujoco_steps: int = 5,
    stress_steps: int = 100,
    allow_build_attempt: bool = True,
    build_timeout_s: float = 120.0,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    root_map = {**default_unitree_local_roots(), **dict(local_roots or {})}
    validation_receipts = build_unitree_ros2_sdk2_build_message_validation_receipts(
        local_roots=root_map,
        scratch_dir=output / "build_probe_scratch",
        allow_build_attempt=allow_build_attempt,
        build_timeout_s=build_timeout_s,
    )
    local_payload = run_prepare_phase4_unitree_local_harnesses(
        output_dir=phase4_unitree_local_harness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        local_roots=root_map,
        sample_count=sample_count,
        timing_iterations=timing_iterations,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    runtime_payload = run_prepare_phase4_unitree_runtime_evidence_bridge(
        output_dir=phase4_unitree_runtime_bridge_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        phase4_unitree_local_harness_dir=phase4_unitree_local_harness_dir,
        local_roots=root_map,
        mujoco_steps=mujoco_steps,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    blocker_payload = run_probe_phase4_unitree_blockers(
        output_dir=phase4_unitree_blocker_stress_probe_dir,
        local_roots=root_map,
        stress_steps=stress_steps,
    )
    artifact_refs = {
        **refs,
        "phase4_unitree_local_harness_report_path": str(
            Path(phase4_unitree_local_harness_dir)
            / "phase4_unitree_local_harness_report_v1.json"
        ),
        "phase4_unitree_runtime_bridge_report_path": str(
            Path(phase4_unitree_runtime_bridge_dir)
            / "phase4_unitree_runtime_evidence_bridge_report_v1.json"
        ),
        "phase4_unitree_blocker_stress_probe_report_path": str(
            Path(phase4_unitree_blocker_stress_probe_dir)
            / "phase4_unitree_blocker_stress_probe_report_v1.json"
        ),
        "local_roots": {key: str(value) for key, value in root_map.items()},
    }
    (
        events,
        decisions,
        episodes,
        steps,
        windows,
        join_rows,
        ingestion_rows,
    ) = build_cpu_august_gap_event_replay_lower_wm_surfaces(
        validation_receipts=validation_receipts,
        phase4_unitree_local_harness_dir=phase4_unitree_local_harness_dir,
        phase4_unitree_runtime_bridge_dir=phase4_unitree_runtime_bridge_dir,
        phase4_unitree_blocker_stress_probe_dir=phase4_unitree_blocker_stress_probe_dir,
        artifact_refs=artifact_refs,
    )
    report = build_cpu_august_gap_execution_report(
        validation_receipts=validation_receipts,
        phase4_unitree_local_harness_report=local_payload,
        phase4_unitree_runtime_bridge_report=runtime_payload,
        phase4_unitree_blocker_stress_probe_report=blocker_payload,
        events=events,
        decisions=decisions,
        episodes=episodes,
        steps=steps,
        windows=windows,
        join_rows=join_rows,
        ingestion_rows=ingestion_rows,
        artifact_refs=artifact_refs,
    )
    saved_refs = save_cpu_august_gap_execution(
        output,
        report=report,
        validation_receipts=validation_receipts,
        events=events,
        decisions=decisions,
        episodes=episodes,
        steps=steps,
        windows=windows,
        join_rows=join_rows,
        ingestion_rows=ingestion_rows,
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
        "--phase4-unitree-local-harness-dir",
        default=str(DEFAULT_PHASE4_UNITREE_LOCAL_HARNESS_DIR),
    )
    parser.add_argument(
        "--phase4-unitree-runtime-bridge-dir",
        default=str(DEFAULT_PHASE4_UNITREE_RUNTIME_BRIDGE_DIR),
    )
    parser.add_argument(
        "--phase4-unitree-blocker-stress-probe-dir",
        default=str(DEFAULT_PHASE4_UNITREE_BLOCKER_STRESS_PROBE_DIR),
    )
    parser.add_argument(
        "--local-root",
        action="append",
        help="Override a Unitree dependency root as KEY=PATH. May be repeated.",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--timing-iterations", type=int, default=32)
    parser.add_argument("--mujoco-steps", type=int, default=5)
    parser.add_argument("--stress-steps", type=int, default=100)
    parser.add_argument("--build-timeout-s", type=float, default=120.0)
    parser.add_argument("--no-build-attempt", action="store_true")
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_cpu_august_gap_tranche(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=args.phase4_downstream_controller_dir,
        phase4_unitree_local_harness_dir=args.phase4_unitree_local_harness_dir,
        phase4_unitree_runtime_bridge_dir=args.phase4_unitree_runtime_bridge_dir,
        phase4_unitree_blocker_stress_probe_dir=(
            args.phase4_unitree_blocker_stress_probe_dir
        ),
        local_roots=_parse_local_root_overrides(args.local_root),
        sample_count=args.sample_count,
        timing_iterations=args.timing_iterations,
        mujoco_steps=args.mujoco_steps,
        stress_steps=args.stress_steps,
        allow_build_attempt=not args.no_build_attempt,
        build_timeout_s=args.build_timeout_s,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["cpu_august_gap_tranche_complete"]
        and not payload["ros2_publish_attempted"]
        and not payload["unitree_sdk2_write_enabled"]
        and not payload["g1pilot_runtime_invoked"]
        and not payload["hardware_executed"]
        and not payload["live_policy_control"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["gpu_training_executed"]
        and not payload["reward_math_mutation"]
        and not payload["phase7_authority_granted"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
