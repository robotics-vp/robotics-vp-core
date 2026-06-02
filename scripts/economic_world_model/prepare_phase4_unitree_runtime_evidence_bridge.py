#!/usr/bin/env python3
"""Materialize Phase 4 Unitree runtime-evidence bridge artifacts."""

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
from src.world_model.embodiment_actuation import load_joint_limit_envelopes  # noqa: E402
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase4_unitree_runtime_evidence_bridge,
    default_unitree_local_roots,
    save_phase4_unitree_runtime_evidence_bridge,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_unitree_runtime_evidence_bridge"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(
            output / "phase4_unitree_runtime_evidence_bridge_report_v1.json"
        ),
        "ros2_runtime_readiness_receipts_path": str(
            output / "unitree_ros2_runtime_readiness_receipts_v1.jsonl"
        ),
        "mujoco_headless_step_receipts_path": str(
            output / "unitree_mujoco_headless_step_receipts_v1.jsonl"
        ),
        "mujoco_headless_trace_rows_path": str(
            output / "unitree_mujoco_headless_trace_rows_v1.jsonl"
        ),
        "trace_import_adapter_receipts_path": str(
            output / "unitree_trace_import_adapter_receipts_v1.jsonl"
        ),
        "safety_envelope_expansion_receipts_path": str(
            output / "unitree_safety_envelope_expansion_receipts_v1.jsonl"
        ),
        "operator_recovery_scenarios_path": str(
            output / "unitree_operator_recovery_scenarios_v1.jsonl"
        ),
        "operator_recovery_drill_transitions_path": str(
            output / "unitree_operator_recovery_drill_transitions_v1.jsonl"
        ),
        "operator_recovery_drill_receipts_path": str(
            output / "unitree_operator_recovery_drill_receipts_v1.jsonl"
        ),
        "markdown_path": str(
            output / "phase4_unitree_runtime_evidence_bridge_v1.md"
        ),
    }


def _input_paths(
    *,
    bipedal_chassis_dir: Path,
    phase4_unitree_local_harness_dir: Path,
) -> dict[str, Path]:
    return {
        "joint_limits": bipedal_chassis_dir / "joint_limit_envelopes_v1.jsonl",
        "local_harness_report": phase4_unitree_local_harness_dir
        / "phase4_unitree_local_harness_report_v1.json",
        "low_state_traces": phase4_unitree_local_harness_dir
        / "unitree_low_state_traces_v1.jsonl",
        "imu_traces": phase4_unitree_local_harness_dir / "unitree_imu_traces_v1.jsonl",
        "wireless_estop_traces": phase4_unitree_local_harness_dir
        / "unitree_wireless_estop_traces_v1.jsonl",
        "contact_traces": phase4_unitree_local_harness_dir
        / "unitree_contact_traces_v1.jsonl",
    }


def _ensure_inputs(
    *,
    bipedal_chassis_dir: Path,
    phase35_bipedal_readiness_dir: Path,
    phase4_downstream_controller_dir: Path,
    phase4_unitree_local_harness_dir: Path,
    local_roots: Mapping[str, str | Path] | None,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _input_paths(
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase4_unitree_local_harness_dir=phase4_unitree_local_harness_dir,
    )
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Unitree runtime bridge inputs: " + ", ".join(missing)
        )
    run_prepare_phase4_unitree_local_harnesses(
        output_dir=phase4_unitree_local_harness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        local_roots=local_roots,
        run_dependencies_if_missing=True,
    )
    paths = _input_paths(
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase4_unitree_local_harness_dir=phase4_unitree_local_harness_dir,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Unitree runtime bridge dependency builders did not materialize: "
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
        "# Phase 4 Unitree Runtime Evidence Bridge",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local runtime evidence bridge complete: "
        f"`{str(payload['local_runtime_evidence_bridge_complete']).lower()}`",
        "- ROS2 runtime preflight complete: "
        f"`{str(payload['ros2_runtime_preflight_complete']).lower()}`",
        "- MuJoCo headless trace attempt complete: "
        f"`{str(payload['mujoco_headless_trace_attempt_complete']).lower()}`",
        "- Minimal MuJoCo headless step executed: "
        f"`{str(payload['minimal_mujoco_headless_step_executed']).lower()}`",
        "- Trace ingestion adapters complete: "
        f"`{str(payload['trace_ingestion_adapters_complete']).lower()}`",
        "- Safety envelope expansion complete: "
        f"`{str(payload['safety_envelope_expansion_complete']).lower()}`",
        "- Operator drill runner complete: "
        f"`{str(payload['operator_drill_runner_complete']).lower()}`",
        f"- ROS2 readiness receipts: "
        f"`{payload['ros2_runtime_readiness_receipt_count']}`",
        f"- MuJoCo trace rows: `{payload['mujoco_trace_row_count']}`",
        f"- Trace adapter receipts: "
        f"`{payload['trace_import_adapter_receipt_count']}`",
        "- Trace unavailable receipts: "
        f"`{payload['trace_import_unavailable_receipt_count']}`",
        "- Trace fixture-shape-only receipts: "
        f"`{payload['trace_fixture_shape_only_count']}`",
        "- rosbag2 real import claimed: "
        f"`{str(payload['rosbag2_real_import_claimed']).lower()}`",
        "- MCAP real import claimed: "
        f"`{str(payload['mcap_real_import_claimed']).lower()}`",
        f"- Safety envelope receipts: "
        f"`{payload['safety_envelope_expansion_receipt_count']}`",
        f"- Operator drill receipts: "
        f"`{payload['operator_recovery_drill_receipt_count']}`",
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
            "This bridge may execute a local no-policy MuJoCo headless step if",
            "the local host supports it. It does not publish ROS2/DDS messages,",
            "write Unitree SDK2 commands, invoke G1Pilot, run hardware, grant",
            "live policy control, train weights, mutate reward math, or promote",
            "authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase4_unitree_runtime_evidence_bridge(
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
    local_roots: Mapping[str, str | Path] | None = None,
    mujoco_steps: int = 5,
    rosbag2_path: str | Path | None = None,
    mcap_path: str | Path | None = None,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    root_map = {**default_unitree_local_roots(), **dict(local_roots or {})}
    input_paths = _ensure_inputs(
        bipedal_chassis_dir=Path(bipedal_chassis_dir),
        phase35_bipedal_readiness_dir=Path(phase35_bipedal_readiness_dir),
        phase4_downstream_controller_dir=Path(phase4_downstream_controller_dir),
        phase4_unitree_local_harness_dir=Path(phase4_unitree_local_harness_dir),
        local_roots=root_map,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    artifact_refs = {
        **refs,
        **{f"input_{key}_path": str(path) for key, path in input_paths.items()},
        "local_roots": {key: str(value) for key, value in root_map.items()},
        "rosbag2_path": str(rosbag2_path or ""),
        "mcap_path": str(mcap_path or ""),
    }
    (
        report,
        ros2_receipts,
        mujoco_receipt,
        mujoco_rows,
        trace_adapters,
        safety_receipts,
        scenarios,
        transitions,
        drill_receipts,
    ) = build_phase4_unitree_runtime_evidence_bridge(
        trace_dir=Path(phase4_unitree_local_harness_dir),
        joint_limits=load_joint_limit_envelopes(input_paths["joint_limits"]),
        output_dir=output,
        local_roots=root_map,
        mujoco_steps=mujoco_steps,
        rosbag2_path=rosbag2_path,
        mcap_path=mcap_path,
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase4_unitree_runtime_evidence_bridge(
        output,
        report=report,
        ros2_receipts=ros2_receipts,
        mujoco_receipt=mujoco_receipt,
        mujoco_rows=mujoco_rows,
        trace_adapters=trace_adapters,
        safety_receipts=safety_receipts,
        scenarios=scenarios,
        transitions=transitions,
        drill_receipts=drill_receipts,
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
        "--local-root",
        action="append",
        help="Override a Unitree dependency root as KEY=PATH. May be repeated.",
    )
    parser.add_argument("--mujoco-steps", type=int, default=5)
    parser.add_argument("--rosbag2-path", default="")
    parser.add_argument("--mcap-path", default="")
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase4_unitree_runtime_evidence_bridge(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=args.phase35_bipedal_readiness_dir,
        phase4_downstream_controller_dir=args.phase4_downstream_controller_dir,
        phase4_unitree_local_harness_dir=args.phase4_unitree_local_harness_dir,
        local_roots=_parse_local_root_overrides(args.local_root),
        mujoco_steps=args.mujoco_steps,
        rosbag2_path=args.rosbag2_path or None,
        mcap_path=args.mcap_path or None,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_runtime_evidence_bridge_complete"]
        and payload["ros2_runtime_preflight_complete"]
        and payload["mujoco_headless_trace_attempt_complete"]
        and payload["trace_ingestion_adapters_complete"]
        and payload["safety_envelope_expansion_complete"]
        and payload["operator_drill_runner_complete"]
        and not payload["ros2_publish_attempted"]
        and not payload["unitree_sdk2_write_enabled"]
        and not payload["g1pilot_runtime_invoked"]
        and not payload["hardware_executed"]
        and not payload["live_policy_control"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
