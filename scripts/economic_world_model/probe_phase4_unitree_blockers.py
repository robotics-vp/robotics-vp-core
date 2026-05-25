#!/usr/bin/env python3
"""Run Phase 4 Unitree blocker stress probes and emit receipts."""

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

from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase4_unitree_blocker_stress_probes,
    default_unitree_local_roots,
    save_phase4_unitree_blocker_stress_probes,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase4_unitree_blocker_stress_probes"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(
            output / "phase4_unitree_blocker_stress_probe_report_v1.json"
        ),
        "probe_receipts_path": str(
            output / "unitree_blocker_stress_probe_receipts_v1.jsonl"
        ),
        "mujoco_model_stress_receipts_path": str(
            output / "unitree_mujoco_model_stress_receipts_v1.jsonl"
        ),
        "markdown_path": str(
            output / "phase4_unitree_blocker_stress_probes_v1.md"
        ),
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
        "# Phase 4 Unitree Blocker Stress Probes",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local Phase 4 probe expansion complete: "
        f"`{str(payload['local_phase4_probe_expansion_complete']).lower()}`",
        "- All local probe attempts complete: "
        f"`{str(payload['all_local_probe_attempts_complete']).lower()}`",
        f"- Probe receipts: `{payload['probe_receipt_count']}`",
        f"- Succeeded probes: `{payload['succeeded_probe_count']}`",
        f"- Blocked probes: `{payload['blocked_probe_count']}`",
        "- MuJoCo model stress successes: "
        f"`{payload['mujoco_model_stress_success_count']}` / "
        f"`{payload['mujoco_model_stress_receipt_count']}`",
        "- G1Pilot static surface succeeded: "
        f"`{str(payload['g1pilot_static_surface_succeeded']).lower()}`",
        "- CycloneDDS header compile succeeded: "
        f"`{str(payload['cyclonedds_header_compile_succeeded']).lower()}`",
        "- Unitree SDK2 header compile succeeded: "
        f"`{str(payload['unitree_sdk2_header_compile_succeeded']).lower()}`",
        "- ROS2 runtime available: "
        f"`{str(payload['ros2_runtime_available']).lower()}`",
        "- rosbag2/MCAP modules available: "
        f"`{str(payload['trace_import_modules_available']).lower()}`",
        "- Policy checkpoint visible: "
        f"`{str(payload['policy_checkpoint_visible']).lower()}`",
        "- IsaacLab task surface visible: "
        f"`{str(payload['isaaclab_task_surface_visible']).lower()}`",
        "- LeRobot adapter surface visible: "
        f"`{str(payload['lerobot_adapter_surface_visible']).lower()}`",
        "",
        "## Unlocked Local Follow-Ups",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["unlocked_local_followups"])
    lines.extend(["", "## Remaining Evidence Blockers", ""])
    lines.extend(f"- `{item}`" for item in payload["remaining_evidence_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These probes are static, compile-only, import-only, or no-policy",
            "MuJoCo checks. They do not publish ROS2/DDS messages, write SDK2",
            "commands, invoke G1Pilot, run hardware, execute policy control,",
            "train, write weights, mutate reward math, or promote authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe_phase4_unitree_blockers(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    local_roots: Mapping[str, str | Path] | None = None,
    stress_steps: int = 100,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    root_map = {**default_unitree_local_roots(), **dict(local_roots or {})}
    artifact_refs = {
        **refs,
        "local_roots": {key: str(value) for key, value in root_map.items()},
        "stress_steps": int(stress_steps),
    }
    report, probe_receipts, mujoco_receipts = (
        build_phase4_unitree_blocker_stress_probes(
            local_roots=root_map,
            stress_steps=stress_steps,
            artifact_refs=artifact_refs,
        )
    )
    saved_refs = save_phase4_unitree_blocker_stress_probes(
        output,
        report=report,
        probe_receipts=probe_receipts,
        mujoco_receipts=mujoco_receipts,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stress-steps", type=int, default=100)
    parser.add_argument(
        "--local-root",
        action="append",
        help="Override a Unitree dependency root as KEY=PATH. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_probe_phase4_unitree_blockers(
        output_dir=args.output_dir,
        local_roots=_parse_local_root_overrides(args.local_root),
        stress_steps=args.stress_steps,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_phase4_probe_expansion_complete"]
        and payload["all_local_probe_attempts_complete"]
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
