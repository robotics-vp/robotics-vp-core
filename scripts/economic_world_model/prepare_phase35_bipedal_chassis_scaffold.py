#!/usr/bin/env python3
"""Materialize Phase 3.5 canonical bipedal chassis scaffold artifacts."""

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

from src.world_model.embodiment_actuation import (  # noqa: E402
    build_bipedal_chassis_scaffold,
    save_bipedal_chassis_scaffold,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase35_bipedal_chassis_scaffold"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "bipedal_chassis_scaffold_report_v1.json"),
        "chassis_profile_path": str(output / "humanoid_chassis_profile_v1.json"),
        "frame_tree_path": str(output / "humanoid_frame_tree_v1.json"),
        "frames_path": str(output / "limb_coordinate_frames_v1.jsonl"),
        "joint_limits_path": str(output / "joint_limit_envelopes_v1.jsonl"),
        "observation_schema_path": str(
            output / "whole_body_observation_schema_v1.json"
        ),
        "action_schema_path": str(output / "whole_body_action_schema_v1.json"),
        "support_states_path": str(output / "bipedal_support_states_v1.jsonl"),
        "balance_receipts_path": str(output / "balance_envelope_receipts_v1.jsonl"),
        "markdown_path": str(output / "bipedal_chassis_scaffold_report_v1.md"),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3.5 Bipedal Chassis Scaffold",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Chassis: `{payload['chassis_id']}`",
        f"- Controlled joints: `{payload['controlled_joint_count']}`",
        f"- Frames: `{payload['frame_count']}`",
        f"- Joint-limit envelopes: `{payload['joint_limit_envelope_count']}`",
        f"- Support states: `{payload['support_state_count']}`",
        f"- Balance receipts: `{payload['balance_receipt_count']}`",
        "- Local structural scaffold complete: "
        f"`{str(payload['local_structural_scaffold_complete']).lower()}`",
        "",
        "## Present Local Surfaces",
        "",
        f"- Canonical bipedal chassis: "
        f"`{str(payload['canonical_bipedal_chassis_present']).lower()}`",
        f"- Limb frame tree: "
        f"`{str(payload['limb_frame_tree_present']).lower()}`",
        f"- Joint-limit envelope: "
        f"`{str(payload['joint_limit_envelope_present']).lower()}`",
        f"- Whole-body observation schema: "
        f"`{str(payload['whole_body_observation_schema_present']).lower()}`",
        f"- Whole-body action schema: "
        f"`{str(payload['whole_body_action_schema_present']).lower()}`",
        f"- Balance envelope: "
        f"`{str(payload['balance_envelope_present']).lower()}`",
        "",
        "## Boundary",
        "",
        "These artifacts are local structure only. Numeric joint envelopes are",
        "planning envelopes, not hardware-calibrated safety limits. This CLI does",
        "not run Unitree sim, hardware, providers, training, weight writes, live",
        "policy control, reward mutation, or promotion.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase35_bipedal_chassis_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    variant: str = "g1_29dof",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    (
        report,
        chassis,
        frame_tree,
        frames,
        joint_limits,
        observation_schema,
        action_schema,
        support_states,
        balance_receipts,
    ) = build_bipedal_chassis_scaffold(variant=variant, artifact_refs=refs)
    saved_refs = save_bipedal_chassis_scaffold(
        output,
        report=report,
        chassis=chassis,
        frame_tree=frame_tree,
        frames=frames,
        joint_limits=joint_limits,
        observation_schema=observation_schema,
        action_schema=action_schema,
        support_states=support_states,
        balance_receipts=balance_receipts,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(Path(refs["markdown_path"]), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variant", default="g1_29dof")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase35_bipedal_chassis_scaffold(
        output_dir=args.output_dir,
        variant=args.variant,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_structural_scaffold_complete"]
        and payload["controlled_joint_count"] >= 21
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
