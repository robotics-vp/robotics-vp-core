#!/usr/bin/env python3
"""Materialize Phase 3.5 humanoid capacity and environment refit artifacts."""

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
    build_phase35_humanoid_refit,
    save_phase35_humanoid_refit,
)
from src.world_model.embodiment_actuation import (  # noqa: E402
    build_bipedal_chassis_scaffold,
    save_bipedal_chassis_scaffold,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase35_humanoid_capacity_env_refit"
)
DEFAULT_BIPEDAL_CHASSIS_DIR = Path(
    "artifacts/economic_world_model/phase35_bipedal_chassis_scaffold"
)


def _artifact_refs(output: Path) -> dict[str, str]:
    return {
        "report_path": str(output / "humanoid_phase35_refit_report_v1.json"),
        "capacity_bands_path": str(
            output / "humanoid_phase35_capacity_band_contracts_v1.jsonl"
        ),
        "schema_deltas_path": str(
            output / "humanoid_phase35_schema_delta_contracts_v1.jsonl"
        ),
        "env_taxonomy_path": str(
            output / "humanoid_phase35_env_taxonomy_receipts_v1.jsonl"
        ),
        "benchmarks_path": str(
            output / "humanoid_phase35_benchmark_taxonomy_v1.jsonl"
        ),
        "markdown_path": str(output / "humanoid_phase35_refit_report_v1.md"),
    }


def _bipedal_chassis_refs(output: Path) -> dict[str, str]:
    return {
        "bipedal_chassis_report_path": str(
            output / "bipedal_chassis_scaffold_report_v1.json"
        ),
        "bipedal_chassis_profile_path": str(
            output / "humanoid_chassis_profile_v1.json"
        ),
        "bipedal_frame_tree_path": str(output / "humanoid_frame_tree_v1.json"),
        "bipedal_frames_path": str(output / "limb_coordinate_frames_v1.jsonl"),
        "bipedal_joint_limits_path": str(output / "joint_limit_envelopes_v1.jsonl"),
        "bipedal_observation_schema_path": str(
            output / "whole_body_observation_schema_v1.json"
        ),
        "bipedal_action_schema_path": str(output / "whole_body_action_schema_v1.json"),
        "bipedal_support_states_path": str(output / "bipedal_support_states_v1.jsonl"),
        "bipedal_balance_receipts_path": str(
            output / "balance_envelope_receipts_v1.jsonl"
        ),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3.5 Humanoid Capacity and Environment Refit Artifacts",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local structural refit complete: "
        f"`{str(payload['local_structural_refit_complete']).lower()}`",
        f"- Capacity bands: `{payload['capacity_band_count']}`",
        f"- Schema deltas: `{payload['schema_delta_count']}`",
        f"- Environment taxonomy receipts: `{payload['env_taxonomy_count']}`",
        f"- Benchmark targets: `{payload['benchmark_target_count']}`",
        f"- Bipedal chassis report: `{payload['bipedal_chassis_report_id']}`",
        f"- Bipedal chassis joints: `{payload['bipedal_chassis_joint_count']}`",
        f"- Bipedal chassis frames: `{payload['bipedal_chassis_frame_count']}`",
        "- Bipedal joint-limit envelopes: "
        f"`{payload['bipedal_chassis_joint_limit_envelope_count']}`",
        f"- Bipedal balance receipts: `{payload['bipedal_balance_receipt_count']}`",
        "",
        "## Boundary",
        "",
        "These are local contracts and receipts only. They do not claim Unitree",
        "sim runtime, hardware execution, provider execution, training, weight",
        "writes, promotion, live policy control, or reward-math mutation.",
        "",
        "## Remaining blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prepare_phase35_humanoid_capacity_env_refit(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bipedal_chassis_dir: str | Path = DEFAULT_BIPEDAL_CHASSIS_DIR,
    variant: str = "g1_29dof",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    refs = _artifact_refs(output)
    chassis_output = Path(bipedal_chassis_dir)
    chassis_output.mkdir(parents=True, exist_ok=True)
    chassis_refs = _bipedal_chassis_refs(chassis_output)
    (
        chassis_report,
        chassis,
        frame_tree,
        frames,
        joint_limits,
        observation_schema,
        action_schema,
        support_states,
        balance_receipts,
    ) = build_bipedal_chassis_scaffold(
        variant=variant,
        artifact_refs=chassis_refs,
    )
    save_bipedal_chassis_scaffold(
        chassis_output,
        report=chassis_report,
        chassis=chassis,
        frame_tree=frame_tree,
        frames=frames,
        joint_limits=joint_limits,
        observation_schema=observation_schema,
        action_schema=action_schema,
        support_states=support_states,
        balance_receipts=balance_receipts,
    )
    refs = {**refs, **chassis_refs}
    report, capacity_bands, schema_deltas, env_receipts, benchmarks = (
        build_phase35_humanoid_refit(
            artifact_refs=refs,
            bipedal_chassis_report=chassis_report.to_dict(),
        )
    )
    saved_refs = save_phase35_humanoid_refit(
        output,
        report,
        capacity_bands,
        schema_deltas,
        env_receipts,
        benchmarks,
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
    parser.add_argument("--variant", default="g1_29dof")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_prepare_phase35_humanoid_capacity_env_refit(
        output_dir=args.output_dir,
        bipedal_chassis_dir=args.bipedal_chassis_dir,
        variant=args.variant,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_structural_refit_complete"]
        and payload["bipedal_chassis_local_scaffold_complete"]
        and payload["bipedal_chassis_joint_count"] >= 21
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
