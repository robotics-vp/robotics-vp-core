#!/usr/bin/env python3
"""Inspect Phase-1 runtime roots, layouts, and policy contracts for sim/synth backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.world_model.sim_synth_physics.adapters.holosoma_deployment import (
    build_holosoma_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_binding import (
    build_holosoma_runtime_binding,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_pack import (
    build_holosoma_runtime_pack,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_deployment import (
    build_isaac_unitree_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_binding import (
    build_isaac_unitree_runtime_binding,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_pack import (
    build_isaac_unitree_runtime_pack,
)
from src.world_model.sim_synth_physics.asset_manifest import (
    normalize_robot_asset_manifest,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import (
    describe_holosoma_runtime_targets,
    describe_isaac_runtime_targets,
)


def _load_mapping(path: str | None) -> Optional[Mapping[str, Any]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected mapping payload in {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Phase-1 backend runtime layouts")
    parser.add_argument("--embodiment-context", help="Optional JSON mapping with embodiment context")
    parser.add_argument(
        "--output-path",
        default="artifacts/sim_synth_runtime_layout_scan.json",
        help="Path to write the scan summary",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    embodiment_context = _load_mapping(args.embodiment_context) or {}
    isaac_runtime_targets = describe_isaac_runtime_targets(embodiment_context)
    isaac_runtime_layouts = describe_isaac_runtime_layouts(embodiment_context)
    isaac_policy_contract = describe_isaac_policy_contract(embodiment_context)
    normalized_asset_manifest = normalize_robot_asset_manifest(embodiment_context)
    isaac_deployment_contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=isaac_runtime_targets,
        runtime_layout_contract=isaac_runtime_layouts,
        policy_contract=isaac_policy_contract,
        normalized_asset_manifest=normalized_asset_manifest,
    )
    isaac_runtime_pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=isaac_runtime_targets,
        runtime_layout_contract=isaac_runtime_layouts,
        policy_contract=isaac_policy_contract,
        deployment_contract=isaac_deployment_contract,
        normalized_robot_asset_manifest=normalized_asset_manifest,
    )
    isaac_launch_specs = [
        {
            "profile_id": "unitree_sim_isaaclab",
            "root": isaac_runtime_pack.get("profile_root", ""),
            "command": (
                "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py "
                "--task scan_runtime --policy ${UNITREE_POLICY_REF} --headless"
            ),
        }
    ]
    isaac_runtime_binding = build_isaac_unitree_runtime_binding(
        task_id="scan_runtime",
        explicit_policy_ref=str(isaac_policy_contract.get("policy_ref", "") or ""),
        preferred_profile=str(isaac_runtime_pack.get("preferred_profile", "") or ""),
        launch_specs=isaac_launch_specs,
        runtime_target_contract=isaac_runtime_targets,
        policy_contract=isaac_policy_contract,
        deployment_contract=isaac_deployment_contract,
        upstream_runtime_pack=isaac_runtime_pack,
    )
    holosoma_runtime_targets = describe_holosoma_runtime_targets(embodiment_context)
    holosoma_runtime_layouts = describe_holosoma_runtime_layouts(embodiment_context)
    holosoma_policy_contract = describe_holosoma_policy_contract(embodiment_context)
    holosoma_deployment_contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=holosoma_runtime_targets,
        runtime_layout_contract=holosoma_runtime_layouts,
        policy_contract=holosoma_policy_contract,
    )
    holosoma_runtime_pack = build_holosoma_runtime_pack(
        runtime_target_contract=holosoma_runtime_targets,
        runtime_layout_contract=holosoma_runtime_layouts,
        policy_contract=holosoma_policy_contract,
        deployment_contract=holosoma_deployment_contract,
        embodiment_context=embodiment_context,
    )
    holosoma_launch_specs = [
        {
            "profile_id": "holosoma_repo",
            "root": holosoma_runtime_pack.get("profile_root", ""),
            "command": "python -m holosoma.eval --task-id scan_runtime --policy ${HOLOSOMA_POLICY_REF}",
        },
        {
            "profile_id": "holosoma_motion_bank",
            "root": holosoma_runtime_pack.get("profile_root", ""),
            "command": "python scripts/local_holosoma_smoke.py --task-id scan_runtime --episodes 1",
        },
    ]
    holosoma_runtime_binding = build_holosoma_runtime_binding(
        task_id="scan_runtime",
        explicit_policy_ref=str(holosoma_policy_contract.get("policy_ref", "") or ""),
        preferred_profile=str(holosoma_runtime_pack.get("preferred_profile", "") or ""),
        launch_specs=holosoma_launch_specs,
        runtime_target_contract=holosoma_runtime_targets,
        policy_contract=holosoma_policy_contract,
        deployment_contract=holosoma_deployment_contract,
        upstream_runtime_pack=holosoma_runtime_pack,
    )
    summary = {
        "version": "phase1_runtime_layout_scan_v1",
        "isaac_runtime_targets": isaac_runtime_targets,
        "isaac_runtime_layouts": isaac_runtime_layouts,
        "isaac_policy_contract": isaac_policy_contract,
        "isaac_deployment_contract": isaac_deployment_contract,
        "isaac_upstream_runtime_pack": isaac_runtime_pack,
        "isaac_runtime_binding": isaac_runtime_binding,
        "holosoma_runtime_targets": holosoma_runtime_targets,
        "holosoma_runtime_layouts": holosoma_runtime_layouts,
        "holosoma_policy_contract": holosoma_policy_contract,
        "holosoma_deployment_contract": holosoma_deployment_contract,
        "holosoma_upstream_runtime_pack": holosoma_runtime_pack,
        "holosoma_runtime_binding": holosoma_runtime_binding,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_path": str(output_path.resolve())}


if __name__ == "__main__":
    main()
