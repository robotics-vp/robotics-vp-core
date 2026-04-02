"""Holosoma execution binding helpers for the sim/synth/physics WM."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Mapping

from src.motor_backend.holosoma_backend import HOLOSOMA_TASK_MAP

from ..common import mapping
from .holosoma_deployment import build_holosoma_deployment_contract
from .holosoma_runtime_pack import build_holosoma_runtime_pack
from ..runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
)
from ..runtime_targets import describe_holosoma_runtime_targets


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def build_holosoma_backend_binding(
    *,
    physics_context: Mapping[str, Any],
    adaptation_policy: Mapping[str, Any],
    embodiment_context: Mapping[str, Any],
) -> Dict[str, Any]:
    available = _has_module("holosoma")
    active_embodiments = list(
        embodiment_context.get("active_embodiments")
        or embodiment_context.get("target_embodiments")
        or embodiment_context.get("robot_families")
        or []
    )
    motion_sources = list(embodiment_context.get("motion_clip_datapacks") or [])
    motion_clips = list(
        embodiment_context.get("motion_clips")
        or embodiment_context.get("motion_clip_paths")
        or []
    )
    retargeting_contract = mapping(
        embodiment_context.get("retargeting_contract")
        or embodiment_context.get("whole_body_retargeting")
    )
    reward_overlay = mapping(embodiment_context.get("whole_body_reward_overlay"))
    runtime_target_contract = describe_holosoma_runtime_targets(embodiment_context)
    runtime_layout_contract = describe_holosoma_runtime_layouts(embodiment_context)
    policy_contract = describe_holosoma_policy_contract(embodiment_context)
    deployment_contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
    )
    upstream_runtime_pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )
    required_assets = ["humanoid_embodiment_context", "holosoma_runtime"]
    optional_assets = [
        "motion_source_bundle",
        "whole_body_retargeting_contract",
        "whole_body_reward_overlay",
        "policy_checkpoint",
    ]
    available_assets = ([] if not active_embodiments else ["humanoid_embodiment_context"])
    if motion_sources or motion_clips:
        available_assets.append("motion_source_bundle")
    if retargeting_contract:
        available_assets.append("whole_body_retargeting_contract")
    if reward_overlay:
        available_assets.append("whole_body_reward_overlay")
    if policy_contract.get("policy_ready", False):
        available_assets.append("policy_checkpoint")
    if available:
        available_assets.append("holosoma_runtime")
    missing_assets = [asset for asset in required_assets if asset not in available_assets]
    ready_modes = list(deployment_contract.get("ready_modes") or [])
    if available and ready_modes and not missing_assets:
        binding_status = "runtime_ready"
    elif ready_modes or active_embodiments:
        binding_status = "shadow_ready"
    else:
        binding_status = "assets_missing"
    return {
        "binding_name": "holosoma_execution_binding_v1",
        "binding_status": binding_status,
        "executor_entrypoint": "src.motor_backend.factory:make_motor_backend",
        "executor_kind": "motor_backend_factory",
        "observation_adapter_entrypoint": "",
        "supports_training": True,
        "supports_evaluation": True,
        "supports_deploy_handle": True,
        "target_runtime_stack": ["holosoma", "isaacgym", "isaacsim"],
        "asset_profile": "unitree_humanoid_shadow_assets",
        "required_assets": required_assets,
        "available_assets": available_assets,
        "missing_assets": missing_assets,
        "metadata": {
            "engine_type": "holosoma",
            "fidelity_tier": str(physics_context.get("fidelity_tier", "")),
            "domain_randomization_profile": str(
                adaptation_policy.get("domain_randomization_profile", "")
            ),
            "system_identification_profile": str(
                adaptation_policy.get("system_identification_profile", "")
            ),
            "shadow_backend_available": True,
            "concrete_runtime_available": available,
            "task_presets": sorted(HOLOSOMA_TASK_MAP.keys()),
            "motion_source_count": len(motion_sources) + len(motion_clips),
            "retargeting_contract_present": bool(retargeting_contract),
            "reward_overlay_present": bool(reward_overlay),
            "runtime_target_contract": runtime_target_contract,
            "runtime_layout_contract": runtime_layout_contract,
            "policy_contract": policy_contract,
            "deployment_contract": deployment_contract,
            "upstream_runtime_pack": upstream_runtime_pack,
            "optional_assets": optional_assets,
            "embodiment_context": mapping(embodiment_context),
        },
    }


__all__ = ["build_holosoma_backend_binding"]
