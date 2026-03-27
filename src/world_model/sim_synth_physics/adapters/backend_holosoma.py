"""Holosoma execution binding helpers for the sim/synth/physics WM."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Mapping

from src.motor_backend.holosoma_backend import HOLOSOMA_TASK_MAP

from ..common import mapping


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
    required_assets = [
        "humanoid_embodiment_context",
        "motion_clip_datapacks",
        "whole_body_reward_overlay",
        "holosoma_runtime",
    ]
    available_assets = ([] if not active_embodiments else ["humanoid_embodiment_context"]) + (
        ["holosoma_runtime"] if available else []
    )
    missing_assets = [asset for asset in required_assets if asset not in available_assets]
    if available and not missing_assets:
        binding_status = "ready"
    elif active_embodiments:
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
            "embodiment_context": mapping(embodiment_context),
        },
    }


__all__ = ["build_holosoma_backend_binding"]
