"""Isaac / Unitree-target binding helpers for the sim/synth/physics WM."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Mapping

from ..asset_manifest import (
    available_assets_for_hardware_class,
    extract_robot_asset_manifest,
    missing_assets_for_hardware_class,
    normalize_robot_asset_manifest,
    required_assets_for_hardware_class,
    recommended_assets_for_hardware_class,
)
from ..common import mapping


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _asset_lists(embodiment_context: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    target_hardware_class = "unitree_g1_r1_class"
    required = required_assets_for_hardware_class(target_hardware_class)
    available = available_assets_for_hardware_class(target_hardware_class, embodiment_context)
    missing = missing_assets_for_hardware_class(target_hardware_class, embodiment_context)
    return required, available, missing


def build_isaac_backend_binding(
    *,
    physics_context: Mapping[str, Any],
    adaptation_policy: Mapping[str, Any],
    embodiment_context: Mapping[str, Any],
) -> Dict[str, Any]:
    isaacsim_available = _has_module("isaacsim") or _has_module("omni.isaac.kit")
    isaacgym_available = _has_module("isaacgym")
    isaaclab_backend_available = _has_module("src.motor_backend.workcell_isaaclab_backend")
    shadow_backend_available = True
    required_assets, available_assets, missing_assets = _asset_lists(embodiment_context)
    manifest = extract_robot_asset_manifest(embodiment_context)
    normalized_manifest = normalize_robot_asset_manifest(embodiment_context)
    adapter_ready = (
        shadow_backend_available
        or isaaclab_backend_available
        or isaacsim_available
        or isaacgym_available
    )
    binding_status = "integration_pending"
    if isaaclab_backend_available and not missing_assets:
        binding_status = "runtime_ready"
    elif isaaclab_backend_available:
        binding_status = "runtime_assets_missing"
    elif adapter_ready and not missing_assets:
        binding_status = "shadow_ready"
    elif adapter_ready:
        binding_status = "assets_missing"
    return {
        "binding_name": "isaac_unitree_execution_binding_v1",
        "binding_status": binding_status,
        "executor_entrypoint": (
            "src.motor_backend.factory:make_motor_backend"
            if isaaclab_backend_available
            else "src.envs.physics.backend_factory:make_backend"
        ),
        "executor_kind": (
            "motor_backend_factory" if isaaclab_backend_available else "physics_backend_factory"
        ),
        "observation_adapter_entrypoint": "src.env.isaac_adapter:IsaacAdapter",
        "supports_training": bool(adapter_ready),
        "supports_evaluation": bool(adapter_ready),
        "supports_deploy_handle": bool(isaaclab_backend_available),
        "target_runtime_stack": ["isaacsim", "isaacgym", "unitree_sdk2"],
        "asset_profile": "unitree_humanoid_shadow_assets",
        "required_assets": required_assets,
        "available_assets": available_assets,
        "missing_assets": missing_assets,
        "metadata": {
            "engine_type": "isaac",
            "fidelity_tier": str(physics_context.get("fidelity_tier", "")),
            "domain_randomization_profile": str(
                adaptation_policy.get("domain_randomization_profile", "")
            ),
            "system_identification_profile": str(
                adaptation_policy.get("system_identification_profile", "")
            ),
            "shadow_backend_available": shadow_backend_available,
            "isaacsim_available": isaacsim_available,
            "isaacgym_available": isaacgym_available,
            "isaaclab_backend_available": isaaclab_backend_available,
            "concrete_runtime_available": bool(
                isaaclab_backend_available or isaacsim_available or isaacgym_available
            ),
            "runtime_backend_name": "workcell_isaaclab" if isaaclab_backend_available else "",
            "required_asset_contracts": required_assets,
            "recommended_asset_contracts": recommended_assets_for_hardware_class(
                "unitree_g1_r1_class"
            ),
            "normalized_asset_manifest": normalized_manifest,
            "raw_asset_manifest": manifest,
            "embodiment_context": mapping(embodiment_context),
        },
    }


__all__ = ["build_isaac_backend_binding"]
