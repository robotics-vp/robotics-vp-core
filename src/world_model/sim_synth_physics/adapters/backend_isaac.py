"""Isaac / Unitree-target binding helpers for the sim/synth/physics WM."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Mapping

from ..common import mapping


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _asset_lists(embodiment_context: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    manifest = mapping(
        embodiment_context.get("robot_asset_manifest")
        or embodiment_context.get("asset_manifest")
        or embodiment_context.get("robot_assets")
    )
    available = [
        str(key)
        for key, value in sorted(manifest.items())
        if value not in (None, "", False, 0, [], {})
    ]
    required = [
        "unitree_robot_description",
        "joint_mapping_contract",
        "sensor_extrinsics",
        "actuator_latency_profile",
    ]
    missing = [asset for asset in required if asset not in available]
    return required, available, missing


def build_isaac_backend_binding(
    *,
    physics_context: Mapping[str, Any],
    adaptation_policy: Mapping[str, Any],
    embodiment_context: Mapping[str, Any],
) -> Dict[str, Any]:
    isaacsim_available = _has_module("isaacsim") or _has_module("omni.isaac.kit")
    isaacgym_available = _has_module("isaacgym")
    required_assets, available_assets, missing_assets = _asset_lists(embodiment_context)
    adapter_ready = isaacsim_available or isaacgym_available
    binding_status = "integration_pending"
    if adapter_ready and not missing_assets:
        binding_status = "shadow_ready"
    elif adapter_ready:
        binding_status = "assets_missing"
    return {
        "binding_name": "isaac_unitree_execution_binding_v1",
        "binding_status": binding_status,
        "executor_entrypoint": "src.envs.physics.backend_factory:make_backend",
        "executor_kind": "physics_backend_factory",
        "observation_adapter_entrypoint": "src.env.isaac_adapter:IsaacAdapter",
        "supports_training": bool(adapter_ready),
        "supports_evaluation": bool(adapter_ready),
        "supports_deploy_handle": False,
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
            "isaacsim_available": isaacsim_available,
            "isaacgym_available": isaacgym_available,
            "embodiment_context": mapping(embodiment_context),
        },
    }


__all__ = ["build_isaac_backend_binding"]
