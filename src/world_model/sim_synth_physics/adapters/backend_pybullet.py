"""Concrete PyBullet binding helpers for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..common import mapping


def build_pybullet_backend_binding(
    *,
    physics_context: Mapping[str, Any],
    adaptation_policy: Mapping[str, Any],
    embodiment_context: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "binding_name": "pybullet_execution_binding_v1",
        "binding_status": "ready",
        "executor_entrypoint": "src.envs.physics.backend_factory:make_backend",
        "executor_kind": "physics_backend_factory",
        "observation_adapter_entrypoint": "",
        "supports_training": True,
        "supports_evaluation": True,
        "supports_deploy_handle": False,
        "target_runtime_stack": ["pybullet"],
        "asset_profile": "tabletop_workcell_assets",
        "required_assets": [],
        "available_assets": ["pybullet_runtime", "episode_summary_contract"],
        "missing_assets": [],
        "metadata": {
            "engine_type": "pybullet",
            "fidelity_tier": str(physics_context.get("fidelity_tier", "")),
            "domain_randomization_profile": str(
                adaptation_policy.get("domain_randomization_profile", "")
            ),
            "embodiment_context": mapping(embodiment_context),
        },
    }


__all__ = ["build_pybullet_backend_binding"]
