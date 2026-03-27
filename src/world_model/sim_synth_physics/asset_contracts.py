"""Robot-asset contract helpers for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .common import clip01, mapping, stable_id
from .state import (
    BackendExecutionBindingState,
    PhysicsAdaptationPolicyState,
    RobotAssetContractState,
)


def _asset_manifest(embodiment_context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = mapping(embodiment_context)
    return mapping(
        payload.get("robot_asset_manifest")
        or payload.get("asset_manifest")
        or payload.get("robot_assets")
    )


def _observation_contracts(target_hardware_class: str) -> list[str]:
    contracts = ["rgb_frame_v1", "joint_state_v1"]
    if target_hardware_class == "unitree_g1_r1_class":
        contracts.extend(
            [
                "imu_state_v1",
                "foot_contact_state_v1",
                "force_torque_state_v1",
                "depth_frame_v1",
            ]
        )
    return sorted(set(contracts))


def _action_contracts(target_hardware_class: str) -> list[str]:
    if target_hardware_class == "unitree_g1_r1_class":
        return ["whole_body_joint_command_v1", "locomotion_mode_command_v1"]
    return ["joint_command_v1"]


def compile_robot_asset_contract(
    backend_execution_binding: BackendExecutionBindingState,
    *,
    adaptation_policy: Optional[PhysicsAdaptationPolicyState],
    embodiment_context: Optional[Mapping[str, Any]] = None,
) -> RobotAssetContractState:
    manifest = _asset_manifest(embodiment_context)
    manifest_available = [
        str(key)
        for key, value in sorted(manifest.items())
        if value not in (None, "", False, 0, [], {})
    ]
    available_assets = sorted(
        set(manifest_available) | set(backend_execution_binding.available_assets)
    )
    required_assets = list(backend_execution_binding.required_assets)
    missing_assets = [asset for asset in required_assets if asset not in available_assets]
    target_hardware_class = (
        ""
        if adaptation_policy is None
        else str(adaptation_policy.target_hardware_class)
    )
    calibration_contracts = (
        []
        if adaptation_policy is None
        else list(adaptation_policy.calibration_targets)
    )
    payload = {
        "asset_profile": backend_execution_binding.asset_profile,
        "target_hardware_class": target_hardware_class,
        "required_assets": required_assets,
        "available_assets": available_assets,
    }
    return RobotAssetContractState(
        contract_id=stable_id("robot_asset_contract", payload),
        asset_profile=backend_execution_binding.asset_profile,
        target_hardware_class=target_hardware_class,
        required_assets=required_assets,
        available_assets=available_assets,
        missing_assets=missing_assets,
        calibration_contracts=calibration_contracts,
        observation_contracts=_observation_contracts(target_hardware_class),
        action_contracts=_action_contracts(target_hardware_class),
        metadata={
            "binding_id": backend_execution_binding.binding_id,
            "binding_status": backend_execution_binding.binding_status,
            "asset_manifest": manifest,
            "embodiment_context": mapping(embodiment_context),
            "asset_readiness_score": clip01(
                1.0
                if not required_assets
                else 1.0 - (len(missing_assets) / float(max(1, len(required_assets))))
            ),
        },
    )


__all__ = ["compile_robot_asset_contract"]
