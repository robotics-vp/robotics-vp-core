"""Provider/runtime contracts for Embodiment / Actuation WM.

These contracts expose provider evidence posture without requiring provider
availability. They intentionally keep Unitree/Isaac/Holosoma as external
families under native WM truth, not as ontology owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import mapping, safe_float, strings, truth_status


@dataclass(frozen=True)
class EmbodimentProviderContract:
    provider_id: str
    provider_family: str
    runtime_refs: dict[str, Any] = field(default_factory=dict)
    action_space_refs: dict[str, Any] = field(default_factory=dict)
    morphology_refs: dict[str, Any] = field(default_factory=dict)
    calibration_refs: dict[str, Any] = field(default_factory=dict)
    safety_refs: dict[str, Any] = field(default_factory=dict)
    compute_refs: dict[str, Any] = field(default_factory=dict)
    truth_class: str = "unavailable"
    missing_components: list[str] = field(default_factory=list)
    degraded_components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_provider_contract_v1"

    def resolved_status(self) -> str:
        return truth_status(self.missing_components, self.degraded_components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_family": self.provider_family,
            "runtime_refs": mapping(self.runtime_refs),
            "action_space_refs": mapping(self.action_space_refs),
            "morphology_refs": mapping(self.morphology_refs),
            "calibration_refs": mapping(self.calibration_refs),
            "safety_refs": mapping(self.safety_refs),
            "compute_refs": mapping(self.compute_refs),
            "truth_class": self.truth_class,
            "resolved_status": self.resolved_status(),
            "missing_components": strings(self.missing_components),
            "degraded_components": strings(self.degraded_components),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentRuntimeResourceSurface:
    surface_id: str
    provider_contracts: list[EmbodimentProviderContract] = field(default_factory=list)
    onboard_compute_available: bool = False
    companion_compute_available: bool = False
    battery_fraction: float = 0.0
    thermal_margin_fraction: float = 0.0
    latency_budget_ms: float = 0.0
    missing_components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_runtime_resource_surface_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "provider_contracts": [contract.to_dict() for contract in self.provider_contracts],
            "onboard_compute_available": bool(self.onboard_compute_available),
            "companion_compute_available": bool(self.companion_compute_available),
            "battery_fraction": max(0.0, min(1.0, safe_float(self.battery_fraction))),
            "thermal_margin_fraction": max(0.0, min(1.0, safe_float(self.thermal_margin_fraction))),
            "latency_budget_ms": max(0.0, safe_float(self.latency_budget_ms)),
            "missing_components": strings(self.missing_components),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


def unitree_g1_contract(
    *,
    policy_ref: str = "",
    runtime_ref: str = "",
    actuator_latency_profile_ref: str = "",
    safety_watchdog_profile_ref: str = "",
    metadata: dict[str, Any] | None = None,
) -> EmbodimentProviderContract:
    missing: list[str] = []
    if not actuator_latency_profile_ref:
        missing.append("actuator_latency_profile")
    if not safety_watchdog_profile_ref:
        missing.append("safety_watchdog_profile")
    return EmbodimentProviderContract(
        provider_id="unitree_g1_embodiment_provider",
        provider_family="unitree_g1",
        runtime_refs={"runtime_ref": runtime_ref, "policy_ref": policy_ref},
        morphology_refs={"robot_family": "unitree_g1"},
        calibration_refs={"actuator_latency_profile_ref": actuator_latency_profile_ref},
        safety_refs={"safety_watchdog_profile_ref": safety_watchdog_profile_ref},
        truth_class="external_blocked" if missing else "local_contract_visible",
        missing_components=missing,
        metadata=mapping(metadata),
    )


def holosoma_contract(
    *,
    policy_ref: str = "",
    runtime_ref: str = "",
    metadata: dict[str, Any] | None = None,
) -> EmbodimentProviderContract:
    missing: list[str] = []
    if not runtime_ref:
        missing.append("native_holosoma_runtime_execution")
    return EmbodimentProviderContract(
        provider_id="holosoma_embodiment_provider",
        provider_family="holosoma",
        runtime_refs={"runtime_ref": runtime_ref, "policy_ref": policy_ref},
        morphology_refs={"robot_family": "g1_29dof"},
        truth_class="local_deploy_smoke" if policy_ref else "unavailable",
        missing_components=missing,
        metadata=mapping(metadata),
    )


def isaac_contract(
    *,
    runtime_ref: str = "",
    asset_ref: str = "",
    metadata: dict[str, Any] | None = None,
) -> EmbodimentProviderContract:
    missing: list[str] = []
    if not runtime_ref:
        missing.append("isaac_runtime_execution")
    if not asset_ref:
        missing.append("embodiment_asset_ref")
    return EmbodimentProviderContract(
        provider_id="isaac_embodiment_provider",
        provider_family="isaac",
        runtime_refs={"runtime_ref": runtime_ref},
        morphology_refs={"asset_ref": asset_ref},
        truth_class="external_blocked" if missing else "local_contract_visible",
        missing_components=missing,
        metadata=mapping(metadata),
    )
