"""Backend adapter registry for the sim/synth/physics world model."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict

from src.motor_backend.holosoma_backend import HOLOSOMA_TASK_MAP

from .common import mapping


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@dataclass(frozen=True)
class BackendAdapterDescriptor:
    backend: str
    adapter_name: str
    adapter_status: str
    supports_execution: bool
    simulator_family: str = ""
    target_hardware_class: str = ""
    execution_envelope: str = ""
    fallback_backend: str = ""
    fallback_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "adapter_name": self.adapter_name,
            "adapter_status": self.adapter_status,
            "supports_execution": bool(self.supports_execution),
            "simulator_family": self.simulator_family,
            "target_hardware_class": self.target_hardware_class,
            "execution_envelope": self.execution_envelope,
            "fallback_backend": self.fallback_backend,
            "fallback_reason": self.fallback_reason,
            "metadata": mapping(self.metadata),
        }


def describe_backend_adapter(backend: str) -> BackendAdapterDescriptor:
    normalized = str(backend or "").strip().lower() or "pybullet"
    if normalized == "pybullet":
        return BackendAdapterDescriptor(
            backend="pybullet",
            adapter_name="backend_pybullet_v2",
            adapter_status="ready",
            supports_execution=True,
            simulator_family="pybullet",
            target_hardware_class="tabletop_fixed_base",
            execution_envelope="fixed_base_tabletop_and_workcell",
            metadata={
                "provider_class": "oss_provider",
                "supports_receipt_harvest": True,
                "supports_domain_randomization": True,
                "supports_system_identification": True,
                "supports_unitree_assets": False,
                "supported_fidelity_tiers": ["fast_scan", "branch_balanced", "high_fidelity"],
                "supported_randomization_regimes": [
                    "steady_state",
                    "coverage_exploration",
                    "calibration_focus",
                    "benchmark_focus",
                ],
                "target_runtime_stack": ["pybullet"],
            },
        )
    if normalized == "holosoma":
        available = _has_module("holosoma")
        simulator_stack = sorted(
            {
                spec.simulator
                for spec in HOLOSOMA_TASK_MAP.values()
                if str(spec.simulator or "").strip()
            }
        )
        return BackendAdapterDescriptor(
            backend="holosoma",
            adapter_name="backend_holosoma_v2",
            adapter_status="ready" if available else "shadow_ready",
            supports_execution=available,
            simulator_family="isaac",
            target_hardware_class="unitree_g1_r1_class",
            execution_envelope="humanoid_locomotion_and_whole_body_tracking",
            fallback_backend="pybullet" if not available else "",
            fallback_reason=(
                ""
                if available
                else "holosoma runtime is not installed on this host; preserve the request but route through pybullet"
            ),
            metadata={
                "provider_class": "external_execution_provider",
                "holosoma_available": available,
                "shadow_backend_available": True,
                "concrete_runtime_available": available,
                "supports_receipt_harvest": True,
                "supports_domain_randomization": True,
                "supports_system_identification": True,
                "supports_unitree_assets": True,
                "requires_companion_gpu": True,
                "supported_fidelity_tiers": ["branch_balanced", "high_fidelity"],
                "supported_randomization_regimes": [
                    "coverage_exploration",
                    "calibration_focus",
                    "benchmark_focus",
                ],
                "target_runtime_stack": simulator_stack or ["isaacgym", "isaacsim"],
                "task_presets": sorted(HOLOSOMA_TASK_MAP.keys()),
            },
        )
    if normalized == "isaac":
        isaacsim_available = _has_module("isaacsim") or _has_module("omni.isaac.kit")
        isaacgym_available = _has_module("isaacgym")
        shadow_backend_available = True
        return BackendAdapterDescriptor(
            backend="isaac",
            adapter_name="backend_isaac_unitree_target_v1",
            adapter_status="shadow_ready",
            supports_execution=False,
            simulator_family="isaac",
            target_hardware_class="unitree_g1_r1_class",
            execution_envelope="humanoid_shadow_and_unitree_target",
            fallback_backend="pybullet",
            fallback_reason=(
                "isaac backend remains an explicit real-runtime integration gap: "
                "shadow execution and adapter routing exist, but concrete Isaac Sim / "
                "Isaac Gym / Unitree asset execution is not wired yet"
            ),
            metadata={
                "provider_class": "explicit_gap",
                "gap_kind": "missing_backend_adapter",
                "stub_backend": False,
                "shadow_backend_available": shadow_backend_available,
                "supports_receipt_harvest": False,
                "supports_domain_randomization": True,
                "supports_system_identification": True,
                "supports_unitree_assets": True,
                "requires_companion_gpu": True,
                "isaacsim_available": isaacsim_available,
                "isaacgym_available": isaacgym_available,
                "target_runtime_stack": ["isaacsim", "isaacgym", "unitree_sdk2"],
                "required_assets": [
                    "unitree_robot_description",
                    "joint_mapping_contract",
                    "sensor_extrinsics",
                    "actuator_latency_profile",
                ],
                "supported_fidelity_tiers": ["branch_balanced", "high_fidelity"],
                "supported_randomization_regimes": [
                    "coverage_exploration",
                    "calibration_focus",
                    "benchmark_focus",
                ],
            },
        )
    return BackendAdapterDescriptor(
        backend=normalized,
        adapter_name=f"backend_{normalized}_unknown_v1",
        adapter_status="fallback_only",
        supports_execution=False,
        simulator_family="unknown",
        target_hardware_class="unknown",
        execution_envelope="unknown",
        fallback_backend="pybullet",
        fallback_reason=f"no sim/synth WM adapter is registered for backend '{normalized}'",
        metadata={
            "provider_class": "explicit_gap",
            "gap_kind": "unknown_backend_adapter",
            "supports_receipt_harvest": False,
            "supports_domain_randomization": False,
            "supports_system_identification": False,
            "supports_unitree_assets": False,
        },
    )


__all__ = ["BackendAdapterDescriptor", "describe_backend_adapter"]
