"""Backend routing for the sim/synth/physics world model."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict

from .common import mapping, stable_id
from .physics_contracts import PhysicsExecutionContract
from .state import SimSynthPhysicsWorldState


@dataclass(frozen=True)
class BackendAdapterDescriptor:
    backend: str
    adapter_name: str
    adapter_status: str
    supports_execution: bool
    fallback_backend: str = ""
    fallback_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "adapter_name": self.adapter_name,
            "adapter_status": self.adapter_status,
            "supports_execution": bool(self.supports_execution),
            "fallback_backend": self.fallback_backend,
            "fallback_reason": self.fallback_reason,
            "metadata": mapping(self.metadata),
        }


def _holosoma_available() -> bool:
    return importlib.util.find_spec("holosoma") is not None


def describe_backend_adapter(backend: str) -> BackendAdapterDescriptor:
    normalized = str(backend or "").strip().lower() or "pybullet"
    if normalized == "pybullet":
        return BackendAdapterDescriptor(
            backend="pybullet",
            adapter_name="backend_pybullet_v1",
            adapter_status="ready",
            supports_execution=True,
            metadata={
                "provider_class": "oss_provider",
                "supports_receipt_harvest": True,
            },
        )
    if normalized == "holosoma":
        available = _holosoma_available()
        return BackendAdapterDescriptor(
            backend="holosoma",
            adapter_name="backend_holosoma_v1",
            adapter_status="ready" if available else "fallback_only",
            supports_execution=available,
            fallback_backend="pybullet" if not available else "",
            fallback_reason=(
                ""
                if available
                else "holosoma runtime is not installed on this host; preserve the request but route through pybullet"
            ),
            metadata={
                "provider_class": "external_execution_provider",
                "holosoma_available": available,
            },
        )
    if normalized == "isaac":
        return BackendAdapterDescriptor(
            backend="isaac",
            adapter_name="backend_isaac_stub_v1",
            adapter_status="fallback_only",
            supports_execution=False,
            fallback_backend="pybullet",
            fallback_reason="isaac backend remains an explicit stub and is not a real execution adapter yet",
            metadata={
                "provider_class": "explicit_gap",
                "gap_kind": "missing_backend_adapter",
                "stub_backend": True,
            },
        )
    return BackendAdapterDescriptor(
        backend=normalized,
        adapter_name=f"backend_{normalized}_unknown_v1",
        adapter_status="fallback_only",
        supports_execution=False,
        fallback_backend="pybullet",
        fallback_reason=f"no sim/synth WM adapter is registered for backend '{normalized}'",
        metadata={
            "provider_class": "explicit_gap",
            "gap_kind": "unknown_backend_adapter",
        },
    )


def build_physics_execution_contract(
    world_state: SimSynthPhysicsWorldState,
    *,
    fallback_backend: str = "pybullet",
) -> PhysicsExecutionContract:
    physics_context = world_state.physics_context
    requested_backend = str(physics_context.backend or fallback_backend)
    requested_descriptor = describe_backend_adapter(requested_backend)
    resolved_backend = requested_backend
    route_status = "ready"
    fallback_reason = ""
    resolved_descriptor = requested_descriptor

    if not requested_descriptor.supports_execution:
        fallback_target = str(requested_descriptor.fallback_backend or fallback_backend or requested_backend)
        resolved_descriptor = describe_backend_adapter(fallback_target)
        if resolved_descriptor.supports_execution:
            resolved_backend = fallback_target
            route_status = "fallback"
            fallback_reason = str(requested_descriptor.fallback_reason or "")
        else:
            resolved_backend = requested_backend
            route_status = "blocked"
            fallback_reason = str(
                requested_descriptor.fallback_reason
                or f"no executable backend adapter available for {requested_backend}"
            )

    payload = {
        "state_id": world_state.state_id,
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "fidelity_tier": physics_context.fidelity_tier,
        "domain_randomization_regime": physics_context.domain_randomization_regime,
        "route_status": route_status,
    }
    return PhysicsExecutionContract(
        contract_id=stable_id("physics_execution_contract", payload),
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        fidelity_tier=physics_context.fidelity_tier,
        domain_randomization_regime=physics_context.domain_randomization_regime,
        calibration_profile=physics_context.calibration_profile,
        backend_selection_policy=physics_context.selection_policy,
        adapter_name=resolved_descriptor.adapter_name,
        route_status=route_status,
        fallback_reason=fallback_reason,
        metadata={
            "requested_adapter": requested_descriptor.to_dict(),
            "resolved_adapter": resolved_descriptor.to_dict(),
            "requested_branch_count": len(world_state.synthetic_branch_plans),
            "benchmark_signals": mapping(
                physics_context.metadata.get("benchmark_signals", {})
            ),
        },
    )


__all__ = [
    "BackendAdapterDescriptor",
    "build_physics_execution_contract",
    "describe_backend_adapter",
]
