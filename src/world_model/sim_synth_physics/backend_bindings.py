"""Backend binding registry for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .adapters.backend_holosoma import build_holosoma_backend_binding
from .adapters.backend_isaac import build_isaac_backend_binding
from .adapters.backend_pybullet import build_pybullet_backend_binding
from .common import mapping, stable_id
from .state import BackendExecutionBindingState, PhysicsAdaptationPolicyState, PhysicsContextState


def compile_backend_execution_binding(
    physics_context: PhysicsContextState,
    *,
    adaptation_policy: Optional[PhysicsAdaptationPolicyState],
    embodiment_context: Optional[Mapping[str, Any]] = None,
) -> BackendExecutionBindingState:
    adaptation_payload = (
        {} if adaptation_policy is None else adaptation_policy.to_dict()
    )
    embodiment_payload = mapping(embodiment_context)
    backend = str(physics_context.backend or "pybullet")
    if backend == "holosoma":
        binding = build_holosoma_backend_binding(
            physics_context=physics_context.to_dict(),
            adaptation_policy=adaptation_payload,
            embodiment_context=embodiment_payload,
        )
    elif backend == "isaac":
        binding = build_isaac_backend_binding(
            physics_context=physics_context.to_dict(),
            adaptation_policy=adaptation_payload,
            embodiment_context=embodiment_payload,
        )
    else:
        binding = build_pybullet_backend_binding(
            physics_context=physics_context.to_dict(),
            adaptation_policy=adaptation_payload,
            embodiment_context=embodiment_payload,
        )
    payload = {
        "backend": backend,
        "binding_name": binding.get("binding_name"),
        "binding_status": binding.get("binding_status"),
        "asset_profile": binding.get("asset_profile"),
    }
    return BackendExecutionBindingState(
        binding_id=stable_id("backend_execution_binding", payload),
        backend=backend,
        binding_name=str(binding.get("binding_name", "")),
        binding_status=str(binding.get("binding_status", "integration_pending")),
        executor_entrypoint=str(binding.get("executor_entrypoint", "")),
        executor_kind=str(binding.get("executor_kind", "")),
        observation_adapter_entrypoint=str(binding.get("observation_adapter_entrypoint", "")),
        target_runtime_stack=list(binding.get("target_runtime_stack", []) or []),
        asset_profile=str(binding.get("asset_profile", "")),
        required_assets=list(binding.get("required_assets", []) or []),
        available_assets=list(binding.get("available_assets", []) or []),
        missing_assets=list(binding.get("missing_assets", []) or []),
        metadata=mapping(binding.get("metadata", {})),
    )


__all__ = ["compile_backend_execution_binding"]
