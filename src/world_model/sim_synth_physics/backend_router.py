"""Backend routing for the sim/synth/physics world model."""

from __future__ import annotations

from .backend_adapters import BackendAdapterDescriptor, describe_backend_adapter
from .common import mapping, stable_id
from .physics_contracts import PhysicsExecutionContract
from .state import SimSynthPhysicsWorldState


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
    adaptation_policy = world_state.physics_adaptation_policy

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
        simulator_family=resolved_descriptor.simulator_family,
        target_hardware_class=(
            ""
            if adaptation_policy is None
            else str(adaptation_policy.target_hardware_class)
        ),
        adaptation_policy_id=(
            "" if adaptation_policy is None else str(adaptation_policy.policy_id)
        ),
        route_status=route_status,
        fallback_reason=fallback_reason,
        metadata={
            "requested_adapter": requested_descriptor.to_dict(),
            "resolved_adapter": resolved_descriptor.to_dict(),
            "requested_branch_count": len(world_state.synthetic_branch_plans),
            "execution_envelope": resolved_descriptor.execution_envelope,
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
