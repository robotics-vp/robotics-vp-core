"""Shadow downstream consumers for Phase 3 Embodiment / Actuation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .common import clip01, mapping, stable_id, strings
from .receipts import EmbodimentReceipt
from .state import EmbodimentActuationWorldState


@dataclass(frozen=True)
class SimEmbodimentTransferContext:
    context_id: str
    source_state_id: str
    active_embodiments: list[str] = field(default_factory=list)
    capability_profile: dict[str, Any] = field(default_factory=dict)
    action_constraints: dict[str, Any] = field(default_factory=dict)
    latency_budget_ms: float = 0.0
    contact_risk_score: float = 0.0
    action_feasibility_score: float = 0.0
    retargeting_readiness_score: float = 0.0
    drift_score: float = 0.0
    safety_status: str = "external_blocked"
    authority_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "sim_embodiment_transfer_context_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "source_state_id": self.source_state_id,
            "active_embodiments": strings(self.active_embodiments),
            "capability_profile": mapping(self.capability_profile),
            "action_constraints": mapping(self.action_constraints),
            "latency_budget_ms": float(self.latency_budget_ms),
            "contact_risk_score": clip01(self.contact_risk_score),
            "action_feasibility_score": clip01(self.action_feasibility_score),
            "retargeting_readiness_score": clip01(self.retargeting_readiness_score),
            "drift_score": clip01(self.drift_score),
            "safety_status": self.safety_status,
            "authority_level": self.authority_level,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class PerceptionEmbodimentFeedbackSurface:
    feedback_id: str
    source_state_id: str
    contact_affordance_summary: dict[str, Any] = field(default_factory=dict)
    object_feedback: list[dict[str, Any]] = field(default_factory=list)
    degraded_modes: list[str] = field(default_factory=list)
    authority_level: str = "none"
    version: str = "perception_embodiment_feedback_surface_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "source_state_id": self.source_state_id,
            "contact_affordance_summary": mapping(self.contact_affordance_summary),
            "object_feedback": [mapping(item) for item in self.object_feedback],
            "degraded_modes": strings(self.degraded_modes),
            "authority_level": self.authority_level,
            "version": self.version,
        }


@dataclass(frozen=True)
class RuntimeAdapterValidationContext:
    validation_id: str
    source_state_id: str
    action_schema_id: str
    observation_schema_id: str
    runtime_validation_status: str = "shadow_only"
    blocking_reasons: list[str] = field(default_factory=list)
    action_adapter_ref: dict[str, Any] = field(default_factory=dict)
    observation_adapter_ref: dict[str, Any] = field(default_factory=dict)
    safety_ref: dict[str, Any] = field(default_factory=dict)
    authority_level: str = "none"
    version: str = "runtime_adapter_validation_context_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "source_state_id": self.source_state_id,
            "action_schema_id": self.action_schema_id,
            "observation_schema_id": self.observation_schema_id,
            "runtime_validation_status": self.runtime_validation_status,
            "blocking_reasons": strings(self.blocking_reasons),
            "action_adapter_ref": mapping(self.action_adapter_ref),
            "observation_adapter_ref": mapping(self.observation_adapter_ref),
            "safety_ref": mapping(self.safety_ref),
            "authority_level": self.authority_level,
            "version": self.version,
        }


@dataclass(frozen=True)
class EconomicEmbodimentReceiptBundle:
    bundle_id: str
    source_state_id: str
    receipt_refs: list[str] = field(default_factory=list)
    cost_summary: dict[str, Any] = field(default_factory=dict)
    safety_summary: dict[str, Any] = field(default_factory=dict)
    drift_summary: dict[str, Any] = field(default_factory=dict)
    allocative_authority: str = "none"
    version: str = "economic_embodiment_receipt_bundle_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "source_state_id": self.source_state_id,
            "receipt_refs": strings(self.receipt_refs),
            "cost_summary": mapping(self.cost_summary),
            "safety_summary": mapping(self.safety_summary),
            "drift_summary": mapping(self.drift_summary),
            "allocative_authority": self.allocative_authority,
            "version": self.version,
        }


def build_sim_embodiment_transfer_context(
    state: EmbodimentActuationWorldState,
) -> SimEmbodimentTransferContext:
    return SimEmbodimentTransferContext(
        context_id=stable_id("sim_embodiment_context", {"state_id": state.state_id}),
        source_state_id=state.state_id,
        active_embodiments=[state.capability.embodiment_id],
        capability_profile=state.capability.to_dict(),
        action_constraints={
            "action_schema_id": state.action_space.schema_id,
            "channels": list(state.action_space.channels),
            "bounds": state.action_space.bounds,
            "retargeting_readiness_score": state.inverse_retarget_trace.readiness_score,
            "safety_status": state.safety_envelope.status,
        },
        latency_budget_ms=state.actuator_configuration.latency_ms,
        contact_risk_score=state.local_dynamics_forecast.contact_transition_risk,
        action_feasibility_score=state.action_proposal_bundle.action_feasibility_score,
        retargeting_readiness_score=state.inverse_retarget_trace.readiness_score,
        drift_score=state.drift_summary.drift_score,
        safety_status=state.safety_envelope.status,
        authority_level="none",
        metadata={"consumer": "sim_synth_physics_shadow"},
    )


def build_perception_embodiment_feedback(
    state: EmbodimentActuationWorldState,
) -> PerceptionEmbodimentFeedbackSurface:
    degraded = []
    if state.safety_envelope.missing_evidence:
        degraded.append("missing_safety_evidence")
    if state.observation_interface.degraded_modes:
        degraded.extend(state.observation_interface.degraded_modes)
    return PerceptionEmbodimentFeedbackSurface(
        feedback_id=stable_id("perception_embodiment_feedback", {"state_id": state.state_id}),
        source_state_id=state.state_id,
        contact_affordance_summary={
            "scene_contact_feasibility": state.contact_affordance_graph.scene_contact_feasibility,
            "scene_affordance_coverage": state.contact_affordance_graph.scene_affordance_coverage,
            "scene_obstruction_severity": state.contact_affordance_graph.scene_obstruction_severity,
            "contact_transition_risk": state.local_dynamics_forecast.contact_transition_risk,
        },
        object_feedback=[
            {
                "track_id": track_id,
                "contact_coverage": state.contact_state.contact_coverage,
                "contact_confidence_mean": state.contact_state.contact_confidence_mean,
            }
            for track_id in state.contact_state.track_ids
        ],
        degraded_modes=degraded,
        authority_level="none",
    )


def build_runtime_adapter_validation_context(
    state: EmbodimentActuationWorldState,
) -> RuntimeAdapterValidationContext:
    blocking = []
    if state.action_space.validation_status != "adapter_validated":
        blocking.append("action_space_not_validated")
    if state.observation_interface.validation_status != "adapter_validated":
        blocking.append("observation_interface_not_validated")
    blocking.extend(state.safety_envelope.missing_evidence)
    status = "shadow_validated" if not blocking else "shadow_blocked"
    return RuntimeAdapterValidationContext(
        validation_id=stable_id("runtime_adapter_validation", {"state_id": state.state_id}),
        source_state_id=state.state_id,
        action_schema_id=state.action_space.schema_id,
        observation_schema_id=state.observation_interface.schema_id,
        runtime_validation_status=status,
        blocking_reasons=blocking,
        action_adapter_ref=state.action_space.to_dict(),
        observation_adapter_ref=state.observation_interface.to_dict(),
        safety_ref=state.safety_envelope.to_dict(),
        authority_level="none",
    )


def build_economic_embodiment_receipt_bundle(
    state: EmbodimentActuationWorldState,
    receipts: Iterable[EmbodimentReceipt] = (),
) -> EconomicEmbodimentReceiptBundle:
    receipt_list = list(receipts)
    return EconomicEmbodimentReceiptBundle(
        bundle_id=stable_id("economic_embodiment_bundle", {"state_id": state.state_id}),
        source_state_id=state.state_id,
        receipt_refs=[receipt.receipt_id for receipt in receipt_list],
        cost_summary=state.cost_vector.to_dict(),
        safety_summary=state.safety_envelope.to_dict(),
        drift_summary=state.drift_summary.to_dict(),
        allocative_authority="none",
    )
