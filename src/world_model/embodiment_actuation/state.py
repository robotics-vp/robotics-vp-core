"""Canonical Embodiment / Actuation world-model state contracts.

Phase 3 state is additive and advisory. It compiles body/action/runtime-adjacent
truth beside the frozen Phase B baseline; it does not control robots, alter
reward math, or promote provider outputs into native truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import clip01, float_mapping, mapping, safe_float, safe_int, strings

EMBODIMENT_ACTUATION_WORLD_STATE_VERSION = "embodiment_actuation_world_state_v1"


@dataclass(frozen=True)
class CapabilityState:
    capability_id: str
    embodiment_id: str = ""
    robot_id: str = ""
    robot_family: str = "unknown"
    sensor_modalities: list[str] = field(default_factory=list)
    action_spaces: list[str] = field(default_factory=list)
    skill_capabilities: dict[str, float] = field(default_factory=dict)
    workspace_bounds: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    safety_envelopes: dict[str, Any] = field(default_factory=dict)
    truth_class: str = "unavailable"
    source_refs: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "capability_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "embodiment_id": self.embodiment_id,
            "robot_id": self.robot_id,
            "robot_family": self.robot_family,
            "sensor_modalities": strings(self.sensor_modalities),
            "action_spaces": strings(self.action_spaces),
            "skill_capabilities": float_mapping(self.skill_capabilities),
            "workspace_bounds": mapping(self.workspace_bounds),
            "timing": float_mapping(self.timing),
            "safety_envelopes": mapping(self.safety_envelopes),
            "truth_class": self.truth_class,
            "source_refs": mapping(self.source_refs),
            "missing_fields": strings(self.missing_fields),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentProfileState:
    profile_id: str
    profile_ref: str = ""
    quality_score: float = 0.0
    contact_coverage: float = 0.0
    semantic_confidence: float = 0.0
    physically_impossible_contacts: int = 0
    trust_override_candidate: bool = False
    missing_inputs: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_profile_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_ref": self.profile_ref,
            "quality_score": clip01(self.quality_score),
            "contact_coverage": clip01(self.contact_coverage),
            "semantic_confidence": clip01(self.semantic_confidence),
            "physically_impossible_contacts": int(self.physically_impossible_contacts),
            "trust_override_candidate": bool(self.trust_override_candidate),
            "missing_inputs": strings(self.missing_inputs),
            "diagnostics": mapping(self.diagnostics),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ActuatorConfigurationState:
    config_id: str
    action_schema_id: str = ""
    control_hz: float = 0.0
    latency_ms: float = 0.0
    channel_order: list[str] = field(default_factory=list)
    translator_ref: str = ""
    embodiment_id: str = ""
    bounds: dict[str, Any] = field(default_factory=dict)
    source_refs: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    authority_level: str = "none"
    version: str = "actuator_configuration_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "action_schema_id": self.action_schema_id,
            "control_hz": safe_float(self.control_hz),
            "latency_ms": safe_float(self.latency_ms),
            "channel_order": strings(self.channel_order),
            "translator_ref": self.translator_ref,
            "embodiment_id": self.embodiment_id,
            "bounds": mapping(self.bounds),
            "source_refs": mapping(self.source_refs),
            "missing_fields": strings(self.missing_fields),
            "authority_level": self.authority_level,
            "version": self.version,
        }


@dataclass(frozen=True)
class JointStateVector:
    vector_id: str
    joint_names: list[str] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    efforts: list[float] = field(default_factory=list)
    timestamp_s: float = 0.0
    truth_class: str = "unavailable"
    missing_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "joint_state_vector_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "joint_names": strings(self.joint_names),
            "positions": [safe_float(v) for v in self.positions],
            "velocities": [safe_float(v) for v in self.velocities],
            "efforts": [safe_float(v) for v in self.efforts],
            "timestamp_s": safe_float(self.timestamp_s),
            "truth_class": self.truth_class,
            "missing_fields": strings(self.missing_fields),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ContactStateVector:
    contact_id: str
    track_ids: list[str] = field(default_factory=list)
    contact_event_count: int = 0
    contact_pair_count: int = 0
    impossible_contact_count: int = 0
    contact_confidence_mean: float = 0.0
    contact_coverage: float = 0.0
    contact_pairs: list[dict[str, Any]] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "contact_state_vector_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "track_ids": strings(self.track_ids),
            "contact_event_count": int(self.contact_event_count),
            "contact_pair_count": int(self.contact_pair_count),
            "impossible_contact_count": int(self.impossible_contact_count),
            "contact_confidence_mean": clip01(self.contact_confidence_mean),
            "contact_coverage": clip01(self.contact_coverage),
            "contact_pairs": [mapping(pair) for pair in self.contact_pairs],
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SafetyEnvelopeState:
    envelope_id: str
    status: str = "external_blocked"
    margin_fraction: float = 0.0
    watchdog_ref: str = ""
    latency_profile_ref: str = ""
    safety_limits: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    authority_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "safety_envelope_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "status": self.status,
            "margin_fraction": clip01(self.margin_fraction),
            "watchdog_ref": self.watchdog_ref,
            "latency_profile_ref": self.latency_profile_ref,
            "safety_limits": mapping(self.safety_limits),
            "missing_evidence": strings(self.missing_evidence),
            "degraded_reasons": strings(self.degraded_reasons),
            "authority_level": self.authority_level,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ActionSpaceState:
    action_space_id: str
    schema_id: str = ""
    dimension: int = 0
    channels: list[str] = field(default_factory=list)
    normalized: bool = False
    bounds: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "unavailable"
    missing_fields: list[str] = field(default_factory=list)
    translator_ref: str = ""
    version: str = "action_space_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_space_id": self.action_space_id,
            "schema_id": self.schema_id,
            "dimension": int(self.dimension),
            "channels": strings(self.channels),
            "normalized": bool(self.normalized),
            "bounds": mapping(self.bounds),
            "validation_status": self.validation_status,
            "missing_fields": strings(self.missing_fields),
            "translator_ref": self.translator_ref,
            "version": self.version,
        }


@dataclass(frozen=True)
class ObservationInterfaceState:
    observation_interface_id: str
    schema_id: str = ""
    proprio_fields: list[str] = field(default_factory=list)
    sensor_refs: list[str] = field(default_factory=list)
    sample_hz: float = 0.0
    latency_ms: float = 0.0
    translator_ref: str = ""
    embodiment_id: str = ""
    degraded_modes: list[str] = field(default_factory=list)
    quality_metrics: dict[str, float] = field(default_factory=dict)
    validation_status: str = "unavailable"
    missing_fields: list[str] = field(default_factory=list)
    version: str = "observation_interface_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_interface_id": self.observation_interface_id,
            "schema_id": self.schema_id,
            "proprio_fields": strings(self.proprio_fields),
            "sensor_refs": strings(self.sensor_refs),
            "sample_hz": safe_float(self.sample_hz),
            "latency_ms": safe_float(self.latency_ms),
            "translator_ref": self.translator_ref,
            "embodiment_id": self.embodiment_id,
            "degraded_modes": strings(self.degraded_modes),
            "quality_metrics": float_mapping(self.quality_metrics),
            "validation_status": self.validation_status,
            "missing_fields": strings(self.missing_fields),
            "version": self.version,
        }


@dataclass(frozen=True)
class ContactAffordanceGraphState:
    graph_id: str
    graph_ref: str = ""
    node_count: int = 0
    edge_count: int = 0
    actionable_object_count: int = 0
    obstructed_object_count: int = 0
    scene_contact_feasibility: float = 0.0
    scene_affordance_coverage: float = 0.0
    scene_obstruction_severity: float = 0.0
    body_object_engagement_summary: dict[str, float] = field(default_factory=dict)
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "contact_affordance_graph_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "graph_ref": self.graph_ref,
            "node_count": int(self.node_count),
            "edge_count": int(self.edge_count),
            "actionable_object_count": int(self.actionable_object_count),
            "obstructed_object_count": int(self.obstructed_object_count),
            "scene_contact_feasibility": clip01(self.scene_contact_feasibility),
            "scene_affordance_coverage": clip01(self.scene_affordance_coverage),
            "scene_obstruction_severity": clip01(self.scene_obstruction_severity),
            "body_object_engagement_summary": float_mapping(self.body_object_engagement_summary),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class LocalDynamicsForecastState:
    forecast_id: str
    forecast_mode: str = "heuristic_shadow"
    horizon_steps: int = 0
    confidence: float = 0.0
    contact_transition_risk: float = 0.0
    promotion_stage: str = "heuristic_fallback"
    blocked_reason: str = "no_promoted_dynamics_seam"
    source_refs: dict[str, Any] = field(default_factory=dict)
    version: str = "local_dynamics_forecast_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "forecast_mode": self.forecast_mode,
            "horizon_steps": int(self.horizon_steps),
            "confidence": clip01(self.confidence),
            "contact_transition_risk": clip01(self.contact_transition_risk),
            "promotion_stage": self.promotion_stage,
            "blocked_reason": self.blocked_reason,
            "source_refs": mapping(self.source_refs),
            "version": self.version,
        }


@dataclass(frozen=True)
class InverseRetargetTraceState:
    trace_id: str
    retarget_mode: str = "shadow_unavailable"
    source_action_space: str = ""
    target_action_space: str = ""
    readiness_score: float = 0.0
    missing_evidence: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    authority_level: str = "none"
    version: str = "inverse_retarget_trace_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "retarget_mode": self.retarget_mode,
            "source_action_space": self.source_action_space,
            "target_action_space": self.target_action_space,
            "readiness_score": clip01(self.readiness_score),
            "missing_evidence": strings(self.missing_evidence),
            "source_refs": mapping(self.source_refs),
            "authority_level": self.authority_level,
            "version": self.version,
        }


@dataclass(frozen=True)
class ActionProposalBundleState:
    bundle_id: str
    proposal_mode: str = "shadow_summary"
    proposal_count: int = 0
    proposal_refs: list[str] = field(default_factory=list)
    action_feasibility_score: float = 0.0
    promotion_stage: str = "heuristic_fallback"
    blocked_reason: str = "no_promoted_action_proposal_seam"
    authority_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "action_proposal_bundle_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "proposal_mode": self.proposal_mode,
            "proposal_count": int(self.proposal_count),
            "proposal_refs": strings(self.proposal_refs),
            "action_feasibility_score": clip01(self.action_feasibility_score),
            "promotion_stage": self.promotion_stage,
            "blocked_reason": self.blocked_reason,
            "authority_level": self.authority_level,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentDriftSummaryState:
    drift_id: str
    drift_score: float = 0.0
    calibration_due: bool = False
    drift_reasons: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    external_blockers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_drift_summary_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "drift_score": clip01(self.drift_score),
            "calibration_due": bool(self.calibration_due),
            "drift_reasons": strings(self.drift_reasons),
            "source_refs": mapping(self.source_refs),
            "external_blockers": strings(self.external_blockers),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentCostVectorState:
    cost_id: str
    energy_wh: float = 0.0
    risk_score: float = 0.0
    latency_ms: float = 0.0
    safety_penalty: float = 0.0
    maintenance_risk: float = 0.0
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_cost_vector_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_id": self.cost_id,
            "energy_wh": max(0.0, safe_float(self.energy_wh)),
            "risk_score": clip01(self.risk_score),
            "latency_ms": max(0.0, safe_float(self.latency_ms)),
            "safety_penalty": clip01(self.safety_penalty),
            "maintenance_risk": clip01(self.maintenance_risk),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class CalibrationTargetState:
    calibration_id: str
    target_refs: dict[str, Any] = field(default_factory=dict)
    priority_score: float = 0.0
    missing_evidence: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    version: str = "calibration_target_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "target_refs": mapping(self.target_refs),
            "priority_score": clip01(self.priority_score),
            "missing_evidence": strings(self.missing_evidence),
            "next_actions": strings(self.next_actions),
            "source_refs": mapping(self.source_refs),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentActuationWorldState:
    state_id: str
    episode_id: str
    frame_index: int
    capability: CapabilityState
    embodiment_profile: EmbodimentProfileState
    actuator_configuration: ActuatorConfigurationState
    joint_state: JointStateVector
    contact_state: ContactStateVector
    safety_envelope: SafetyEnvelopeState
    action_space: ActionSpaceState
    observation_interface: ObservationInterfaceState
    contact_affordance_graph: ContactAffordanceGraphState
    local_dynamics_forecast: LocalDynamicsForecastState
    inverse_retarget_trace: InverseRetargetTraceState
    action_proposal_bundle: ActionProposalBundleState
    drift_summary: EmbodimentDriftSummaryState
    cost_vector: EmbodimentCostVectorState
    calibration_targets: CalibrationTargetState
    provider_runtime_surface: dict[str, Any] = field(default_factory=dict)
    receipt_manifest: dict[str, Any] = field(default_factory=dict)
    downstream_preconditions: dict[str, Any] = field(default_factory=dict)
    source_refs: dict[str, Any] = field(default_factory=dict)
    authority_level: str = "none"
    compilation_mode: str = "shadow_advisory"
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = EMBODIMENT_ACTUATION_WORLD_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "episode_id": self.episode_id,
            "frame_index": safe_int(self.frame_index),
            "capability": self.capability.to_dict(),
            "embodiment_profile": self.embodiment_profile.to_dict(),
            "actuator_configuration": self.actuator_configuration.to_dict(),
            "joint_state": self.joint_state.to_dict(),
            "contact_state": self.contact_state.to_dict(),
            "safety_envelope": self.safety_envelope.to_dict(),
            "action_space": self.action_space.to_dict(),
            "observation_interface": self.observation_interface.to_dict(),
            "contact_affordance_graph": self.contact_affordance_graph.to_dict(),
            "local_dynamics_forecast": self.local_dynamics_forecast.to_dict(),
            "inverse_retarget_trace": self.inverse_retarget_trace.to_dict(),
            "action_proposal_bundle": self.action_proposal_bundle.to_dict(),
            "drift_summary": self.drift_summary.to_dict(),
            "cost_vector": self.cost_vector.to_dict(),
            "calibration_targets": self.calibration_targets.to_dict(),
            "provider_runtime_surface": mapping(self.provider_runtime_surface),
            "receipt_manifest": mapping(self.receipt_manifest),
            "downstream_preconditions": mapping(self.downstream_preconditions),
            "source_refs": mapping(self.source_refs),
            "authority_level": self.authority_level,
            "compilation_mode": self.compilation_mode,
            "metadata": mapping(self.metadata),
            "version": self.version,
        }
