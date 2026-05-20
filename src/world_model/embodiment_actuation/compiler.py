"""Shadow compiler for canonical Embodiment / Actuation WM state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from src.embodiment.registry import CapabilityProfile, EmbodimentRegistryEntry
from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2

from .common import clip01, float_mapping, mapping, safe_float, safe_int, stable_id, strings
from .provider_contracts import EmbodimentProviderContract, EmbodimentRuntimeResourceSurface
from .receipts import (
    ActionProposalReceipt,
    ActionSpaceValidationReceipt,
    CalibrationTargetReceipt,
    CapabilityProfileReceipt,
    ContactAffordanceReceipt,
    EmbodimentCompilationReceipt,
    EmbodimentCostReceipt,
    EmbodimentDriftReceipt,
    EmbodimentReceipt,
    InverseRetargetReceipt,
    LocalDynamicsReceipt,
    ObservationInterfaceReceipt,
    SafetyEnvelopeReceipt,
    SimEmbodimentTransferReceipt,
)
from .state import (
    ActionProposalBundleState,
    ActionSpaceState,
    ActuatorConfigurationState,
    CalibrationTargetState,
    CapabilityState,
    ContactAffordanceGraphState,
    ContactStateVector,
    EmbodimentActuationWorldState,
    EmbodimentCostVectorState,
    EmbodimentDriftSummaryState,
    EmbodimentProfileState,
    InverseRetargetTraceState,
    JointStateVector,
    LocalDynamicsForecastState,
    ObservationInterfaceState,
    SafetyEnvelopeState,
)


@dataclass(frozen=True)
class EmbodimentActuationCompilationInputs:
    episode_id: str
    frame_index: int = 0
    embodiment_registry_entry: Optional[EmbodimentRegistryEntry] = None
    capability_profile: Optional[CapabilityProfile] = None
    advisory_embodiment_result: Any = None
    action_adapter: Optional[ActionAdapterV2] = None
    observation_adapter: Optional[ObservationAdapterV2] = None
    perception_shadow_surface: Any = None
    sim_embodiment_context: Optional[Mapping[str, Any]] = None
    provider_contracts: list[EmbodimentProviderContract] = field(default_factory=list)
    runtime_resource_surface: Optional[EmbodimentRuntimeResourceSurface] = None
    joint_state: Optional[Mapping[str, Any]] = None
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbodimentActuationCompilationResult:
    state: EmbodimentActuationWorldState
    receipts: list[EmbodimentReceipt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
        }


def compile_embodiment_actuation_world_state(
    inputs: EmbodimentActuationCompilationInputs | None = None,
    **kwargs: Any,
) -> EmbodimentActuationWorldState:
    """Compile canonical Phase 3 state and return state only.

    Use ``compile_embodiment_actuation_with_receipts`` when receipt emission is
    needed by downstream consumers or audits.
    """
    return compile_embodiment_actuation_with_receipts(inputs, **kwargs).state


def compile_embodiment_actuation_with_receipts(
    inputs: EmbodimentActuationCompilationInputs | None = None,
    **kwargs: Any,
) -> EmbodimentActuationCompilationResult:
    if inputs is None:
        inputs = EmbodimentActuationCompilationInputs(**kwargs)
    elif kwargs:
        merged = {**inputs.__dict__, **kwargs}
        inputs = EmbodimentActuationCompilationInputs(**merged)

    state_seed = {
        "episode_id": inputs.episode_id,
        "frame_index": inputs.frame_index,
        "embodiment_id": _embodiment_id(inputs),
        "source_refs": inputs.source_refs,
    }
    state_id = stable_id("embodiment_actuation_state", state_seed)

    capability = _compile_capability_state(state_id, inputs)
    profile = _compile_profile_state(state_id, inputs)
    actuator = _compile_actuator_configuration_state(state_id, inputs)
    joint_state = _compile_joint_state(state_id, inputs)
    contact_state = _compile_contact_state(state_id, inputs)
    safety = _compile_safety_envelope_state(state_id, inputs)
    action_space = _compile_action_space_state(state_id, inputs, actuator)
    observation = _compile_observation_interface_state(state_id, inputs)
    affordance = _compile_contact_affordance_graph_state(state_id, inputs, contact_state)
    dynamics = _compile_local_dynamics_state(state_id, inputs, contact_state)
    retarget = _compile_inverse_retarget_state(state_id, inputs, action_space)
    proposals = _compile_action_proposal_state(state_id, inputs, affordance, retarget)
    drift = _compile_drift_state(state_id, inputs, profile, safety)
    costs = _compile_cost_state(state_id, inputs, safety, proposals)
    calibration = _compile_calibration_state(state_id, inputs, drift, safety)
    provider_surface = _compile_provider_surface(state_id, inputs)

    preconditions = _downstream_preconditions(
        safety=safety,
        retarget=retarget,
        proposals=proposals,
        calibration=calibration,
    )

    receipts = _build_receipts(
        state_id=state_id,
        inputs=inputs,
        capability=capability,
        action_space=action_space,
        observation=observation,
        affordance=affordance,
        dynamics=dynamics,
        retarget=retarget,
        proposals=proposals,
        safety=safety,
        drift=drift,
        calibration=calibration,
        costs=costs,
    )
    receipt_manifest = {
        "schema_version": "embodiment_actuation_receipt_manifest_v1",
        "receipt_count": len(receipts),
        "receipt_types": [receipt.version for receipt in receipts],
        "receipt_ids": [receipt.receipt_id for receipt in receipts],
        "authority_level": "none",
    }

    state = EmbodimentActuationWorldState(
        state_id=state_id,
        episode_id=str(inputs.episode_id),
        frame_index=int(inputs.frame_index),
        capability=capability,
        embodiment_profile=profile,
        actuator_configuration=actuator,
        joint_state=joint_state,
        contact_state=contact_state,
        safety_envelope=safety,
        action_space=action_space,
        observation_interface=observation,
        contact_affordance_graph=affordance,
        local_dynamics_forecast=dynamics,
        inverse_retarget_trace=retarget,
        action_proposal_bundle=proposals,
        drift_summary=drift,
        cost_vector=costs,
        calibration_targets=calibration,
        provider_runtime_surface=provider_surface,
        receipt_manifest=receipt_manifest,
        downstream_preconditions=preconditions,
        source_refs=mapping(inputs.source_refs),
        authority_level="none",
        compilation_mode="shadow_advisory",
        metadata={
            **mapping(inputs.metadata),
            "compiler_version": "embodiment_actuation_shadow_compiler_v1",
            "phase": "phase3",
            "phase_tranche": "3.1-3.3",
        },
    )
    compilation_receipt = EmbodimentCompilationReceipt(
        receipt_id=stable_id("embodiment_compilation_receipt", {"state_id": state_id}),
        state_id=state_id,
        compiled_surface_count=15,
        receipt_count=len(receipts) + 1,
        compilation_mode="shadow_advisory",
        source_refs=inputs.source_refs,
        downstream_preconditions=list(preconditions.keys()),
        metadata={"authority_level": "none"},
    )
    return EmbodimentActuationCompilationResult(
        state=state,
        receipts=[compilation_receipt, *receipts],
    )


def _embodiment_id(inputs: EmbodimentActuationCompilationInputs) -> str:
    if inputs.embodiment_registry_entry is not None:
        return inputs.embodiment_registry_entry.embodiment_id
    if inputs.action_adapter is not None and inputs.action_adapter.embodiment_id:
        return str(inputs.action_adapter.embodiment_id)
    if inputs.observation_adapter is not None and inputs.observation_adapter.embodiment_id:
        return str(inputs.observation_adapter.embodiment_id)
    return str(mapping(inputs.metadata).get("embodiment_id", "unknown_embodiment"))


def _capability_profile(inputs: EmbodimentActuationCompilationInputs) -> Optional[CapabilityProfile]:
    if inputs.capability_profile is not None:
        return inputs.capability_profile
    if inputs.embodiment_registry_entry is not None:
        return inputs.embodiment_registry_entry.capability_profile
    return None


def _compile_capability_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> CapabilityState:
    entry = inputs.embodiment_registry_entry
    profile = _capability_profile(inputs)
    missing: list[str] = []
    if profile is None:
        missing.append("capability_profile")
    if entry is None:
        missing.append("embodiment_registry_entry")

    return CapabilityState(
        capability_id=stable_id("capability_state", {"state_id": state_id}),
        embodiment_id=entry.embodiment_id if entry else _embodiment_id(inputs),
        robot_id=entry.robot_id if entry else "",
        robot_family=(entry.robot_family if entry else (profile.robot_family if profile else "unknown")),
        sensor_modalities=list(profile.sensor_modalities) if profile else [],
        action_spaces=list(profile.action_spaces) if profile else [],
        skill_capabilities=dict(profile.skill_capabilities) if profile else {},
        workspace_bounds=dict(profile.workspace_bounds) if profile else {},
        timing=dict(profile.timing) if profile else {},
        safety_envelopes=dict(profile.safety_envelopes) if profile else {},
        truth_class="registry_backed" if profile else "unavailable",
        source_refs={"registry_entry": entry.embodiment_id if entry else ""},
        missing_fields=missing,
    )


def _summary(inputs: EmbodimentActuationCompilationInputs) -> dict[str, Any]:
    result = inputs.advisory_embodiment_result
    return mapping(getattr(result, "summary", None)) if result is not None else {}


def _compile_profile_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> EmbodimentProfileState:
    summary = _summary(inputs)
    result = inputs.advisory_embodiment_result
    missing = strings(summary.get("missing_inputs", []))
    if result is None:
        missing.append("advisory_embodiment_result")
    return EmbodimentProfileState(
        profile_id=stable_id("embodiment_profile_state", {"state_id": state_id}),
        profile_ref=str(mapping(inputs.source_refs).get("embodiment_profile_ref", "")),
        quality_score=summary.get("embodiment_quality_score", summary.get("w_embodiment", 0.0)),
        contact_coverage=summary.get("contact_coverage_pct", 0.0),
        semantic_confidence=summary.get("semantic_confidence_mean", 0.0),
        physically_impossible_contacts=safe_int(summary.get("physically_impossible_contacts", 0)),
        trust_override_candidate=bool(summary.get("trust_override_candidate", False)),
        missing_inputs=missing,
        diagnostics=mapping(summary.get("diagnostics")),
        source_refs=mapping(inputs.source_refs),
    )


def _compile_actuator_configuration_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> ActuatorConfigurationState:
    adapter = inputs.action_adapter
    missing = [] if adapter else ["action_adapter_v2"]
    return ActuatorConfigurationState(
        config_id=stable_id("actuator_config_state", {"state_id": state_id}),
        action_schema_id=adapter.schema_id if adapter else "",
        control_hz=adapter.control_hz if adapter else 0.0,
        latency_ms=adapter.latency_ms if adapter else 0.0,
        channel_order=list(adapter.channel_order) if adapter else [],
        translator_ref=str(adapter.translator_ref or "") if adapter else "",
        embodiment_id=str(adapter.embodiment_id or "") if adapter else _embodiment_id(inputs),
        bounds=dict(adapter.bounds) if adapter else {},
        source_refs={"action_adapter_schema": adapter.schema_id if adapter else ""},
        missing_fields=missing,
        authority_level="none",
    )


def _compile_joint_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> JointStateVector:
    payload = mapping(inputs.joint_state)
    missing: list[str] = []
    if not payload:
        missing.append("joint_state")
    names = strings(payload.get("joint_names"))
    positions = [safe_float(v) for v in payload.get("positions", []) or []]
    velocities = [safe_float(v) for v in payload.get("velocities", []) or []]
    efforts = [safe_float(v) for v in payload.get("efforts", []) or []]
    return JointStateVector(
        vector_id=stable_id("joint_state_vector", {"state_id": state_id}),
        joint_names=names,
        positions=positions,
        velocities=velocities,
        efforts=efforts,
        timestamp_s=safe_float(payload.get("timestamp_s", 0.0)),
        truth_class="provided" if payload else "unavailable",
        missing_fields=missing,
        metadata={"joint_count": len(names)},
    )


def _compile_contact_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> ContactStateVector:
    result = inputs.advisory_embodiment_result
    profile = getattr(result, "profile", None) if result is not None else None
    contact_matrix = getattr(profile, "contact_matrix", None)
    contact_confidence = getattr(profile, "contact_confidence", None)
    contact_impossible = getattr(profile, "contact_impossible", None)
    track_ids = strings(getattr(profile, "track_ids", []) if profile is not None else [])
    event_count = int(np.asarray(contact_matrix).sum()) if contact_matrix is not None else 0
    impossible_count = int(np.asarray(contact_impossible).sum()) if contact_impossible is not None else 0
    confidence_mean = (
        float(np.asarray(contact_confidence, dtype=np.float32).mean())
        if contact_confidence is not None and np.asarray(contact_confidence).size
        else 0.0
    )
    summary = _summary(inputs)
    pair_count = safe_int(mapping(summary.get("diagnostics")).get("contact_pair_count", 0))
    return ContactStateVector(
        contact_id=stable_id("contact_state", {"state_id": state_id}),
        track_ids=track_ids,
        contact_event_count=event_count,
        contact_pair_count=pair_count,
        impossible_contact_count=impossible_count,
        contact_confidence_mean=confidence_mean,
        contact_coverage=summary.get("contact_coverage_pct", 0.0),
        source_refs={"embodiment_profile_ref": mapping(inputs.source_refs).get("embodiment_profile_ref", "")},
        metadata={"source": "advisory_embodiment_result" if result is not None else "missing"},
    )


def _compile_safety_envelope_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> SafetyEnvelopeState:
    profile = _capability_profile(inputs)
    safety = mapping(profile.safety_envelopes if profile else {})
    watchdog_ref = str(safety.get("watchdog_ref", safety.get("safety_watchdog_profile_ref", "")))
    latency_ref = str(safety.get("latency_profile_ref", safety.get("actuator_latency_profile_ref", "")))
    missing: list[str] = []
    if not watchdog_ref:
        missing.append("safety_watchdog_profile")
    if not latency_ref:
        missing.append("actuator_latency_profile")
    margin = safe_float(safety.get("margin_fraction", 0.0), 0.0)
    status = "available" if not missing else "external_blocked"
    return SafetyEnvelopeState(
        envelope_id=stable_id("safety_envelope", {"state_id": state_id}),
        status=status,
        margin_fraction=margin,
        watchdog_ref=watchdog_ref,
        latency_profile_ref=latency_ref,
        safety_limits=safety,
        missing_evidence=missing,
        authority_level="none",
    )


def _compile_action_space_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    actuator: ActuatorConfigurationState,
) -> ActionSpaceState:
    missing = list(actuator.missing_fields)
    validation = "adapter_validated" if actuator.action_schema_id else "unavailable"
    if not actuator.channel_order:
        missing.append("action_channels")
        validation = "unavailable"
    return ActionSpaceState(
        action_space_id=stable_id("action_space", {"state_id": state_id}),
        schema_id=actuator.action_schema_id,
        dimension=len(actuator.channel_order),
        channels=list(actuator.channel_order),
        normalized=bool(actuator.channel_order),
        bounds=dict(actuator.bounds),
        validation_status=validation,
        missing_fields=missing,
        translator_ref=actuator.translator_ref,
    )


def _compile_observation_interface_state(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> ObservationInterfaceState:
    adapter = inputs.observation_adapter
    missing = [] if adapter else ["observation_adapter_v2"]
    degraded: list[str] = []
    if adapter and not adapter.sensor_refs:
        degraded.append("no_sensor_refs")
    return ObservationInterfaceState(
        observation_interface_id=stable_id("observation_interface", {"state_id": state_id}),
        schema_id=adapter.schema_id if adapter else "",
        proprio_fields=list(adapter.proprio_fields) if adapter else [],
        sensor_refs=list(adapter.sensor_refs) if adapter else [],
        sample_hz=adapter.sample_hz if adapter else 0.0,
        latency_ms=adapter.latency_ms if adapter else 0.0,
        translator_ref=str(adapter.translator_ref or "") if adapter else "",
        embodiment_id=str(adapter.embodiment_id or "") if adapter else _embodiment_id(inputs),
        degraded_modes=degraded,
        quality_metrics=_perception_quality(inputs),
        validation_status="adapter_validated" if adapter else "unavailable",
        missing_fields=missing,
    )


def _perception_quality(inputs: EmbodimentActuationCompilationInputs) -> dict[str, float]:
    surface = inputs.perception_shadow_surface
    quality = mapping(getattr(surface, "evidence_quality_for_embodiment", {}))
    return float_mapping(quality)


def _compile_contact_affordance_graph_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    contact_state: ContactStateVector,
) -> ContactAffordanceGraphState:
    surface = inputs.perception_shadow_surface
    result = inputs.advisory_embodiment_result
    graph = getattr(result, "affordance_graph", None) if result is not None else None
    node_ids = getattr(graph, "node_ids", None) if graph is not None else None
    node_count = len(node_ids) if node_ids is not None else len(contact_state.track_ids)
    edge_index = getattr(graph, "edge_index", None)
    edge_count = int(np.asarray(edge_index).shape[-1]) if edge_index is not None else 0
    return ContactAffordanceGraphState(
        graph_id=stable_id("contact_affordance_graph", {"state_id": state_id}),
        graph_ref=str(mapping(inputs.source_refs).get("affordance_graph_ref", "")),
        node_count=node_count,
        edge_count=edge_count,
        actionable_object_count=safe_int(getattr(surface, "actionable_object_count", 0)),
        obstructed_object_count=safe_int(getattr(surface, "obstructed_object_count", 0)),
        scene_contact_feasibility=getattr(surface, "scene_contact_feasibility", 0.0),
        scene_affordance_coverage=getattr(surface, "scene_affordance_coverage", 0.0),
        scene_obstruction_severity=getattr(surface, "scene_obstruction_severity", 0.0),
        body_object_engagement_summary=float_mapping(
            getattr(surface, "body_object_engagement_summary", {})
        ),
        source_refs={
            "perception_shadow_surface_id": str(getattr(surface, "surface_id", "")),
            "affordance_graph_ref": mapping(inputs.source_refs).get("affordance_graph_ref", ""),
        },
    )


def _compile_local_dynamics_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    contact_state: ContactStateVector,
) -> LocalDynamicsForecastState:
    risk = clip01(
        0.5 * (contact_state.impossible_contact_count / max(contact_state.contact_event_count, 1))
        + 0.5 * (1.0 - contact_state.contact_confidence_mean)
    )
    return LocalDynamicsForecastState(
        forecast_id=stable_id("local_dynamics", {"state_id": state_id}),
        forecast_mode="heuristic_shadow",
        horizon_steps=safe_int(mapping(inputs.metadata).get("local_dynamics_horizon_steps", 1), 1),
        confidence=clip01(contact_state.contact_confidence_mean),
        contact_transition_risk=risk,
        promotion_stage="heuristic_fallback",
        blocked_reason="no_promoted_dynamics_seam",
        source_refs={"contact_state_id": contact_state.contact_id},
    )


def _compile_inverse_retarget_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    action_space: ActionSpaceState,
) -> InverseRetargetTraceState:
    missing: list[str] = []
    if not action_space.schema_id:
        missing.append("target_action_schema")
    if not action_space.translator_ref:
        missing.append("retarget_translator_ref")
    readiness = 1.0 if not missing else 0.0 if len(missing) >= 2 else 0.4
    return InverseRetargetTraceState(
        trace_id=stable_id("inverse_retarget", {"state_id": state_id}),
        retarget_mode="adapter_shadow" if readiness > 0.0 else "shadow_unavailable",
        source_action_space=str(mapping(inputs.metadata).get("source_action_space", "task_space")),
        target_action_space=action_space.schema_id,
        readiness_score=readiness,
        missing_evidence=missing,
        source_refs={"action_space_id": action_space.action_space_id},
        authority_level="none",
    )


def _compile_action_proposal_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    affordance: ContactAffordanceGraphState,
    retarget: InverseRetargetTraceState,
) -> ActionProposalBundleState:
    feasibility = clip01(
        0.45 * affordance.scene_affordance_coverage
        + 0.35 * affordance.scene_contact_feasibility
        + 0.20 * retarget.readiness_score
    )
    return ActionProposalBundleState(
        bundle_id=stable_id("action_proposal_bundle", {"state_id": state_id}),
        proposal_mode="shadow_summary",
        proposal_count=affordance.actionable_object_count,
        proposal_refs=strings(mapping(inputs.metadata).get("proposal_refs", [])),
        action_feasibility_score=feasibility,
        promotion_stage="heuristic_fallback",
        blocked_reason="no_promoted_action_proposal_seam",
        authority_level="none",
    )


def _compile_drift_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    profile: EmbodimentProfileState,
    safety: SafetyEnvelopeState,
) -> EmbodimentDriftSummaryState:
    result = inputs.advisory_embodiment_result
    drift_report = mapping(getattr(result, "drift_report", {})) if result is not None else {}
    drift_score = drift_report.get("drift_score", profile.diagnostics.get("drift_score", 0.0))
    reasons = strings(drift_report.get("drift_reasons", []))
    if profile.trust_override_candidate:
        reasons.append("trust_override_candidate")
    return EmbodimentDriftSummaryState(
        drift_id=stable_id("embodiment_drift", {"state_id": state_id}),
        drift_score=drift_score,
        calibration_due=bool(clip01(drift_score) > 0.35 or safety.missing_evidence),
        drift_reasons=reasons,
        source_refs={"embodiment_profile_state_id": profile.profile_id},
        external_blockers=list(safety.missing_evidence),
        metadata=drift_report,
    )


def _compile_cost_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    safety: SafetyEnvelopeState,
    proposals: ActionProposalBundleState,
) -> EmbodimentCostVectorState:
    result = inputs.advisory_embodiment_result
    costs = mapping(getattr(result, "cost_breakdown", {})) if result is not None else {}
    energy = safe_float(costs.get("total_energy_Wh", costs.get("energy_Wh", 0.0)))
    risk = clip01(costs.get("risk_score", 1.0 - proposals.action_feasibility_score))
    safety_penalty = 0.35 if safety.missing_evidence else 0.0
    latency_ms = safe_float(
        costs.get("latency_ms", inputs.action_adapter.latency_ms if inputs.action_adapter else 0.0)
    )
    return EmbodimentCostVectorState(
        cost_id=stable_id("embodiment_cost", {"state_id": state_id}),
        energy_wh=energy,
        risk_score=risk,
        latency_ms=latency_ms,
        safety_penalty=safety_penalty,
        maintenance_risk=clip01(costs.get("maintenance_risk", risk * 0.5)),
        source_refs={"cost_breakdown": mapping(inputs.source_refs).get("cost_breakdown_ref", "")},
        metadata=costs,
    )


def _compile_calibration_state(
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    drift: EmbodimentDriftSummaryState,
    safety: SafetyEnvelopeState,
) -> CalibrationTargetState:
    result = inputs.advisory_embodiment_result
    targets = mapping(getattr(result, "calibration_targets", {})) if result is not None else {}
    missing = list(safety.missing_evidence)
    if not targets:
        missing.append("calibration_targets")
    next_actions = [f"collect_{item}" for item in missing]
    return CalibrationTargetState(
        calibration_id=stable_id("calibration_targets", {"state_id": state_id}),
        target_refs=targets,
        priority_score=clip01(max(drift.drift_score, 0.45 if missing else 0.0)),
        missing_evidence=missing,
        next_actions=next_actions,
        source_refs={"drift_id": drift.drift_id},
    )


def _compile_provider_surface(
    state_id: str, inputs: EmbodimentActuationCompilationInputs
) -> dict[str, Any]:
    if inputs.runtime_resource_surface is not None:
        return inputs.runtime_resource_surface.to_dict()
    missing = [component for contract in inputs.provider_contracts for component in contract.missing_components]
    surface = EmbodimentRuntimeResourceSurface(
        surface_id=stable_id("embodiment_runtime_surface", {"state_id": state_id}),
        provider_contracts=list(inputs.provider_contracts),
        missing_components=missing,
        metadata={"source": "compiler_provider_contracts"},
    )
    return surface.to_dict()


def _downstream_preconditions(
    *,
    safety: SafetyEnvelopeState,
    retarget: InverseRetargetTraceState,
    proposals: ActionProposalBundleState,
    calibration: CalibrationTargetState,
) -> dict[str, Any]:
    return {
        "sim_synth_transfer": {
            "requires": ["action_feasibility_score", "retargeting_readiness_score", "drift_score"],
            "ready": bool(proposals.action_feasibility_score > 0.0),
        },
        "perception_feedback": {
            "requires": ["contact_affordance_graph", "contact_state"],
            "ready": True,
        },
        "runtime_validation": {
            "requires": ["action_space", "observation_interface", "safety_envelope"],
            "ready": bool(not safety.missing_evidence and retarget.readiness_score > 0.0),
        },
        "economic_receipt_ingest": {
            "requires": ["embodiment_cost_receipt", "safety_envelope_receipt", "drift_receipt"],
            "ready": True,
        },
        "calibration_followup": {
            "requires": list(calibration.missing_evidence),
            "ready": not calibration.missing_evidence,
        },
    }


def _base_missing(*states: Any) -> list[str]:
    missing: list[str] = []
    for state in states:
        missing.extend(strings(getattr(state, "missing_fields", [])))
        missing.extend(strings(getattr(state, "missing_evidence", [])))
        missing.extend(strings(getattr(state, "missing_inputs", [])))
    return sorted(set(missing))


def _build_receipts(
    *,
    state_id: str,
    inputs: EmbodimentActuationCompilationInputs,
    capability: CapabilityState,
    action_space: ActionSpaceState,
    observation: ObservationInterfaceState,
    affordance: ContactAffordanceGraphState,
    dynamics: LocalDynamicsForecastState,
    retarget: InverseRetargetTraceState,
    proposals: ActionProposalBundleState,
    safety: SafetyEnvelopeState,
    drift: EmbodimentDriftSummaryState,
    calibration: CalibrationTargetState,
    costs: EmbodimentCostVectorState,
) -> list[EmbodimentReceipt]:
    src = mapping(inputs.source_refs)
    return [
        CapabilityProfileReceipt(
            receipt_id=stable_id("capability_receipt", {"state_id": state_id}),
            state_id=state_id,
            robot_family=capability.robot_family,
            action_space_count=len(capability.action_spaces),
            sensor_modality_count=len(capability.sensor_modalities),
            source_refs=src,
            missing_evidence=list(capability.missing_fields),
        ),
        ActionSpaceValidationReceipt(
            receipt_id=stable_id("action_space_receipt", {"state_id": state_id}),
            state_id=state_id,
            schema_id=action_space.schema_id,
            dimension=action_space.dimension,
            validation_status=action_space.validation_status,
            source_refs=src,
            missing_evidence=list(action_space.missing_fields),
        ),
        ObservationInterfaceReceipt(
            receipt_id=stable_id("observation_interface_receipt", {"state_id": state_id}),
            state_id=state_id,
            schema_id=observation.schema_id,
            sensor_ref_count=len(observation.sensor_refs),
            proprio_field_count=len(observation.proprio_fields),
            validation_status=observation.validation_status,
            source_refs=src,
            missing_evidence=list(observation.missing_fields),
            degraded_reasons=list(observation.degraded_modes),
        ),
        ContactAffordanceReceipt(
            receipt_id=stable_id("contact_affordance_receipt", {"state_id": state_id}),
            state_id=state_id,
            node_count=affordance.node_count,
            edge_count=affordance.edge_count,
            scene_contact_feasibility=affordance.scene_contact_feasibility,
            scene_affordance_coverage=affordance.scene_affordance_coverage,
            source_refs=affordance.source_refs,
        ),
        LocalDynamicsReceipt(
            receipt_id=stable_id("local_dynamics_receipt", {"state_id": state_id}),
            state_id=state_id,
            forecast_mode=dynamics.forecast_mode,
            confidence=dynamics.confidence,
            promotion_stage=dynamics.promotion_stage,
            source_refs=dynamics.source_refs,
            missing_evidence=[dynamics.blocked_reason],
        ),
        InverseRetargetReceipt(
            receipt_id=stable_id("inverse_retarget_receipt", {"state_id": state_id}),
            state_id=state_id,
            retarget_mode=retarget.retarget_mode,
            readiness_score=retarget.readiness_score,
            source_refs=retarget.source_refs,
            missing_evidence=list(retarget.missing_evidence),
        ),
        ActionProposalReceipt(
            receipt_id=stable_id("action_proposal_receipt", {"state_id": state_id}),
            state_id=state_id,
            proposal_mode=proposals.proposal_mode,
            proposal_count=proposals.proposal_count,
            action_feasibility_score=proposals.action_feasibility_score,
            missing_evidence=[proposals.blocked_reason],
        ),
        SafetyEnvelopeReceipt(
            receipt_id=stable_id("safety_envelope_receipt", {"state_id": state_id}),
            state_id=state_id,
            safety_status=safety.status,
            margin_fraction=safety.margin_fraction,
            watchdog_ref_present=bool(safety.watchdog_ref),
            latency_profile_ref_present=bool(safety.latency_profile_ref),
            missing_evidence=list(safety.missing_evidence),
        ),
        EmbodimentDriftReceipt(
            receipt_id=stable_id("embodiment_drift_receipt", {"state_id": state_id}),
            state_id=state_id,
            drift_score=drift.drift_score,
            calibration_due=drift.calibration_due,
            source_refs=drift.source_refs,
            missing_evidence=list(drift.external_blockers),
        ),
        CalibrationTargetReceipt(
            receipt_id=stable_id("calibration_target_receipt", {"state_id": state_id}),
            state_id=state_id,
            priority_score=calibration.priority_score,
            target_count=len(calibration.target_refs),
            missing_evidence=list(calibration.missing_evidence),
            source_refs=calibration.source_refs,
        ),
        EmbodimentCostReceipt(
            receipt_id=stable_id("embodiment_cost_receipt", {"state_id": state_id}),
            state_id=state_id,
            energy_wh=costs.energy_wh,
            risk_score=costs.risk_score,
            latency_ms=costs.latency_ms,
            source_refs=costs.source_refs,
            missing_evidence=_base_missing(safety, retarget),
        ),
        SimEmbodimentTransferReceipt(
            receipt_id=stable_id("sim_embodiment_transfer_receipt", {"state_id": state_id}),
            state_id=state_id,
            transfer_status="shadow_advisory",
            action_feasibility_score=proposals.action_feasibility_score,
            retargeting_readiness_score=retarget.readiness_score,
            drift_score=drift.drift_score,
            missing_evidence=_base_missing(safety, retarget, calibration),
            downstream_preconditions=["sim_synth_transfer", "runtime_validation"],
        ),
    ]
