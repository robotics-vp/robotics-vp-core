"""Bio/neuro-inspired Embodiment WM substrate surfaces.

These are typed local surfaces for the doctrine in
``doctrine_bio_neuro_architecture_inspirations.md``. They are deterministic,
receipt-shaped, and advisory. They do not train models, write weights, dispatch
commands, mutate reward math, or promote anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .common import clip01, float_mapping, mapping, safe_float, stable_id, strings
from .state import EmbodimentActuationWorldState

DENIED_BIO_NEURO_AUTHORITIES = (
    "training_executed",
    "weights_written",
    "provider_executed",
    "hardware_executed",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
)


def _float_list(values: Optional[Sequence[Any]]) -> list[float]:
    if values is None:
        return []
    return [safe_float(value) for value in values]


@dataclass(frozen=True)
class SelfMotionExpectation:
    """Embodiment-owned prediction of self-caused sensory/body changes."""

    expectation_id: str
    embodiment_state_id: str
    episode_id: str
    frame_index: int
    predicted_body_delta: dict[str, float] = field(default_factory=dict)
    predicted_camera_ego_motion: dict[str, float] = field(default_factory=dict)
    predicted_occlusion_regions: list[dict[str, Any]] = field(default_factory=list)
    predicted_force_contact_changes: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_status: str = "heuristic_shadow_only"
    missing_evidence: list[str] = field(default_factory=list)
    authority_level: str = "none"
    promotion_eligible: bool = False
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "self_motion_expectation_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "embodiment_state_id": self.embodiment_state_id,
            "episode_id": self.episode_id,
            "frame_index": int(self.frame_index),
            "predicted_body_delta": float_mapping(self.predicted_body_delta),
            "predicted_camera_ego_motion": float_mapping(self.predicted_camera_ego_motion),
            "predicted_occlusion_regions": [
                mapping(region) for region in self.predicted_occlusion_regions
            ],
            "predicted_force_contact_changes": float_mapping(
                self.predicted_force_contact_changes
            ),
            "confidence": clip01(self.confidence),
            "evidence_status": self.evidence_status,
            "missing_evidence": strings(self.missing_evidence),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ActiveSensingProposal:
    """Bounded embodiment-local proposal to gather better evidence."""

    proposal_id: str
    embodiment_state_id: str
    action_type: str
    target_uncertainty_region: dict[str, Any] = field(default_factory=dict)
    expected_information_gain: float = 0.0
    cost_vector: dict[str, float] = field(default_factory=dict)
    safety_margin_required: float = 0.0
    value_of_information_prior: float = 0.0
    confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    authority_level: str = "none"
    promotion_eligible: bool = False
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "active_sensing_proposal_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "embodiment_state_id": self.embodiment_state_id,
            "action_type": self.action_type,
            "target_uncertainty_region": mapping(self.target_uncertainty_region),
            "expected_information_gain": clip01(self.expected_information_gain),
            "cost_vector": float_mapping(self.cost_vector),
            "safety_margin_required": clip01(self.safety_margin_required),
            "value_of_information_prior": clip01(self.value_of_information_prior),
            "confidence": clip01(self.confidence),
            "reason_codes": strings(self.reason_codes),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class SynergyCodebookEntry:
    """Heuristic placeholder for future learned motor synergy codebooks."""

    synergy_id: str
    embodiment_id: str
    posture_tag: str
    intended_function: str
    joint_group_mask: list[str] = field(default_factory=list)
    activation_pattern: list[float] = field(default_factory=list)
    confidence: float = 0.0
    learned: bool = False
    training_corpus_ref: str = ""
    blockers: list[str] = field(default_factory=list)
    authority_level: str = "none"
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "synergy_codebook_entry_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "synergy_id": self.synergy_id,
            "embodiment_id": self.embodiment_id,
            "posture_tag": self.posture_tag,
            "intended_function": self.intended_function,
            "joint_group_mask": strings(self.joint_group_mask),
            "activation_pattern": _float_list(self.activation_pattern),
            "confidence": clip01(self.confidence),
            "learned": bool(self.learned),
            "training_corpus_ref": self.training_corpus_ref,
            "blockers": strings(self.blockers),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class InteroceptiveState:
    """Typed resource/body telemetry state for Embodiment-local use."""

    interoceptive_state_id: str
    embodiment_state_id: str
    thermal_vector: dict[str, float] = field(default_factory=dict)
    battery_state: dict[str, float] = field(default_factory=dict)
    wear_estimates: dict[str, float] = field(default_factory=dict)
    compute_headroom: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    safety_margin: float = 0.0
    telemetry_quality: float = 0.0
    missing_evidence: list[str] = field(default_factory=list)
    authority_level: str = "none"
    source_refs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "interoceptive_state_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "interoceptive_state_id": self.interoceptive_state_id,
            "embodiment_state_id": self.embodiment_state_id,
            "thermal_vector": float_mapping(self.thermal_vector),
            "battery_state": float_mapping(self.battery_state),
            "wear_estimates": float_mapping(self.wear_estimates),
            "compute_headroom": float_mapping(self.compute_headroom),
            "latency_ms": max(0.0, safe_float(self.latency_ms)),
            "safety_margin": clip01(self.safety_margin),
            "telemetry_quality": clip01(self.telemetry_quality),
            "missing_evidence": strings(self.missing_evidence),
            "authority_level": self.authority_level,
            "source_refs": mapping(self.source_refs),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentBioNeuroSubstrateReceipt:
    """Receipt proving local bio/neuro substrate surfaces were materialized."""

    receipt_id: str
    embodiment_state_id: str
    surface_versions: list[str] = field(default_factory=list)
    surface_count: int = 0
    status: str = "local_substrate_emitted"
    missing_evidence: list[str] = field(default_factory=list)
    denied_authorities: list[str] = field(
        default_factory=lambda: list(DENIED_BIO_NEURO_AUTHORITIES)
    )
    authority_level: str = "none"
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_bio_neuro_substrate_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "embodiment_state_id": self.embodiment_state_id,
            "surface_versions": strings(self.surface_versions),
            "surface_count": int(self.surface_count),
            "status": self.status,
            "missing_evidence": strings(self.missing_evidence),
            "denied_authorities": strings(self.denied_authorities),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentBioNeuroSubstrateBundle:
    """Bundle of Embodiment-owned bio/neuro surfaces for downstream receipts."""

    self_motion_expectation: SelfMotionExpectation
    active_sensing_proposals: list[ActiveSensingProposal] = field(default_factory=list)
    synergy_codebook_entries: list[SynergyCodebookEntry] = field(default_factory=list)
    interoceptive_state: InteroceptiveState | None = None
    receipt: EmbodimentBioNeuroSubstrateReceipt | None = None
    version: str = "embodiment_bio_neuro_substrate_bundle_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "self_motion_expectation": self.self_motion_expectation.to_dict(),
            "active_sensing_proposals": [
                proposal.to_dict() for proposal in self.active_sensing_proposals
            ],
            "synergy_codebook_entries": [
                entry.to_dict() for entry in self.synergy_codebook_entries
            ],
            "interoceptive_state": (
                self.interoceptive_state.to_dict()
                if self.interoceptive_state is not None
                else {}
            ),
            "receipt": self.receipt.to_dict() if self.receipt is not None else {},
            "version": self.version,
        }


def build_self_motion_expectation(
    state: EmbodimentActuationWorldState,
) -> SelfMotionExpectation:
    velocities = _float_list(state.joint_state.velocities)
    efforts = _float_list(state.joint_state.efforts)
    missing: list[str] = []
    if not velocities:
        missing.append("joint_velocity_vector")
    if not efforts:
        missing.append("joint_effort_vector")
    if not state.observation_interface.sensor_refs:
        missing.append("sensor_frame_refs")

    velocity_l1 = sum(abs(value) for value in velocities)
    effort_l1 = sum(abs(value) for value in efforts)
    contact_change = float(state.contact_state.contact_event_count)
    confidence_terms = [
        1.0 if velocities else 0.0,
        1.0 if state.action_proposal_bundle.proposal_count else 0.0,
        clip01(state.local_dynamics_forecast.confidence),
        clip01(state.contact_state.contact_confidence_mean),
    ]
    confidence = sum(confidence_terms) / len(confidence_terms)
    occlusion_regions: list[dict[str, Any]] = []
    if state.contact_affordance_graph.obstructed_object_count:
        occlusion_regions.append(
            {
                "region_id": "affordance_obstruction_summary",
                "obstructed_object_count": state.contact_affordance_graph.obstructed_object_count,
                "severity": clip01(
                    state.contact_affordance_graph.scene_obstruction_severity
                ),
            }
        )

    return SelfMotionExpectation(
        expectation_id=stable_id(
            "self_motion_expectation",
            {
                "state_id": state.state_id,
                "frame_index": state.frame_index,
                "velocity_l1": velocity_l1,
                "contact_change": contact_change,
            },
        ),
        embodiment_state_id=state.state_id,
        episode_id=state.episode_id,
        frame_index=state.frame_index,
        predicted_body_delta={
            "joint_velocity_l1": velocity_l1,
            "contact_transition_risk": state.local_dynamics_forecast.contact_transition_risk,
            "action_feasibility_shift": state.action_proposal_bundle.action_feasibility_score,
        },
        predicted_camera_ego_motion={
            "egocentric_motion_available": 1.0
            if state.observation_interface.sensor_refs
            else 0.0,
            "heuristic_motion_magnitude": min(velocity_l1, 10.0) / 10.0,
        },
        predicted_occlusion_regions=occlusion_regions,
        predicted_force_contact_changes={
            "contact_event_delta": contact_change,
            "joint_effort_l1": effort_l1,
        },
        confidence=confidence,
        missing_evidence=missing,
        source_refs={"embodiment_state_id": state.state_id},
        metadata={
            "posture": "bipedal_whole_body",
            "truth_boundary": "heuristic expectation only; perception remains truth owner for observations",
        },
    )


def build_active_sensing_proposals(
    state: EmbodimentActuationWorldState,
    perception_uncertainty: Optional[Mapping[str, Any]] = None,
    value_of_information_prior: float = 0.0,
) -> list[ActiveSensingProposal]:
    uncertainty = float_mapping(perception_uncertainty or {})
    if not uncertainty:
        uncertainty = {
            "occlusion": state.contact_affordance_graph.scene_obstruction_severity,
            "semantic_confidence_gap": 1.0 - state.embodiment_profile.semantic_confidence,
            "contact_ambiguity": 1.0 - state.contact_state.contact_confidence_mean,
        }
    ranked = sorted(uncertainty.items(), key=lambda item: item[1], reverse=True)
    top_region, top_uncertainty = ranked[0] if ranked else ("unknown", 0.0)
    safety_margin = clip01(state.safety_envelope.margin_fraction)
    feasibility = clip01(state.action_proposal_bundle.action_feasibility_score)
    proposals: list[ActiveSensingProposal] = []

    if top_uncertainty > 0.35:
        action_type = "reposition_sensor"
        if "contact" in top_region and feasibility > 0.35 and safety_margin > 0.25:
            action_type = "cautious_exploratory_contact"
        elif "depth" in top_region or "occlusion" in top_region:
            action_type = "sensor_mode_switch_depth"
        expected_gain = clip01(top_uncertainty * (0.5 + 0.5 * feasibility))
        proposals.append(
            ActiveSensingProposal(
                proposal_id=stable_id(
                    "active_sensing_proposal",
                    {
                        "state_id": state.state_id,
                        "region": top_region,
                        "action_type": action_type,
                    },
                ),
                embodiment_state_id=state.state_id,
                action_type=action_type,
                target_uncertainty_region={
                    "region_id": top_region,
                    "uncertainty": top_uncertainty,
                },
                expected_information_gain=expected_gain,
                cost_vector={
                    "time_s": 1.0 if action_type != "cautious_exploratory_contact" else 2.5,
                    "energy_wh": 0.05 + 0.1 * feasibility,
                    "wear_risk": 0.02
                    if action_type != "cautious_exploratory_contact"
                    else 0.08,
                },
                safety_margin_required=0.25
                if action_type == "cautious_exploratory_contact"
                else 0.1,
                value_of_information_prior=value_of_information_prior,
                confidence=min(expected_gain, safety_margin if safety_margin else 0.5),
                reason_codes=["uncertainty_above_threshold", f"target:{top_region}"],
                source_refs={"embodiment_state_id": state.state_id},
                metadata={"generic_exploration_bonus": False},
            )
        )

    if not proposals:
        proposals.append(
            ActiveSensingProposal(
                proposal_id=stable_id(
                    "active_sensing_proposal",
                    {"state_id": state.state_id, "region": "none", "action_type": "observe_only"},
                ),
                embodiment_state_id=state.state_id,
                action_type="observe_only",
                expected_information_gain=0.0,
                cost_vector={"time_s": 0.0, "energy_wh": 0.0, "wear_risk": 0.0},
                confidence=0.5,
                reason_codes=["uncertainty_below_threshold"],
                source_refs={"embodiment_state_id": state.state_id},
            )
        )
    return proposals


def build_synergy_codebook_entries(
    state: EmbodimentActuationWorldState,
) -> list[SynergyCodebookEntry]:
    joint_names = strings(state.joint_state.joint_names)
    leg_joints = [name for name in joint_names if "hip" in name or "knee" in name or "ankle" in name]
    arm_joints = [
        name
        for name in joint_names
        if "shoulder" in name or "elbow" in name or "wrist" in name
    ]
    waist_joints = [name for name in joint_names if "waist" in name or "torso" in name]
    embodiment_id = state.capability.embodiment_id or state.actuator_configuration.embodiment_id
    blockers = [
        "no_learned_synergy_training_run",
        "no_real_interoceptive_telemetry_corpus",
        "no_hardware_calibrated_activation_patterns",
    ]
    entries = [
        SynergyCodebookEntry(
            synergy_id=stable_id(
                "synergy_codebook_entry",
                {"state_id": state.state_id, "function": "standing_balance_guard"},
            ),
            embodiment_id=embodiment_id,
            posture_tag="bipedal_whole_body",
            intended_function="standing_balance_guard",
            joint_group_mask=leg_joints + waist_joints,
            activation_pattern=[0.0 for _ in leg_joints + waist_joints],
            confidence=0.25 if leg_joints else 0.0,
            blockers=blockers,
            metadata={"heuristic_grouping_only": True},
        ),
        SynergyCodebookEntry(
            synergy_id=stable_id(
                "synergy_codebook_entry",
                {"state_id": state.state_id, "function": "bimanual_reach_stabilizer"},
            ),
            embodiment_id=embodiment_id,
            posture_tag="bipedal_whole_body",
            intended_function="bimanual_reach_stabilizer",
            joint_group_mask=arm_joints + waist_joints,
            activation_pattern=[0.0 for _ in arm_joints + waist_joints],
            confidence=0.25 if arm_joints else 0.0,
            blockers=blockers,
            metadata={"heuristic_grouping_only": True},
        ),
    ]
    return entries


def build_interoceptive_state(
    state: EmbodimentActuationWorldState,
) -> InteroceptiveState:
    provider_surface = mapping(state.provider_runtime_surface)
    missing = [
        "measured_battery_telemetry",
        "measured_thermal_telemetry",
        "measured_joint_wear_telemetry",
    ]
    telemetry_quality = 0.0
    if provider_surface:
        telemetry_quality = 0.2
    return InteroceptiveState(
        interoceptive_state_id=stable_id(
            "interoceptive_state", {"state_id": state.state_id}
        ),
        embodiment_state_id=state.state_id,
        thermal_vector={
            "thermal_headroom_heuristic": 1.0 - clip01(state.cost_vector.risk_score)
        },
        battery_state={
            "reserve_unknown": 0.0,
            "energy_cost_wh": state.cost_vector.energy_wh,
        },
        wear_estimates={
            "maintenance_risk": state.cost_vector.maintenance_risk,
            "safety_penalty": state.cost_vector.safety_penalty,
        },
        compute_headroom={
            "provider_surface_present": 1.0 if provider_surface else 0.0,
            "latency_budget_headroom": clip01(
                1.0 - (state.cost_vector.latency_ms / 1000.0)
            ),
        },
        latency_ms=state.cost_vector.latency_ms,
        safety_margin=state.safety_envelope.margin_fraction,
        telemetry_quality=telemetry_quality,
        missing_evidence=missing,
        source_refs={"embodiment_state_id": state.state_id},
        metadata={"truth_class": "heuristic_shadow_only"},
    )


def build_embodiment_bio_neuro_substrate(
    state: EmbodimentActuationWorldState,
    perception_uncertainty: Optional[Mapping[str, Any]] = None,
    value_of_information_prior: float = 0.0,
) -> EmbodimentBioNeuroSubstrateBundle:
    expectation = build_self_motion_expectation(state)
    proposals = build_active_sensing_proposals(
        state,
        perception_uncertainty=perception_uncertainty,
        value_of_information_prior=value_of_information_prior,
    )
    codebook = build_synergy_codebook_entries(state)
    interoception = build_interoceptive_state(state)
    missing = sorted(
        {
            *expectation.missing_evidence,
            *interoception.missing_evidence,
            "learned_synergy_codebook",
            "active_sensing_outcome_receipts",
        }
    )
    surface_versions = [
        expectation.version,
        *(proposal.version for proposal in proposals),
        *(entry.version for entry in codebook),
        interoception.version,
    ]
    receipt = EmbodimentBioNeuroSubstrateReceipt(
        receipt_id=stable_id(
            "embodiment_bio_neuro_substrate_receipt",
            {"state_id": state.state_id, "surface_versions": surface_versions},
        ),
        embodiment_state_id=state.state_id,
        surface_versions=surface_versions,
        surface_count=len(surface_versions),
        missing_evidence=missing,
        metadata={
            "local_only": True,
            "provider_or_hardware_proof": False,
            "phase_boundary": "lower_wm_substrate_only",
        },
    )
    return EmbodimentBioNeuroSubstrateBundle(
        self_motion_expectation=expectation,
        active_sensing_proposals=proposals,
        synergy_codebook_entries=codebook,
        interoceptive_state=interoception,
        receipt=receipt,
    )


__all__ = [
    "ActiveSensingProposal",
    "DENIED_BIO_NEURO_AUTHORITIES",
    "EmbodimentBioNeuroSubstrateBundle",
    "EmbodimentBioNeuroSubstrateReceipt",
    "InteroceptiveState",
    "SelfMotionExpectation",
    "SynergyCodebookEntry",
    "build_active_sensing_proposals",
    "build_embodiment_bio_neuro_substrate",
    "build_interoceptive_state",
    "build_self_motion_expectation",
    "build_synergy_codebook_entries",
]
