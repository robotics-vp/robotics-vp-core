"""Phase 7 Meta-Regal-Node / control WM scaffold surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.world_model.humanoid_readiness.closure import Phase35465LocalClosureAudit
from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase65 import (
    Phase65MetaNodeNeuralizationReport,
)

PHASE7_REPORT_VERSION = "phase7_meta_regal_control_scaffold_report_v1"
PHASE7_GOVERNANCE_NODE_VERSION = "phase7_governance_node_surface_v1"
PHASE7_COMPOSITION_MODE_VERSION = "phase7_composition_mode_spec_v1"
PHASE7_CONFLICT_RECEIPT_VERSION = "phase7_conflict_override_receipt_v1"
PHASE7_ADMISSIBLE_REGION_VERSION = "phase7_admissible_region_spec_v1"
PHASE7_CONTROL_FIELD_VERSION = "phase7_control_field_slot_v1"
PHASE7_TRAINING_ROW_VERSION = "phase7_training_row_slot_v1"
PHASE7_PROMOTION_GATE_VERSION = "phase7_promotion_gate_v1"

PHASE7_REMAINING_BLOCKERS = (
    "lower_wm_bounded_runtime_authority_missing",
    "governance_node_training_and_benchmark_evidence_missing",
    "cross_wm_governance_corpus_density_missing",
    "meta_composition_learning_not_trained",
    "real_governance_benchmark_evidence_missing",
    "live_runtime_wiring_not_executed",
    "provider_hardware_deployment_evidence_missing",
)

DENIED_PHASE7_AUTHORITIES = (
    "training_execution",
    "weight_write",
    "provider_execution",
    "hardware_execution",
    "unitree_sim_runtime",
    "live_policy_control",
    "reward_math_mutation",
    "promotion",
    "live_cross_wm_control",
    "hard_veto_dispatch",
    "lower_wm_replacement",
    "scalar_governance_collapse",
    "phase7_runtime_authority",
)


def _phase7_denied_gate_map(
    extra: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    return denied_gate_map(
        {
            "phase7_runtime_authority": False,
            "live_cross_wm_control": False,
            "hard_veto_dispatch": False,
            "lower_wm_replacement": False,
            "scalar_governance_collapse": False,
            **dict(extra or {}),
        }
    )


@dataclass(frozen=True)
class Phase7GovernanceNodeSurface:
    surface_id: str
    node_key: str
    domain_key: str
    composition_role: str
    maturity_stage: str
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    confidence_prior: float = 0.0
    hard_constraint_capable: bool = False
    advisory_only: bool = True
    bounded_helper_ready: bool = False
    training_aware: bool = True
    authority_class: str = "phase7_governance_node_surface_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_GOVERNANCE_NODE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "version": self.version,
            "node_key": self.node_key,
            "domain_key": self.domain_key,
            "composition_role": self.composition_role,
            "maturity_stage": self.maturity_stage,
            "input_refs": list(self.input_refs),
            "output_refs": list(self.output_refs),
            "confidence_prior": float(self.confidence_prior),
            "hard_constraint_capable": bool(self.hard_constraint_capable),
            "advisory_only": bool(self.advisory_only),
            "bounded_helper_ready": bool(self.bounded_helper_ready),
            "training_aware": bool(self.training_aware),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7GovernanceNodeSurface":
        return cls(
            surface_id=str(payload.get("surface_id", "")),
            node_key=str(payload.get("node_key", "")),
            domain_key=str(payload.get("domain_key", "")),
            composition_role=str(payload.get("composition_role", "")),
            maturity_stage=str(payload.get("maturity_stage", "")),
            input_refs=strings(payload.get("input_refs")),
            output_refs=strings(payload.get("output_refs")),
            confidence_prior=float(payload.get("confidence_prior", 0.0) or 0.0),
            hard_constraint_capable=bool(
                payload.get("hard_constraint_capable", False)
            ),
            advisory_only=bool(payload.get("advisory_only", True)),
            bounded_helper_ready=bool(payload.get("bounded_helper_ready", False)),
            training_aware=bool(payload.get("training_aware", True)),
            authority_class=str(
                payload.get("authority_class", "phase7_governance_node_surface_only")
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_GOVERNANCE_NODE_VERSION)),
        )


@dataclass(frozen=True)
class Phase7CompositionModeSpec:
    mode_id: str
    mode_key: str
    composition_family: str
    description: str
    required_input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)
    allowed_authority: str = "shadow_composition_only"
    hard_constraint_semantics: bool = False
    shadow_only: bool = True
    training_aware: bool = True
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_COMPOSITION_MODE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "version": self.version,
            "mode_key": self.mode_key,
            "composition_family": self.composition_family,
            "description": self.description,
            "required_input_fields": list(self.required_input_fields),
            "output_fields": list(self.output_fields),
            "allowed_authority": self.allowed_authority,
            "hard_constraint_semantics": bool(self.hard_constraint_semantics),
            "shadow_only": bool(self.shadow_only),
            "training_aware": bool(self.training_aware),
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7CompositionModeSpec":
        return cls(
            mode_id=str(payload.get("mode_id", "")),
            mode_key=str(payload.get("mode_key", "")),
            composition_family=str(payload.get("composition_family", "")),
            description=str(payload.get("description", "")),
            required_input_fields=strings(payload.get("required_input_fields")),
            output_fields=strings(payload.get("output_fields")),
            allowed_authority=str(
                payload.get("allowed_authority", "shadow_composition_only")
            ),
            hard_constraint_semantics=bool(
                payload.get("hard_constraint_semantics", False)
            ),
            shadow_only=bool(payload.get("shadow_only", True)),
            training_aware=bool(payload.get("training_aware", True)),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_COMPOSITION_MODE_VERSION)),
        )


@dataclass(frozen=True)
class Phase7ConflictOverrideReceipt:
    receipt_id: str
    conflict_key: str
    source_node_ids: list[str]
    composition_mode: str
    severity_prior: float
    override_policy: str
    provenance_refs: list[str] = field(default_factory=list)
    observational_only: bool = True
    shadow_only: bool = True
    training_aware: bool = True
    authority_class: str = "phase7_conflict_receipt_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_CONFLICT_RECEIPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "conflict_key": self.conflict_key,
            "source_node_ids": list(self.source_node_ids),
            "composition_mode": self.composition_mode,
            "severity_prior": float(self.severity_prior),
            "override_policy": self.override_policy,
            "provenance_refs": list(self.provenance_refs),
            "observational_only": bool(self.observational_only),
            "shadow_only": bool(self.shadow_only),
            "training_aware": bool(self.training_aware),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7ConflictOverrideReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            conflict_key=str(payload.get("conflict_key", "")),
            source_node_ids=strings(payload.get("source_node_ids")),
            composition_mode=str(payload.get("composition_mode", "")),
            severity_prior=float(payload.get("severity_prior", 0.0) or 0.0),
            override_policy=str(payload.get("override_policy", "")),
            provenance_refs=strings(payload.get("provenance_refs")),
            observational_only=bool(payload.get("observational_only", True)),
            shadow_only=bool(payload.get("shadow_only", True)),
            training_aware=bool(payload.get("training_aware", True)),
            authority_class=str(
                payload.get("authority_class", "phase7_conflict_receipt_only")
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_CONFLICT_RECEIPT_VERSION)),
        )


@dataclass(frozen=True)
class Phase7AdmissibleRegionSpec:
    region_id: str
    regime_key: str
    active_node_ids: list[str]
    admissibility_conditions: list[str] = field(default_factory=list)
    hard_veto_sources: list[str] = field(default_factory=list)
    pareto_dimensions: list[str] = field(default_factory=list)
    lower_wm_refs: list[str] = field(default_factory=list)
    output_ref: str = ""
    evaluation_only: bool = True
    promotion_eligible: bool = False
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_ADMISSIBLE_REGION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "version": self.version,
            "regime_key": self.regime_key,
            "active_node_ids": list(self.active_node_ids),
            "admissibility_conditions": list(self.admissibility_conditions),
            "hard_veto_sources": list(self.hard_veto_sources),
            "pareto_dimensions": list(self.pareto_dimensions),
            "lower_wm_refs": list(self.lower_wm_refs),
            "output_ref": self.output_ref,
            "evaluation_only": bool(self.evaluation_only),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7AdmissibleRegionSpec":
        return cls(
            region_id=str(payload.get("region_id", "")),
            regime_key=str(payload.get("regime_key", "")),
            active_node_ids=strings(payload.get("active_node_ids")),
            admissibility_conditions=strings(
                payload.get("admissibility_conditions")
            ),
            hard_veto_sources=strings(payload.get("hard_veto_sources")),
            pareto_dimensions=strings(payload.get("pareto_dimensions")),
            lower_wm_refs=strings(payload.get("lower_wm_refs")),
            output_ref=str(payload.get("output_ref", "")),
            evaluation_only=bool(payload.get("evaluation_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_ADMISSIBLE_REGION_VERSION)),
        )


@dataclass(frozen=True)
class Phase7ControlFieldSlot:
    slot_id: str
    field_key: str
    target_surface: str
    source_node_ids: list[str]
    composition_mode: str
    field_schema: dict[str, Any] = field(default_factory=dict)
    output_authority: str = "shadow_field_only"
    shadow_only: bool = True
    live_dispatch_allowed: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_authority: list[str] = field(default_factory=list)
    version: str = PHASE7_CONTROL_FIELD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "version": self.version,
            "field_key": self.field_key,
            "target_surface": self.target_surface,
            "source_node_ids": list(self.source_node_ids),
            "composition_mode": self.composition_mode,
            "field_schema": mapping(self.field_schema),
            "output_authority": self.output_authority,
            "shadow_only": bool(self.shadow_only),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7ControlFieldSlot":
        return cls(
            slot_id=str(payload.get("slot_id", "")),
            field_key=str(payload.get("field_key", "")),
            target_surface=str(payload.get("target_surface", "")),
            source_node_ids=strings(payload.get("source_node_ids")),
            composition_mode=str(payload.get("composition_mode", "")),
            field_schema=mapping(payload.get("field_schema")),
            output_authority=str(payload.get("output_authority", "shadow_field_only")),
            shadow_only=bool(payload.get("shadow_only", True)),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", PHASE7_CONTROL_FIELD_VERSION)),
        )


@dataclass(frozen=True)
class Phase7TrainingRowSlot:
    row_id: str
    row_family: str
    source_refs: list[str] = field(default_factory=list)
    label_slots: dict[str, Any] = field(default_factory=dict)
    outcome_join_slots: dict[str, Any] = field(default_factory=dict)
    replay_export_ready: bool = True
    training_target_only: bool = True
    weights_written: bool = False
    promotion_eligible: bool = False
    version: str = PHASE7_TRAINING_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "row_family": self.row_family,
            "source_refs": list(self.source_refs),
            "label_slots": mapping(self.label_slots),
            "outcome_join_slots": mapping(self.outcome_join_slots),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_target_only": bool(self.training_target_only),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7TrainingRowSlot":
        return cls(
            row_id=str(payload.get("row_id", "")),
            row_family=str(payload.get("row_family", "")),
            source_refs=strings(payload.get("source_refs")),
            label_slots=mapping(payload.get("label_slots")),
            outcome_join_slots=mapping(payload.get("outcome_join_slots")),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            training_target_only=bool(payload.get("training_target_only", True)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_TRAINING_ROW_VERSION)),
        )


@dataclass(frozen=True)
class Phase7PromotionGate:
    gate_id: str
    requested_authority: str
    gate_status: str = "denied"
    missing_evidence: list[str] = field(default_factory=list)
    authority_granted: bool = False
    promotion_eligible: bool = False
    version: str = PHASE7_PROMOTION_GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "version": self.version,
            "requested_authority": self.requested_authority,
            "gate_status": self.gate_status,
            "missing_evidence": list(self.missing_evidence),
            "authority_granted": bool(self.authority_granted),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7PromotionGate":
        return cls(
            gate_id=str(payload.get("gate_id", "")),
            requested_authority=str(payload.get("requested_authority", "")),
            gate_status=str(payload.get("gate_status", "denied")),
            missing_evidence=strings(payload.get("missing_evidence")),
            authority_granted=bool(payload.get("authority_granted", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_PROMOTION_GATE_VERSION)),
        )


@dataclass(frozen=True)
class Phase7MetaRegalControlScaffoldReport:
    report_id: str
    phase65_report_id: str
    closure_audit_id: str
    status: str
    governance_node_surface_count: int
    composition_mode_count: int
    conflict_override_receipt_count: int
    admissible_region_count: int
    control_field_slot_count: int
    training_row_slot_count: int
    promotion_gate_count: int
    local_phase7_scaffold_complete: bool
    ready_for_runtime_wiring: bool
    runtime_wiring_executed: bool = False
    phase7_authority_granted: bool = False
    live_control_authority: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_phase7_denied_gate_map)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase65_report_id": self.phase65_report_id,
            "closure_audit_id": self.closure_audit_id,
            "status": self.status,
            "governance_node_surface_count": int(
                self.governance_node_surface_count
            ),
            "composition_mode_count": int(self.composition_mode_count),
            "conflict_override_receipt_count": int(
                self.conflict_override_receipt_count
            ),
            "admissible_region_count": int(self.admissible_region_count),
            "control_field_slot_count": int(self.control_field_slot_count),
            "training_row_slot_count": int(self.training_row_slot_count),
            "promotion_gate_count": int(self.promotion_gate_count),
            "local_phase7_scaffold_complete": bool(
                self.local_phase7_scaffold_complete
            ),
            "ready_for_runtime_wiring": bool(self.ready_for_runtime_wiring),
            "runtime_wiring_executed": bool(self.runtime_wiring_executed),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "live_control_authority": bool(self.live_control_authority),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _phase7_denied_gate_map(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7MetaRegalControlScaffoldReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase65_report_id=str(payload.get("phase65_report_id", "")),
            closure_audit_id=str(payload.get("closure_audit_id", "")),
            status=str(payload.get("status", "blocked")),
            governance_node_surface_count=int(
                payload.get("governance_node_surface_count", 0) or 0
            ),
            composition_mode_count=int(payload.get("composition_mode_count", 0) or 0),
            conflict_override_receipt_count=int(
                payload.get("conflict_override_receipt_count", 0) or 0
            ),
            admissible_region_count=int(
                payload.get("admissible_region_count", 0) or 0
            ),
            control_field_slot_count=int(
                payload.get("control_field_slot_count", 0) or 0
            ),
            training_row_slot_count=int(
                payload.get("training_row_slot_count", 0) or 0
            ),
            promotion_gate_count=int(payload.get("promotion_gate_count", 0) or 0),
            local_phase7_scaffold_complete=bool(
                payload.get("local_phase7_scaffold_complete", False)
            ),
            ready_for_runtime_wiring=bool(
                payload.get("ready_for_runtime_wiring", False)
            ),
            runtime_wiring_executed=bool(
                payload.get("runtime_wiring_executed", False)
            ),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            live_control_authority=bool(payload.get("live_control_authority", False)),
            training_executed=bool(payload.get("training_executed", False)),
            weights_written=bool(payload.get("weights_written", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            unitree_sim_runtime_executed=bool(
                payload.get("unitree_sim_runtime_executed", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            denied_gates=_phase7_denied_gate_map(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE7_REPORT_VERSION)),
        )


def _surface_id(node_key: str) -> str:
    return f"phase7_node_{node_key}"


def build_phase7_meta_regal_control_scaffold(
    *,
    phase65_report: Phase65MetaNodeNeuralizationReport,
    closure_audit: Phase35465LocalClosureAudit,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase7MetaRegalControlScaffoldReport,
    list[Phase7GovernanceNodeSurface],
    list[Phase7CompositionModeSpec],
    list[Phase7ConflictOverrideReceipt],
    list[Phase7AdmissibleRegionSpec],
    list[Phase7ControlFieldSlot],
    list[Phase7TrainingRowSlot],
    list[Phase7PromotionGate],
]:
    denied = list(DENIED_PHASE7_AUTHORITIES)
    common_refs = [
        phase65_report.report_id,
        closure_audit.audit_id,
        "economic_wm_allocation_envelope_ref",
        "phase6_transport_quality_ref",
    ]
    node_specs = [
        (
            "economic_allocation_governance",
            "economic_wm",
            "pareto_budget_voice",
            ["allocation_envelope", "opportunity_cost", "value_of_information"],
            ["budget_constraint_candidate", "resource_tradeoff_hint"],
            0.42,
            False,
        ),
        (
            "reward_integrity_governance",
            "anti_reward_hacking",
            "hard_integrity_guard",
            ["reward_channel_consistency", "exploit_suspicion_receipt"],
            ["reward_integrity_veto_candidate", "integrity_escalation_hint"],
            0.34,
            True,
        ),
        (
            "plausibility_geometry_governance",
            "plausibility_geometry",
            "physical_truth_guard",
            ["grounding_quality", "geometry_consistency", "sim_real_delta"],
            ["plausibility_constraint_candidate", "geometry_conflict_hint"],
            0.38,
            True,
        ),
        (
            "deployment_truth_governance",
            "deployment_truth",
            "runtime_evidence_guard",
            ["provider_truth", "hardware_runtime_evidence", "preflight_receipt"],
            ["deployment_truth_veto_candidate", "runtime_readiness_hint"],
            0.36,
            True,
        ),
        (
            "safety_constraint_governance",
            "safety",
            "hard_safety_guard",
            ["estop_state", "joint_limit_envelope", "recovery_state"],
            ["safety_veto_candidate", "degraded_mode_hint"],
            0.5,
            True,
        ),
        (
            "data_value_governance",
            "data_value",
            "advisory_data_voice",
            ["replay_fidelity", "annotation_integrity", "corpus_gap_receipt"],
            ["data_collection_priority_candidate", "label_need_hint"],
            0.32,
            False,
        ),
        (
            "embodiment_limit_governance",
            "embodiment_limits",
            "posture_and_capacity_guard",
            ["bipedal_balance_state", "limb_coordinate_frame", "capacity_band"],
            ["embodiment_feasibility_constraint", "posture_demote_hint"],
            0.45,
            True,
        ),
        (
            "coordination_operator_governance",
            "coordination_operator",
            "handoff_and_recovery_guard",
            ["operator_intent", "teleop_recovery_state", "comms_qos"],
            ["operator_handoff_candidate", "coordination_conflict_hint"],
            0.35,
            True,
        ),
    ]
    surfaces = [
        Phase7GovernanceNodeSurface(
            surface_id=_surface_id(node_key),
            node_key=node_key,
            domain_key=domain_key,
            composition_role=role,
            maturity_stage="stage_a_typed_non_neural_scaffold",
            input_refs=[*common_refs, *inputs],
            output_refs=outputs,
            confidence_prior=confidence_prior,
            hard_constraint_capable=hard_constraint_capable,
            denied_authority=denied,
        )
        for (
            node_key,
            domain_key,
            role,
            inputs,
            outputs,
            confidence_prior,
            hard_constraint_capable,
        ) in node_specs
    ]
    node_ids = [surface.surface_id for surface in surfaces]
    node_by_key = {surface.node_key: surface.surface_id for surface in surfaces}

    modes = [
        Phase7CompositionModeSpec(
            mode_id="composition_pareto_relation",
            mode_key="pareto_relation",
            composition_family="inter_domain_tradeoff",
            description="Preserve multiple non-dominated node objectives.",
            required_input_fields=["node_output", "confidence", "tradeoff_axis"],
            output_fields=["pareto_region_ref", "active_tradeoff_receipt"],
            denied_authority=denied,
        ),
        Phase7CompositionModeSpec(
            mode_id="composition_lexicographic_priority",
            mode_key="lexicographic_priority",
            composition_family="ordered_domain_priority",
            description="Let one domain constrain lower-priority domains.",
            required_input_fields=["priority_node", "bounded_candidate_set"],
            output_fields=["priority_order_ref", "bounded_feasible_region_ref"],
            denied_authority=denied,
        ),
        Phase7CompositionModeSpec(
            mode_id="composition_veto_constraint",
            mode_key="veto_constraint",
            composition_family="hard_constraint_relation",
            description="Emit a typed hard-constraint candidate without dispatch.",
            required_input_fields=["veto_source", "candidate_action", "evidence_ref"],
            output_fields=["veto_candidate_ref", "blocked_region_ref"],
            hard_constraint_semantics=True,
            denied_authority=denied,
        ),
        Phase7CompositionModeSpec(
            mode_id="composition_advisory_evidence",
            mode_key="advisory_evidence",
            composition_family="information_only_relation",
            description="Carry non-binding evidence into a shadow field.",
            required_input_fields=["advisory_signal", "provenance_ref"],
            output_fields=["advisory_signal_ref", "nonbinding_context_ref"],
            denied_authority=denied,
        ),
        Phase7CompositionModeSpec(
            mode_id="composition_confidence_weighted",
            mode_key="confidence_weighted",
            composition_family="epistemic_uncertainty_relation",
            description="Weight node influence by confidence and evidence quality.",
            required_input_fields=["node_confidence", "evidence_quality"],
            output_fields=["confidence_weight_ref", "uncertainty_receipt_ref"],
            denied_authority=denied,
        ),
    ]

    conflicts = [
        (
            "safety_vs_economic_throughput",
            [
                node_by_key["safety_constraint_governance"],
                node_by_key["economic_allocation_governance"],
            ],
            "veto_constraint",
            0.9,
            "safety_candidate_bounds_economic_candidate_in_shadow",
        ),
        (
            "deployment_truth_vs_data_collection",
            [
                node_by_key["deployment_truth_governance"],
                node_by_key["data_value_governance"],
            ],
            "lexicographic_priority",
            0.8,
            "deployment_truth_blocks_collection_claims_without_runtime_evidence",
        ),
        (
            "reward_integrity_vs_economic_value",
            [
                node_by_key["reward_integrity_governance"],
                node_by_key["economic_allocation_governance"],
            ],
            "veto_constraint",
            0.85,
            "integrity_suspicion_prevents_reward_channel_trust_in_shadow",
        ),
        (
            "plausibility_vs_data_value",
            [
                node_by_key["plausibility_geometry_governance"],
                node_by_key["data_value_governance"],
            ],
            "confidence_weighted",
            0.65,
            "low_plausibility_reduces_data_value_without_deleting_receipts",
        ),
        (
            "embodiment_limit_vs_economic_plan",
            [
                node_by_key["embodiment_limit_governance"],
                node_by_key["economic_allocation_governance"],
            ],
            "veto_constraint",
            0.88,
            "body_capacity_bounds_allocation_candidate_in_shadow",
        ),
        (
            "operator_recovery_vs_autonomy",
            [
                node_by_key["coordination_operator_governance"],
                node_by_key["deployment_truth_governance"],
            ],
            "lexicographic_priority",
            0.75,
            "operator_recovery_state_preempts_autonomy_candidate_in_shadow",
        ),
    ]
    conflict_receipts = [
        Phase7ConflictOverrideReceipt(
            receipt_id=f"phase7_conflict_{conflict_key}",
            conflict_key=conflict_key,
            source_node_ids=source_node_ids,
            composition_mode=composition_mode,
            severity_prior=severity_prior,
            override_policy=override_policy,
            provenance_refs=common_refs,
            denied_authority=denied,
        )
        for (
            conflict_key,
            source_node_ids,
            composition_mode,
            severity_prior,
            override_policy,
        ) in conflicts
    ]

    regions = [
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_nominal_bipedal_shadow",
            regime_key="nominal_bipedal_shadow",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "bipedal_whole_body_primary_standard",
                "no_live_dispatch",
                "all_hard_constraints_shadow_only",
            ],
            hard_veto_sources=[
                node_by_key["safety_constraint_governance"],
                node_by_key["deployment_truth_governance"],
                node_by_key["reward_integrity_governance"],
            ],
            pareto_dimensions=[
                "economic_value",
                "safety_margin",
                "plausibility",
                "deployment_truth",
                "data_value",
            ],
            lower_wm_refs=common_refs,
            output_ref="shadow_nominal_bipedal_region",
            denied_authority=denied,
        ),
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_degraded_stable_base_fallback",
            regime_key="degraded_stable_base_fallback",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "stable_base_is_degraded_mode",
                "balance_or_recovery_evidence_required",
                "fixed_base_tabletop_not_primary_deployment",
            ],
            hard_veto_sources=[
                node_by_key["embodiment_limit_governance"],
                node_by_key["safety_constraint_governance"],
            ],
            pareto_dimensions=["safety_margin", "recovery_cost", "economic_value"],
            lower_wm_refs=common_refs,
            output_ref="shadow_stable_base_demote_region",
            denied_authority=denied,
        ),
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_operator_recovery_required",
            regime_key="operator_recovery_required",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "estop_or_recovery_latch_present",
                "operator_handoff_trace_required",
                "autonomy_candidate_shadow_only",
            ],
            hard_veto_sources=[
                node_by_key["coordination_operator_governance"],
                node_by_key["safety_constraint_governance"],
            ],
            pareto_dimensions=["recovery_time", "human_supervision", "data_value"],
            lower_wm_refs=common_refs,
            output_ref="shadow_operator_recovery_region",
            denied_authority=denied,
        ),
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_deployment_truth_blocked",
            regime_key="deployment_truth_blocked",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "runtime_claim_missing_evidence",
                "provider_or_hardware_claim_denied",
                "artifact_receipts_remain_shadow_only",
            ],
            hard_veto_sources=[node_by_key["deployment_truth_governance"]],
            pareto_dimensions=["deployment_truth", "economic_value", "data_value"],
            lower_wm_refs=common_refs,
            output_ref="shadow_deployment_truth_blocked_region",
            denied_authority=denied,
        ),
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_reward_integrity_suspect",
            regime_key="reward_integrity_suspect",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "reward_hack_suspicion_present",
                "reward_math_not_mutated",
                "economic_allocation_not_sovereign",
            ],
            hard_veto_sources=[node_by_key["reward_integrity_governance"]],
            pareto_dimensions=[
                "integrity_confidence",
                "economic_value",
                "plausibility",
            ],
            lower_wm_refs=common_refs,
            output_ref="shadow_reward_integrity_region",
            denied_authority=denied,
        ),
        Phase7AdmissibleRegionSpec(
            region_id="phase7_region_resource_degraded",
            regime_key="compute_battery_comms_degraded",
            active_node_ids=node_ids,
            admissibility_conditions=[
                "battery_or_thermal_or_comms_degradation",
                "companion_compute_contract_required",
                "local_only_no_provider_claim",
            ],
            hard_veto_sources=[
                node_by_key["deployment_truth_governance"],
                node_by_key["embodiment_limit_governance"],
            ],
            pareto_dimensions=["compute_budget", "latency", "battery", "safety"],
            lower_wm_refs=common_refs,
            output_ref="shadow_resource_degraded_region",
            denied_authority=denied,
        ),
    ]

    control_fields = [
        Phase7ControlFieldSlot(
            slot_id="phase7_field_cross_wm_shaping",
            field_key="cross_wm_shaping_field",
            target_surface="lower_wm_shadow_shaping_bus",
            source_node_ids=node_ids,
            composition_mode="pareto_relation",
            field_schema={
                "shape": "typed_shadow_vector",
                "provenance": "required",
                "dispatch": "denied",
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_budget_constraint",
            field_key="economic_budget_constraint_field",
            target_surface="economic_wm_shadow_budget_bus",
            source_node_ids=[
                node_by_key["economic_allocation_governance"],
                node_by_key["deployment_truth_governance"],
            ],
            composition_mode="pareto_relation",
            field_schema={
                "budget_envelope_ref": "required",
                "safety_filter_ref": "required",
                "reward_math_mutation": False,
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_safety_veto",
            field_key="safety_veto_field",
            target_surface="safety_shadow_veto_bus",
            source_node_ids=[
                node_by_key["safety_constraint_governance"],
                node_by_key["embodiment_limit_governance"],
            ],
            composition_mode="veto_constraint",
            field_schema={
                "veto_candidate": "typed",
                "evidence_ref": "required",
                "hard_dispatch": "denied",
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_deployment_truth_veto",
            field_key="deployment_truth_veto_field",
            target_surface="deployment_truth_shadow_bus",
            source_node_ids=[node_by_key["deployment_truth_governance"]],
            composition_mode="veto_constraint",
            field_schema={
                "truth_gap_ref": "required",
                "runtime_claim_allowed": False,
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_operator_handoff",
            field_key="operator_recovery_handoff_field",
            target_surface="operator_recovery_shadow_bus",
            source_node_ids=[
                node_by_key["coordination_operator_governance"],
                node_by_key["safety_constraint_governance"],
            ],
            composition_mode="lexicographic_priority",
            field_schema={
                "handoff_state": "latched_or_clear",
                "recovery_trace_ref": "required",
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_data_collection_priority",
            field_key="data_collection_priority_field",
            target_surface="data_value_shadow_bus",
            source_node_ids=[
                node_by_key["data_value_governance"],
                node_by_key["plausibility_geometry_governance"],
            ],
            composition_mode="confidence_weighted",
            field_schema={
                "collection_priority": "shadow_label",
                "corpus_gap_ref": "required",
                "training_dispatch": "denied",
            },
            denied_authority=denied,
        ),
        Phase7ControlFieldSlot(
            slot_id="phase7_field_embodiment_mode_demote",
            field_key="embodiment_mode_demote_field",
            target_surface="humanoid_posture_shadow_bus",
            source_node_ids=[
                node_by_key["embodiment_limit_governance"],
                node_by_key["safety_constraint_governance"],
            ],
            composition_mode="veto_constraint",
            field_schema={
                "primary_posture": "bipedal_whole_body",
                "fallback_posture": "stable_base_mobile_manipulator",
                "fixed_base_tabletop": "curriculum_regression_only",
            },
            denied_authority=denied,
        ),
    ]

    training_rows = [
        Phase7TrainingRowSlot(
            row_id="phase7_row_governance_node_snapshot",
            row_family="governance_node_snapshot",
            source_refs=common_refs,
            label_slots={
                "node_confidence": "awaiting_governance_benchmark_label",
                "node_output_quality": "awaiting_shadow_outcome_join",
            },
            outcome_join_slots={
                "downstream_improvement": None,
                "governance_satisfaction": None,
            },
        ),
        Phase7TrainingRowSlot(
            row_id="phase7_row_conflict_override",
            row_family="conflict_override",
            source_refs=[receipt.receipt_id for receipt in conflict_receipts],
            label_slots={
                "override_correctness": "awaiting_postmortem_label",
                "regime_fit": "awaiting_shadow_eval",
            },
            outcome_join_slots={
                "counterfactual_policy_delta": None,
                "operator_recovery_delta": None,
            },
        ),
        Phase7TrainingRowSlot(
            row_id="phase7_row_admissible_region",
            row_family="admissible_region",
            source_refs=[region.region_id for region in regions],
            label_slots={
                "region_membership": "awaiting_runtime_trace",
                "region_stability": "awaiting_jitter_and_replay_trace",
            },
            outcome_join_slots={
                "region_exit_reason": None,
                "veto_false_positive": None,
            },
        ),
        Phase7TrainingRowSlot(
            row_id="phase7_row_control_field_shadow_outcome",
            row_family="control_field_shadow_outcome",
            source_refs=[field.slot_id for field in control_fields],
            label_slots={
                "field_legibility": "awaiting_reviewer_or_metric_label",
                "field_effectiveness": "awaiting_shadow_comparison",
            },
            outcome_join_slots={
                "live_action_delta": None,
                "policy_regret_delta": None,
                "reward_math_changed": False,
            },
        ),
        Phase7TrainingRowSlot(
            row_id="phase7_row_counterfactual_composition_target",
            row_family="counterfactual_composition_target",
            source_refs=common_refs,
            label_slots={
                "pareto_mode_target": "awaiting_counterfactual_corpus",
                "veto_mode_target": "awaiting_safety_case_label",
            },
            outcome_join_slots={
                "alternative_composition_outcome": None,
                "blocked_action_outcome": None,
            },
        ),
        Phase7TrainingRowSlot(
            row_id="phase7_row_failure_recovery",
            row_family="governance_failure_and_recovery",
            source_refs=common_refs,
            label_slots={
                "failure_mode": "awaiting_runtime_or_sim_evidence",
                "recovery_quality": "awaiting_operator_trace",
            },
            outcome_join_slots={
                "time_to_recovery": None,
                "safety_margin_after_recovery": None,
            },
        ),
    ]

    gates = [
        Phase7PromotionGate(
            gate_id="phase7_gate_runtime_authority",
            requested_authority="phase7_runtime_authority",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_live_cross_wm_shaping",
            requested_authority="live_cross_wm_shaping",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_hard_veto_dispatch",
            requested_authority="hard_veto_dispatch",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_lower_wm_replacement",
            requested_authority="lower_wm_replacement",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_reward_math_mutation",
            requested_authority="reward_math_mutation",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_training_execution",
            requested_authority="training_execution",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_weight_write",
            requested_authority="weight_write",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
        Phase7PromotionGate(
            gate_id="phase7_gate_promotion",
            requested_authority="promotion",
            missing_evidence=list(PHASE7_REMAINING_BLOCKERS),
        ),
    ]

    input_ready = (
        phase65_report.status == "ok"
        and phase65_report.local_meta_node_scaffold_complete
        and phase65_report.ready_for_phase7_scaffold
        and not phase65_report.phase7_authority_granted
        and closure_audit.status == "ok"
        and closure_audit.all_local_structures_complete
        and closure_audit.ready_for_phase7_scaffold
        and not closure_audit.phase7_authority_granted
    )
    complete = (
        input_ready
        and len(surfaces) >= 7
        and len(modes) == 5
        and len(conflict_receipts) >= 5
        and len(regions) >= 5
        and len(control_fields) >= 6
        and len(training_rows) >= 5
        and len(gates) >= 5
    )
    report_payload = {
        "phase65_report_id": phase65_report.report_id,
        "closure_audit_id": closure_audit.audit_id,
        "governance_node_surface_count": len(surfaces),
        "composition_mode_count": len(modes),
        "artifact_refs": mapping(artifact_refs),
    }
    report = Phase7MetaRegalControlScaffoldReport(
        report_id=stable_id("phase7_meta_regal", report_payload),
        phase65_report_id=phase65_report.report_id,
        closure_audit_id=closure_audit.audit_id,
        status="ok" if complete else "blocked",
        governance_node_surface_count=len(surfaces),
        composition_mode_count=len(modes),
        conflict_override_receipt_count=len(conflict_receipts),
        admissible_region_count=len(regions),
        control_field_slot_count=len(control_fields),
        training_row_slot_count=len(training_rows),
        promotion_gate_count=len(gates),
        local_phase7_scaffold_complete=complete,
        ready_for_runtime_wiring=complete,
        denied_gates=_phase7_denied_gate_map(),
        remaining_blockers=list(PHASE7_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return (
        report,
        surfaces,
        modes,
        conflict_receipts,
        regions,
        control_fields,
        training_rows,
        gates,
    )


def save_phase7_meta_regal_control_scaffold(
    output_dir: str | Path,
    report: Phase7MetaRegalControlScaffoldReport,
    surfaces: list[Phase7GovernanceNodeSurface],
    modes: list[Phase7CompositionModeSpec],
    conflict_receipts: list[Phase7ConflictOverrideReceipt],
    regions: list[Phase7AdmissibleRegionSpec],
    control_fields: list[Phase7ControlFieldSlot],
    training_rows: list[Phase7TrainingRowSlot],
    gates: list[Phase7PromotionGate],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase7_meta_regal_control_scaffold_report_v1.json",
        "governance_node_surfaces_path": output
        / "phase7_governance_node_surfaces_v1.jsonl",
        "composition_modes_path": output / "phase7_composition_mode_specs_v1.jsonl",
        "conflict_receipts_path": output
        / "phase7_conflict_override_receipts_v1.jsonl",
        "admissible_regions_path": output
        / "phase7_admissible_region_specs_v1.jsonl",
        "control_fields_path": output / "phase7_control_field_slots_v1.jsonl",
        "training_rows_path": output / "phase7_training_row_slots_v1.jsonl",
        "promotion_gates_path": output / "phase7_promotion_gates_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["governance_node_surfaces_path"],
        [item.to_dict() for item in surfaces],
    )
    write_jsonl(paths["composition_modes_path"], [item.to_dict() for item in modes])
    write_jsonl(
        paths["conflict_receipts_path"],
        [item.to_dict() for item in conflict_receipts],
    )
    write_jsonl(
        paths["admissible_regions_path"],
        [item.to_dict() for item in regions],
    )
    write_jsonl(
        paths["control_fields_path"],
        [item.to_dict() for item in control_fields],
    )
    write_jsonl(
        paths["training_rows_path"],
        [item.to_dict() for item in training_rows],
    )
    write_jsonl(paths["promotion_gates_path"], [item.to_dict() for item in gates])
    return {key: str(value) for key, value in paths.items()}


def load_phase7_meta_regal_control_scaffold_report(
    path: str | Path,
) -> Phase7MetaRegalControlScaffoldReport:
    return Phase7MetaRegalControlScaffoldReport.from_dict(load_json(path))


def load_phase7_governance_node_surfaces(
    path: str | Path,
) -> list[Phase7GovernanceNodeSurface]:
    return [Phase7GovernanceNodeSurface.from_dict(row) for row in load_jsonl(path)]


def load_phase7_composition_mode_specs(
    path: str | Path,
) -> list[Phase7CompositionModeSpec]:
    return [Phase7CompositionModeSpec.from_dict(row) for row in load_jsonl(path)]


def load_phase7_conflict_override_receipts(
    path: str | Path,
) -> list[Phase7ConflictOverrideReceipt]:
    return [Phase7ConflictOverrideReceipt.from_dict(row) for row in load_jsonl(path)]


def load_phase7_admissible_region_specs(
    path: str | Path,
) -> list[Phase7AdmissibleRegionSpec]:
    return [Phase7AdmissibleRegionSpec.from_dict(row) for row in load_jsonl(path)]


def load_phase7_control_field_slots(
    path: str | Path,
) -> list[Phase7ControlFieldSlot]:
    return [Phase7ControlFieldSlot.from_dict(row) for row in load_jsonl(path)]


def load_phase7_training_row_slots(path: str | Path) -> list[Phase7TrainingRowSlot]:
    return [Phase7TrainingRowSlot.from_dict(row) for row in load_jsonl(path)]


def load_phase7_promotion_gates(path: str | Path) -> list[Phase7PromotionGate]:
    return [Phase7PromotionGate.from_dict(row) for row in load_jsonl(path)]
