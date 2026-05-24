"""Phase 6.5 local meta-node neuralization and robustness surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.world_model.humanoid_readiness.common import (
    denied_gate_map,
    float_mapping,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase35 import HumanoidPhase35RefitReport
from src.world_model.humanoid_readiness.phase4 import (
    Phase4DeploymentEnablerSweepReport,
)
from src.world_model.transport import WMTransportPhase6ClosureAuditReport

PHASE65_REPORT_VERSION = "phase65_meta_node_neuralization_report_v1"
META_NODE_STATE_VERSION = "meta_node_state_v1"
META_NODE_TRAJECTORY_VERSION = "meta_node_trajectory_receipt_v1"
META_NODE_INTERVENTION_VERSION = "meta_node_intervention_receipt_v1"
META_NODE_COUNTERFACTUAL_VERSION = "meta_node_counterfactual_target_v1"
META_NODE_ROBUSTNESS_VERSION = "meta_node_robustness_report_v1"
META_NODE_PROMOTION_GATE_VERSION = "meta_node_promotion_gate_v1"

PHASE65_REMAINING_BLOCKERS = (
    "counterfactual_meta_node_corpus_density_missing",
    "trained_meta_node_weights_missing",
    "heldout_robustness_benchmarks_missing",
    "provider_hardware_deployment_evidence_missing",
    "real_governance_benchmark_evidence_missing",
)


@dataclass(frozen=True)
class MetaNodeState:
    node_id: str
    node_family: str
    activation_scope: str
    posture_scope: str
    input_refs: list[str] = field(default_factory=list)
    target_refs: list[str] = field(default_factory=list)
    neighbor_node_ids: list[str] = field(default_factory=list)
    confidence_prior: float = 0.0
    activation_strength_prior: float = 0.0
    authority_class: str = "meta_node_state_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = META_NODE_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "version": self.version,
            "node_family": self.node_family,
            "activation_scope": self.activation_scope,
            "posture_scope": self.posture_scope,
            "input_refs": list(self.input_refs),
            "target_refs": list(self.target_refs),
            "neighbor_node_ids": list(self.neighbor_node_ids),
            "confidence_prior": float(self.confidence_prior),
            "activation_strength_prior": float(self.activation_strength_prior),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodeState":
        return cls(
            node_id=str(payload.get("node_id", "")),
            node_family=str(payload.get("node_family", "")),
            activation_scope=str(payload.get("activation_scope", "")),
            posture_scope=str(payload.get("posture_scope", "unknown")),
            input_refs=strings(payload.get("input_refs")),
            target_refs=strings(payload.get("target_refs")),
            neighbor_node_ids=strings(payload.get("neighbor_node_ids")),
            confidence_prior=float(payload.get("confidence_prior", 0.0) or 0.0),
            activation_strength_prior=float(
                payload.get("activation_strength_prior", 0.0) or 0.0
            ),
            authority_class=str(payload.get("authority_class", "meta_node_state_only")),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", META_NODE_STATE_VERSION)),
        )


@dataclass(frozen=True)
class MetaNodeTrajectoryReceipt:
    receipt_id: str
    node_id: str
    trajectory_events: list[str] = field(default_factory=list)
    replay_refs: list[str] = field(default_factory=list)
    observational_only: bool = True
    training_aware: bool = True
    version: str = META_NODE_TRAJECTORY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "node_id": self.node_id,
            "trajectory_events": list(self.trajectory_events),
            "replay_refs": list(self.replay_refs),
            "observational_only": bool(self.observational_only),
            "training_aware": bool(self.training_aware),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodeTrajectoryReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            node_id=str(payload.get("node_id", "")),
            trajectory_events=strings(payload.get("trajectory_events")),
            replay_refs=strings(payload.get("replay_refs")),
            observational_only=bool(payload.get("observational_only", True)),
            training_aware=bool(payload.get("training_aware", True)),
            version=str(payload.get("version", META_NODE_TRAJECTORY_VERSION)),
        )


@dataclass(frozen=True)
class MetaNodeInterventionReceipt:
    receipt_id: str
    node_id: str
    intervention_kind: str
    rationale: str
    target_refs: list[str] = field(default_factory=list)
    advisory_only: bool = True
    shadow_only: bool = True
    authority_class: str = "meta_node_intervention_advisory_only"
    denied_authority: list[str] = field(default_factory=list)
    version: str = META_NODE_INTERVENTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "node_id": self.node_id,
            "intervention_kind": self.intervention_kind,
            "rationale": self.rationale,
            "target_refs": list(self.target_refs),
            "advisory_only": bool(self.advisory_only),
            "shadow_only": bool(self.shadow_only),
            "authority_class": self.authority_class,
            "denied_authority": list(self.denied_authority),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodeInterventionReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            node_id=str(payload.get("node_id", "")),
            intervention_kind=str(payload.get("intervention_kind", "")),
            rationale=str(payload.get("rationale", "")),
            target_refs=strings(payload.get("target_refs")),
            advisory_only=bool(payload.get("advisory_only", True)),
            shadow_only=bool(payload.get("shadow_only", True)),
            authority_class=str(
                payload.get("authority_class", "meta_node_intervention_advisory_only")
            ),
            denied_authority=strings(payload.get("denied_authority")),
            version=str(payload.get("version", META_NODE_INTERVENTION_VERSION)),
        )


@dataclass(frozen=True)
class MetaNodeCounterfactualTarget:
    target_id: str
    node_id: str
    target_family: str
    label_slots: dict[str, Any] = field(default_factory=dict)
    downstream_effect_slots: dict[str, Any] = field(default_factory=dict)
    outcome_join_status: str = "awaiting_real_counterfactual_corpus"
    training_target_only: bool = True
    promotion_eligible: bool = False
    version: str = META_NODE_COUNTERFACTUAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "version": self.version,
            "node_id": self.node_id,
            "target_family": self.target_family,
            "label_slots": mapping(self.label_slots),
            "downstream_effect_slots": mapping(self.downstream_effect_slots),
            "outcome_join_status": self.outcome_join_status,
            "training_target_only": bool(self.training_target_only),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodeCounterfactualTarget":
        return cls(
            target_id=str(payload.get("target_id", "")),
            node_id=str(payload.get("node_id", "")),
            target_family=str(payload.get("target_family", "")),
            label_slots=mapping(payload.get("label_slots")),
            downstream_effect_slots=mapping(payload.get("downstream_effect_slots")),
            outcome_join_status=str(
                payload.get(
                    "outcome_join_status", "awaiting_real_counterfactual_corpus"
                )
            ),
            training_target_only=bool(payload.get("training_target_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", META_NODE_COUNTERFACTUAL_VERSION)),
        )


@dataclass(frozen=True)
class MetaNodeRobustnessReport:
    report_id: str
    node_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    metric_status: str = "local_surface_only"
    blockers: list[str] = field(default_factory=list)
    evaluation_only: bool = True
    promotion_eligible: bool = False
    version: str = META_NODE_ROBUSTNESS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "node_id": self.node_id,
            "metrics": float_mapping(self.metrics),
            "metric_status": self.metric_status,
            "blockers": list(self.blockers),
            "evaluation_only": bool(self.evaluation_only),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodeRobustnessReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            node_id=str(payload.get("node_id", "")),
            metrics=float_mapping(payload.get("metrics")),
            metric_status=str(payload.get("metric_status", "local_surface_only")),
            blockers=strings(payload.get("blockers")),
            evaluation_only=bool(payload.get("evaluation_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", META_NODE_ROBUSTNESS_VERSION)),
        )


@dataclass(frozen=True)
class MetaNodePromotionGate:
    gate_id: str
    node_id: str
    requested_authority: str
    gate_status: str = "denied"
    missing_evidence: list[str] = field(default_factory=list)
    phase7_authority_granted: bool = False
    promotion_eligible: bool = False
    version: str = META_NODE_PROMOTION_GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "version": self.version,
            "node_id": self.node_id,
            "requested_authority": self.requested_authority,
            "gate_status": self.gate_status,
            "missing_evidence": list(self.missing_evidence),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetaNodePromotionGate":
        return cls(
            gate_id=str(payload.get("gate_id", "")),
            node_id=str(payload.get("node_id", "")),
            requested_authority=str(payload.get("requested_authority", "")),
            gate_status=str(payload.get("gate_status", "denied")),
            missing_evidence=strings(payload.get("missing_evidence")),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", META_NODE_PROMOTION_GATE_VERSION)),
        )


@dataclass(frozen=True)
class Phase65MetaNodeNeuralizationReport:
    report_id: str
    phase35_report_id: str
    phase4_report_id: str
    phase6_closure_audit_id: str
    status: str
    node_state_count: int
    trajectory_receipt_count: int
    intervention_receipt_count: int
    counterfactual_target_count: int
    robustness_report_count: int
    promotion_gate_count: int
    local_meta_node_scaffold_complete: bool
    ready_for_phase7_scaffold: bool
    phase7_authority_granted: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=denied_gate_map)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE65_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase35_report_id": self.phase35_report_id,
            "phase4_report_id": self.phase4_report_id,
            "phase6_closure_audit_id": self.phase6_closure_audit_id,
            "status": self.status,
            "node_state_count": int(self.node_state_count),
            "trajectory_receipt_count": int(self.trajectory_receipt_count),
            "intervention_receipt_count": int(self.intervention_receipt_count),
            "counterfactual_target_count": int(self.counterfactual_target_count),
            "robustness_report_count": int(self.robustness_report_count),
            "promotion_gate_count": int(self.promotion_gate_count),
            "local_meta_node_scaffold_complete": bool(
                self.local_meta_node_scaffold_complete
            ),
            "ready_for_phase7_scaffold": bool(self.ready_for_phase7_scaffold),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": denied_gate_map(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase65MetaNodeNeuralizationReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase35_report_id=str(payload.get("phase35_report_id", "")),
            phase4_report_id=str(payload.get("phase4_report_id", "")),
            phase6_closure_audit_id=str(payload.get("phase6_closure_audit_id", "")),
            status=str(payload.get("status", "blocked")),
            node_state_count=int(payload.get("node_state_count", 0) or 0),
            trajectory_receipt_count=int(
                payload.get("trajectory_receipt_count", 0) or 0
            ),
            intervention_receipt_count=int(
                payload.get("intervention_receipt_count", 0) or 0
            ),
            counterfactual_target_count=int(
                payload.get("counterfactual_target_count", 0) or 0
            ),
            robustness_report_count=int(
                payload.get("robustness_report_count", 0) or 0
            ),
            promotion_gate_count=int(payload.get("promotion_gate_count", 0) or 0),
            local_meta_node_scaffold_complete=bool(
                payload.get("local_meta_node_scaffold_complete", False)
            ),
            ready_for_phase7_scaffold=bool(
                payload.get("ready_for_phase7_scaffold", False)
            ),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
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
            denied_gates=denied_gate_map(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(payload.get("version", PHASE65_REPORT_VERSION)),
        )


def build_phase65_meta_node_neuralization(
    *,
    phase35_report: HumanoidPhase35RefitReport,
    phase4_report: Phase4DeploymentEnablerSweepReport,
    phase6_closure_audit: WMTransportPhase6ClosureAuditReport,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase65MetaNodeNeuralizationReport,
    list[MetaNodeState],
    list[MetaNodeTrajectoryReceipt],
    list[MetaNodeInterventionReceipt],
    list[MetaNodeCounterfactualTarget],
    list[MetaNodeRobustnessReport],
    list[MetaNodePromotionGate],
]:
    denied = [
        "training_execution",
        "weight_write",
        "provider_execution",
        "hardware_execution",
        "live_policy_control",
        "reward_math_mutation",
        "promotion",
        "phase7_control_wm_authority",
    ]
    node_specs = [
        (
            "economic_allocation_guard",
            "economic_allocation_and_resource_routing",
            ["economic_wm_allocation_envelope_ref", "phase6_shadow_join_slot_ref"],
            "shape",
        ),
        (
            "transport_quality_guard",
            "wm_transport_eval_quality",
            ["phase6_decomposed_eval_report_ref", "phase6_closure_audit_ref"],
            "defer",
        ),
        (
            "humanoid_posture_guard",
            "bipedal_posture_and_schema_scope",
            ["phase35_posture_taxonomy_ref", "humanoid_schema_delta_ref"],
            "fallback",
        ),
        (
            "deployment_resource_guard",
            "timing_compute_battery_comms_scope",
            ["phase4_timing_contract_ref", "phase4_compute_placement_ref"],
            "veto",
        ),
        (
            "operator_recovery_guard",
            "operator_handoff_and_recovery_scope",
            ["phase4_operator_handoff_ref", "phase4_recovery_trace_ref"],
            "operator_handoff",
        ),
    ]
    states = [
        MetaNodeState(
            node_id=f"meta_node_{name}",
            node_family=name,
            activation_scope=scope,
            posture_scope="bipedal_whole_body",
            input_refs=inputs,
            target_refs=[
                phase35_report.report_id,
                phase4_report.report_id,
                phase6_closure_audit.audit_id,
            ],
            confidence_prior=0.5,
            activation_strength_prior=0.25,
            denied_authority=denied,
        )
        for name, scope, inputs, _ in node_specs
    ]
    neighbor_ids = [state.node_id for state in states]
    states = [
        MetaNodeState(
            node_id=state.node_id,
            node_family=state.node_family,
            activation_scope=state.activation_scope,
            posture_scope=state.posture_scope,
            input_refs=state.input_refs,
            target_refs=state.target_refs,
            neighbor_node_ids=[
                node_id for node_id in neighbor_ids if node_id != state.node_id
            ],
            confidence_prior=state.confidence_prior,
            activation_strength_prior=state.activation_strength_prior,
            denied_authority=state.denied_authority,
        )
        for state in states
    ]
    intervention_by_node = {f"meta_node_{name}": kind for name, _, _, kind in node_specs}
    trajectories = [
        MetaNodeTrajectoryReceipt(
            receipt_id=f"trajectory_{state.node_id}",
            node_id=state.node_id,
            trajectory_events=[
                "activation_candidate_created",
                "intervention_slot_created",
                "promotion_gate_denied",
            ],
            replay_refs=["event_spine_ref_required", "governance_trace_ref_required"],
        )
        for state in states
    ]
    interventions = [
        MetaNodeInterventionReceipt(
            receipt_id=f"intervention_{state.node_id}",
            node_id=state.node_id,
            intervention_kind=intervention_by_node[state.node_id],
            rationale="local_meta_node_training_target_slot_only",
            target_refs=state.target_refs,
            denied_authority=denied,
        )
        for state in states
    ]
    counterfactual_targets = [
        MetaNodeCounterfactualTarget(
            target_id=f"counterfactual_{state.node_id}",
            node_id=state.node_id,
            target_family="activation_timing_strength_and_downstream_effect",
            label_slots={
                "activation_timing": "awaiting_postmortem_label",
                "activation_strength": "awaiting_postmortem_label",
                "target_selection": "awaiting_postmortem_label",
                "operator_handoff": "awaiting_recovery_trace",
            },
            downstream_effect_slots={
                "counterfactual_downstream_improvement": None,
                "governance_satisfaction": None,
                "rollback_sensitivity": None,
            },
        )
        for state in states
    ]
    robustness_reports = [
        MetaNodeRobustnessReport(
            report_id=f"robustness_{state.node_id}",
            node_id=state.node_id,
            metrics={
                "surface_completeness": 1.0,
                "activation_calibration_evidence": 0.0,
                "replay_shift_benchmark_evidence": 0.0,
                "neighbor_consistency_benchmark_evidence": 0.0,
                "deployment_robustness_evidence": 0.0,
            },
            blockers=list(PHASE65_REMAINING_BLOCKERS),
        )
        for state in states
    ]
    gates = [
        MetaNodePromotionGate(
            gate_id=f"promotion_gate_{state.node_id}",
            node_id=state.node_id,
            requested_authority="phase7_control_wm_authority",
            missing_evidence=list(PHASE65_REMAINING_BLOCKERS),
        )
        for state in states
    ]
    complete = (
        phase35_report.local_structural_refit_complete
        and phase4_report.local_non_hardware_scaffold_complete
        and phase6_closure_audit.local_phase6_structurally_closed
        and len(states) >= 5
        and len(trajectories) == len(states)
        and len(interventions) == len(states)
        and len(counterfactual_targets) == len(states)
        and len(robustness_reports) == len(states)
        and len(gates) == len(states)
    )
    report_payload = {
        "phase35_report_id": phase35_report.report_id,
        "phase4_report_id": phase4_report.report_id,
        "phase6_closure_audit_id": phase6_closure_audit.audit_id,
        "node_state_count": len(states),
        "artifact_refs": mapping(artifact_refs),
    }
    report = Phase65MetaNodeNeuralizationReport(
        report_id=stable_id("phase65_meta_node", report_payload),
        phase35_report_id=phase35_report.report_id,
        phase4_report_id=phase4_report.report_id,
        phase6_closure_audit_id=phase6_closure_audit.audit_id,
        status="ok" if complete else "blocked",
        node_state_count=len(states),
        trajectory_receipt_count=len(trajectories),
        intervention_receipt_count=len(interventions),
        counterfactual_target_count=len(counterfactual_targets),
        robustness_report_count=len(robustness_reports),
        promotion_gate_count=len(gates),
        local_meta_node_scaffold_complete=complete,
        ready_for_phase7_scaffold=complete,
        denied_gates=denied_gate_map(),
        remaining_blockers=list(PHASE65_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return report, states, trajectories, interventions, counterfactual_targets, robustness_reports, gates


def save_phase65_meta_node_neuralization(
    output_dir: str | Path,
    report: Phase65MetaNodeNeuralizationReport,
    states: list[MetaNodeState],
    trajectories: list[MetaNodeTrajectoryReceipt],
    interventions: list[MetaNodeInterventionReceipt],
    targets: list[MetaNodeCounterfactualTarget],
    robustness_reports: list[MetaNodeRobustnessReport],
    gates: list[MetaNodePromotionGate],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase65_meta_node_neuralization_report_v1.json",
        "states_path": output / "meta_node_states_v1.jsonl",
        "trajectories_path": output / "meta_node_trajectory_receipts_v1.jsonl",
        "interventions_path": output / "meta_node_intervention_receipts_v1.jsonl",
        "targets_path": output / "meta_node_counterfactual_targets_v1.jsonl",
        "robustness_path": output / "meta_node_robustness_reports_v1.jsonl",
        "gates_path": output / "meta_node_promotion_gates_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(paths["states_path"], [item.to_dict() for item in states])
    write_jsonl(paths["trajectories_path"], [item.to_dict() for item in trajectories])
    write_jsonl(paths["interventions_path"], [item.to_dict() for item in interventions])
    write_jsonl(paths["targets_path"], [item.to_dict() for item in targets])
    write_jsonl(
        paths["robustness_path"], [item.to_dict() for item in robustness_reports]
    )
    write_jsonl(paths["gates_path"], [item.to_dict() for item in gates])
    return {key: str(value) for key, value in paths.items()}


def load_phase65_meta_node_neuralization_report(
    path: str | Path,
) -> Phase65MetaNodeNeuralizationReport:
    return Phase65MetaNodeNeuralizationReport.from_dict(load_json(path))


def load_meta_node_states(path: str | Path) -> list[MetaNodeState]:
    return [MetaNodeState.from_dict(row) for row in load_jsonl(path)]


def load_meta_node_trajectory_receipts(
    path: str | Path,
) -> list[MetaNodeTrajectoryReceipt]:
    return [MetaNodeTrajectoryReceipt.from_dict(row) for row in load_jsonl(path)]


def load_meta_node_intervention_receipts(
    path: str | Path,
) -> list[MetaNodeInterventionReceipt]:
    return [MetaNodeInterventionReceipt.from_dict(row) for row in load_jsonl(path)]


def load_meta_node_counterfactual_targets(
    path: str | Path,
) -> list[MetaNodeCounterfactualTarget]:
    return [MetaNodeCounterfactualTarget.from_dict(row) for row in load_jsonl(path)]


def load_meta_node_robustness_reports(
    path: str | Path,
) -> list[MetaNodeRobustnessReport]:
    return [MetaNodeRobustnessReport.from_dict(row) for row in load_jsonl(path)]


def load_meta_node_promotion_gates(path: str | Path) -> list[MetaNodePromotionGate]:
    return [MetaNodePromotionGate.from_dict(row) for row in load_jsonl(path)]
