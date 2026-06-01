"""Shadow evaluation and outcome joins for Phase 7 Meta-Regal runtime events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.runtime.event_spine import DecisionLedgerEntry, RuntimeEvent
from src.world_model.humanoid_readiness.common import (
    float_mapping,
    load_json,
    load_jsonl,
    mapping,
    stable_id,
    strings,
    write_json,
    write_jsonl,
)
from src.world_model.humanoid_readiness.phase7 import PHASE7_REMAINING_BLOCKERS
from src.world_model.humanoid_readiness.phase7_runtime import (
    Phase7ConflictRuntimeJoinReceipt,
    Phase7ControlFieldRuntimeReceipt,
    load_phase7_conflict_runtime_join_receipts,
    load_phase7_control_field_runtime_receipts,
)

PHASE7_META_GOVERNANCE_EVAL_REPORT_VERSION = (
    "phase7_meta_governance_evaluation_report_v1"
)
PHASE7_CONTROL_FIELD_EVAL_VERSION = "phase7_control_field_eval_report_v1"
PHASE7_CONFLICT_JOIN_EVAL_VERSION = "phase7_conflict_join_eval_report_v1"
PHASE7_PARETO_REGIME_EVAL_VERSION = "phase7_pareto_regime_eval_report_v1"
PHASE7_OUTCOME_JOIN_ROW_VERSION = "phase7_outcome_join_row_v1"

PHASE7_EVAL_REMAINING_BLOCKERS = (
    *PHASE7_REMAINING_BLOCKERS,
    "ground_truth_outcome_labels_missing",
    "false_veto_false_allow_labels_missing",
    "counterfactual_composition_benchmarks_missing",
    "trained_meta_composition_policy_missing",
)


def _phase7_eval_denied_gates(
    extra: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    gates = {
        "training_executed": False,
        "weights_written": False,
        "provider_executed": False,
        "hardware_executed": False,
        "unitree_sim_runtime_executed": False,
        "live_policy_control": False,
        "reward_math_mutation": False,
        "promotion_eligible": False,
        "phase7_authority_granted": False,
        "live_dispatch_allowed": False,
        "hard_veto_dispatch": False,
        "lower_wm_replacement": False,
        "scalar_governance_collapse": False,
    }
    gates.update({str(key): bool(value) for key, value in dict(extra or {}).items()})
    return gates


@dataclass(frozen=True)
class Phase7ControlFieldEvalReport:
    report_id: str
    receipt_id: str
    slot_id: str
    field_key: str
    runtime_event_id: str
    decision_id: str
    episode_id: str
    eval_status: str
    metrics: dict[str, float] = field(default_factory=dict)
    outcome_join_slots: dict[str, Any] = field(default_factory=dict)
    evaluation_only: bool = True
    replay_export_ready: bool = True
    training_target_only: bool = True
    promotion_eligible: bool = False
    version: str = PHASE7_CONTROL_FIELD_EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "receipt_id": self.receipt_id,
            "slot_id": self.slot_id,
            "field_key": self.field_key,
            "runtime_event_id": self.runtime_event_id,
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "eval_status": self.eval_status,
            "metrics": float_mapping(self.metrics),
            "outcome_join_slots": mapping(self.outcome_join_slots),
            "evaluation_only": bool(self.evaluation_only),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_target_only": bool(self.training_target_only),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7ControlFieldEvalReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            receipt_id=str(payload.get("receipt_id", "")),
            slot_id=str(payload.get("slot_id", "")),
            field_key=str(payload.get("field_key", "")),
            runtime_event_id=str(payload.get("runtime_event_id", "")),
            decision_id=str(payload.get("decision_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            eval_status=str(payload.get("eval_status", "blocked")),
            metrics=float_mapping(payload.get("metrics")),
            outcome_join_slots=mapping(payload.get("outcome_join_slots")),
            evaluation_only=bool(payload.get("evaluation_only", True)),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            training_target_only=bool(payload.get("training_target_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_CONTROL_FIELD_EVAL_VERSION)),
        )


@dataclass(frozen=True)
class Phase7ConflictJoinEvalReport:
    report_id: str
    receipt_id: str
    conflict_receipt_id: str
    conflict_key: str
    runtime_event_id: str
    decision_id: str
    episode_id: str
    composition_mode: str
    eval_status: str
    related_control_field_event_ids: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    outcome_join_slots: dict[str, Any] = field(default_factory=dict)
    evaluation_only: bool = True
    replay_export_ready: bool = True
    training_target_only: bool = True
    promotion_eligible: bool = False
    version: str = PHASE7_CONFLICT_JOIN_EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "receipt_id": self.receipt_id,
            "conflict_receipt_id": self.conflict_receipt_id,
            "conflict_key": self.conflict_key,
            "runtime_event_id": self.runtime_event_id,
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "composition_mode": self.composition_mode,
            "eval_status": self.eval_status,
            "related_control_field_event_ids": list(
                self.related_control_field_event_ids
            ),
            "metrics": float_mapping(self.metrics),
            "outcome_join_slots": mapping(self.outcome_join_slots),
            "evaluation_only": bool(self.evaluation_only),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_target_only": bool(self.training_target_only),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7ConflictJoinEvalReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            receipt_id=str(payload.get("receipt_id", "")),
            conflict_receipt_id=str(payload.get("conflict_receipt_id", "")),
            conflict_key=str(payload.get("conflict_key", "")),
            runtime_event_id=str(payload.get("runtime_event_id", "")),
            decision_id=str(payload.get("decision_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            composition_mode=str(payload.get("composition_mode", "")),
            eval_status=str(payload.get("eval_status", "blocked")),
            related_control_field_event_ids=strings(
                payload.get("related_control_field_event_ids")
            ),
            metrics=float_mapping(payload.get("metrics")),
            outcome_join_slots=mapping(payload.get("outcome_join_slots")),
            evaluation_only=bool(payload.get("evaluation_only", True)),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            training_target_only=bool(payload.get("training_target_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_CONFLICT_JOIN_EVAL_VERSION)),
        )


@dataclass(frozen=True)
class Phase7ParetoRegimeEvalReport:
    report_id: str
    episode_id: str
    regime_key: str
    composition_modes_seen: list[str] = field(default_factory=list)
    active_conflict_keys: list[str] = field(default_factory=list)
    pareto_dimensions: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    regime_label_slots: dict[str, Any] = field(default_factory=dict)
    outcome_join_slots: dict[str, Any] = field(default_factory=dict)
    eval_status: str = "ok"
    evaluation_only: bool = True
    replay_export_ready: bool = True
    training_target_only: bool = True
    promotion_eligible: bool = False
    version: str = PHASE7_PARETO_REGIME_EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "episode_id": self.episode_id,
            "regime_key": self.regime_key,
            "composition_modes_seen": list(self.composition_modes_seen),
            "active_conflict_keys": list(self.active_conflict_keys),
            "pareto_dimensions": list(self.pareto_dimensions),
            "metrics": float_mapping(self.metrics),
            "regime_label_slots": mapping(self.regime_label_slots),
            "outcome_join_slots": mapping(self.outcome_join_slots),
            "eval_status": self.eval_status,
            "evaluation_only": bool(self.evaluation_only),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_target_only": bool(self.training_target_only),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7ParetoRegimeEvalReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            regime_key=str(payload.get("regime_key", "")),
            composition_modes_seen=strings(payload.get("composition_modes_seen")),
            active_conflict_keys=strings(payload.get("active_conflict_keys")),
            pareto_dimensions=strings(payload.get("pareto_dimensions")),
            metrics=float_mapping(payload.get("metrics")),
            regime_label_slots=mapping(payload.get("regime_label_slots")),
            outcome_join_slots=mapping(payload.get("outcome_join_slots")),
            eval_status=str(payload.get("eval_status", "blocked")),
            evaluation_only=bool(payload.get("evaluation_only", True)),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            training_target_only=bool(payload.get("training_target_only", True)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_PARETO_REGIME_EVAL_VERSION)),
        )


@dataclass(frozen=True)
class Phase7OutcomeJoinRow:
    row_id: str
    row_family: str
    episode_id: str
    source_report_id: str
    source_event_ids: list[str] = field(default_factory=list)
    source_decision_ids: list[str] = field(default_factory=list)
    label_slots: dict[str, Any] = field(default_factory=dict)
    outcome_join_slots: dict[str, Any] = field(default_factory=dict)
    replay_export_ready: bool = True
    training_target_only: bool = True
    weights_written: bool = False
    promotion_eligible: bool = False
    version: str = PHASE7_OUTCOME_JOIN_ROW_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "version": self.version,
            "row_family": self.row_family,
            "episode_id": self.episode_id,
            "source_report_id": self.source_report_id,
            "source_event_ids": list(self.source_event_ids),
            "source_decision_ids": list(self.source_decision_ids),
            "label_slots": mapping(self.label_slots),
            "outcome_join_slots": mapping(self.outcome_join_slots),
            "replay_export_ready": bool(self.replay_export_ready),
            "training_target_only": bool(self.training_target_only),
            "weights_written": bool(self.weights_written),
            "promotion_eligible": bool(self.promotion_eligible),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Phase7OutcomeJoinRow":
        return cls(
            row_id=str(payload.get("row_id", "")),
            row_family=str(payload.get("row_family", "")),
            episode_id=str(payload.get("episode_id", "")),
            source_report_id=str(payload.get("source_report_id", "")),
            source_event_ids=strings(payload.get("source_event_ids")),
            source_decision_ids=strings(payload.get("source_decision_ids")),
            label_slots=mapping(payload.get("label_slots")),
            outcome_join_slots=mapping(payload.get("outcome_join_slots")),
            replay_export_ready=bool(payload.get("replay_export_ready", True)),
            training_target_only=bool(payload.get("training_target_only", True)),
            weights_written=bool(payload.get("weights_written", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            version=str(payload.get("version", PHASE7_OUTCOME_JOIN_ROW_VERSION)),
        )


@dataclass(frozen=True)
class Phase7MetaGovernanceEvaluationReport:
    report_id: str
    run_id: str
    status: str
    control_field_eval_count: int
    conflict_join_eval_count: int
    pareto_regime_eval_count: int
    outcome_join_row_count: int
    phase7_event_count: int
    phase7_decision_count: int
    control_field_only_eval_complete: bool
    conflict_join_eval_complete: bool
    pareto_regime_eval_complete: bool
    outcome_join_slots_complete: bool
    local_meta_governance_eval_complete: bool
    replay_export_ready: bool
    phase7_authority_granted: bool = False
    live_dispatch_allowed: bool = False
    hard_veto_dispatch: bool = False
    training_executed: bool = False
    weights_written: bool = False
    provider_executed: bool = False
    hardware_executed: bool = False
    unitree_sim_runtime_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    denied_gates: dict[str, bool] = field(default_factory=_phase7_eval_denied_gates)
    remaining_blockers: list[str] = field(default_factory=list)
    artifact_refs: dict[str, Any] = field(default_factory=dict)
    version: str = PHASE7_META_GOVERNANCE_EVAL_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "run_id": self.run_id,
            "status": self.status,
            "control_field_eval_count": int(self.control_field_eval_count),
            "conflict_join_eval_count": int(self.conflict_join_eval_count),
            "pareto_regime_eval_count": int(self.pareto_regime_eval_count),
            "outcome_join_row_count": int(self.outcome_join_row_count),
            "phase7_event_count": int(self.phase7_event_count),
            "phase7_decision_count": int(self.phase7_decision_count),
            "control_field_only_eval_complete": bool(
                self.control_field_only_eval_complete
            ),
            "conflict_join_eval_complete": bool(self.conflict_join_eval_complete),
            "pareto_regime_eval_complete": bool(self.pareto_regime_eval_complete),
            "outcome_join_slots_complete": bool(self.outcome_join_slots_complete),
            "local_meta_governance_eval_complete": bool(
                self.local_meta_governance_eval_complete
            ),
            "replay_export_ready": bool(self.replay_export_ready),
            "phase7_authority_granted": bool(self.phase7_authority_granted),
            "live_dispatch_allowed": bool(self.live_dispatch_allowed),
            "hard_veto_dispatch": bool(self.hard_veto_dispatch),
            "training_executed": bool(self.training_executed),
            "weights_written": bool(self.weights_written),
            "provider_executed": bool(self.provider_executed),
            "hardware_executed": bool(self.hardware_executed),
            "unitree_sim_runtime_executed": bool(self.unitree_sim_runtime_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "denied_gates": _phase7_eval_denied_gates(self.denied_gates),
            "remaining_blockers": list(self.remaining_blockers),
            "artifact_refs": mapping(self.artifact_refs),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "Phase7MetaGovernanceEvaluationReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            run_id=str(payload.get("run_id", "")),
            status=str(payload.get("status", "blocked")),
            control_field_eval_count=int(
                payload.get("control_field_eval_count", 0) or 0
            ),
            conflict_join_eval_count=int(
                payload.get("conflict_join_eval_count", 0) or 0
            ),
            pareto_regime_eval_count=int(
                payload.get("pareto_regime_eval_count", 0) or 0
            ),
            outcome_join_row_count=int(payload.get("outcome_join_row_count", 0) or 0),
            phase7_event_count=int(payload.get("phase7_event_count", 0) or 0),
            phase7_decision_count=int(payload.get("phase7_decision_count", 0) or 0),
            control_field_only_eval_complete=bool(
                payload.get("control_field_only_eval_complete", False)
            ),
            conflict_join_eval_complete=bool(
                payload.get("conflict_join_eval_complete", False)
            ),
            pareto_regime_eval_complete=bool(
                payload.get("pareto_regime_eval_complete", False)
            ),
            outcome_join_slots_complete=bool(
                payload.get("outcome_join_slots_complete", False)
            ),
            local_meta_governance_eval_complete=bool(
                payload.get("local_meta_governance_eval_complete", False)
            ),
            replay_export_ready=bool(payload.get("replay_export_ready", False)),
            phase7_authority_granted=bool(
                payload.get("phase7_authority_granted", False)
            ),
            live_dispatch_allowed=bool(payload.get("live_dispatch_allowed", False)),
            hard_veto_dispatch=bool(payload.get("hard_veto_dispatch", False)),
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
            denied_gates=_phase7_eval_denied_gates(payload.get("denied_gates")),
            remaining_blockers=strings(payload.get("remaining_blockers")),
            artifact_refs=mapping(payload.get("artifact_refs")),
            version=str(
                payload.get("version", PHASE7_META_GOVERNANCE_EVAL_REPORT_VERSION)
            ),
        )


def build_phase7_meta_governance_evaluation(
    *,
    run_id: str,
    field_receipts: Sequence[Phase7ControlFieldRuntimeReceipt],
    conflict_receipts: Sequence[Phase7ConflictRuntimeJoinReceipt],
    runtime_events: Sequence[RuntimeEvent],
    decision_entries: Sequence[DecisionLedgerEntry],
    summary_payload: Mapping[str, Any] | None = None,
    artifact_refs: Mapping[str, Any] | None = None,
) -> tuple[
    Phase7MetaGovernanceEvaluationReport,
    list[Phase7ControlFieldEvalReport],
    list[Phase7ConflictJoinEvalReport],
    list[Phase7ParetoRegimeEvalReport],
    list[Phase7OutcomeJoinRow],
]:
    event_by_id = {event.event_id: event for event in runtime_events}
    decision_by_id = {decision.decision_id: decision for decision in decision_entries}
    phase7_events = [
        event for event in runtime_events if str(event.event_kind).startswith("phase7_")
    ]
    phase7_decisions = [
        decision
        for decision in decision_entries
        if str(decision.decision_kind).startswith("phase7_")
    ]

    field_evals = [
        _build_control_field_eval(
            receipt=receipt,
            event_by_id=event_by_id,
            decision_by_id=decision_by_id,
        )
        for receipt in field_receipts
    ]
    conflict_evals = [
        _build_conflict_join_eval(
            receipt=receipt,
            event_by_id=event_by_id,
            decision_by_id=decision_by_id,
        )
        for receipt in conflict_receipts
    ]
    regime_evals = _build_pareto_regime_evals(
        runtime_events=phase7_events,
        decision_entries=phase7_decisions,
        summary_payload=summary_payload or {},
    )
    outcome_rows = _build_outcome_join_rows(
        field_evals=field_evals,
        conflict_evals=conflict_evals,
        regime_evals=regime_evals,
    )

    control_eval_complete = bool(field_evals) and all(
        item.eval_status == "ok" for item in field_evals
    )
    conflict_eval_complete = bool(conflict_evals) and all(
        item.eval_status == "ok" for item in conflict_evals
    )
    regime_eval_complete = bool(regime_evals) and all(
        item.eval_status == "ok" for item in regime_evals
    )
    outcome_complete = bool(outcome_rows) and all(
        row.replay_export_ready
        and row.training_target_only
        and not row.weights_written
        and not row.promotion_eligible
        for row in outcome_rows
    )
    complete = (
        control_eval_complete
        and conflict_eval_complete
        and regime_eval_complete
        and outcome_complete
    )
    report_payload = {
        "run_id": run_id,
        "control_field_eval_count": len(field_evals),
        "conflict_join_eval_count": len(conflict_evals),
        "pareto_regime_eval_count": len(regime_evals),
        "phase7_event_count": len(phase7_events),
        "phase7_decision_count": len(phase7_decisions),
        "artifact_refs": mapping(artifact_refs),
    }
    report = Phase7MetaGovernanceEvaluationReport(
        report_id=stable_id("phase7_meta_governance_eval", report_payload),
        run_id=run_id,
        status="ok" if complete else "blocked",
        control_field_eval_count=len(field_evals),
        conflict_join_eval_count=len(conflict_evals),
        pareto_regime_eval_count=len(regime_evals),
        outcome_join_row_count=len(outcome_rows),
        phase7_event_count=len(phase7_events),
        phase7_decision_count=len(phase7_decisions),
        control_field_only_eval_complete=control_eval_complete,
        conflict_join_eval_complete=conflict_eval_complete,
        pareto_regime_eval_complete=regime_eval_complete,
        outcome_join_slots_complete=outcome_complete,
        local_meta_governance_eval_complete=complete,
        replay_export_ready=outcome_complete,
        denied_gates=_phase7_eval_denied_gates(),
        remaining_blockers=list(PHASE7_EVAL_REMAINING_BLOCKERS),
        artifact_refs=mapping(artifact_refs),
    )
    return report, field_evals, conflict_evals, regime_evals, outcome_rows


def _build_control_field_eval(
    *,
    receipt: Phase7ControlFieldRuntimeReceipt,
    event_by_id: Mapping[str, RuntimeEvent],
    decision_by_id: Mapping[str, DecisionLedgerEntry],
) -> Phase7ControlFieldEvalReport:
    event = event_by_id.get(receipt.runtime_event_id)
    decision = decision_by_id.get(receipt.decision_id)
    event_metadata = event.metadata if event is not None else {}
    decision_metadata = decision.metadata if decision is not None else {}
    checks = {
        "event_join_present": float(event is not None),
        "decision_join_present": float(decision is not None),
        "event_kind_valid": float(
            event is not None
            and event.event_kind == "phase7_control_field_shadow_emitted"
        ),
        "decision_kind_valid": float(
            decision is not None
            and decision.decision_kind == "phase7_control_field_shadow_recorded"
        ),
        "decision_sources_event": float(
            decision is not None
            and receipt.runtime_event_id in decision.source_event_ids
        ),
        "shadow_only_verified": float(
            receipt.shadow_only and bool(event_metadata.get("shadow_only", False))
        ),
        "live_dispatch_denied": float(
            not receipt.live_dispatch_allowed
            and not bool(event_metadata.get("live_dispatch_allowed", True))
            and not bool(decision_metadata.get("live_dispatch_allowed", True))
        ),
        "reward_mutation_denied": float(
            not receipt.reward_math_mutation
            and not bool(event_metadata.get("reward_math_mutation", True))
        ),
        "promotion_denied": float(
            not receipt.promotion_eligible
            and not bool(event_metadata.get("promotion_eligible", True))
        ),
        "node_signal_receipt_join_count": float(
            len(strings(event_metadata.get("node_signal_receipt_ids")))
        ),
        "lower_wm_signal_backed": float(
            bool(event_metadata.get("lower_wm_signal_backed", False))
        ),
    }
    required_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"node_signal_receipt_join_count", "lower_wm_signal_backed"}
    }
    status = (
        "ok" if all(value == 1.0 for value in required_checks.values()) else "blocked"
    )
    outcome_slots = {
        "false_allow_observed": None,
        "false_veto_observed": None,
        "shadow_downstream_effect": None,
        "policy_regret_delta": None,
        "operator_recovery_delta": None,
        "node_signal_receipt_ids": strings(
            event_metadata.get("node_signal_receipt_ids")
        ),
        "ground_truth_join_status": "awaiting_runtime_or_benchmark_outcome",
    }
    return Phase7ControlFieldEvalReport(
        report_id=stable_id(
            "phase7_field_eval",
            {
                "receipt_id": receipt.receipt_id,
                "event_id": receipt.runtime_event_id,
                "decision_id": receipt.decision_id,
            },
        ),
        receipt_id=receipt.receipt_id,
        slot_id=receipt.slot_id,
        field_key=receipt.field_key,
        runtime_event_id=receipt.runtime_event_id,
        decision_id=receipt.decision_id,
        episode_id=event.episode_id if event is not None else "",
        eval_status=status,
        metrics=checks,
        outcome_join_slots=outcome_slots,
    )


def _build_conflict_join_eval(
    *,
    receipt: Phase7ConflictRuntimeJoinReceipt,
    event_by_id: Mapping[str, RuntimeEvent],
    decision_by_id: Mapping[str, DecisionLedgerEntry],
) -> Phase7ConflictJoinEvalReport:
    event = event_by_id.get(receipt.runtime_event_id)
    decision = decision_by_id.get(receipt.decision_id)
    event_metadata = event.metadata if event is not None else {}
    checks = {
        "event_join_present": float(event is not None),
        "decision_join_present": float(decision is not None),
        "event_kind_valid": float(
            event is not None
            and event.event_kind == "phase7_conflict_override_shadow_joined"
        ),
        "decision_kind_valid": float(
            decision is not None
            and decision.decision_kind == "phase7_conflict_override_shadow_recorded"
        ),
        "decision_sources_event": float(
            decision is not None
            and receipt.runtime_event_id in decision.source_event_ids
        ),
        "related_field_join_present": float(
            bool(receipt.related_control_field_event_ids)
            and all(
                event_id in event_by_id
                for event_id in receipt.related_control_field_event_ids
            )
        ),
        "hard_veto_dispatch_denied": float(
            not receipt.hard_veto_dispatch
            and not bool(event_metadata.get("hard_veto_dispatch", True))
        ),
        "live_dispatch_denied": float(
            not receipt.live_dispatch_allowed
            and not bool(event_metadata.get("live_dispatch_allowed", True))
        ),
        "promotion_denied": float(
            not receipt.promotion_eligible
            and not bool(event_metadata.get("promotion_eligible", True))
        ),
        "node_signal_receipt_join_count": float(
            len(strings(event_metadata.get("node_signal_receipt_ids")))
        ),
        "lower_wm_signal_backed": float(
            bool(event_metadata.get("lower_wm_signal_backed", False))
        ),
    }
    required_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"node_signal_receipt_join_count", "lower_wm_signal_backed"}
    }
    status = (
        "ok" if all(value == 1.0 for value in required_checks.values()) else "blocked"
    )
    outcome_slots = {
        "false_veto_observed": None,
        "false_allow_observed": None,
        "override_correctness": None,
        "conflict_resolution_quality": None,
        "counterfactual_composition_delta": None,
        "node_signal_receipt_ids": strings(
            event_metadata.get("node_signal_receipt_ids")
        ),
        "ground_truth_join_status": "awaiting_counterfactual_governance_benchmark",
    }
    return Phase7ConflictJoinEvalReport(
        report_id=stable_id(
            "phase7_conflict_eval",
            {
                "receipt_id": receipt.receipt_id,
                "event_id": receipt.runtime_event_id,
                "decision_id": receipt.decision_id,
            },
        ),
        receipt_id=receipt.receipt_id,
        conflict_receipt_id=receipt.conflict_receipt_id,
        conflict_key=receipt.conflict_key,
        runtime_event_id=receipt.runtime_event_id,
        decision_id=receipt.decision_id,
        episode_id=event.episode_id if event is not None else "",
        composition_mode=receipt.composition_mode,
        eval_status=status,
        related_control_field_event_ids=list(receipt.related_control_field_event_ids),
        metrics=checks,
        outcome_join_slots=outcome_slots,
    )


def _build_pareto_regime_evals(
    *,
    runtime_events: Sequence[RuntimeEvent],
    decision_entries: Sequence[DecisionLedgerEntry],
    summary_payload: Mapping[str, Any],
) -> list[Phase7ParetoRegimeEvalReport]:
    events_by_episode: dict[str, list[RuntimeEvent]] = defaultdict(list)
    decisions_by_episode: dict[str, list[DecisionLedgerEntry]] = defaultdict(list)
    for event in runtime_events:
        events_by_episode[event.episode_id].append(event)
    for decision in decision_entries:
        decisions_by_episode[decision.episode_id].append(decision)

    summary_by_episode = {
        str(row.get("episode_id", "")): dict(row)
        for row in summary_payload.get("episode_summaries", []) or []
    }
    reports: list[Phase7ParetoRegimeEvalReport] = []
    for episode_id, events in sorted(events_by_episode.items()):
        decisions = decisions_by_episode.get(episode_id, [])
        modes = sorted(
            {
                str(event.metadata.get("composition_mode", ""))
                or str(event.scope.get("composition_mode", ""))
                for event in events
                if event.metadata.get("composition_mode")
                or event.scope.get("composition_mode")
            }
        )
        conflict_keys = sorted(
            {
                str(event.scope.get("conflict_key", ""))
                for event in events
                if event.scope.get("conflict_key")
            }
        )
        veto_count = sum(1 for mode in modes if mode == "veto_constraint")
        pareto_count = sum(1 for mode in modes if mode == "pareto_relation")
        lexicographic_count = sum(
            1 for mode in modes if mode == "lexicographic_priority"
        )
        confidence_count = sum(1 for mode in modes if mode == "confidence_weighted")
        episode_summary = summary_by_episode.get(episode_id, {})
        regime_key = _infer_regime_key(episode_summary=episode_summary, events=events)
        metrics = {
            "phase7_event_count": float(len(events)),
            "phase7_decision_count": float(len(decisions)),
            "composition_mode_count": float(len(modes)),
            "active_conflict_count": float(len(conflict_keys)),
            "veto_mode_present": float(veto_count > 0),
            "pareto_mode_present": float(pareto_count > 0),
            "lexicographic_mode_present": float(lexicographic_count > 0),
            "confidence_weighted_mode_present": float(confidence_count > 0),
            "live_dispatch_denied": float(
                all(
                    not bool(event.metadata.get("live_dispatch_allowed", True))
                    for event in events
                )
            ),
        }
        outcome_slots = {
            "pareto_frontier_delta": None,
            "regime_label_confirmed": None,
            "false_veto_rate": None,
            "false_allow_rate": None,
            "counterfactual_best_mode": None,
            "ground_truth_join_status": "awaiting_labeled_governance_outcomes",
        }
        reports.append(
            Phase7ParetoRegimeEvalReport(
                report_id=stable_id(
                    "phase7_pareto_regime_eval",
                    {
                        "episode_id": episode_id,
                        "regime_key": regime_key,
                        "modes": modes,
                        "conflicts": conflict_keys,
                    },
                ),
                episode_id=episode_id,
                regime_key=regime_key,
                composition_modes_seen=modes,
                active_conflict_keys=conflict_keys,
                pareto_dimensions=[
                    "economic_value",
                    "safety_margin",
                    "deployment_truth",
                    "reward_integrity",
                    "embodiment_feasibility",
                    "data_value",
                ],
                metrics=metrics,
                regime_label_slots={
                    "deploy_recommendation": episode_summary.get(
                        "deploy_recommendation"
                    ),
                    "pricing_recommendation": episode_summary.get(
                        "pricing_recommendation"
                    ),
                    "datapack_recommendation": episode_summary.get(
                        "datapack_recommendation"
                    ),
                    "label_source": "shadow_runtime_summary_only",
                },
                outcome_join_slots=outcome_slots,
            )
        )
    return reports


def _infer_regime_key(
    *,
    episode_summary: Mapping[str, Any],
    events: Sequence[RuntimeEvent],
) -> str:
    deploy = str(episode_summary.get("deploy_recommendation", "allow_shadow"))
    if deploy == "require_review":
        return "operator_or_deployment_review_shadow"
    if deploy == "deny_shadow":
        return "deployment_truth_or_safety_blocked_shadow"
    if any(
        str(event.scope.get("conflict_key", ""))
        in {"reward_integrity_vs_economic_value", "safety_vs_economic_throughput"}
        for event in events
    ):
        return "nominal_bipedal_with_shadow_conflicts"
    return "nominal_bipedal_shadow"


def _build_outcome_join_rows(
    *,
    field_evals: Sequence[Phase7ControlFieldEvalReport],
    conflict_evals: Sequence[Phase7ConflictJoinEvalReport],
    regime_evals: Sequence[Phase7ParetoRegimeEvalReport],
) -> list[Phase7OutcomeJoinRow]:
    rows: list[Phase7OutcomeJoinRow] = []
    for field_item in field_evals:
        rows.append(
            Phase7OutcomeJoinRow(
                row_id=stable_id(
                    "phase7_outcome_row",
                    {
                        "source_report_id": field_item.report_id,
                        "family": "control_field",
                    },
                ),
                row_family="control_field_shadow_outcome_join",
                episode_id=field_item.episode_id,
                source_report_id=field_item.report_id,
                source_event_ids=[field_item.runtime_event_id],
                source_decision_ids=[field_item.decision_id],
                label_slots={
                    "field_effectiveness": "awaiting_shadow_outcome_label",
                    "field_legibility": "awaiting_reviewer_or_metric_label",
                    "dispatch_was_denied": True,
                },
                outcome_join_slots=dict(field_item.outcome_join_slots),
            )
        )
    for conflict_item in conflict_evals:
        rows.append(
            Phase7OutcomeJoinRow(
                row_id=stable_id(
                    "phase7_outcome_row",
                    {
                        "source_report_id": conflict_item.report_id,
                        "family": "conflict_join",
                    },
                ),
                row_family="conflict_join_shadow_outcome_join",
                episode_id=conflict_item.episode_id,
                source_report_id=conflict_item.report_id,
                source_event_ids=[
                    conflict_item.runtime_event_id,
                    *conflict_item.related_control_field_event_ids,
                ],
                source_decision_ids=[conflict_item.decision_id],
                label_slots={
                    "override_correctness": "awaiting_counterfactual_label",
                    "false_veto": "awaiting_governance_benchmark_label",
                    "false_allow": "awaiting_governance_benchmark_label",
                },
                outcome_join_slots=dict(conflict_item.outcome_join_slots),
            )
        )
    for regime_item in regime_evals:
        rows.append(
            Phase7OutcomeJoinRow(
                row_id=stable_id(
                    "phase7_outcome_row",
                    {
                        "source_report_id": regime_item.report_id,
                        "family": "pareto_regime",
                    },
                ),
                row_family="pareto_regime_shadow_outcome_join",
                episode_id=regime_item.episode_id,
                source_report_id=regime_item.report_id,
                source_event_ids=[],
                source_decision_ids=[],
                label_slots={
                    "regime_key": regime_item.regime_key,
                    "composition_modes_seen": regime_item.composition_modes_seen,
                    "pareto_label": "awaiting_labeled_pareto_front",
                },
                outcome_join_slots=dict(regime_item.outcome_join_slots),
            )
        )
    return rows


def save_phase7_meta_governance_evaluation(
    output_dir: str | Path,
    report: Phase7MetaGovernanceEvaluationReport,
    field_evals: Sequence[Phase7ControlFieldEvalReport],
    conflict_evals: Sequence[Phase7ConflictJoinEvalReport],
    regime_evals: Sequence[Phase7ParetoRegimeEvalReport],
    outcome_rows: Sequence[Phase7OutcomeJoinRow],
) -> dict[str, str]:
    output = Path(output_dir)
    paths = {
        "report_path": output / "phase7_meta_governance_evaluation_report_v1.json",
        "control_field_evals_path": output
        / "phase7_control_field_eval_reports_v1.jsonl",
        "conflict_join_evals_path": output
        / "phase7_conflict_join_eval_reports_v1.jsonl",
        "pareto_regime_evals_path": output
        / "phase7_pareto_regime_eval_reports_v1.jsonl",
        "outcome_join_rows_path": output / "phase7_outcome_join_rows_v1.jsonl",
    }
    write_json(paths["report_path"], report.to_dict())
    write_jsonl(
        paths["control_field_evals_path"],
        [item.to_dict() for item in field_evals],
    )
    write_jsonl(
        paths["conflict_join_evals_path"],
        [item.to_dict() for item in conflict_evals],
    )
    write_jsonl(
        paths["pareto_regime_evals_path"],
        [item.to_dict() for item in regime_evals],
    )
    write_jsonl(
        paths["outcome_join_rows_path"],
        [item.to_dict() for item in outcome_rows],
    )
    return {key: str(value) for key, value in paths.items()}


def load_phase7_meta_governance_evaluation_report(
    path: str | Path,
) -> Phase7MetaGovernanceEvaluationReport:
    return Phase7MetaGovernanceEvaluationReport.from_dict(load_json(path))


def load_phase7_control_field_eval_reports(
    path: str | Path,
) -> list[Phase7ControlFieldEvalReport]:
    return [Phase7ControlFieldEvalReport.from_dict(row) for row in load_jsonl(path)]


def load_phase7_conflict_join_eval_reports(
    path: str | Path,
) -> list[Phase7ConflictJoinEvalReport]:
    return [Phase7ConflictJoinEvalReport.from_dict(row) for row in load_jsonl(path)]


def load_phase7_pareto_regime_eval_reports(
    path: str | Path,
) -> list[Phase7ParetoRegimeEvalReport]:
    return [Phase7ParetoRegimeEvalReport.from_dict(row) for row in load_jsonl(path)]


def load_phase7_outcome_join_rows(path: str | Path) -> list[Phase7OutcomeJoinRow]:
    return [Phase7OutcomeJoinRow.from_dict(row) for row in load_jsonl(path)]


def load_phase7_runtime_eval_inputs(
    runtime_dir: str | Path,
) -> tuple[
    list[Phase7ControlFieldRuntimeReceipt],
    list[Phase7ConflictRuntimeJoinReceipt],
    list[RuntimeEvent],
    list[DecisionLedgerEntry],
    dict[str, Any],
]:
    root = Path(runtime_dir)
    field_receipts = load_phase7_control_field_runtime_receipts(
        root / "phase7_control_field_runtime_receipts.jsonl"
    )
    conflict_receipts = load_phase7_conflict_runtime_join_receipts(
        root / "phase7_conflict_runtime_join_receipts.jsonl"
    )
    event_payload = load_json(root / "event_spine.json")
    decision_payload = load_json(root / "decision_ledger.json")
    summary_payload = load_json(root / "summary.json")
    runtime_events = [
        RuntimeEvent.from_dict(row) for row in event_payload.get("events", []) or []
    ]
    decisions = [
        DecisionLedgerEntry.from_dict(row)
        for row in decision_payload.get("decisions", []) or []
    ]
    return field_receipts, conflict_receipts, runtime_events, decisions, summary_payload
