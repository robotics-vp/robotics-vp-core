"""Deterministic deployment-economics representation router.

The router chooses the evidence source with the best usable evidence for a
task under current economic, uncertainty, time, compute, battery, failure-cost,
availability, and sufficiency constraints. It is additive and CPU-only: no
training, provider SDK, GPU, hardware, or promotion path is invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from src.deployment.task_economics import (
    CONCRETE_EVIDENCE_SOURCES,
    DETERMINISTIC_TIE_BREAK_ORDER,
    DecisionClass,
    EvidenceSource,
    EvidenceSourceState,
    TaskEconomics,
    bounded_unit,
    coerce_evidence_source,
)
from src.utils.config_digest import sha256_json

REPRESENTATION_ROUTER_SCHEMA_VERSION = "representation_router_decision_v1"

HIGH_STAKES_ALIGNMENT: Dict[EvidenceSource, float] = {
    EvidenceSource.REAL_OBSERVATION: 0.34,
    EvidenceSource.HUMAN_OPERATOR_INPUT: 0.32,
    EvidenceSource.SIMULATION: 0.08,
    EvidenceSource.GEOMETRY: 0.05,
    EvidenceSource.PRIOR_REPLAY: 0.02,
    EvidenceSource.GENERATED_VIDEO: 0.0,
}

LOW_STAKES_REUSE_ALIGNMENT: Dict[EvidenceSource, float] = {
    EvidenceSource.PRIOR_REPLAY: 0.38,
    EvidenceSource.GEOMETRY: 0.18,
    EvidenceSource.SIMULATION: 0.12,
    EvidenceSource.HUMAN_OPERATOR_INPUT: 0.08,
    EvidenceSource.GENERATED_VIDEO: 0.04,
    EvidenceSource.REAL_OBSERVATION: 0.0,
}

RESOURCE_ALIGNMENT: Dict[EvidenceSource, float] = {
    EvidenceSource.GEOMETRY: 0.22,
    EvidenceSource.PRIOR_REPLAY: 0.18,
    EvidenceSource.HUMAN_OPERATOR_INPUT: 0.04,
    EvidenceSource.REAL_OBSERVATION: 0.0,
    EvidenceSource.SIMULATION: -0.04,
    EvidenceSource.GENERATED_VIDEO: -0.18,
}

DEFAULT_SOURCE_COSTS: Dict[EvidenceSource, Dict[str, float]] = {
    EvidenceSource.REAL_OBSERVATION: {
        "time_cost": 0.28,
        "battery_cost": 0.18,
        "compute_cost": 0.08,
        "failure_risk": 0.08,
    },
    EvidenceSource.HUMAN_OPERATOR_INPUT: {
        "time_cost": 0.34,
        "battery_cost": 0.02,
        "compute_cost": 0.02,
        "failure_risk": 0.04,
    },
    EvidenceSource.PRIOR_REPLAY: {
        "time_cost": 0.05,
        "battery_cost": 0.01,
        "compute_cost": 0.02,
        "failure_risk": 0.10,
    },
    EvidenceSource.GEOMETRY: {
        "time_cost": 0.08,
        "battery_cost": 0.01,
        "compute_cost": 0.05,
        "failure_risk": 0.18,
    },
    EvidenceSource.SIMULATION: {
        "time_cost": 0.24,
        "battery_cost": 0.08,
        "compute_cost": 0.24,
        "failure_risk": 0.22,
    },
    EvidenceSource.GENERATED_VIDEO: {
        "time_cost": 0.46,
        "battery_cost": 0.34,
        "compute_cost": 0.78,
        "failure_risk": 0.42,
    },
}


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {str(key): value for key, value in dict(payload or {}).items()}


def _round_score(value: float) -> float:
    return round(float(value), 6)


def _tie_rank(source: EvidenceSource) -> int:
    try:
        return DETERMINISTIC_TIE_BREAK_ORDER.index(source)
    except ValueError:
        return len(DETERMINISTIC_TIE_BREAK_ORDER)


def _default_state(source: EvidenceSource) -> EvidenceSourceState:
    return EvidenceSourceState(
        source=source,
        available=False,
        source_sufficiency=0.0,
        hard_blockers=["source_not_declared_available"],
    )


def _source_map(
    source_states: Sequence[EvidenceSourceState | Mapping[str, Any]]
    | Mapping[Any, EvidenceSourceState | Mapping[str, Any]],
) -> Dict[EvidenceSource, EvidenceSourceState]:
    if isinstance(source_states, Mapping):
        iterable: Iterable[EvidenceSourceState | Mapping[str, Any]] = []
        rows: list[EvidenceSourceState | Mapping[str, Any]] = []
        for key, value in source_states.items():
            if isinstance(value, EvidenceSourceState):
                rows.append(value)
            else:
                payload = dict(value)
                payload.setdefault("source", key)
                rows.append(payload)
        iterable = rows
    else:
        iterable = source_states

    states: Dict[EvidenceSource, EvidenceSourceState] = {}
    for row in iterable:
        state = row if isinstance(row, EvidenceSourceState) else EvidenceSourceState.from_dict(row)
        source = coerce_evidence_source(state.source)
        if source != EvidenceSource.UNAVAILABLE:
            states[source] = state
    return states


def _resolved_cost(state: EvidenceSourceState, key: str) -> float:
    source_defaults = DEFAULT_SOURCE_COSTS.get(state.source, {})
    value = getattr(state, key)
    if value is None:
        value = source_defaults.get(key, 0.0)
    return bounded_unit(value)


def _cost_payload(state: EvidenceSourceState) -> Dict[str, float]:
    return {
        "time_cost": _resolved_cost(state, "time_cost"),
        "battery_cost": _resolved_cost(state, "battery_cost"),
        "compute_cost": _resolved_cost(state, "compute_cost"),
        "failure_risk": _resolved_cost(state, "failure_risk"),
    }


def _economic_score(task: TaskEconomics, state: EvidenceSourceState) -> tuple[float, Dict[str, float]]:
    costs = _cost_payload(state)
    direct_resource_pressure = (
        costs["time_cost"] * task.normalized_time_cost
        + costs["battery_cost"] * task.normalized_battery_cost
        + costs["compute_cost"] * task.normalized_compute_cost
    ) / 3.0
    sufficiency = state.normalized_source_sufficiency
    evidence_value = sufficiency * (0.75 + 0.25 * task.normalized_task_value)
    sufficiency_margin = max(0.0, sufficiency - task.required_evidence_sufficiency)
    sufficiency_bonus = sufficiency_margin * 0.10
    high_stakes_bonus = (
        task.high_stakes_pressure
        * HIGH_STAKES_ALIGNMENT.get(state.source, 0.0)
        * 0.25
    )
    reuse_bonus = (
        task.low_stakes_reuse_pressure
        * LOW_STAKES_REUSE_ALIGNMENT.get(state.source, 0.0)
        * 0.22
    )
    resource_fit = (
        task.resource_pressure * RESOURCE_ALIGNMENT.get(state.source, 0.0) * 0.12
    )
    time_penalty = costs["time_cost"] * (0.25 + 0.75 * task.normalized_time_cost)
    battery_penalty = costs["battery_cost"] * (
        0.25 + 0.75 * task.normalized_battery_cost
    )
    compute_penalty = costs["compute_cost"] * (
        0.25 + 0.75 * task.normalized_compute_cost
    )
    failure_penalty = costs["failure_risk"] * (
        0.20 + 0.70 * task.normalized_failure_cost
    )
    cost_penalty = (
        time_penalty + battery_penalty + compute_penalty + failure_penalty
    ) * 0.30
    score = (
        evidence_value
        + sufficiency_bonus
        + high_stakes_bonus
        + reuse_bonus
        + resource_fit
        - cost_penalty
    )
    components = {
        "evidence_value": _round_score(evidence_value),
        "sufficiency_bonus": _round_score(sufficiency_bonus),
        "high_stakes_bonus": _round_score(high_stakes_bonus),
        "reuse_bonus": _round_score(reuse_bonus),
        "resource_fit": _round_score(resource_fit),
        "cost_penalty": _round_score(cost_penalty),
        "direct_resource_pressure": _round_score(direct_resource_pressure),
        "net_score": _round_score(score),
    }
    return _round_score(score), components


@dataclass(frozen=True)
class RejectedSource:
    """Rejected source plus deterministic reasons and score context."""

    source: EvidenceSource
    reasons: list[str]
    score: float
    source_sufficiency: float
    available: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "reasons": list(self.reasons),
            "score": _round_score(self.score),
            "source_sufficiency": _round_score(self.source_sufficiency),
            "available": bool(self.available),
            "details": _mapping(self.details),
        }


@dataclass(frozen=True)
class RepresentationRoutingDecision:
    """Typed deterministic output of the representation router."""

    selected_source: EvidenceSource
    decision_class: DecisionClass
    score_by_source: Dict[str, float]
    rejected_sources: list[RejectedSource]
    sufficiency_summary: Dict[str, Any]
    blocker_summary: Dict[str, Any]
    receipt: Dict[str, Any]
    receipt_sha: str
    input_sha: str
    schema_version: str = REPRESENTATION_ROUTER_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selected_source": self.selected_source.value,
            "decision_class": self.decision_class.value,
            "score_by_source": dict(self.score_by_source),
            "rejected_sources": [source.to_dict() for source in self.rejected_sources],
            "sufficiency_summary": _mapping(self.sufficiency_summary),
            "blocker_summary": _mapping(self.blocker_summary),
            "receipt": _mapping(self.receipt),
            "receipt_sha": self.receipt_sha,
            "input_sha": self.input_sha,
        }


def _input_payload(
    task: TaskEconomics,
    states: Mapping[EvidenceSource, EvidenceSourceState],
) -> Dict[str, Any]:
    return {
        "schema_version": REPRESENTATION_ROUTER_SCHEMA_VERSION,
        "task_economics": task.to_dict(),
        "sources": [
            {
                **states.get(source, _default_state(source)).to_dict(),
                "resolved_costs": _cost_payload(states.get(source, _default_state(source))),
            }
            for source in DETERMINISTIC_TIE_BREAK_ORDER
        ],
        "tie_break_order": [source.value for source in DETERMINISTIC_TIE_BREAK_ORDER],
    }


def _decision_class(selected: EvidenceSource, task: TaskEconomics) -> DecisionClass:
    if selected == EvidenceSource.UNAVAILABLE:
        return DecisionClass.UNAVAILABLE
    if selected == EvidenceSource.HUMAN_OPERATOR_INPUT:
        return DecisionClass.REQUIRE_HUMAN_REVIEW
    if task.high_stakes_pressure >= 0.85 and selected not in {
        EvidenceSource.REAL_OBSERVATION,
        EvidenceSource.HUMAN_OPERATOR_INPUT,
    }:
        return DecisionClass.REQUIRE_HUMAN_REVIEW
    return DecisionClass.USE


def route_representation_source(
    task: TaskEconomics,
    source_states: Sequence[EvidenceSourceState | Mapping[str, Any]]
    | Mapping[Any, EvidenceSourceState | Mapping[str, Any]],
) -> RepresentationRoutingDecision:
    """Choose the economically justified evidence source deterministically."""

    states = _source_map(source_states)
    input_payload = _input_payload(task, states)
    input_sha = sha256_json(input_payload)
    score_by_source: Dict[str, float] = {}
    score_components_by_source: Dict[str, Dict[str, float]] = {}
    rejection_reasons: Dict[EvidenceSource, list[str]] = {}
    sufficient_sources: list[str] = []
    unavailable_sources: list[str] = []
    insufficient_sources: list[str] = []
    hard_blocked_sources: Dict[str, list[str]] = {}
    viable_sources: list[EvidenceSource] = []

    for source in DETERMINISTIC_TIE_BREAK_ORDER:
        state = states.get(source, _default_state(source))
        reasons: list[str] = []
        if not state.available:
            reasons.append("source_unavailable")
            unavailable_sources.append(source.value)
        if state.hard_blockers:
            blockers = [str(blocker) for blocker in state.hard_blockers]
            hard_blocked_sources[source.value] = blockers
            reasons.extend([f"source_blocker:{blocker}" for blocker in blockers])
        if state.normalized_source_sufficiency < task.required_evidence_sufficiency:
            reasons.append(
                "source_sufficiency_below_required:"
                f"{state.normalized_source_sufficiency:.3f}<"
                f"{task.required_evidence_sufficiency:.3f}"
            )
            insufficient_sources.append(source.value)

        score, components = _economic_score(task, state)
        score_by_source[source.value] = score
        score_components_by_source[source.value] = components
        if (
            components["direct_resource_pressure"] >= 0.75 or score <= 0.0
        ) and state.available and not reasons:
            reasons.append("economic_cost_dominates_evidence_value")

        if reasons:
            rejection_reasons[source] = reasons
        else:
            viable_sources.append(source)
            sufficient_sources.append(source.value)

    if viable_sources:
        selected_source = sorted(
            viable_sources,
            key=lambda item: (-score_by_source[item.value], _tie_rank(item)),
        )[0]
    else:
        selected_source = EvidenceSource.UNAVAILABLE

    selected_score = (
        score_by_source.get(selected_source.value, 0.0)
        if selected_source != EvidenceSource.UNAVAILABLE
        else 0.0
    )
    rejected_sources: list[RejectedSource] = []
    for source in DETERMINISTIC_TIE_BREAK_ORDER:
        state = states.get(source, _default_state(source))
        reasons = list(rejection_reasons.get(source, []))
        if source != selected_source and not reasons:
            if score_by_source[source.value] == selected_score:
                reasons.append("tie_lost_to_deterministic_order")
            else:
                reasons.append("lower_score_than_selected")
        if source != selected_source:
            rejected_sources.append(
                RejectedSource(
                    source=source,
                    reasons=reasons,
                    score=score_by_source[source.value],
                    source_sufficiency=state.normalized_source_sufficiency,
                    available=state.available,
                    details={
                        "score_components": score_components_by_source[source.value],
                        "resolved_costs": _cost_payload(state),
                    },
                )
            )

    decision_class = _decision_class(selected_source, task)
    sufficiency_summary = {
        "required_evidence_sufficiency": task.required_evidence_sufficiency,
        "source_sufficiency_by_source": {
            source.value: states.get(
                source, _default_state(source)
            ).normalized_source_sufficiency
            for source in DETERMINISTIC_TIE_BREAK_ORDER
        },
        "sufficient_sources": sufficient_sources,
        "insufficient_sources": sorted(set(insufficient_sources)),
        "selected_source_sufficiency": (
            0.0
            if selected_source == EvidenceSource.UNAVAILABLE
            else states[selected_source].normalized_source_sufficiency
        ),
    }
    blocker_summary = {
        "available_source_count": len(
            [source for source in CONCRETE_EVIDENCE_SOURCES if states.get(source, _default_state(source)).available]
        ),
        "sufficient_source_count": len(sufficient_sources),
        "unavailable_sources": sorted(set(unavailable_sources)),
        "hard_blocked_sources": hard_blocked_sources,
        "selected_source_available": selected_source != EvidenceSource.UNAVAILABLE,
        "no_sufficient_source_available": selected_source == EvidenceSource.UNAVAILABLE,
    }
    receipt_core = {
        "schema_version": REPRESENTATION_ROUTER_SCHEMA_VERSION,
        "router_kind": "deployment_economics_representation_router",
        "selected_source": selected_source.value,
        "decision_class": decision_class.value,
        "score_by_source": score_by_source,
        "score_components_by_source": score_components_by_source,
        "rejected_sources": [source.to_dict() for source in rejected_sources],
        "sufficiency_summary": sufficiency_summary,
        "blocker_summary": blocker_summary,
        "input_sha": input_sha,
        "tie_break_order": [
            source.value for source in DETERMINISTIC_TIE_BREAK_ORDER
        ],
        "doctrine": (
            "choose_best_usable_evidence_for_task_under_current_economic_"
            "uncertainty_time_compute_battery_and_failure_cost_constraints"
        ),
        "cpu_only": True,
        "deterministic": True,
        "training_run": False,
        "gpu_execution": False,
        "provider_bringup": False,
        "hardware_execution": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
    }
    receipt_sha = sha256_json(receipt_core)
    receipt = {**receipt_core, "receipt_sha": receipt_sha}
    return RepresentationRoutingDecision(
        selected_source=selected_source,
        decision_class=decision_class,
        score_by_source=score_by_source,
        rejected_sources=rejected_sources,
        sufficiency_summary=sufficiency_summary,
        blocker_summary=blocker_summary,
        receipt=receipt,
        receipt_sha=receipt_sha,
        input_sha=input_sha,
    )


__all__ = [
    "DEFAULT_SOURCE_COSTS",
    "HIGH_STAKES_ALIGNMENT",
    "LOW_STAKES_REUSE_ALIGNMENT",
    "REPRESENTATION_ROUTER_SCHEMA_VERSION",
    "RESOURCE_ALIGNMENT",
    "RejectedSource",
    "RepresentationRoutingDecision",
    "route_representation_source",
]
