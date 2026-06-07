"""Typed task-economics inputs for deployment evidence-source routing.

This module is CPU-only and deterministic. It carries normalized local
contracts for deciding which evidence source is economically justified before
any GPU, provider, training, or hardware loop is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class EvidenceSource(str, Enum):
    """Evidence/representation source classes considered by the router."""

    REAL_OBSERVATION = "real_observation"
    SIMULATION = "simulation"
    GEOMETRY = "geometry"
    GENERATED_VIDEO = "generated_video"
    HUMAN_OPERATOR_INPUT = "human_operator_input"
    PRIOR_REPLAY = "prior_replay"
    UNAVAILABLE = "unavailable"


class DecisionClass(str, Enum):
    """Decision classes emitted by representation routing."""

    USE = "use"
    REQUIRE_HUMAN_REVIEW = "require_human_review"
    UNAVAILABLE = "unavailable"


CONCRETE_EVIDENCE_SOURCES: tuple[EvidenceSource, ...] = (
    EvidenceSource.REAL_OBSERVATION,
    EvidenceSource.SIMULATION,
    EvidenceSource.GEOMETRY,
    EvidenceSource.GENERATED_VIDEO,
    EvidenceSource.HUMAN_OPERATOR_INPUT,
    EvidenceSource.PRIOR_REPLAY,
)

DETERMINISTIC_TIE_BREAK_ORDER: tuple[EvidenceSource, ...] = (
    EvidenceSource.REAL_OBSERVATION,
    EvidenceSource.HUMAN_OPERATOR_INPUT,
    EvidenceSource.PRIOR_REPLAY,
    EvidenceSource.GEOMETRY,
    EvidenceSource.SIMULATION,
    EvidenceSource.GENERATED_VIDEO,
)


def bounded_unit(value: Any, *, default: float = 0.0) -> float:
    """Coerce numeric input into the router's normalized [0, 1] range."""

    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return max(0.0, min(1.0, resolved))


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return {str(key): value for key, value in dict(payload or {}).items()}


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in list(values or []) if str(value)]


def coerce_evidence_source(value: Any) -> EvidenceSource:
    """Coerce strings/enums into an EvidenceSource value."""

    if isinstance(value, EvidenceSource):
        return value
    try:
        return EvidenceSource(str(value))
    except ValueError:
        return EvidenceSource.UNAVAILABLE


@dataclass(frozen=True)
class TaskEconomics:
    """Normalized task context used to route representation/evidence sources.

    All scalar costs and pressure terms are expected in [0, 1]. Values outside
    that range are clamped when scored and serialized for receipts.
    """

    task_id: str
    task_value: float
    uncertainty: float
    failure_cost: float
    time_cost: float
    battery_cost: float
    compute_cost: float
    evidence_sufficiency: float
    task_kind: str = "deployment_evidence_routing"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_task_value(self) -> float:
        return bounded_unit(self.task_value)

    @property
    def normalized_uncertainty(self) -> float:
        return bounded_unit(self.uncertainty)

    @property
    def normalized_failure_cost(self) -> float:
        return bounded_unit(self.failure_cost)

    @property
    def normalized_time_cost(self) -> float:
        return bounded_unit(self.time_cost)

    @property
    def normalized_battery_cost(self) -> float:
        return bounded_unit(self.battery_cost)

    @property
    def normalized_compute_cost(self) -> float:
        return bounded_unit(self.compute_cost)

    @property
    def required_evidence_sufficiency(self) -> float:
        return bounded_unit(self.evidence_sufficiency)

    @property
    def stakes_index(self) -> float:
        return (
            self.normalized_task_value
            + self.normalized_uncertainty
            + self.normalized_failure_cost
        ) / 3.0

    @property
    def high_stakes_pressure(self) -> float:
        return bounded_unit((self.stakes_index - 0.5) * 2.0)

    @property
    def low_stakes_reuse_pressure(self) -> float:
        return bounded_unit((0.5 - self.stakes_index) * 2.0)

    @property
    def resource_pressure(self) -> float:
        return (
            self.normalized_time_cost
            + self.normalized_battery_cost
            + self.normalized_compute_cost
        ) / 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "task_value": self.normalized_task_value,
            "uncertainty": self.normalized_uncertainty,
            "failure_cost": self.normalized_failure_cost,
            "time_cost": self.normalized_time_cost,
            "battery_cost": self.normalized_battery_cost,
            "compute_cost": self.normalized_compute_cost,
            "evidence_sufficiency": self.required_evidence_sufficiency,
            "stakes_index": self.stakes_index,
            "high_stakes_pressure": self.high_stakes_pressure,
            "low_stakes_reuse_pressure": self.low_stakes_reuse_pressure,
            "resource_pressure": self.resource_pressure,
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceSourceState:
    """Availability, sufficiency, and source-local cost state."""

    source: EvidenceSource
    available: bool
    source_sufficiency: float
    time_cost: Optional[float] = None
    battery_cost: Optional[float] = None
    compute_cost: Optional[float] = None
    failure_risk: Optional[float] = None
    hard_blockers: list[str] = field(default_factory=list)
    lineage_refs: list[str] = field(default_factory=list)
    functional_contribution: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_source_sufficiency(self) -> float:
        return bounded_unit(self.source_sufficiency)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "available": bool(self.available),
            "source_sufficiency": self.normalized_source_sufficiency,
            "time_cost": None if self.time_cost is None else bounded_unit(self.time_cost),
            "battery_cost": (
                None if self.battery_cost is None else bounded_unit(self.battery_cost)
            ),
            "compute_cost": (
                None if self.compute_cost is None else bounded_unit(self.compute_cost)
            ),
            "failure_risk": (
                None if self.failure_risk is None else bounded_unit(self.failure_risk)
            ),
            "hard_blockers": _strings(self.hard_blockers),
            "lineage_refs": _strings(self.lineage_refs),
            "functional_contribution": self.functional_contribution,
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceSourceState":
        return cls(
            source=coerce_evidence_source(payload.get("source")),
            available=bool(payload.get("available", False)),
            source_sufficiency=bounded_unit(payload.get("source_sufficiency", 0.0)),
            time_cost=(
                None
                if payload.get("time_cost") is None
                else bounded_unit(payload.get("time_cost"))
            ),
            battery_cost=(
                None
                if payload.get("battery_cost") is None
                else bounded_unit(payload.get("battery_cost"))
            ),
            compute_cost=(
                None
                if payload.get("compute_cost") is None
                else bounded_unit(payload.get("compute_cost"))
            ),
            failure_risk=(
                None
                if payload.get("failure_risk") is None
                else bounded_unit(payload.get("failure_risk"))
            ),
            hard_blockers=_strings(payload.get("hard_blockers")),
            lineage_refs=_strings(payload.get("lineage_refs")),
            functional_contribution=str(payload.get("functional_contribution", "")),
            metadata=_mapping(payload.get("metadata")),
        )


__all__ = [
    "CONCRETE_EVIDENCE_SOURCES",
    "DETERMINISTIC_TIE_BREAK_ORDER",
    "DecisionClass",
    "EvidenceSource",
    "EvidenceSourceState",
    "TaskEconomics",
    "bounded_unit",
    "coerce_evidence_source",
]
