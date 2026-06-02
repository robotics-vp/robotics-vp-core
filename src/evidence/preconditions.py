"""Execution preconditions and work-order artifacts for self-improvement flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class PreconditionCheck:
    """One typed readiness/precondition check."""

    precondition_id: str
    satisfied: bool
    hard: bool = True
    detail: str = ""
    observed_value: Any = None
    expected_value: Any = None
    artifact_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precondition_id": self.precondition_id,
            "satisfied": bool(self.satisfied),
            "hard": bool(self.hard),
            "detail": self.detail,
            "observed_value": to_json_safe(self.observed_value),
            "expected_value": to_json_safe(self.expected_value),
            "artifact_ref": self.artifact_ref,
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreconditionCheck":
        return cls(
            precondition_id=str(payload.get("precondition_id", "")),
            satisfied=bool(payload.get("satisfied", False)),
            hard=bool(payload.get("hard", True)),
            detail=str(payload.get("detail", "")),
            observed_value=payload.get("observed_value"),
            expected_value=payload.get("expected_value"),
            artifact_ref=payload.get("artifact_ref"),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class ExecutionPreconditionsReport:
    """Normalized execution-readiness artifact."""

    subject_id: str
    subject_kind: str
    ready: bool
    readiness_score: float
    checks: list[PreconditionCheck]
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "execution_preconditions_v1"

    @property
    def blocking_preconditions(self) -> list[str]:
        return [
            check.precondition_id
            for check in self.checks
            if check.hard and not check.satisfied
        ]

    @property
    def satisfied_preconditions(self) -> list[str]:
        return [
            check.precondition_id
            for check in self.checks
            if check.satisfied
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "ready": bool(self.ready),
            "readiness_score": float(self.readiness_score),
            "checks": [check.to_dict() for check in self.checks],
            "blocking_preconditions": list(self.blocking_preconditions),
            "satisfied_preconditions": list(self.satisfied_preconditions),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionPreconditionsReport":
        checks = [
            PreconditionCheck.from_dict(row)
            for row in list(payload.get("checks", []) or [])
        ]
        return cls(
            subject_id=str(payload.get("subject_id", "")),
            subject_kind=str(payload.get("subject_kind", "")),
            ready=bool(payload.get("ready", False)),
            readiness_score=float(payload.get("readiness_score", 0.0)),
            checks=checks,
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "execution_preconditions_v1")),
        )


@dataclass(frozen=True)
class ExecutionWorkOrder:
    """Normalized work order emitted once a decision meets explicit preconditions."""

    work_order_id: str
    order_type: str
    subject_id: str
    subject_kind: str
    decision: str
    ready: bool
    priority: float
    recommended_mode: str
    reasons: list[str]
    blocking_preconditions: list[str]
    required_preconditions: list[str]
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "execution_work_order_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "order_type": self.order_type,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "decision": self.decision,
            "ready": bool(self.ready),
            "priority": float(self.priority),
            "recommended_mode": self.recommended_mode,
            "reasons": list(self.reasons),
            "blocking_preconditions": list(self.blocking_preconditions),
            "required_preconditions": list(self.required_preconditions),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


def build_execution_preconditions(
    *,
    subject_id: str,
    subject_kind: str,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    required_artifact_refs: Optional[Sequence[str]] = None,
    soft_required_artifact_refs: Optional[Sequence[str]] = None,
    signal_values: Optional[Mapping[str, Any]] = None,
    min_signal_thresholds: Optional[Mapping[str, float]] = None,
    max_signal_thresholds: Optional[Mapping[str, float]] = None,
    required_boolean_signals: Optional[Mapping[str, bool]] = None,
    soft_min_signal_thresholds: Optional[Mapping[str, float]] = None,
    soft_max_signal_thresholds: Optional[Mapping[str, float]] = None,
    soft_boolean_signals: Optional[Mapping[str, bool]] = None,
    blocked_reasons: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ExecutionPreconditionsReport:
    """Build a stable readiness report from refs, signals, and explicit blockers."""

    refs = _mapping(artifact_refs)
    signals = _mapping(signal_values)
    checks: list[PreconditionCheck] = []

    for key in _strings(required_artifact_refs):
        value = refs.get(key)
        checks.append(
            PreconditionCheck(
                precondition_id=f"artifact::{key}",
                satisfied=_has_value(value),
                hard=True,
                detail="required_artifact_present" if _has_value(value) else "required_artifact_missing",
                observed_value=value,
                artifact_ref=str(value) if isinstance(value, str) else None,
            )
        )

    for key in _strings(soft_required_artifact_refs):
        value = refs.get(key)
        checks.append(
            PreconditionCheck(
                precondition_id=f"artifact::{key}",
                satisfied=_has_value(value),
                hard=False,
                detail="optional_artifact_present" if _has_value(value) else "optional_artifact_missing",
                observed_value=value,
                artifact_ref=str(value) if isinstance(value, str) else None,
            )
        )

    for key, threshold in sorted(dict(min_signal_thresholds or {}).items()):
        value = signals.get(str(key))
        threshold_value = _safe_float(threshold)
        satisfied = _safe_float(value) >= threshold_value
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_min::{key}",
                satisfied=satisfied,
                hard=True,
                detail="min_signal_threshold" if satisfied else "signal_below_min",
                observed_value=value,
                expected_value=threshold_value,
            )
        )

    for key, threshold in sorted(dict(soft_min_signal_thresholds or {}).items()):
        value = signals.get(str(key))
        threshold_value = _safe_float(threshold)
        satisfied = _safe_float(value) >= threshold_value
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_min::{key}",
                satisfied=satisfied,
                hard=False,
                detail="optional_min_signal_threshold" if satisfied else "optional_signal_below_min",
                observed_value=value,
                expected_value=threshold_value,
            )
        )

    for key, threshold in sorted(dict(max_signal_thresholds or {}).items()):
        value = signals.get(str(key))
        threshold_value = _safe_float(threshold)
        satisfied = _safe_float(value) <= threshold_value
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_max::{key}",
                satisfied=satisfied,
                hard=True,
                detail="max_signal_threshold" if satisfied else "signal_above_max",
                observed_value=value,
                expected_value=threshold_value,
            )
        )

    for key, threshold in sorted(dict(soft_max_signal_thresholds or {}).items()):
        value = signals.get(str(key))
        threshold_value = _safe_float(threshold)
        satisfied = _safe_float(value) <= threshold_value
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_max::{key}",
                satisfied=satisfied,
                hard=False,
                detail="optional_max_signal_threshold" if satisfied else "optional_signal_above_max",
                observed_value=value,
                expected_value=threshold_value,
            )
        )

    for key, expected in sorted(dict(required_boolean_signals or {}).items()):
        value = bool(signals.get(str(key), False))
        expected_bool = bool(expected)
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_bool::{key}",
                satisfied=value == expected_bool,
                hard=True,
                detail="required_boolean_signal" if value == expected_bool else "boolean_signal_mismatch",
                observed_value=value,
                expected_value=expected_bool,
            )
        )

    for key, expected in sorted(dict(soft_boolean_signals or {}).items()):
        value = bool(signals.get(str(key), False))
        expected_bool = bool(expected)
        checks.append(
            PreconditionCheck(
                precondition_id=f"signal_bool::{key}",
                satisfied=value == expected_bool,
                hard=False,
                detail="optional_boolean_signal" if value == expected_bool else "optional_boolean_signal_mismatch",
                observed_value=value,
                expected_value=expected_bool,
            )
        )

    for reason in _strings(blocked_reasons):
        checks.append(
            PreconditionCheck(
                precondition_id=f"blocked::{reason}",
                satisfied=False,
                hard=True,
                detail="explicit_blocker",
                observed_value=reason,
                expected_value="absent",
            )
        )

    hard_checks = [check for check in checks if check.hard]
    satisfied_hard = sum(1 for check in hard_checks if check.satisfied)
    readiness_score = (
        float(satisfied_hard) / float(len(hard_checks))
        if hard_checks
        else 1.0
    )
    ready = all(check.satisfied for check in hard_checks)
    return ExecutionPreconditionsReport(
        subject_id=str(subject_id),
        subject_kind=str(subject_kind),
        ready=ready,
        readiness_score=readiness_score,
        checks=checks,
        artifact_refs=refs,
        metadata={
            **_mapping(metadata),
            "signal_values": signals,
        },
    )


def summarize_execution_preconditions(
    reports: Sequence[ExecutionPreconditionsReport | Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate readiness across a batch of reports."""

    normalized = [
        report
        if isinstance(report, ExecutionPreconditionsReport)
        else ExecutionPreconditionsReport.from_dict(report)
        for report in reports
    ]
    if not normalized:
        return {
            "report_count": 0,
            "ready_count": 0,
            "blocked_count": 0,
            "mean_readiness_score": 0.0,
            "blocking_preconditions": {},
            "satisfied_preconditions": {},
        }

    blocking_counts: Dict[str, int] = {}
    satisfied_counts: Dict[str, int] = {}
    for report in normalized:
        for key in report.blocking_preconditions:
            blocking_counts[key] = blocking_counts.get(key, 0) + 1
        for key in report.satisfied_preconditions:
            satisfied_counts[key] = satisfied_counts.get(key, 0) + 1
    return {
        "report_count": len(normalized),
        "ready_count": sum(1 for report in normalized if report.ready),
        "blocked_count": sum(1 for report in normalized if not report.ready),
        "mean_readiness_score": sum(report.readiness_score for report in normalized) / float(len(normalized)),
        "blocking_preconditions": dict(sorted(blocking_counts.items())),
        "satisfied_preconditions": dict(sorted(satisfied_counts.items())),
    }


def build_execution_work_order(
    *,
    order_type: str,
    subject_id: str,
    subject_kind: str,
    decision: str,
    priority: float,
    recommended_mode: str,
    readiness: ExecutionPreconditionsReport | Mapping[str, Any],
    reasons: Optional[Sequence[str]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ExecutionWorkOrder:
    """Create a work order that is executable only when readiness is satisfied."""

    report = (
        readiness
        if isinstance(readiness, ExecutionPreconditionsReport)
        else ExecutionPreconditionsReport.from_dict(readiness)
    )
    payload = {
        "order_type": str(order_type),
        "subject_id": str(subject_id),
        "subject_kind": str(subject_kind),
        "decision": str(decision),
        "priority": float(priority),
        "recommended_mode": str(recommended_mode),
        "blocking_preconditions": list(report.blocking_preconditions),
        "required_preconditions": sorted(
            set(list(report.satisfied_preconditions) + list(report.blocking_preconditions))
        ),
        "reasons": _strings(reasons),
        "artifact_refs": _mapping(artifact_refs) or dict(report.artifact_refs),
        "metadata": _mapping(metadata),
    }
    work_order_id = f"work_{sha256_json(payload)[:16]}"
    return ExecutionWorkOrder(
        work_order_id=work_order_id,
        order_type=str(order_type),
        subject_id=str(subject_id),
        subject_kind=str(subject_kind),
        decision=str(decision),
        ready=bool(report.ready),
        priority=float(priority),
        recommended_mode=str(recommended_mode),
        reasons=_strings(reasons),
        blocking_preconditions=list(report.blocking_preconditions),
        required_preconditions=sorted(
            set(list(report.satisfied_preconditions) + list(report.blocking_preconditions))
        ),
        artifact_refs=_mapping(artifact_refs) or dict(report.artifact_refs),
        metadata={
            **_mapping(metadata),
            "readiness_score": float(report.readiness_score),
            "readiness_version": report.version,
        },
    )


__all__ = [
    "ExecutionPreconditionsReport",
    "ExecutionWorkOrder",
    "PreconditionCheck",
    "build_execution_preconditions",
    "build_execution_work_order",
    "summarize_execution_preconditions",
]
