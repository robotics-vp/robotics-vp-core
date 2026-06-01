"""Typed shell-activation backlog and readiness evaluation for higher-order planners."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.evidence.preconditions import (
    ExecutionPreconditionsReport,
    ExecutionWorkOrder,
    build_execution_preconditions,
    build_execution_work_order,
)
from src.utils.json_safe import to_json_safe


DEFAULT_SHELL_ACTIVATION_BACKLOG_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "SHELL_ACTIVATION_BACKLOG.json"
)


def _string_list(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


def _count_mapping(value: Any) -> Dict[str, int]:
    if isinstance(value, Mapping):
        counts: Dict[str, int] = {}
        for key, count in value.items():
            try:
                counts[str(key)] = int(count)
            except (TypeError, ValueError):
                counts[str(key)] = 0
        return counts
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence_counts: Dict[str, int] = {}
        for key in value:
            normalized = str(key)
            sequence_counts[normalized] = sequence_counts.get(normalized, 0) + 1
        return sequence_counts
    return {}


def normalize_execution_summary(summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize aggregated or single-report readiness payloads into one shape."""

    payload = dict(summary or {})
    blocking_counts = _count_mapping(payload.get("blocking_preconditions"))
    satisfied_counts = _count_mapping(payload.get("satisfied_preconditions"))

    inferred_report_count = 0
    if "ready" in payload or "readiness_score" in payload:
        inferred_report_count = 1

    report_count = int(payload.get("report_count", inferred_report_count) or 0)
    ready_flag = bool(payload.get("ready", False))
    ready_count = int(
        payload.get("ready_count", 1 if ready_flag and report_count else 0) or 0
    )
    if report_count <= 0 and (ready_flag or blocking_counts or satisfied_counts):
        report_count = 1
        ready_count = 1 if ready_flag else 0

    default_blocked = max(report_count - ready_count, 0)
    blocked_count = int(payload.get("blocked_count", default_blocked) or 0)
    mean_readiness_score = float(
        payload.get(
            "mean_readiness_score",
            payload.get("readiness_score", 1.0 if ready_flag and report_count else 0.0),
        )
        or 0.0
    )

    return {
        "report_count": report_count,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "mean_readiness_score": mean_readiness_score,
        "blocking_preconditions": dict(sorted(blocking_counts.items())),
        "satisfied_preconditions": dict(sorted(satisfied_counts.items())),
    }


@dataclass(frozen=True)
class ActivationThresholds:
    min_report_count: int = 0
    min_ready_count: int = 0
    max_blocked_count: int = 0
    min_mean_readiness_score: float = 0.0
    required_satisfied_preconditions: Dict[str, int] = field(default_factory=dict)
    forbidden_blocking_preconditions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ActivationThresholds":
        payload = dict(payload or {})
        required = payload.get("required_satisfied_preconditions", {})
        if isinstance(required, Sequence) and not isinstance(
            required, (str, bytes, bytearray)
        ):
            required = {str(value): 1 for value in required}
        return cls(
            min_report_count=int(payload.get("min_report_count", 0) or 0),
            min_ready_count=int(payload.get("min_ready_count", 0) or 0),
            max_blocked_count=int(payload.get("max_blocked_count", 0) or 0),
            min_mean_readiness_score=float(
                payload.get("min_mean_readiness_score", 0.0) or 0.0
            ),
            required_satisfied_preconditions=_count_mapping(required),
            forbidden_blocking_preconditions=_string_list(
                payload.get("forbidden_blocking_preconditions")
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_report_count": self.min_report_count,
            "min_ready_count": self.min_ready_count,
            "max_blocked_count": self.max_blocked_count,
            "min_mean_readiness_score": self.min_mean_readiness_score,
            "required_satisfied_preconditions": dict(
                sorted(self.required_satisfied_preconditions.items())
            ),
            "forbidden_blocking_preconditions": list(
                self.forbidden_blocking_preconditions
            ),
        }


@dataclass(frozen=True)
class ShellActivationBacklogItem:
    activation_id: str
    module_key: str
    module_path: str
    title: str
    current_mode: str
    target_mode: str
    activation_decision: str
    recommended_mode: str
    priority: str = "P1"
    owner: str = "codex"
    auto_activate: bool = True
    future_training_only: bool = False
    bounded_actions: list[str] = field(default_factory=list)
    thresholds: ActivationThresholds = field(default_factory=ActivationThresholds)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShellActivationBacklogItem":
        return cls(
            activation_id=str(payload.get("activation_id", "")),
            module_key=str(payload.get("module_key", "")),
            module_path=str(payload.get("module_path", "")),
            title=str(payload.get("title", "")),
            current_mode=str(payload.get("current_mode", "advisory")),
            target_mode=str(payload.get("target_mode", "advisory")),
            activation_decision=str(
                payload.get("activation_decision", "activate_shell")
            ),
            recommended_mode=str(payload.get("recommended_mode", "bounded_execution")),
            priority=str(payload.get("priority", "P1")),
            owner=str(payload.get("owner", "codex")),
            auto_activate=bool(payload.get("auto_activate", True)),
            future_training_only=bool(payload.get("future_training_only", False)),
            bounded_actions=_string_list(payload.get("bounded_actions")),
            thresholds=ActivationThresholds.from_dict(
                payload.get("activation_thresholds")
            ),
            notes=str(payload.get("notes", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_id": self.activation_id,
            "module_key": self.module_key,
            "module_path": self.module_path,
            "title": self.title,
            "current_mode": self.current_mode,
            "target_mode": self.target_mode,
            "activation_decision": self.activation_decision,
            "recommended_mode": self.recommended_mode,
            "priority": self.priority,
            "owner": self.owner,
            "auto_activate": self.auto_activate,
            "future_training_only": self.future_training_only,
            "bounded_actions": list(self.bounded_actions),
            "activation_thresholds": self.thresholds.to_dict(),
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ShellActivationAssessment:
    item: ShellActivationBacklogItem
    readiness: ExecutionPreconditionsReport
    state: str
    pending_requirements: list[str] = field(default_factory=list)

    @property
    def activated(self) -> bool:
        return self.state == "activated"

    @property
    def activation_ready(self) -> bool:
        return self.state in {"activated", "activation_ready"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_id": self.item.activation_id,
            "module_key": self.item.module_key,
            "module_path": self.item.module_path,
            "title": self.item.title,
            "current_mode": self.item.current_mode,
            "target_mode": self.item.target_mode,
            "activation_decision": self.item.activation_decision,
            "recommended_mode": self.item.recommended_mode,
            "priority": self.item.priority,
            "auto_activate": self.item.auto_activate,
            "future_training_only": self.item.future_training_only,
            "bounded_actions": list(self.item.bounded_actions),
            "state": self.state,
            "pending_requirements": list(self.pending_requirements),
            "notes": self.item.notes,
            "readiness": self.readiness.to_dict(),
        }


def load_shell_activation_backlog(
    backlog_path: Optional[Path] = None,
) -> list[ShellActivationBacklogItem]:
    """Load the typed higher-shell activation backlog."""

    path = backlog_path or DEFAULT_SHELL_ACTIVATION_BACKLOG_PATH
    if not path.exists():
        return []
    with open(path, "r") as handle:
        payload = json.load(handle)
    return [
        ShellActivationBacklogItem.from_dict(row)
        for row in list(payload.get("backlog", []) or [])
    ]


def _selected_items(
    *,
    backlog_items: Optional[Sequence[ShellActivationBacklogItem]] = None,
    backlog_path: Optional[Path] = None,
    module_keys: Optional[Sequence[str]] = None,
) -> list[ShellActivationBacklogItem]:
    items = list(backlog_items or load_shell_activation_backlog(backlog_path))
    module_filter = {str(value) for value in (module_keys or [])}
    if module_filter:
        items = [item for item in items if item.module_key in module_filter]
    return items


def _pending_requirements(
    summary: Mapping[str, Any],
    item: ShellActivationBacklogItem,
) -> list[str]:
    pending: list[str] = []
    thresholds = item.thresholds
    if summary.get("report_count", 0) < thresholds.min_report_count:
        pending.append(f"report_count<{thresholds.min_report_count}")
    if summary.get("ready_count", 0) < thresholds.min_ready_count:
        pending.append(f"ready_count<{thresholds.min_ready_count}")
    if summary.get("blocked_count", 0) > thresholds.max_blocked_count:
        pending.append(f"blocked_count>{thresholds.max_blocked_count}")
    if (
        float(summary.get("mean_readiness_score", 0.0) or 0.0)
        < thresholds.min_mean_readiness_score
    ):
        pending.append(
            f"mean_readiness_score<{thresholds.min_mean_readiness_score:.2f}"
        )

    satisfied_counts = _count_mapping(summary.get("satisfied_preconditions"))
    for key, expected in sorted(thresholds.required_satisfied_preconditions.items()):
        if satisfied_counts.get(key, 0) < expected:
            pending.append(f"satisfied::{key}<{expected}")

    blocking_counts = _count_mapping(summary.get("blocking_preconditions"))
    for key in thresholds.forbidden_blocking_preconditions:
        if blocking_counts.get(key, 0) > 0:
            pending.append(f"blocking::{key}>0")
    return pending


def build_shell_activation_readiness(
    item: ShellActivationBacklogItem,
    execution_summary: Optional[Mapping[str, Any]],
    *,
    subject_id: Optional[str] = None,
) -> ExecutionPreconditionsReport:
    """Compile a shell-level readiness report from aggregated execution summaries."""

    normalized = normalize_execution_summary(execution_summary)
    thresholds = item.thresholds

    signal_values: Dict[str, Any] = {
        "report_count": normalized["report_count"],
        "ready_count": normalized["ready_count"],
        "blocked_count": normalized["blocked_count"],
        "mean_readiness_score": normalized["mean_readiness_score"],
    }
    min_thresholds: Dict[str, float] = {
        "report_count": float(thresholds.min_report_count),
        "ready_count": float(thresholds.min_ready_count),
        "mean_readiness_score": float(thresholds.min_mean_readiness_score),
    }
    max_thresholds: Dict[str, float] = {
        "blocked_count": float(thresholds.max_blocked_count),
    }

    for key, count in normalized["satisfied_preconditions"].items():
        signal_values[f"satisfied::{key}"] = count
    for key, expected in thresholds.required_satisfied_preconditions.items():
        signal_values.setdefault(f"satisfied::{key}", 0)
        min_thresholds[f"satisfied::{key}"] = float(expected)

    for key, count in normalized["blocking_preconditions"].items():
        signal_values[f"blocking::{key}"] = count
    for key in thresholds.forbidden_blocking_preconditions:
        signal_values.setdefault(f"blocking::{key}", 0)
        max_thresholds[f"blocking::{key}"] = 0.0

    return build_execution_preconditions(
        subject_id=subject_id or item.activation_id,
        subject_kind=item.module_key,
        signal_values=signal_values,
        min_signal_thresholds=min_thresholds,
        max_signal_thresholds=max_thresholds,
        metadata={
            "activation_id": item.activation_id,
            "module_path": item.module_path,
            "current_mode": item.current_mode,
            "target_mode": item.target_mode,
            "future_training_only": item.future_training_only,
            "summary": normalized,
        },
    )


def build_shell_activation_work_order(
    assessment: ShellActivationAssessment,
    *,
    subject_id: str,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ExecutionWorkOrder:
    """Create a shell-activation work order from an assessment."""

    return build_execution_work_order(
        order_type="shell_activation",
        subject_id=subject_id,
        subject_kind=assessment.item.module_key,
        decision=assessment.item.activation_decision,
        priority=max(0.2, float(assessment.readiness.readiness_score)),
        recommended_mode=assessment.item.recommended_mode,
        readiness=assessment.readiness,
        reasons=assessment.item.bounded_actions or [assessment.item.title],
        artifact_refs=artifact_refs,
        metadata={
            "activation_id": assessment.item.activation_id,
            "current_mode": assessment.item.current_mode,
            "target_mode": assessment.item.target_mode,
            "future_training_only": assessment.item.future_training_only,
            **dict(metadata or {}),
        },
    )


def evaluate_shell_activation_backlog(
    execution_summary: Optional[Mapping[str, Any]],
    *,
    module_keys: Optional[Sequence[str]] = None,
    backlog_items: Optional[Sequence[ShellActivationBacklogItem]] = None,
    backlog_path: Optional[Path] = None,
    subject_prefix: str = "shell",
) -> Dict[str, Any]:
    """Evaluate the activation backlog against the current execution summary."""

    items = _selected_items(
        backlog_items=backlog_items,
        backlog_path=backlog_path,
        module_keys=module_keys,
    )
    normalized = normalize_execution_summary(execution_summary)
    assessments: list[ShellActivationAssessment] = []
    for item in items:
        readiness = build_shell_activation_readiness(
            item,
            normalized,
            subject_id=f"{subject_prefix}:{item.activation_id}",
        )
        pending = _pending_requirements(normalized, item)
        if readiness.ready and item.auto_activate and not item.future_training_only:
            state = "activated"
        elif readiness.ready:
            state = "activation_ready"
        elif item.future_training_only:
            state = "future_pending"
        else:
            state = "advisory"
        assessments.append(
            ShellActivationAssessment(
                item=item,
                readiness=readiness,
                state=state,
                pending_requirements=pending or list(readiness.blocking_preconditions),
            )
        )

    return to_json_safe(
        {
            "schema_version": "shell_activation_assessment_v1",
            "execution_summary": normalized,
            "activated": [a.to_dict() for a in assessments if a.state == "activated"],
            "activation_ready": [
                a.to_dict() for a in assessments if a.state == "activation_ready"
            ],
            "pending": [a.to_dict() for a in assessments if a.state == "advisory"],
            "future_training": [
                a.to_dict() for a in assessments if a.state == "future_pending"
            ],
            "assessments": [a.to_dict() for a in assessments],
        }
    )


def get_shell_activation_assessment(
    activation_payload: Mapping[str, Any],
    activation_id: str,
) -> Optional[Dict[str, Any]]:
    """Extract one activation assessment from an evaluated backlog payload."""

    target = str(activation_id)
    for row in list(activation_payload.get("assessments", []) or []):
        if str(row.get("activation_id", "")) == target:
            return dict(row)
    return None


__all__ = [
    "ActivationThresholds",
    "DEFAULT_SHELL_ACTIVATION_BACKLOG_PATH",
    "ShellActivationAssessment",
    "ShellActivationBacklogItem",
    "build_shell_activation_readiness",
    "build_shell_activation_work_order",
    "evaluate_shell_activation_backlog",
    "get_shell_activation_assessment",
    "load_shell_activation_backlog",
    "normalize_execution_summary",
]
