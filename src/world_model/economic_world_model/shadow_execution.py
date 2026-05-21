"""Shadow execution work orders for Economic WM outputs.

The Economic WM may now emit work orders and allocation recommendations that can
be compared against later outcomes. These artifacts are advisory only: they do
not control reward math, live policy, or hardware execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.allocation_eval import (
    EconomicWMAllocationCandidate,
    EconomicWMShadowAllocationEval,
    load_economic_wm_shadow_allocation_eval,
)
from src.world_model.economic_world_model.phase5_local_prep import (
    EconomicWMPhase5LocalPrepManifest,
    load_economic_wm_datapack_composition_rows,
    load_economic_wm_phase5_local_prep_manifest,
    load_economic_wm_temporal_window_rows,
)

ECONOMIC_WM_SHADOW_WORK_ORDER_VERSION = "economic_wm_shadow_work_order_v1"
ECONOMIC_WM_SHADOW_OUTCOME_COMPARISON_VERSION = (
    "economic_wm_shadow_outcome_comparison_v1"
)
ECONOMIC_WM_SHADOW_EXECUTION_REPORT_VERSION = "economic_wm_shadow_execution_report_v1"

DENIED_SHADOW_AUTHORITIES = (
    "live_policy_control",
    "reward_math_mutation",
    "provider_truth_substitution",
    "gpu_training_execution",
    "promotion_decision",
)

SHADOW_EXECUTION_BLOCKERS = (
    "outcome_receipts_not_collected",
    "gpu_training_not_run",
    "provider_bringup_not_run",
    "promotion_grade_shadow_benchmarks_missing",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EconomicWMShadowWorkOrder:
    """Advisory Economic WM work order for lower-WM shadow planning."""

    work_order_id: str
    allocation_candidate_id: str
    allocation_label: str
    recommended_action: str
    priority: float
    source_row_ids: list[str] = field(default_factory=list)
    temporal_window_ids: list[str] = field(default_factory=list)
    target_lower_wm_producers: list[str] = field(default_factory=list)
    resource_request: Dict[str, float] = field(default_factory=dict)
    expected_effects: Dict[str, float] = field(default_factory=dict)
    comparison_refs: Dict[str, Any] = field(default_factory=dict)
    denied_authority: list[str] = field(
        default_factory=lambda: list(DENIED_SHADOW_AUTHORITIES)
    )
    authority_class: str = "shadow_work_order_only"
    ready_for_outcome_comparison: bool = True
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_WORK_ORDER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "version": self.version,
            "allocation_candidate_id": self.allocation_candidate_id,
            "allocation_label": self.allocation_label,
            "recommended_action": self.recommended_action,
            "priority": float(self.priority),
            "source_row_ids": list(self.source_row_ids),
            "temporal_window_ids": list(self.temporal_window_ids),
            "target_lower_wm_producers": list(self.target_lower_wm_producers),
            "resource_request": _float_dict(self.resource_request),
            "expected_effects": _float_dict(self.expected_effects),
            "comparison_refs": _mapping(self.comparison_refs),
            "denied_authority": list(self.denied_authority),
            "authority_class": self.authority_class,
            "ready_for_outcome_comparison": bool(self.ready_for_outcome_comparison),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMShadowWorkOrder":
        return cls(
            work_order_id=str(payload.get("work_order_id", "")),
            allocation_candidate_id=str(payload.get("allocation_candidate_id", "")),
            allocation_label=str(payload.get("allocation_label", "")),
            recommended_action=str(payload.get("recommended_action", "")),
            priority=float(payload.get("priority", 0.0)),
            source_row_ids=[
                str(item) for item in list(payload.get("source_row_ids", []) or [])
            ],
            temporal_window_ids=[
                str(item) for item in list(payload.get("temporal_window_ids", []) or [])
            ],
            target_lower_wm_producers=[
                str(item)
                for item in list(payload.get("target_lower_wm_producers", []) or [])
            ],
            resource_request=_float_dict(payload.get("resource_request", {})),
            expected_effects=_float_dict(payload.get("expected_effects", {})),
            comparison_refs=_mapping(payload.get("comparison_refs")),
            denied_authority=[
                str(item) for item in list(payload.get("denied_authority", []) or [])
            ],
            authority_class=str(
                payload.get("authority_class", "shadow_work_order_only")
            ),
            ready_for_outcome_comparison=bool(
                payload.get("ready_for_outcome_comparison", True)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", ECONOMIC_WM_SHADOW_WORK_ORDER_VERSION)),
        )


@dataclass(frozen=True)
class EconomicWMShadowOutcomeComparison:
    """Outcome-comparison placeholder for one shadow work order."""

    comparison_id: str
    work_order_id: str
    comparison_status: str = "awaiting_outcome_receipts"
    expected_effect_keys: list[str] = field(default_factory=list)
    observed_outcome_refs: list[str] = field(default_factory=list)
    comparison_metrics: Dict[str, float] = field(default_factory=dict)
    authority_class: str = "shadow_outcome_comparison_only"
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_OUTCOME_COMPARISON_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "version": self.version,
            "work_order_id": self.work_order_id,
            "comparison_status": self.comparison_status,
            "expected_effect_keys": list(self.expected_effect_keys),
            "observed_outcome_refs": list(self.observed_outcome_refs),
            "comparison_metrics": _float_dict(self.comparison_metrics),
            "authority_class": self.authority_class,
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMShadowOutcomeComparison":
        return cls(
            comparison_id=str(payload.get("comparison_id", "")),
            work_order_id=str(payload.get("work_order_id", "")),
            comparison_status=str(
                payload.get("comparison_status", "awaiting_outcome_receipts")
            ),
            expected_effect_keys=[
                str(item)
                for item in list(payload.get("expected_effect_keys", []) or [])
            ],
            observed_outcome_refs=[
                str(item)
                for item in list(payload.get("observed_outcome_refs", []) or [])
            ],
            comparison_metrics=_float_dict(payload.get("comparison_metrics", {})),
            authority_class=str(
                payload.get("authority_class", "shadow_outcome_comparison_only")
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SHADOW_OUTCOME_COMPARISON_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMShadowExecutionReport:
    """Manifest for Economic WM shadow work-order execution surfaces."""

    report_id: str
    phase5_manifest_id: str
    allocation_eval_id: str
    trainer_scaffold_id: str
    recommended_candidate: str
    work_order_count: int
    outcome_comparison_count: int
    work_orders_path: str
    outcome_comparisons_path: str
    status: str
    authority_class: str = "shadow_execution_only"
    ready_for_shadow_comparison: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_EXECUTION_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "phase5_manifest_id": self.phase5_manifest_id,
            "allocation_eval_id": self.allocation_eval_id,
            "trainer_scaffold_id": self.trainer_scaffold_id,
            "recommended_candidate": self.recommended_candidate,
            "work_order_count": int(self.work_order_count),
            "outcome_comparison_count": int(self.outcome_comparison_count),
            "work_orders_path": self.work_orders_path,
            "outcome_comparisons_path": self.outcome_comparisons_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "ready_for_shadow_comparison": bool(self.ready_for_shadow_comparison),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMShadowExecutionReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            phase5_manifest_id=str(payload.get("phase5_manifest_id", "")),
            allocation_eval_id=str(payload.get("allocation_eval_id", "")),
            trainer_scaffold_id=str(payload.get("trainer_scaffold_id", "")),
            recommended_candidate=str(payload.get("recommended_candidate", "")),
            work_order_count=int(payload.get("work_order_count", 0) or 0),
            outcome_comparison_count=int(
                payload.get("outcome_comparison_count", 0) or 0
            ),
            work_orders_path=str(payload.get("work_orders_path", "")),
            outcome_comparisons_path=str(payload.get("outcome_comparisons_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "shadow_execution_only")
            ),
            ready_for_shadow_comparison=bool(
                payload.get("ready_for_shadow_comparison", False)
            ),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SHADOW_EXECUTION_REPORT_VERSION)
            ),
        )


def _action_for_candidate(label: str) -> str:
    if label == "curate_benchmark_ready_replay":
        return "materialize_benchmark_replay_fixture"
    if label == "close_shadow_gap_replay":
        return "request_lower_wm_gap_closure_receipts"
    if label == "prepare_teacher_provider_evidence_contracts":
        return "prepare_non_stub_provider_receipt_runbook"
    return f"shadow_review_{label}"


def _target_producers_for_candidate(label: str) -> list[str]:
    if label == "prepare_teacher_provider_evidence_contracts":
        return ["provider_runbook", "perception_grounding", "sim_synth_physics"]
    if label == "close_shadow_gap_replay":
        return ["perception_grounding", "sim_synth_physics", "embodiment_actuation"]
    if label == "curate_benchmark_ready_replay":
        return ["economic_wm_dataset_prep", "lower_wm_replay_export"]
    return ["economic_wm_shadow_queue"]


def _work_order(
    *,
    candidate: EconomicWMAllocationCandidate,
    source_row_ids: list[str],
    temporal_window_ids: list[str],
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
) -> EconomicWMShadowWorkOrder:
    expected_effects = {
        "expected_value": candidate.expected_value,
        "phase5_composition_rows_covered": float(phase5_manifest.composition_row_count),
        "phase5_temporal_windows_covered": float(phase5_manifest.temporal_window_count),
        "shadow_only_rows_covered": phase5_manifest.aggregate_counts.get(
            "shadow_only_count", 0.0
        ),
        "benchmark_ready_rows_covered": phase5_manifest.aggregate_counts.get(
            "benchmark_ready_count", 0.0
        ),
    }
    payload = {
        "candidate_id": candidate.candidate_id,
        "label": candidate.label,
        "source_row_ids": source_row_ids,
        "temporal_window_ids": temporal_window_ids,
        "expected_effects": expected_effects,
    }
    return EconomicWMShadowWorkOrder(
        work_order_id=f"ewm_shadow_work_order_{sha256_json(payload)[:16]}",
        allocation_candidate_id=candidate.candidate_id,
        allocation_label=candidate.label,
        recommended_action=_action_for_candidate(candidate.label),
        priority=float(candidate.expected_value),
        source_row_ids=source_row_ids,
        temporal_window_ids=temporal_window_ids,
        target_lower_wm_producers=_target_producers_for_candidate(candidate.label),
        resource_request=candidate.resource_request,
        expected_effects=expected_effects,
        comparison_refs={
            "phase5_manifest_id": phase5_manifest.manifest_id,
            "composition_rows_path": phase5_manifest.composition_rows_path,
            "temporal_windows_path": phase5_manifest.temporal_windows_path,
            "outcome_receipt_slot": "future_shadow_outcome_receipts_v1",
        },
        blockers=list(SHADOW_EXECUTION_BLOCKERS),
        metadata={"candidate_rationale": candidate.rationale},
    )


def _comparison(
    work_order: EconomicWMShadowWorkOrder,
) -> EconomicWMShadowOutcomeComparison:
    payload = {
        "work_order_id": work_order.work_order_id,
        "expected_effect_keys": sorted(work_order.expected_effects.keys()),
    }
    return EconomicWMShadowOutcomeComparison(
        comparison_id=f"ewm_shadow_outcome_{sha256_json(payload)[:16]}",
        work_order_id=work_order.work_order_id,
        expected_effect_keys=sorted(work_order.expected_effects.keys()),
        comparison_metrics={
            "outcome_receipts_observed": 0.0,
            "counterfactual_accuracy_observed": 0.0,
            "pareto_quality_observed": 0.0,
        },
        blockers=["outcome_receipts_not_collected"],
        metadata={"comparison_scope": "awaits later shadow outcomes"},
    )


def build_economic_wm_shadow_execution_report(
    *,
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    allocation_eval: EconomicWMShadowAllocationEval,
    trainer_scaffold_manifest: Mapping[str, Any],
    work_orders_path: str | Path,
    outcome_comparisons_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    EconomicWMShadowExecutionReport,
    list[EconomicWMShadowWorkOrder],
    list[EconomicWMShadowOutcomeComparison],
]:
    compositions = load_economic_wm_datapack_composition_rows(
        phase5_manifest.composition_rows_path
    )
    windows = load_economic_wm_temporal_window_rows(
        phase5_manifest.temporal_windows_path
    )
    source_row_ids = [item.source_row_id for item in compositions]
    temporal_window_ids = [item.window_id for item in windows]
    allowed_candidates = [
        candidate for candidate in allocation_eval.candidates if candidate.allowed
    ]
    if not allowed_candidates and allocation_eval.candidates:
        allowed_candidates = [allocation_eval.candidates[0]]
    orders = [
        _work_order(
            candidate=candidate,
            source_row_ids=source_row_ids,
            temporal_window_ids=temporal_window_ids,
            phase5_manifest=phase5_manifest,
        )
        for candidate in sorted(
            allowed_candidates, key=lambda item: item.expected_value, reverse=True
        )
    ]
    comparisons = [_comparison(order) for order in orders]
    ready_for_shadow_comparison = bool(
        orders
        and comparisons
        and phase5_manifest.ready_for_trainer_scaffold
        and trainer_scaffold_manifest.get("cpu_smoke_forward_passed", False)
    )
    status = "ok" if ready_for_shadow_comparison else "blocked"
    payload = {
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "allocation_eval_id": allocation_eval.eval_id,
        "trainer_scaffold_id": trainer_scaffold_manifest.get("trainer_scaffold_id", ""),
        "order_ids": [item.work_order_id for item in orders],
        "comparison_ids": [item.comparison_id for item in comparisons],
    }
    report = EconomicWMShadowExecutionReport(
        report_id=f"ewm_shadow_execution_{sha256_json(payload)[:16]}",
        phase5_manifest_id=phase5_manifest.manifest_id,
        allocation_eval_id=allocation_eval.eval_id,
        trainer_scaffold_id=str(
            trainer_scaffold_manifest.get("trainer_scaffold_id", "")
        ),
        recommended_candidate=allocation_eval.recommended_candidate,
        work_order_count=len(orders),
        outcome_comparison_count=len(comparisons),
        work_orders_path=str(work_orders_path),
        outcome_comparisons_path=str(outcome_comparisons_path),
        status=status,
        ready_for_shadow_comparison=ready_for_shadow_comparison,
        blockers=list(SHADOW_EXECUTION_BLOCKERS),
        aggregate_counts={
            "work_order_count": float(len(orders)),
            "outcome_comparison_count": float(len(comparisons)),
            "source_row_count": float(len(source_row_ids)),
            "temporal_window_count": float(len(temporal_window_ids)),
            "allowed_candidate_count": float(len(allowed_candidates)),
            "live_policy_control_count": 0.0,
            "reward_math_mutation_count": 0.0,
        },
        artifact_refs={
            **_mapping(artifact_refs),
            "work_orders_path": str(work_orders_path),
            "outcome_comparisons_path": str(outcome_comparisons_path),
        },
        metadata={
            **_mapping(metadata),
            "boundary": "shadow execution orders only; no live authority",
        },
    )
    return report, orders, comparisons


def save_economic_wm_shadow_execution_report(
    *,
    report_path: str | Path,
    report: EconomicWMShadowExecutionReport,
    work_orders: Iterable[EconomicWMShadowWorkOrder],
    outcome_comparisons: Iterable[EconomicWMShadowOutcomeComparison],
) -> None:
    _write_json(report_path, report.to_dict())
    _write_jsonl(report.work_orders_path, [item.to_dict() for item in work_orders])
    _write_jsonl(
        report.outcome_comparisons_path,
        [item.to_dict() for item in outcome_comparisons],
    )


def load_economic_wm_shadow_execution_report(
    path: str | Path,
) -> EconomicWMShadowExecutionReport:
    return EconomicWMShadowExecutionReport.from_dict(_load_json(path))


def load_economic_wm_shadow_work_orders(
    path: str | Path,
) -> list[EconomicWMShadowWorkOrder]:
    return [EconomicWMShadowWorkOrder.from_dict(row) for row in _load_jsonl(path)]


def load_economic_wm_shadow_outcome_comparisons(
    path: str | Path,
) -> list[EconomicWMShadowOutcomeComparison]:
    return [
        EconomicWMShadowOutcomeComparison.from_dict(row) for row in _load_jsonl(path)
    ]


def build_economic_wm_shadow_execution_report_from_paths(
    *,
    phase5_prep_path: str | Path,
    allocation_eval_path: str | Path,
    trainer_scaffold_path: str | Path,
    report_path: str | Path,
    work_orders_path: str | Path,
    outcome_comparisons_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMShadowExecutionReport:
    phase5_manifest = load_economic_wm_phase5_local_prep_manifest(phase5_prep_path)
    allocation_eval = load_economic_wm_shadow_allocation_eval(allocation_eval_path)
    trainer_scaffold = _load_json(trainer_scaffold_path)
    report, orders, comparisons = build_economic_wm_shadow_execution_report(
        phase5_manifest=phase5_manifest,
        allocation_eval=allocation_eval,
        trainer_scaffold_manifest=trainer_scaffold,
        work_orders_path=work_orders_path,
        outcome_comparisons_path=outcome_comparisons_path,
        artifact_refs={
            "phase5_prep_path": str(phase5_prep_path),
            "allocation_eval_path": str(allocation_eval_path),
            "trainer_scaffold_path": str(trainer_scaffold_path),
            "report_path": str(report_path),
        },
        metadata=metadata,
    )
    save_economic_wm_shadow_execution_report(
        report_path=report_path,
        report=report,
        work_orders=orders,
        outcome_comparisons=comparisons,
    )
    return report
