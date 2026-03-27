"""Canonical learnability and inferential admission contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.economics.inferential_reward import (
    InferentialSignalYield,
    compile_signal_yield,
    compute_inferential_replay_weight,
)
from src.evidence.preconditions import (
    ExecutionWorkOrder,
    ExecutionPreconditionsReport,
    build_execution_work_order,
)
from src.utils.json_safe import to_json_safe


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off", ""}:
            return False
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


@dataclass(frozen=True)
class InferentialLearnabilityContract:
    """Portable learnability/evidence summary used across replay and training."""

    subject_id: str
    subject_kind: str
    datapack_id: str
    learnability_class: str
    signal_yield: Dict[str, float]
    inferential_replay_weight: float
    frontier_gain: float
    epiplexity_delta: float
    epiplexity_confidence: float
    transfer_score: float
    data_quality: float
    provenance_quality: float
    trust_score: float
    benchmark_eligible: bool = False
    semantic_grounding_non_heuristic: bool = False
    promotion_trace_complete: bool = False
    budget_settlement_live: bool = False
    overlay_joined: bool = False
    summary_present: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "inferential_learnability_contract_v1"

    @property
    def receipt_backed(self) -> bool:
        return bool(
            self.overlay_joined
            or self.promotion_trace_complete
            or self.budget_settlement_live
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "datapack_id": self.datapack_id,
            "learnability_class": self.learnability_class,
            "signal_yield": dict(self.signal_yield),
            "inferential_replay_weight": float(self.inferential_replay_weight),
            "frontier_gain": float(self.frontier_gain),
            "epiplexity_delta": float(self.epiplexity_delta),
            "epiplexity_confidence": float(self.epiplexity_confidence),
            "transfer_score": float(self.transfer_score),
            "data_quality": float(self.data_quality),
            "provenance_quality": float(self.provenance_quality),
            "trust_score": float(self.trust_score),
            "benchmark_eligible": bool(self.benchmark_eligible),
            "semantic_grounding_non_heuristic": bool(self.semantic_grounding_non_heuristic),
            "promotion_trace_complete": bool(self.promotion_trace_complete),
            "budget_settlement_live": bool(self.budget_settlement_live),
            "overlay_joined": bool(self.overlay_joined),
            "summary_present": bool(self.summary_present),
            "receipt_backed": bool(self.receipt_backed),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferentialLearnabilityContract":
        return cls(
            subject_id=str(payload.get("subject_id", "")),
            subject_kind=str(payload.get("subject_kind", "")),
            datapack_id=str(payload.get("datapack_id", "")),
            learnability_class=str(payload.get("learnability_class", "missing")),
            signal_yield={
                str(key): _safe_float(value)
                for key, value in dict(payload.get("signal_yield", {}) or {}).items()
            },
            inferential_replay_weight=_safe_float(payload.get("inferential_replay_weight", 0.0)),
            frontier_gain=_safe_float(payload.get("frontier_gain", 0.0)),
            epiplexity_delta=_safe_float(payload.get("epiplexity_delta", 0.0)),
            epiplexity_confidence=_safe_float(payload.get("epiplexity_confidence", 0.0)),
            transfer_score=_safe_float(payload.get("transfer_score", 0.0)),
            data_quality=_safe_float(payload.get("data_quality", 0.0)),
            provenance_quality=_safe_float(payload.get("provenance_quality", 0.0)),
            trust_score=_safe_float(payload.get("trust_score", 0.0)),
            benchmark_eligible=_safe_bool(payload.get("benchmark_eligible", False)),
            semantic_grounding_non_heuristic=_safe_bool(
                payload.get("semantic_grounding_non_heuristic", False)
            ),
            promotion_trace_complete=_safe_bool(payload.get("promotion_trace_complete", False)),
            budget_settlement_live=_safe_bool(payload.get("budget_settlement_live", False)),
            overlay_joined=_safe_bool(payload.get("overlay_joined", False)),
            summary_present=_safe_bool(payload.get("summary_present", False)),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "inferential_learnability_contract_v1")),
        )


def coerce_inferential_learnability_contract(
    payload: Optional[Mapping[str, Any] | InferentialLearnabilityContract],
) -> Optional[InferentialLearnabilityContract]:
    if payload is None:
        return None
    if isinstance(payload, InferentialLearnabilityContract):
        return payload
    if isinstance(payload, Mapping):
        return InferentialLearnabilityContract.from_dict(payload)
    return None


def classify_learnability_class(
    *,
    summary_present: bool,
    overlay_joined: bool,
    benchmark_eligible: bool,
    semantic_grounding_non_heuristic: bool,
    promotion_trace_complete: bool,
    budget_settlement_live: bool,
) -> str:
    if not summary_present:
        return "missing"
    if (
        benchmark_eligible
        and semantic_grounding_non_heuristic
        and (promotion_trace_complete or budget_settlement_live)
    ):
        return "benchmark_receipt_backed"
    if overlay_joined or promotion_trace_complete or budget_settlement_live:
        return "portable_receipt_backed"
    return "summary_only"


def build_inferential_learnability_contract(
    *,
    subject_id: str,
    subject_kind: str,
    datapack_id: str,
    frontier_gain: float = 0.0,
    epiplexity_delta: float = 0.0,
    epiplexity_confidence: float = 0.0,
    transfer_score: float = 0.0,
    data_quality: float = 0.0,
    provenance_quality: float = 0.0,
    trust_score: float = 0.5,
    overlay_joined: bool = False,
    benchmark_eligible: bool = False,
    semantic_grounding_non_heuristic: bool = False,
    promotion_trace_complete: bool = False,
    budget_settlement_live: bool = False,
    summary_present: Optional[bool] = None,
    signal_yield: Optional[InferentialSignalYield | Mapping[str, Any]] = None,
    inferential_replay_weight: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> InferentialLearnabilityContract:
    summary_known = bool(
        summary_present
        if summary_present is not None
        else overlay_joined
        or abs(_safe_float(epiplexity_delta)) > 1e-12
        or abs(_safe_float(epiplexity_confidence)) > 1e-12
    )
    if isinstance(signal_yield, InferentialSignalYield):
        signal = signal_yield
    elif isinstance(signal_yield, Mapping):
        signal = InferentialSignalYield(
            frontier_term=_safe_float(signal_yield.get("frontier_term", frontier_gain)),
            epiplexity_term=_safe_float(signal_yield.get("epiplexity_term", 0.0)),
            transfer_term=_safe_float(signal_yield.get("transfer_term", transfer_score)),
            quality_factor=_safe_float(signal_yield.get("quality_factor", 1.0), 1.0),
            score=_safe_float(signal_yield.get("score", 0.0)),
        )
    else:
        signal = compile_signal_yield(
            frontier_gain=frontier_gain,
            epiplexity_delta=epiplexity_delta,
            epiplexity_confidence=epiplexity_confidence,
            transfer_score=transfer_score,
            data_quality=data_quality,
            provenance_quality=provenance_quality,
        )
    replay_weight = (
        _safe_float(inferential_replay_weight)
        if inferential_replay_weight is not None
        else compute_inferential_replay_weight(
            signal_yield_score=signal.score,
            trust_score=trust_score,
        )
    )
    learnability_class = classify_learnability_class(
        summary_present=summary_known,
        overlay_joined=overlay_joined,
        benchmark_eligible=benchmark_eligible,
        semantic_grounding_non_heuristic=semantic_grounding_non_heuristic,
        promotion_trace_complete=promotion_trace_complete,
        budget_settlement_live=budget_settlement_live,
    )
    return InferentialLearnabilityContract(
        subject_id=subject_id,
        subject_kind=subject_kind,
        datapack_id=datapack_id,
        learnability_class=learnability_class,
        signal_yield=signal.to_dict(),
        inferential_replay_weight=float(replay_weight),
        frontier_gain=_safe_float(frontier_gain),
        epiplexity_delta=_safe_float(epiplexity_delta),
        epiplexity_confidence=_safe_float(epiplexity_confidence),
        transfer_score=_safe_float(transfer_score),
        data_quality=_safe_float(data_quality),
        provenance_quality=_safe_float(provenance_quality),
        trust_score=_safe_float(trust_score, 0.5),
        benchmark_eligible=bool(benchmark_eligible),
        semantic_grounding_non_heuristic=bool(semantic_grounding_non_heuristic),
        promotion_trace_complete=bool(promotion_trace_complete),
        budget_settlement_live=bool(budget_settlement_live),
        overlay_joined=bool(overlay_joined),
        summary_present=bool(summary_known),
        metadata=_mapping(metadata),
    )


def summarize_inferential_learnability_contracts(
    contracts: Sequence[InferentialLearnabilityContract | Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [
        contract
        for contract in (
            coerce_inferential_learnability_contract(row)
            for row in list(contracts or [])
        )
        if contract is not None
    ]
    class_counts: Dict[str, int] = {}
    for row in rows:
        class_counts[row.learnability_class] = class_counts.get(row.learnability_class, 0) + 1
    total = len(rows)
    return {
        "contract_count": total,
        "learnability_class_counts": dict(sorted(class_counts.items())),
        "receipt_backed_count": sum(1 for row in rows if row.receipt_backed),
        "benchmark_receipt_backed_count": sum(
            1 for row in rows if row.learnability_class == "benchmark_receipt_backed"
        ),
        "summary_present_count": sum(1 for row in rows if row.summary_present),
        "mean_signal_yield_score": (
            sum(_safe_float(row.signal_yield.get("score", 0.0)) for row in rows) / float(max(total, 1))
        ),
        "mean_inferential_replay_weight": (
            sum(float(row.inferential_replay_weight) for row in rows) / float(max(total, 1))
        ),
    }


def build_inferential_execution_work_order(
    *,
    decision: Any,
    readiness: ExecutionPreconditionsReport | Mapping[str, Any],
    run_id: str,
    episode_id: str,
    objective_profile_id: str,
    source_domain: str,
    datapack_id: Optional[str] = None,
    learnability_contract: Optional[InferentialLearnabilityContract | Mapping[str, Any]] = None,
) -> ExecutionWorkOrder:
    readiness_report = (
        readiness
        if isinstance(readiness, ExecutionPreconditionsReport)
        else ExecutionPreconditionsReport.from_dict(readiness)
    )
    contract = coerce_inferential_learnability_contract(learnability_contract)
    if contract is None and isinstance(getattr(decision, "artifact_summary", None), Mapping):
        contract = coerce_inferential_learnability_contract(
            decision.artifact_summary.get("inferential_learnability_contract")
        )
    order_type = {
        "adapt_now": "adaptation_training",
        "collect_more_data": "data_collection",
        "require_review": "human_review",
    }.get(str(getattr(decision, "decision", "advisory_followup")), "advisory_followup")
    return build_execution_work_order(
        order_type=order_type,
        subject_id=episode_id,
        subject_kind="replay_episode",
        decision=str(getattr(decision, "decision", "no_op")),
        priority=max(
            0.0,
            _safe_float(
                getattr(decision, "allowed_budget", 0.0),
                _safe_float(getattr(decision, "net_benefit", 0.0)),
            ),
        ),
        recommended_mode=str(getattr(decision, "recommended_training_mode", "no_training")),
        readiness=readiness_report,
        reasons=list(getattr(decision, "reasons", []) or []),
        artifact_refs={
            "run_id": run_id,
            "episode_id": episode_id,
            "datapack_id": datapack_id or (contract.datapack_id if contract is not None else episode_id),
        },
        metadata={
            "contract_kind": "inferential_execution_work_order_v1",
            "decision_id": getattr(decision, "decision_id", ""),
            "objective_profile_id": objective_profile_id,
            "source_domain": source_domain,
            "inferential_reward": _mapping(
                getattr(decision, "artifact_summary", {}).get("inferential_reward", {})
                if isinstance(getattr(decision, "artifact_summary", None), Mapping)
                else {}
            ),
            "inferential_learnability_contract": (
                contract.to_dict() if contract is not None else None
            ),
        },
    )


__all__ = [
    "InferentialLearnabilityContract",
    "build_inferential_execution_work_order",
    "build_inferential_learnability_contract",
    "classify_learnability_class",
    "coerce_inferential_learnability_contract",
    "summarize_inferential_learnability_contracts",
]
