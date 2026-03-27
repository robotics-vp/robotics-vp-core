"""Advisory wrappers for inferential training budget gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.economics.inferential_contract import build_inferential_execution_work_order
from src.economics.inferential_training_gate import (
    InferentialTrainingCandidate,
    InferentialTrainingGate,
)


@dataclass(frozen=True)
class AdaptationBudgetArtifact:
    """JSON-safe summary for orchestration consumers."""

    decisions: list[Dict[str, Any]]
    summary: Dict[str, Any]
    work_orders: list[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": list(self.decisions),
            "summary": dict(self.summary),
            "work_orders": list(self.work_orders),
        }


def evaluate_adaptation_budget(
    *,
    gate: InferentialTrainingGate,
    candidates: Sequence[InferentialTrainingCandidate],
    execution_preconditions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> AdaptationBudgetArtifact:
    decisions = [gate.evaluate(candidate) for candidate in candidates]
    payloads = [decision.to_dict() for decision in decisions]
    work_orders: list[Dict[str, Any]] = []
    preconditions_by_episode = {
        str(key): dict(value)
        for key, value in dict(execution_preconditions or {}).items()
        if value is not None
    }
    for candidate, decision in zip(candidates, decisions):
        readiness = preconditions_by_episode.get(candidate.episode_id)
        if readiness is None:
            continue
        if decision.decision == "no_op":
            continue
        work_order = build_inferential_execution_work_order(
            decision=decision,
            readiness=readiness,
            run_id=candidate.run_id,
            episode_id=candidate.episode_id,
            objective_profile_id=candidate.objective_profile_id,
            source_domain=candidate.source_domain,
            datapack_id=str(candidate.metadata.get("datapack_id") or candidate.episode_id),
            learnability_contract=candidate.metadata.get("inferential_learnability_contract"),
        )
        work_orders.append(work_order.to_dict())
    return AdaptationBudgetArtifact(
        decisions=payloads,
        summary={
            "num_candidates": len(payloads),
            "adapt_now": sum(1 for decision in decisions if decision.decision == "adapt_now"),
            "collect_more_data": sum(1 for decision in decisions if decision.decision == "collect_more_data"),
            "require_review": sum(1 for decision in decisions if decision.decision == "require_review"),
            "no_op": sum(1 for decision in decisions if decision.decision == "no_op"),
            "work_orders": len(work_orders),
            "executable_work_orders": sum(1 for row in work_orders if row.get("ready")),
            "blocked_work_orders": sum(1 for row in work_orders if not row.get("ready")),
        },
        work_orders=work_orders,
    )
