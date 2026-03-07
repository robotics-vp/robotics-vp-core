"""Advisory wrappers for inferential training budget gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from src.economics.inferential_training_gate import (
    InferentialTrainingCandidate,
    InferentialTrainingDecision,
    InferentialTrainingGate,
)


@dataclass(frozen=True)
class AdaptationBudgetArtifact:
    """JSON-safe summary for orchestration consumers."""

    decisions: list[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": list(self.decisions),
            "summary": dict(self.summary),
        }


def evaluate_adaptation_budget(
    *,
    gate: InferentialTrainingGate,
    candidates: Sequence[InferentialTrainingCandidate],
) -> AdaptationBudgetArtifact:
    decisions = [gate.evaluate(candidate) for candidate in candidates]
    payloads = [decision.to_dict() for decision in decisions]
    return AdaptationBudgetArtifact(
        decisions=payloads,
        summary={
            "num_candidates": len(payloads),
            "adapt_now": sum(1 for decision in decisions if decision.decision == "adapt_now"),
            "collect_more_data": sum(1 for decision in decisions if decision.decision == "collect_more_data"),
            "require_review": sum(1 for decision in decisions if decision.decision == "require_review"),
            "no_op": sum(1 for decision in decisions if decision.decision == "no_op"),
        },
    )
