"""Objective scalarization profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ObjectiveProfile:
    """Contract-time scalarization profile for objective tensors."""

    profile_id: str = "default"
    scalarizer: str = "weighted_sum"
    weights: Dict[str, float] = field(default_factory=dict)
    maximize: Dict[str, bool] = field(default_factory=dict)
    constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lexicographic_order: List[str] = field(default_factory=list)
    epsilon: Dict[str, float] = field(default_factory=dict)
    chebyshev_target: Dict[str, float] = field(default_factory=dict)
    penalty_weight: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def weight_for(self, axis: str, default: float = 1.0) -> float:
        return float(self.weights.get(axis, default))

    def maximize_axis(self, axis: str, default: bool = True) -> bool:
        return bool(self.maximize.get(axis, default))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "scalarizer": self.scalarizer,
            "weights": dict(self.weights),
            "maximize": dict(self.maximize),
            "constraints": dict(self.constraints),
            "lexicographic_order": list(self.lexicographic_order),
            "epsilon": dict(self.epsilon),
            "chebyshev_target": dict(self.chebyshev_target),
            "penalty_weight": float(self.penalty_weight),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "ObjectiveProfile":
        if not payload:
            return cls()
        return cls(
            profile_id=str(payload.get("profile_id", "default")),
            scalarizer=str(payload.get("scalarizer", "weighted_sum")),
            weights=dict(payload.get("weights", {}) or {}),
            maximize=dict(payload.get("maximize", {}) or {}),
            constraints=dict(payload.get("constraints", {}) or {}),
            lexicographic_order=list(payload.get("lexicographic_order", []) or []),
            epsilon=dict(payload.get("epsilon", {}) or {}),
            chebyshev_target=dict(payload.get("chebyshev_target", {}) or {}),
            penalty_weight=float(payload.get("penalty_weight", 10.0)),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def weighted_sum(
        cls,
        weights: Mapping[str, float],
        *,
        profile_id: str = "weighted_sum_default",
        maximize: Optional[Mapping[str, bool]] = None,
    ) -> "ObjectiveProfile":
        return cls(
            profile_id=profile_id,
            scalarizer="weighted_sum",
            weights=dict(weights),
            maximize=dict(maximize or {}),
        )
