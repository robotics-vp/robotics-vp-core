"""
Reward term definitions for workcell environments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class WorkcellRewardTerms:
    """
    Reward term weights for workcell tasks.
    """
    sparse_success: float = 1.0
    dense_progress: float = 0.0
    time_penalty: float = -0.01
    error_penalty: float = -0.1
    tolerance_bonus: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sparse_success": self.sparse_success,
            "dense_progress": self.dense_progress,
            "time_penalty": self.time_penalty,
            "error_penalty": self.error_penalty,
            "tolerance_bonus": self.tolerance_bonus,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkcellRewardTerms":
        """Deserialize from dictionary."""
        return cls(
            sparse_success=data.get("sparse_success", 1.0),
            dense_progress=data.get("dense_progress", 0.0),
            time_penalty=data.get("time_penalty", -0.01),
            error_penalty=data.get("error_penalty", -0.1),
            tolerance_bonus=data.get("tolerance_bonus", 0.0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkcellRewardTerms":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_reward(
    terms: WorkcellRewardTerms,
    *,
    success: bool = False,
    progress: float = 0.0,
    time_cost: float = 0.0,
    error_count: int = 0,
    tolerance_met: bool = False,
) -> float:
    """
    Combine reward terms into a single scalar.
    """
    reward = 0.0
    if success:
        reward += terms.sparse_success
    reward += terms.dense_progress * float(progress)
    reward += terms.time_penalty * float(time_cost)
    reward += terms.error_penalty * float(error_count)
    if tolerance_met:
        reward += terms.tolerance_bonus
    return float(reward)
