"""
ConditionVector: centralized conditioning signal for policies/diffusion/SIMA-2.

Frozen dataclass, JSON-safe serialization, and deterministic hashing for
string fields to preserve reproducibility.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import hashlib
import numpy as np

from src.utils.json_safe import to_json_safe


def _hash_to_unit(value: str) -> float:
    """Stable hash -> [0,1] float for categorical values."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _flatten_sequence(seq: Optional[Sequence[Union[int, float]]]) -> List[float]:
    if seq is None:
        return []
    flat: List[float] = []
    for item in seq:
        try:
            flat.append(float(item))
        except Exception:
            continue
    return flat


@dataclass(frozen=True)
class ConditionVector:
    # Task identification
    task_id: str
    env_id: str
    backend_id: str

    # Economic state (read-only from econ layer)
    target_mpl: float
    current_wage_parity: float
    energy_budget_wh: float

    # Semantic emphasis (from curriculum + SIMA-2)
    skill_mode: str
    ood_risk_level: float
    recovery_priority: float
    novelty_tier: int

    # SIMA-2 / RECAP state
    sima2_trust_score: float
    recap_goodness_bucket: str

    # Objective preset (maps to multi-objective vector)
    objective_preset: str
    objective_vector: Optional[Sequence[float]] = None

    # Timestep / phase info
    episode_step: int = 0
    curriculum_phase: str = "warmup"

    # Phase H economic learner signals (flag-gated)
    exploration_uplift: Optional[float] = None
    skill_roi_estimate: Optional[float] = None

    # Free-form metadata (not consumed by policies)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization."""
        payload = {
            "task_id": self.task_id,
            "env_id": self.env_id,
            "backend_id": self.backend_id,
            "target_mpl": float(self.target_mpl),
            "current_wage_parity": float(self.current_wage_parity),
            "energy_budget_wh": float(self.energy_budget_wh),
            "skill_mode": self.skill_mode,
            "ood_risk_level": float(self.ood_risk_level),
            "recovery_priority": float(self.recovery_priority),
            "novelty_tier": int(self.novelty_tier),
            "sima2_trust_score": float(self.sima2_trust_score),
            "recap_goodness_bucket": self.recap_goodness_bucket,
            "objective_preset": self.objective_preset,
            "objective_vector": list(self.objective_vector) if self.objective_vector is not None else None,
            "episode_step": int(self.episode_step),
            "curriculum_phase": self.curriculum_phase,
            "metadata": to_json_safe(self.metadata),
        }
        # Phase H fields (only if present)
        if self.exploration_uplift is not None:
            payload["exploration_uplift"] = float(self.exploration_uplift)
        if self.skill_roi_estimate is not None:
            payload["skill_roi_estimate"] = float(self.skill_roi_estimate)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionVector":
        """Deserialize from JSON-safe dict."""
        return cls(
            task_id=str(data.get("task_id") or ""),
            env_id=str(data.get("env_id") or ""),
            backend_id=str(data.get("backend_id") or ""),
            target_mpl=float(data.get("target_mpl", 0.0)),
            current_wage_parity=float(data.get("current_wage_parity", 0.0)),
            energy_budget_wh=float(data.get("energy_budget_wh", 0.0)),
            skill_mode=str(data.get("skill_mode") or "efficiency_throughput"),
            ood_risk_level=float(data.get("ood_risk_level", 0.0)),
            recovery_priority=float(data.get("recovery_priority", 0.0)),
            novelty_tier=int(data.get("novelty_tier", 0)),
            sima2_trust_score=float(data.get("sima2_trust_score", 0.0)),
            recap_goodness_bucket=str(data.get("recap_goodness_bucket") or "bronze"),
            objective_preset=str(data.get("objective_preset") or "balanced"),
            objective_vector=_flatten_sequence(data.get("objective_vector")),
            episode_step=int(data.get("episode_step", 0)),
            curriculum_phase=str(data.get("curriculum_phase") or "warmup"),
            exploration_uplift=float(data["exploration_uplift"]) if "exploration_uplift" in data else None,
            skill_roi_estimate=float(data["skill_roi_estimate"]) if "skill_roi_estimate" in data else None,
            metadata=to_json_safe(data.get("metadata") or {}),
        )

    def to_vector(self) -> np.ndarray:
        """
        Deterministic numeric representation for neural modules.
        Strings hashed into [0,1]; arrays flattened in a fixed order.
        """
        fields: List[float] = [
            float(self.target_mpl),
            float(self.current_wage_parity),
            float(self.energy_budget_wh),
            float(self.ood_risk_level),
            float(self.recovery_priority),
            float(self.novelty_tier),
            float(self.sima2_trust_score),
            float(self.episode_step),
        ]

        # Hash categorical/string fields to stable floats
        fields.extend(
            [
                _hash_to_unit(self.task_id),
                _hash_to_unit(self.env_id),
                _hash_to_unit(self.backend_id),
                _hash_to_unit(self.skill_mode),
                _hash_to_unit(self.recap_goodness_bucket),
                _hash_to_unit(self.objective_preset),
                _hash_to_unit(self.curriculum_phase),
            ]
        )

        if self.objective_vector is not None:
            fields.extend(_flatten_sequence(self.objective_vector))

        return np.array(fields, dtype=np.float32)
