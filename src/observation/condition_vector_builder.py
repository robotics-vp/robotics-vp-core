"""
ConditionVectorBuilder: single source of truth for constructing ConditionVector.

Combines episode/task metadata, econ state, curriculum phase, SIMA-2 trust,
and datapack tags. Reads inputs, never mutates them, and falls back to
deterministic defaults when fields are missing.
"""
from typing import Any, Dict, Optional

from src.observation.condition_vector import ConditionVector, _flatten_sequence


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Graceful attribute/dict lookup."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ConditionVectorBuilder:
    """
    Builds a ConditionVector per episode/rollout.

    This is the only fusion point for task/env metadata, econ state, and
    semantic curriculum signals.
    """

    DEFAULT_SKILL_MODE = "efficiency_throughput"
    DEFAULT_OBJECTIVE = "balanced"
    DEFAULT_RECAP_BUCKET = "bronze"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.skill_mode_order = (
            self.config.get("skill_mode_order")
            or [
                "frontier_exploration",
                "safety_critical",
                "efficiency_throughput",
                "recovery_heavy",
            ]
        )

    def build(
        self,
        *,
        episode_config: Any,
        econ_state: Any,
        curriculum_phase: str,
        sima2_trust: Optional[Any],
        datapack_metadata: Optional[Dict[str, Any]] = None,
        episode_step: int = 0,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ConditionVector:
        """
        Construct a ConditionVector with deterministic fallbacks.
        """
        overrides = overrides or {}
        meta = datapack_metadata or {}

        skill_mode = overrides.get("skill_mode") or meta.get("skill_mode") or self._select_skill_mode(
            curriculum_phase, meta, sima2_trust
        )

        objective_vector = overrides.get("objective_vector") or _get(episode_config, "objective_vector")
        objective_vector = _flatten_sequence(objective_vector) if objective_vector is not None else None

        return ConditionVector(
            task_id=str(overrides.get("task_id") or _get(episode_config, "task_id", "")),
            env_id=str(overrides.get("env_id") or _get(episode_config, "env_id", "")),
            backend_id=str(overrides.get("backend_id") or _get(episode_config, "backend_id", _get(episode_config, "backend", ""))),
            target_mpl=float(overrides.get("target_mpl", _get(econ_state, "target_mpl", 0.0))),
            current_wage_parity=float(overrides.get("current_wage_parity", _get(econ_state, "current_wage_parity", 0.0))),
            energy_budget_wh=float(overrides.get("energy_budget_wh", _get(econ_state, "energy_budget_wh", 0.0))),
            skill_mode=str(skill_mode or self.DEFAULT_SKILL_MODE),
            ood_risk_level=float(overrides.get("ood_risk_level", meta.get("ood_risk_level", 0.0))),
            recovery_priority=float(overrides.get("recovery_priority", meta.get("recovery_priority", 0.0))),
            novelty_tier=int(overrides.get("novelty_tier", meta.get("novelty_tier", 0))),
            sima2_trust_score=float(overrides.get("sima2_trust_score", self._get_trust(sima2_trust))),
            recap_goodness_bucket=str(overrides.get("recap_goodness_bucket", meta.get("recap_goodness_bucket", self.DEFAULT_RECAP_BUCKET))),
            objective_preset=str(overrides.get("objective_preset", _get(episode_config, "objective_preset", self.DEFAULT_OBJECTIVE))),
            objective_vector=objective_vector,
            episode_step=int(overrides.get("episode_step", episode_step)),
            curriculum_phase=str(overrides.get("curriculum_phase", curriculum_phase or "warmup")),
            metadata=self._build_metadata(meta),
        )

    def _get_trust(self, sima2_trust: Optional[Any]) -> float:
        if sima2_trust is None:
            return 0.0
        if isinstance(sima2_trust, (int, float)):
            return float(sima2_trust)
        if isinstance(sima2_trust, dict) and "trust_score" in sima2_trust:
            return float(sima2_trust.get("trust_score", 0.0))
        return float(_get(sima2_trust, "trust_score", 0.0))

    def _build_metadata(self, datapack_metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Keep only JSON-safe, low-risk fields
        allowed_keys = ["tags", "datapack_id", "backend_id", "phase", "pack_tier"]
        return {k: v for k, v in (datapack_metadata or {}).items() if k in allowed_keys}

    def _select_skill_mode(self, phase: str, datapack_meta: Dict[str, Any], sima2_trust: Optional[Any]) -> str:
        datapack_tags = set(datapack_meta.get("tags", []) or [])
        trust_score = self._get_trust(sima2_trust)
        if phase == "frontier" or "novelty_tier_2" in datapack_tags:
            return "frontier_exploration"
        if "fragile" in datapack_tags or "high_damage_risk" in datapack_tags:
            return "safety_critical"
        if phase == "refinement" and trust_score > 0.8:
            return "efficiency_throughput"
        if "ood_recovery" in datapack_tags or trust_score < 0.5:
            return "recovery_heavy"
        return self.DEFAULT_SKILL_MODE
