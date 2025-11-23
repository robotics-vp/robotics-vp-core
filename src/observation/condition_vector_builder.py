"""
ConditionVectorBuilder: single source of truth for constructing ConditionVector.

Combines episode/task metadata, econ state, curriculum phase, SIMA-2 trust,
and datapack tags. Reads inputs, never mutates them, and falls back to
deterministic defaults when fields are missing.
"""
from typing import Any, Dict, Optional, Sequence

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
        self.skill_mode_order = self.config.get("skill_mode_order") or [
            "frontier_exploration",
            "safety_critical",
            "efficiency_throughput",
            "recovery_heavy",
            "default",
        ]

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
        econ_slice: Optional[Any] = None,
        semantic_tags: Optional[Dict[str, float]] = None,
        recap_scores: Optional[Any] = None,
        trust_summary: Optional[Dict[str, Any]] = None,
        episode_metadata: Optional[Dict[str, Any]] = None,
        advisory_context: Optional[Dict[str, Any]] = None,
    ) -> ConditionVector:
        """
        Construct a ConditionVector with deterministic fallbacks.
        """
        overrides = overrides or {}
        meta = datapack_metadata or {}
        semantic_tags = semantic_tags or {}
        episode_metadata = episode_metadata or {}
        advisory_context = advisory_context or {}

        phase = str(overrides.get("curriculum_phase", curriculum_phase or "warmup"))
        tags = self._merge_tags(meta.get("tags"), semantic_tags)
        trust_score = self._summarize_trust(trust_summary, sima2_trust)

        skill_mode = overrides.get("skill_mode") or advisory_context.get("skill_mode")
        if not skill_mode:
            skill_mode = select_skill_mode(
                tags=tags,
                trust_matrix=trust_summary,
                curriculum_phase=phase,
                default=self.DEFAULT_SKILL_MODE,
                advisory=advisory_context,
                skill_mode_order=self.skill_mode_order,
            )

        objective_vector = overrides.get("objective_vector") or _get(episode_config, "objective_vector")
        objective_vector = _flatten_sequence(objective_vector) if objective_vector is not None else None

        recap_bucket = overrides.get("recap_goodness_bucket")
        if recap_bucket is None:
            recap_bucket = self._bucketize_recap(recap_scores)

        novelty_tier = overrides.get("novelty_tier")
        if novelty_tier is None:
            novelty_tier = self._novelty_tier_from_context(advisory_context, tags, recap_scores)

        episode_step_val = overrides.get("episode_step")
        if episode_step_val is None:
            episode_step_val = episode_metadata.get("timestep", episode_metadata.get("step", episode_step))

        sampler_strategy = episode_metadata.get("sampler_strategy") or meta.get("sampler_strategy")
        wage_parity = overrides.get("current_wage_parity", _get(econ_state, "current_wage_parity", 0.0))
        wage_parity = self._safe_float(wage_parity, self._safe_float(_get(econ_slice, "wage_parity", None)))

        return ConditionVector(
            task_id=str(overrides.get("task_id") or _get(episode_config, "task_id", "")),
            env_id=str(overrides.get("env_id") or _get(episode_config, "env_id", "")),
            backend_id=str(overrides.get("backend_id") or _get(episode_config, "backend_id", _get(episode_config, "backend", ""))),
            target_mpl=self._safe_float(
                overrides.get("target_mpl", _get(econ_state, "target_mpl", _get(econ_slice, "mpl", 0.0)))
            ),
            current_wage_parity=self._safe_float(wage_parity),
            energy_budget_wh=self._safe_float(
                overrides.get("energy_budget_wh", _get(econ_state, "energy_budget_wh", _get(econ_slice, "energy_wh", 0.0)))
            ),
            skill_mode=str(skill_mode or self.DEFAULT_SKILL_MODE),
            ood_risk_level=self._safe_float(
                overrides.get(
                    "ood_risk_level",
                    self._safe_float(meta.get("ood_risk_level", episode_metadata.get("ood_score", 0.0))),
                )
            ),
            recovery_priority=self._safe_float(
                overrides.get(
                    "recovery_priority",
                    self._safe_float(meta.get("recovery_priority", episode_metadata.get("recovery_score", 0.0))),
                )
            ),
            novelty_tier=int(novelty_tier or 0),
            sima2_trust_score=self._safe_float(overrides.get("sima2_trust_score", trust_score)),
            recap_goodness_bucket=str(recap_bucket or self.DEFAULT_RECAP_BUCKET),
            objective_preset=str(overrides.get("objective_preset", _get(episode_config, "objective_preset", self.DEFAULT_OBJECTIVE))),
            objective_vector=objective_vector,
            episode_step=int(episode_step_val or 0),
            curriculum_phase=str(phase),
            metadata=self._build_metadata(
                datapack_metadata=meta,
                episode_metadata=episode_metadata,
                econ_slice=econ_slice,
                semantic_tags=tags,
                recap_scores=recap_scores,
                advisory_context=advisory_context,
                sampler_strategy=sampler_strategy,
            ),
        )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            try:
                return float(default)
            except Exception:
                return 0.0

    def _get_trust(self, sima2_trust: Optional[Any]) -> float:
        if sima2_trust is None:
            return 0.0
        if isinstance(sima2_trust, (int, float)):
            return float(sima2_trust)
        if isinstance(sima2_trust, dict) and "trust_score" in sima2_trust:
            return float(sima2_trust.get("trust_score", 0.0))
        return float(_get(sima2_trust, "trust_score", 0.0))

    def _build_metadata(
        self,
        *,
        datapack_metadata: Dict[str, Any],
        episode_metadata: Dict[str, Any],
        econ_slice: Optional[Any],
        semantic_tags: Dict[str, float],
        recap_scores: Optional[Any],
        advisory_context: Dict[str, Any],
        sampler_strategy: Optional[str],
    ) -> Dict[str, Any]:
        # Keep only JSON-safe, low-risk fields
        allowed_keys = ["tags", "datapack_id", "backend_id", "phase", "pack_tier", "pack_id"]
        meta: Dict[str, Any] = {k: v for k, v in (datapack_metadata or {}).items() if k in allowed_keys}
        if episode_metadata.get("episode_id"):
            meta["episode_id"] = episode_metadata["episode_id"]
        if sampler_strategy:
            meta["sampler_strategy"] = sampler_strategy
        if advisory_context:
            meta["advisory"] = {
                "frontier_score": advisory_context.get("frontier_score"),
                "priority": advisory_context.get("priority"),
                "skill_mode": advisory_context.get("skill_mode"),
            }
        if semantic_tags:
            meta["semantic_tags"] = {k: float(v) for k, v in sorted(semantic_tags.items(), key=lambda kv: kv[0])}
        if recap_scores is not None:
            try:
                recap_score = _get(recap_scores, "recap_goodness_score", recap_scores if isinstance(recap_scores, (int, float)) else None)
                meta["recap"] = {"recap_goodness_score": float(recap_score or 0.0)}
            except Exception:
                pass
        if econ_slice is not None:
            econ_payload = {
                "mpl": self._safe_float(_get(econ_slice, "mpl", None)),
                "energy_wh": self._safe_float(_get(econ_slice, "energy_wh", None)),
                "damage_cost": self._safe_float(_get(econ_slice, "damage_cost", None)),
                "wage_parity": self._safe_float(_get(econ_slice, "wage_parity", None)),
            }
            meta["econ_slice"] = econ_payload
        return meta

    def _summarize_trust(self, trust_summary: Optional[Dict[str, Any]], sima2_trust: Optional[Any]) -> float:
        if trust_summary and isinstance(trust_summary, dict) and trust_summary:
            try:
                vals = [float(v) for v in trust_summary.values() if isinstance(v, (int, float))]
                if vals:
                    return float(sum(vals) / len(vals))
            except Exception:
                pass
        return self._get_trust(sima2_trust)

    def _merge_tags(self, datapack_tags: Optional[Sequence[Any]], semantic_tags: Dict[str, float]) -> Dict[str, float]:
        tags: Dict[str, float] = {}
        for tag in datapack_tags or []:
            tags[str(tag)] = tags.get(str(tag), 0.0) + 1.0
        for key, val in semantic_tags.items():
            try:
                tags[str(key)] = tags.get(str(key), 0.0) + float(val)
            except Exception:
                continue
        return tags

    def _bucketize_recap(self, recap_scores: Optional[Any]) -> str:
        score = None
        if recap_scores is None:
            return self.DEFAULT_RECAP_BUCKET
        if isinstance(recap_scores, (int, float)):
            score = float(recap_scores)
        elif isinstance(recap_scores, dict):
            score = recap_scores.get("recap_goodness_score")
        else:
            score = _get(recap_scores, "recap_goodness_score")
        if score is None:
            return self.DEFAULT_RECAP_BUCKET
        try:
            score_f = float(score)
        except Exception:
            return self.DEFAULT_RECAP_BUCKET
        if score_f >= 0.85:
            return "platinum"
        if score_f >= 0.65:
            return "gold"
        if score_f >= 0.45:
            return "silver"
        return self.DEFAULT_RECAP_BUCKET

    def _novelty_tier_from_context(
        self,
        advisory_context: Dict[str, Any],
        semantic_tags: Dict[str, float],
        recap_scores: Optional[Any],
    ) -> int:
        frontier_score = advisory_context.get("frontier_score")
        if frontier_score is not None:
            try:
                return int(min(3, max(0, round(float(frontier_score)))))
            except Exception:
                pass
        if recap_scores and isinstance(recap_scores, dict):
            try:
                adv_bins = recap_scores.get("advantage_bin_probs") or []
                if adv_bins:
                    return int(min(len(adv_bins), max(0, adv_bins.index(max(adv_bins))) + 1))
            except Exception:
                pass
        if semantic_tags:
            try:
                return int(min(3, max(0, round(max(semantic_tags.values())))))
            except Exception:
                return 0
        return 0


def select_skill_mode(
    *,
    tags: Optional[Dict[str, float]],
    trust_matrix: Optional[Dict[str, Any]],
    curriculum_phase: str,
    default: str = ConditionVectorBuilder.DEFAULT_SKILL_MODE,
    advisory: Optional[Dict[str, Any]] = None,
    skill_mode_order: Optional[Sequence[str]] = None,
) -> str:
    """
    Deterministic skill_mode resolver shared by samplers and ConditionVector.
    """
    advisory = advisory or {}
    if advisory.get("skill_mode"):
        return str(advisory["skill_mode"])

    skill_mode_order = list(skill_mode_order or [])
    tags = tags or {}
    phase = (curriculum_phase or "warmup").lower()

    frontier_flag = advisory.get("frontier") or advisory.get("is_frontier")
    frontier_flag = frontier_flag or any("frontier" in str(k) for k in tags.keys())
    if frontier_flag or phase == "frontier":
        return _coerce_skill_mode("frontier_exploration", skill_mode_order, default)

    safety_flag = advisory.get("safety_critical") or any("fragile" in str(k) or "damage" in str(k) for k in tags.keys())
    if safety_flag:
        return _coerce_skill_mode("safety_critical", skill_mode_order, default)

    trust_score = 0.0
    if trust_matrix:
        try:
            vals = [float(v) for v in trust_matrix.values() if isinstance(v, (int, float))]
            if vals:
                trust_score = sum(vals) / len(vals)
        except Exception:
            trust_score = 0.0
    if advisory.get("trust_score") is not None:
        try:
            trust_score = float(advisory["trust_score"])
        except Exception:
            pass

    if trust_score > 0.8 and phase in {"skill_building", "frontier", "fine_tuning"}:
        return _coerce_skill_mode("efficiency_throughput", skill_mode_order, default)
    if trust_score < 0.5 or any("recovery" in str(k) for k in tags.keys()):
        return _coerce_skill_mode("recovery_heavy", skill_mode_order, default)

    return _coerce_skill_mode(default, skill_mode_order, default)


def _coerce_skill_mode(candidate: str, order: Sequence[str], default: str) -> str:
    if candidate in order:
        return candidate
    if default in order:
        return default
    return str(candidate or default)
