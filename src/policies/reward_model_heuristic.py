"""
Heuristic RewardModelPolicy.

Deterministically scores episodes using econ vectors, recap scores, and
semantic tags. This is advisory-only and does not alter rewards or control.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from src.ontology.models import Episode, EconVector
from src.policies.interfaces import RewardModelPolicy
from src.policies.reward_model_types import RewardModelEpisodeScores
from src.vla.recap_inference import RecapEpisodeScores


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


class HeuristicRewardModelPolicy(RewardModelPolicy):
    def score_episode(
        self,
        episode: Episode,
        econ: EconVector,
        tags: Dict[str, Any],
        recap_scores: Optional[RecapEpisodeScores] = None,
    ) -> RewardModelEpisodeScores:
        status = (getattr(episode, "status", "") or "").lower()
        success_bonus = 0.2 if status == "success" else 0.0
        target_mpl = float(econ.metadata.get("target_mpl_units_per_hour", econ.mpl_units_per_hour or 1.0))
        mpl_norm = _clamp01(econ.mpl_units_per_hour / max(target_mpl, 1e-6))
        wage_norm = _clamp01(econ.wage_parity / 2.0)
        damage_penalty = _clamp01(self._damage_proxy(econ))
        energy_penalty = _clamp01(self._energy_proxy(econ))

        recap_quality_hint = 0.0
        recap_error_hint = 0.0
        if recap_scores:
            recap_quality_hint = _clamp01(recap_scores.recap_goodness_score) if hasattr(recap_scores, "recap_goodness_score") else 0.0
            recap_error_hint = self._extract_recap_error_prob(recap_scores)

        quality_score = _clamp01(
            0.45 * (1.0 - damage_penalty)
            + 0.25 * mpl_norm
            + 0.2 * wage_norm
            + 0.1 * recap_quality_hint
        )
        error_probability = _clamp01(0.25 + 0.5 * damage_penalty + 0.15 * energy_penalty + 0.1 * recap_error_hint)
        progress_estimate = _clamp01(
            0.35 * mpl_norm + 0.2 * (1.0 - error_probability) + 0.15 * wage_norm + success_bonus + 0.1 * recap_quality_hint
        )

        subtasks = self._derive_subtasks(tags, episode.task_id)

        return RewardModelEpisodeScores(
            episode_id=episode.episode_id,
            progress_estimate=progress_estimate,
            quality_score=quality_score,
            error_probability=error_probability,
            subtask_labels=sorted(subtasks),
        )

    def _damage_proxy(self, econ: EconVector) -> float:
        collision = econ.components.get("collision_penalty", 0.0) if hasattr(econ, "components") else 0.0
        damage = abs(econ.damage_cost)
        denom = abs(econ.reward_scalar_sum) + 1e-6
        raw = max(damage, collision) / denom
        return raw

    def _energy_proxy(self, econ: EconVector) -> float:
        energy = abs(econ.energy_cost)
        denom = abs(econ.reward_scalar_sum) + 1e-6
        return energy / denom

    def _extract_recap_error_prob(self, recap_scores: RecapEpisodeScores) -> float:
        metric_map = getattr(recap_scores, "metric_distributions", {}) or {}
        error_keys = [k for k in metric_map.keys() if "error" in k.lower() or "damage" in k.lower()]
        if not error_keys:
            return 0.0
        vals: List[float] = []
        for key in sorted(error_keys):
            dist = metric_map.get(key, [])
            vals.append(_safe_mean(dist))
        return _clamp01(_safe_mean(vals))

    def _derive_subtasks(self, tags: Dict[str, Any], task_id: str) -> Set[str]:
        labels: Set[str] = set()
        if isinstance(tags, dict):
            labels.update(self._from_object_focus(tags))
            labels.update(self._from_semantic_lists(tags))
        if task_id:
            labels.update(self._task_default_labels(task_id))
        return {lbl for lbl in labels if lbl}

    def _from_object_focus(self, tags: Dict[str, Any]) -> Set[str]:
        labels: Set[str] = set()
        for key in ("objects_present", "object_focus", "focus_objects"):
            objs = tags.get(key, [])
            if isinstance(objs, str):
                objs = [objs]
            for obj in objs or []:
                if not obj:
                    continue
                if "drawer" in str(obj):
                    labels.update(["approach_drawer", "grasp_handle", "pull"])
                if "vase" in str(obj):
                    labels.update(["avoid_vase", "stabilize_carry"])
                if "cup" in str(obj) or "mug" in str(obj):
                    labels.update(["approach_cup", "grasp", "transport"])
        return labels

    def _from_semantic_lists(self, tags: Dict[str, Any]) -> Set[str]:
        labels: Set[str] = set()
        semantic_lists: List[Iterable[Any]] = []
        for key in ("semantic_tags", "affordance_tags", "risk_tags"):
            val = tags.get(key, [])
            if isinstance(val, dict):
                val = list(val.values())
            if isinstance(val, list):
                semantic_lists.append(val)
        for seq in semantic_lists:
            for item in seq:
                item_str = str(item)
                if "open" in item_str and "drawer" in item_str:
                    labels.update(["approach_drawer", "pull"])
                if "grasp" in item_str:
                    labels.add("grasp")
                if "recover" in item_str:
                    labels.add("recover_object")
        return labels

    def _task_default_labels(self, task_id: str) -> Set[str]:
        tid = task_id.lower()
        defaults: Set[str] = set()
        if "drawer" in tid:
            defaults.update(["approach_drawer", "pull"])
        if "dish" in tid or "cup" in tid:
            defaults.update(["approach_cup", "grasp", "transport"])
        return defaults
