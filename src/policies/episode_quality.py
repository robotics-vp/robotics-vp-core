"""
Heuristic EpisodeQualityPolicy for recap/alignment signals.

Wraps existing recap goodness or reward statistics; advisory-only.
"""
from typing import Any, Dict, Optional, Sequence
import math

from src.policies.interfaces import EpisodeQualityPolicy
from src.utils.json_safe import to_json_safe


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


class HeuristicEpisodeQualityPolicy(EpisodeQualityPolicy):
    def build_features(
        self,
        rewards: Sequence[float],
        reward_components: Sequence[Dict[str, Any]],
        collisions: Sequence[Any],
        recap_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rewards = list(rewards or [])
        mean_reward = sum(float(r) for r in rewards) / len(rewards) if rewards else 0.0
        variance = 0.0
        if rewards:
            variance = sum((float(r) - mean_reward) ** 2 for r in rewards) / len(rewards)
        recap_goodness = None
        if recap_scores:
            recap_goodness = _safe_float(recap_scores.get("recap_goodness_score"))
        collision_count = len(collisions or [])
        metrics = {}
        if reward_components:
            merged: Dict[str, float] = {}
            for comp in reward_components:
                for k, v in comp.items():
                    merged[k] = merged.get(k, 0.0) + _safe_float(v)
            metrics = merged
        return {
            "mean_reward": mean_reward,
            "reward_variance": variance,
            "recap_goodness_score": recap_goodness,
            "collision_count": collision_count,
            "metrics": metrics,
        }

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        recap_score = features.get("recap_goodness_score")
        quality_score = _safe_float(recap_score) if recap_score is not None else _safe_float(features.get("mean_reward"))
        variance = _safe_float(features.get("reward_variance"))
        collision_count = _safe_float(features.get("collision_count"))
        anomaly_score = math.tanh(variance + collision_count)
        metadata = {
            "reward_variance": variance,
            "collision_count": collision_count,
            "metrics": features.get("metrics", {}),
        }
        return {
            "quality_score": quality_score,
            "anomaly_score": anomaly_score,
            "metadata": to_json_safe(metadata),
        }
