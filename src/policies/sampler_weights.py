"""
Heuristic SamplerWeightPolicy that mirrors DataPackRLSampler weighting.
"""
from typing import Any, Dict, List, Sequence

from src.policies.interfaces import SamplerWeightPolicy
from src.rl import episode_sampling as sampler_utils


def _episode_key(ep: Dict[str, Any]) -> str:
    desc = ep.get("descriptor", {})
    return str(desc.get("pack_id") or desc.get("episode_id") or desc.get("id") or "")


class HeuristicSamplerWeightPolicy(SamplerWeightPolicy):
    def __init__(self, trust_matrix: Dict[str, Any] = None):
        self.trust_matrix = trust_matrix or {}

    def build_features(self, descriptors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Preserve ordering and content for deterministic downstream sampling.
        return [dict(d) for d in descriptors]

    def evaluate(self, features: List[Dict[str, Any]], strategy: str) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for ep in features:
            key = _episode_key(ep)
            if not key:
                continue
            if strategy == "balanced":
                weight = sampler_utils._balanced_weight(ep)
            elif strategy == "frontier_prioritized":
                weight = max(float(ep.get("frontier_score", 0.0)), 1e-6) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "econ_urgency":
                weight = max(float(ep.get("econ_urgency_score", 0.0)), 1e-6) * float(ep.get("recap_weight_multiplier", 1.0))
            else:
                weight = sampler_utils._balanced_weight(ep)
            trust_scale = self._trust_scale(ep)
            weights[key] = float(weight * trust_scale)
        return weights

    def _trust_scale(self, descriptor: Dict[str, Any]) -> float:
        tags = descriptor.get("semantic_tags") or descriptor.get("tags") or []
        scores: List[float] = []
        for tag in tags:
            tag_name = tag.get("tag_type") if isinstance(tag, dict) else None
            if not tag_name and isinstance(tag, str):
                tag_name = tag
            if tag_name and tag_name in self.trust_matrix:
                try:
                    scores.append(float(self.trust_matrix[tag_name].get("trust_score", 1.0)))
                except Exception:
                    continue
        if not scores:
            return 1.0
        return max(0.1, sum(scores) / len(scores))
