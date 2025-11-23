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
        multiplier = 1.0
        for tag in tags:
            tag_name = tag.get("tag_type") if isinstance(tag, dict) else None
            if not tag_name and isinstance(tag, str):
                tag_name = tag
            if not tag_name or tag_name not in self.trust_matrix:
                continue
            entry = self.trust_matrix.get(tag_name, {})
            trust_score = float(entry.get("trust_score", 0.0))
            sampling_mult = float(entry.get("sampling_multiplier", 1.0))
            if sampling_mult != 1.0:
                multiplier = max(multiplier, sampling_mult)
                continue
            if trust_score > 0.8:
                multiplier = max(multiplier, 5.0)
            elif trust_score > 0.5:
                multiplier = max(multiplier, 1.5)
        return float(multiplier)
