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
            weights[key] = float(weight)
        return weights
