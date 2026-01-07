"""
Heuristic SamplerWeightPolicy that mirrors DataPackRLSampler weighting.

Includes process_reward-based sampling strategies:
- "process_reward_conf": Weight by process reward confidence
- "process_reward_progress": Weight by phi_star progress (delta)
- "process_reward_quality": Weight by combined confidence and progress
- "embodiment_quality": Weight by embodiment confidence (w_embodiment)
- "embodiment_drift_penalty": Penalize high embodiment drift
- "embodiment_quality_drift": Combine embodiment weight and drift penalty
"""
from typing import Any, Dict, List, Sequence

from src.policies.interfaces import SamplerWeightPolicy
from src.rl import episode_sampling as sampler_utils


def _process_reward_metric(ep: Dict[str, Any], key: str, fallback_keys: List[str], default: float) -> float:
    for k in [key] + fallback_keys:
        if k in ep:
            return float(ep.get(k, default))
    desc = ep.get("descriptor", {}) if isinstance(ep.get("descriptor"), dict) else {}
    for k in [key] + fallback_keys:
        if k in desc:
            return float(desc.get(k, default))
    pr_profile = desc.get("process_reward_profile") or ep.get("process_reward_profile")
    if isinstance(pr_profile, dict):
        for k in [key] + fallback_keys:
            if k in pr_profile:
                return float(pr_profile.get(k, default))
    return float(default)


def _episode_key(ep: Dict[str, Any]) -> str:
    desc = ep.get("descriptor", {})
    return str(desc.get("pack_id") or desc.get("episode_id") or desc.get("id") or "")


def _process_reward_conf_weight(ep: Dict[str, Any]) -> float:
    """Weight based on process reward confidence.

    Higher confidence = more reliable episode for training.
    Uses conf_mean from process reward logging.
    """
    conf = _process_reward_metric(ep, "conf_mean", ["pr_conf_mean"], 0.5)
    # Avoid zero weights
    return max(conf, 0.1)


def _process_reward_progress_weight(ep: Dict[str, Any]) -> float:
    """Weight based on phi_star progress (delta).

    Higher progress = more successful trajectory.
    Uses phi_star_delta from process reward logging.
    """
    delta = _process_reward_metric(ep, "phi_star_delta", ["pr_phi_delta"], 0.0)
    # Scale delta to reasonable weight range [0.1, 2.0]
    # Positive delta = progress, negative = regression
    weight = 1.0 + delta  # delta in [-1, 1] → weight in [0, 2]
    return max(weight, 0.1)


def _process_reward_quality_weight(ep: Dict[str, Any]) -> float:
    """Combined weight using confidence AND progress.

    Quality = confidence * (1 + progress_factor)
    High confidence + good progress = high quality training data.
    """
    conf = _process_reward_metric(ep, "conf_mean", ["pr_conf_mean"], 0.5)
    delta = _process_reward_metric(ep, "phi_star_delta", ["pr_phi_delta"], 0.0)

    # Progress factor: scale delta contribution
    progress_factor = max(0.0, delta)  # Only boost for positive progress

    # Quality = confidence * (1 + progress boost)
    quality = conf * (1.0 + progress_factor)
    return max(quality, 0.1)


def _embodiment_metric(ep: Dict[str, Any], key: str, default: float) -> float:
    if key in ep:
        return float(ep.get(key, default))
    desc = ep.get("descriptor", {}) if isinstance(ep.get("descriptor"), dict) else {}
    if key in desc:
        return float(desc.get(key, default))
    emb = desc.get("embodiment_profile") or ep.get("embodiment_profile")
    if isinstance(emb, dict) and key in emb:
        return float(emb.get(key, default))
    return float(default)


def _embodiment_quality_weight(ep: Dict[str, Any]) -> float:
    w_emb = _embodiment_metric(ep, "w_embodiment", 1.0)
    return max(w_emb, 0.1)


def _embodiment_drift_penalty_weight(ep: Dict[str, Any]) -> float:
    drift = _embodiment_metric(ep, "embodiment_drift_score", 0.0)
    return max(1.0 - drift, 0.1)


def _embodiment_quality_drift_weight(ep: Dict[str, Any]) -> float:
    w_emb = _embodiment_quality_weight(ep)
    drift = _embodiment_metric(ep, "embodiment_drift_score", 0.0)
    return max(w_emb * (1.0 - drift), 0.1)


class HeuristicSamplerWeightPolicy(SamplerWeightPolicy):
    def __init__(self, trust_matrix: Dict[str, Any] = None):
        self.trust_matrix = trust_matrix or {}

    def build_features(self, descriptors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Preserve ordering and content for deterministic downstream sampling.
        return [dict(d) for d in descriptors]

    def evaluate(self, features: List[Dict[str, Any]], strategy: str) -> Dict[str, float]:
        """Evaluate sampling weights for episodes using the given strategy.

        Supported strategies:
            - "balanced": Standard balanced weighting
            - "frontier_prioritized": Weight by frontier score
            - "econ_urgency": Weight by economic urgency score
            - "process_reward_conf": Weight by process reward confidence
            - "process_reward_progress": Weight by phi_star progress (delta)
            - "process_reward_quality": Combined confidence + progress weighting
            - "embodiment_quality": Weight by embodiment confidence
            - "embodiment_drift_penalty": Penalize high embodiment drift
            - "embodiment_quality_drift": Combine embodiment and drift

        Args:
            features: List of episode descriptors/features.
            strategy: Strategy name to use.

        Returns:
            Dict mapping episode keys to sampling weights.
        """
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
            elif strategy == "process_reward_conf":
                weight = _process_reward_conf_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "process_reward_progress":
                weight = _process_reward_progress_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "process_reward_quality":
                weight = _process_reward_quality_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "embodiment_quality":
                weight = _embodiment_quality_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "embodiment_drift_penalty":
                weight = _embodiment_drift_penalty_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            elif strategy == "embodiment_quality_drift":
                weight = _embodiment_quality_drift_weight(ep) * float(ep.get("recap_weight_multiplier", 1.0))
            else:
                weight = sampler_utils._balanced_weight(ep)
            trust_scale = self._trust_scale(ep)
            weights[key] = float(max(0.0, weight * trust_scale))
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
