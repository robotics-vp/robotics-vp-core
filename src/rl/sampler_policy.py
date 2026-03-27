from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


SAMPLER_POLICY_STRATEGIES: tuple[str, ...] = (
    "balanced",
    "frontier_prioritized",
    "econ_urgency",
    "process_reward_conf",
    "process_reward_progress",
    "process_reward_quality",
    "embodiment_quality",
    "embodiment_drift_penalty",
    "embodiment_quality_drift",
    "epiplexity_roi",
    "inferential_yield",
)

SAMPLER_PLAN_PARAMETER_NAMES: tuple[str, ...] = (
    "frontier_threshold_quantile",
    "frontier_focus_ratio",
    "econ_threshold_quantile",
    "econ_focus_ratio",
)

SAMPLER_POOL_FEATURE_NAMES: tuple[str, ...] = (
    "num_episodes_norm",
    "eligible_fraction",
    "mean_trust_score",
    "high_trust_fraction",
    "mean_tier_norm",
    "tier2_fraction",
    "mean_frontier_score_norm",
    "max_frontier_score_norm",
    "mean_econ_urgency_score_norm",
    "max_econ_urgency_score_norm",
    "mean_novelty_score",
    "max_novelty_score",
    "mean_expected_mpl_gain_norm",
    "mean_unified_quality_weight",
    "mean_recap_weight_multiplier_norm",
    "mean_inferential_replay_weight_norm",
    "mean_epiplexity_roi_norm",
    "mean_embodiment_weight_norm",
    "mean_embodiment_drift_penalty",
    "prior_balanced",
    "prior_frontier_prioritized",
    "prior_econ_urgency",
    "prior_process_reward_conf",
    "prior_process_reward_progress",
    "prior_process_reward_quality",
    "prior_embodiment_quality",
    "prior_embodiment_drift_penalty",
    "prior_embodiment_quality_drift",
    "prior_epiplexity_roi",
    "prior_inferential_yield",
)

SAMPLER_EPISODE_FEATURE_NAMES: tuple[str, ...] = (
    "trust_score",
    "tier_norm",
    "sampling_weight_norm",
    "frontier_score_norm",
    "econ_urgency_score_norm",
    "novelty_score",
    "expected_mpl_gain_norm",
    "recap_weight_multiplier_norm",
    "unified_quality_weight",
    "inferential_replay_weight_norm",
    "epiplexity_roi_norm",
    "embodiment_weight_norm",
    "embodiment_drift_penalty",
    "delta_j_norm",
    "episode_length_norm",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _norm(value: Any, *, scale: float) -> float:
    return _clamp01(_safe_float(value, 0.0) / max(float(scale), 1e-6))


def normalize_strategy_distribution(
    weights: Mapping[str, Any] | None,
) -> Dict[str, float]:
    distribution: Dict[str, float] = {}
    total = 0.0
    for strategy in SAMPLER_POLICY_STRATEGIES:
        weight = max(0.0, _safe_float((weights or {}).get(strategy, 0.0), 0.0))
        distribution[strategy] = weight
        total += weight
    if total <= 0.0:
        return {strategy: (1.0 if strategy == "balanced" else 0.0) for strategy in SAMPLER_POLICY_STRATEGIES}
    return {strategy: float(weight / total) for strategy, weight in distribution.items()}


def build_default_sampling_plan() -> Dict[str, float]:
    return {
        "frontier_threshold_quantile": 0.65,
        "frontier_focus_ratio": 0.70,
        "econ_threshold_quantile": 0.60,
        "econ_focus_ratio": 0.65,
    }


def _descriptor(episode: Mapping[str, Any]) -> Mapping[str, Any]:
    descriptor = episode.get("descriptor")
    if isinstance(descriptor, Mapping):
        return descriptor
    return episode


def _episode_metric(episode: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if key in episode:
        return _safe_float(episode.get(key), default)
    desc = _descriptor(episode)
    return _safe_float(desc.get(key), default)


def build_sampler_pool_feature_map(
    episodes: Sequence[Mapping[str, Any]],
    *,
    heuristic_strategy_distribution: Mapping[str, Any] | None = None,
) -> Dict[str, float]:
    episode_list = list(episodes or [])
    num_episodes = len(episode_list)
    eligible_fraction = (
        sum(1 for episode in episode_list if bool(episode.get("unified_quality_eligible", True))) / float(max(num_episodes, 1))
    )
    trust_scores = [_episode_metric(episode, "trust_score", 0.5) for episode in episode_list]
    tiers = [_episode_metric(episode, "tier", 1.0) for episode in episode_list]
    frontier_scores = [_episode_metric(episode, "frontier_score", 0.0) for episode in episode_list]
    econ_scores = [_episode_metric(episode, "econ_urgency_score", 0.0) for episode in episode_list]
    novelty_scores = [_episode_metric(episode, "novelty_score", 0.0) for episode in episode_list]
    expected_gains = [_episode_metric(episode, "expected_mpl_gain", 0.0) for episode in episode_list]
    unified_quality = [_episode_metric(episode, "unified_quality_weight", 1.0) for episode in episode_list]
    recap_multipliers = [_episode_metric(episode, "recap_weight_multiplier", 1.0) for episode in episode_list]
    inferential_weights = [_episode_metric(episode, "inferential_replay_weight", 0.0) for episode in episode_list]
    epiplexity_scores = [_episode_metric(episode, "w_epi", 0.0) for episode in episode_list]
    embodiment_weights = [_episode_metric(episode, "w_embodiment", 1.0) for episode in episode_list]
    embodiment_drift_penalties = [
        max(0.0, 1.0 - _episode_metric(episode, "embodiment_drift_score", 0.0))
        for episode in episode_list
    ]

    def _mean(values: Sequence[float], default: float = 0.0) -> float:
        return float(sum(values) / max(len(values), 1)) if values else float(default)

    distribution = normalize_strategy_distribution(heuristic_strategy_distribution)
    feature_map = {
        "num_episodes_norm": _norm(num_episodes, scale=64.0),
        "eligible_fraction": _clamp01(eligible_fraction),
        "mean_trust_score": _clamp01(_mean(trust_scores, 0.5)),
        "high_trust_fraction": _clamp01(
            sum(1 for value in trust_scores if value >= 0.8) / float(max(len(trust_scores), 1))
        ),
        "mean_tier_norm": _clamp01(_mean(tiers, 1.0) / 2.0),
        "tier2_fraction": _clamp01(sum(1 for value in tiers if value >= 2.0) / float(max(len(tiers), 1))),
        "mean_frontier_score_norm": _norm(_mean(frontier_scores), scale=5.0),
        "max_frontier_score_norm": _norm(max(frontier_scores) if frontier_scores else 0.0, scale=5.0),
        "mean_econ_urgency_score_norm": _norm(_mean(econ_scores), scale=4.0),
        "max_econ_urgency_score_norm": _norm(max(econ_scores) if econ_scores else 0.0, scale=4.0),
        "mean_novelty_score": _clamp01(_mean(novelty_scores)),
        "max_novelty_score": _clamp01(max(novelty_scores) if novelty_scores else 0.0),
        "mean_expected_mpl_gain_norm": _norm(_mean(expected_gains), scale=10.0),
        "mean_unified_quality_weight": _clamp01(_mean(unified_quality, 1.0)),
        "mean_recap_weight_multiplier_norm": _norm(_mean(recap_multipliers, 1.0), scale=1.5),
        "mean_inferential_replay_weight_norm": _norm(_mean(inferential_weights), scale=2.0),
        "mean_epiplexity_roi_norm": _norm(_mean(epiplexity_scores), scale=2.0),
        "mean_embodiment_weight_norm": _norm(_mean(embodiment_weights, 1.0), scale=1.5),
        "mean_embodiment_drift_penalty": _clamp01(_mean(embodiment_drift_penalties, 1.0)),
    }
    for strategy in SAMPLER_POLICY_STRATEGIES:
        feature_map[f"prior_{strategy}"] = _clamp01(distribution.get(strategy, 0.0))
    return feature_map


def build_sampler_episode_feature_map(episode: Mapping[str, Any]) -> Dict[str, float]:
    descriptor = _descriptor(episode)
    feature_map = {
        "trust_score": _clamp01(_episode_metric(episode, "trust_score", 0.5)),
        "tier_norm": _clamp01(_episode_metric(episode, "tier", 1.0) / 2.0),
        "sampling_weight_norm": _norm(_episode_metric(episode, "sampling_weight", 1.0), scale=3.0),
        "frontier_score_norm": _norm(_episode_metric(episode, "frontier_score", 0.0), scale=5.0),
        "econ_urgency_score_norm": _norm(_episode_metric(episode, "econ_urgency_score", 0.0), scale=4.0),
        "novelty_score": _clamp01(_episode_metric(episode, "novelty_score", 0.0)),
        "expected_mpl_gain_norm": _norm(_episode_metric(episode, "expected_mpl_gain", 0.0), scale=10.0),
        "recap_weight_multiplier_norm": _norm(_episode_metric(episode, "recap_weight_multiplier", 1.0), scale=1.5),
        "unified_quality_weight": _clamp01(_episode_metric(episode, "unified_quality_weight", 1.0)),
        "inferential_replay_weight_norm": _norm(
            _episode_metric(episode, "inferential_replay_weight", descriptor.get("inferential_replay_weight", 0.0)),
            scale=2.0,
        ),
        "epiplexity_roi_norm": _norm(_episode_metric(episode, "w_epi", descriptor.get("w_epi", 0.0)), scale=2.0),
        "embodiment_weight_norm": _norm(_episode_metric(episode, "w_embodiment", 1.0), scale=1.5),
        "embodiment_drift_penalty": _clamp01(
            1.0 - _episode_metric(episode, "embodiment_drift_score", descriptor.get("embodiment_drift_score", 0.0))
        ),
        "delta_j_norm": _norm(_episode_metric(episode, "delta_J", descriptor.get("delta_J", 0.0)), scale=5.0),
        "episode_length_norm": _norm(_episode_metric(episode, "episode_length", descriptor.get("episode_length", 0.0)), scale=2000.0),
    }
    return feature_map


__all__ = [
    "SAMPLER_EPISODE_FEATURE_NAMES",
    "SAMPLER_PLAN_PARAMETER_NAMES",
    "SAMPLER_POLICY_STRATEGIES",
    "SAMPLER_POOL_FEATURE_NAMES",
    "build_default_sampling_plan",
    "build_sampler_episode_feature_map",
    "build_sampler_pool_feature_map",
    "normalize_strategy_distribution",
]
