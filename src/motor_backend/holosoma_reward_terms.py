from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def economic_overlay_reward(env: Any, reward_scales: Mapping[str, float]) -> "torch.Tensor":
    """Compute an additive reward overlay from economic metrics.

    Expected sources (best-effort):
    - env.econ_metrics: Mapping[str, Tensor/float]
    - env.extras["econ_metrics"]: Mapping[str, Tensor/float]
    """
    import torch

    device = getattr(env, "device", "cpu")
    num_envs = int(getattr(env, "num_envs", 1))
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)

    metrics = {}
    if hasattr(env, "econ_metrics") and isinstance(env.econ_metrics, dict):
        metrics = env.econ_metrics
    elif hasattr(env, "extras") and isinstance(env.extras, dict):
        metrics = env.extras.get("econ_metrics", {}) or {}
    if hasattr(env, "log_dict") and isinstance(env.log_dict, dict):
        metrics = {**metrics, **env.log_dict}

    for key, scale in reward_scales.items():
        value = metrics.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.as_tensor(value, device=device, dtype=reward.dtype)
            except Exception:
                continue
        if value.ndim == 0:
            value = value.repeat(num_envs)
        reward = reward + value * float(scale)

    return reward


@dataclass
class AntiRewardHackingReport:
    is_suspicious: bool
    reasons: list[str]
    summary_metrics: Mapping[str, float]


def analyze_anti_reward_hacking(log_dict: Mapping[str, float]) -> AntiRewardHackingReport:
    reasons: list[str] = []
    suspicious = False

    reward = _first_metric(log_dict, ("mean_reward", "episode_reward_mean", "reward_mean"))
    success_rate = _first_metric(log_dict, ("success_rate",))
    energy = _first_metric(log_dict, ("energy_kwh_mean", "energy_kwh"))
    duration = _first_metric(log_dict, ("mean_episode_length", "episode_length_mean", "mean_episode_length_s"))
    expected_duration = _first_metric(log_dict, ("expected_duration",))

    if reward is not None and success_rate is not None and duration is not None:
        duration_threshold = expected_duration * 0.1 if expected_duration else duration * 0.1
        if reward > 10 * max(success_rate, 1e-3) and duration <= duration_threshold:
            suspicious = True
            reasons.append("Reward disproportionately high relative to success and duration")

    if reward is not None and energy is not None and energy <= 0.0 and reward > 0.0:
        suspicious = True
        reasons.append("Positive reward with near-zero energy usage")

    summary = {
        "episode_reward_mean": reward or 0.0,
        "success_rate": success_rate or 0.0,
        "energy_kwh_mean": energy or 0.0,
        "episode_length_mean": duration or 0.0,
    }
    return AntiRewardHackingReport(is_suspicious=suspicious, reasons=reasons, summary_metrics=summary)


def _first_metric(log_dict: Mapping[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in log_dict and log_dict[key] is not None:
            try:
                return float(log_dict[key])
            except (TypeError, ValueError):
                continue
    return None
