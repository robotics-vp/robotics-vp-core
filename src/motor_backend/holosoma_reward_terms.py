from __future__ import annotations

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
