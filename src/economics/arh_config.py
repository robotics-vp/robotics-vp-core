from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ARHPenaltyConfig:
    """Configuration for anti-reward-hacking adjustments."""

    suspicious_penalty_factor: float = 0.5
    hard_exclusion_threshold: float | None = None


def current_arh_config() -> ARHPenaltyConfig:
    penalty = _env_float("ARH_PENALTY_FACTOR")
    if penalty is None:
        penalty = 0.5
    threshold = _env_float("ARH_EXCLUSION_THRESH")
    if threshold is None:
        threshold = _env_float("ARH_EXCLUSION_THRESHOLD")
    return ARHPenaltyConfig(
        suspicious_penalty_factor=_clamp01(penalty),
        hard_exclusion_threshold=threshold,
    )


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
