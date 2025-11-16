import numpy as np
from typing import Optional, Tuple

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


TERMINATION_REASONS = [
    "max_steps",
    "sla_violation",
    "catastrophic_error",
    "zero_throughput",
    "unknown",
]


def _termination_one_hot(reason: str) -> np.ndarray:
    vec = np.zeros(len(TERMINATION_REASONS), dtype=np.float32)
    try:
        idx = TERMINATION_REASONS.index(reason or "unknown")
    except ValueError:
        idx = TERMINATION_REASONS.index("unknown")
    vec[idx] = 1.0
    return vec


def make_episode_feature_vector(
    summary: EpisodeInfoSummary,
    econ: EconParams,
    baseline: Optional[EpisodeInfoSummary] = None,
) -> np.ndarray:
    """
    Build a fixed-length feature vector for meta-learning or controllers.

    Features:
        - normalized MPL (episode MPL / baseline MPL or human proxy)
        - normalized error_rate (episode error / baseline target)
        - normalized EP (episode EP / baseline EP if available)
        - normalized energy (Wh) (episode energy / baseline energy)
        - wage parity (or 0 if unavailable)
        - termination reason one-hot
    """
    baseline_mpl = baseline.mpl_episode if baseline else None
    baseline_ep = baseline.ep_episode if baseline else None
    baseline_err = baseline.error_rate_episode if baseline else econ.max_error_rate_sla
    baseline_energy = baseline.energy_Wh if baseline else max(1e-6, econ.energy_Wh_per_attempt * econ.max_steps)

    mpl_norm = summary.mpl_episode / max(1e-6, baseline_mpl if baseline_mpl else summary.mpl_episode or 1e-6)
    ep_norm = summary.ep_episode / max(1e-6, baseline_ep if baseline_ep else summary.ep_episode or 1e-6)
    err_norm = summary.error_rate_episode / max(1e-6, baseline_err)
    energy_norm = summary.energy_Wh / max(1e-6, baseline_energy)
    wage_parity = summary.wage_parity if summary.wage_parity is not None else 0.0

    term_one_hot = _termination_one_hot(summary.termination_reason)

    features = np.array([
        mpl_norm,
        ep_norm,
        err_norm,
        energy_norm,
        wage_parity,
    ], dtype=np.float32)

    return np.concatenate([features, term_one_hot])
