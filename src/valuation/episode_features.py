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
    semantic_energy_drivers: Optional[list] = None,
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
    baseline_energy_unit = (
        baseline.energy_Wh_per_unit if baseline else econ.energy_Wh_per_attempt
    )

    # Defaults to own value if no baseline provided
    mpl_norm = summary.mpl_episode / max(1e-6, baseline_mpl if baseline_mpl else summary.mpl_episode or 1e-6)

    # Energy productivity (units per Wh)
    default_ep_baseline = 1.0 / max(econ.energy_Wh_per_attempt, 1e-6)
    ep_baseline = baseline_ep if baseline_ep else default_ep_baseline
    ep_norm = summary.ep_episode / max(1e-6, ep_baseline)

    err_norm = summary.error_rate_episode / max(1e-6, baseline_err)

    # Lower is better for energy per unit
    energy_norm = summary.energy_Wh_per_unit / max(1e-6, baseline_energy_unit)

    wage_parity = summary.wage_parity if summary.wage_parity is not None else 0.0

    term_one_hot = _termination_one_hot(summary.termination_reason)

    # Limb energy fractions
    limb_energy = summary.limb_energy_Wh if hasattr(summary, "limb_energy_Wh") else {}
    total_limb = sum(limb_energy.values()) if limb_energy else 0.0
    limb_fractions = []
    for limb in ["shoulder", "elbow", "wrist", "gripper"]:
        val = limb_energy.get(limb, 0.0)
        limb_fractions.append(val / max(total_limb, 1e-6) if total_limb > 0 else 0.0)

    # Semantic energy driver multi-hot
    drivers = semantic_energy_drivers or []
    driver_order = [
        "long_reach",
        "high_friction",
        "cautious_fragility",
        "replanning_occlusion",
        "high_speed_execution",
    ]
    driver_hot = [1.0 if d in drivers else 0.0 for d in driver_order]

    features = np.array([
        mpl_norm,
        ep_norm,
        err_norm,
        energy_norm,
        wage_parity,
        *limb_fractions,
        *driver_hot,
    ], dtype=np.float32)

    return np.concatenate([features, term_one_hot])
