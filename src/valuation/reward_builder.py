from dataclasses import dataclass
from typing import Dict, Any, List

from src.config.econ_params import EconParams
from src.envs.dishwashing_env import EpisodeInfoSummary


@dataclass
class ObjectiveRewardWeights:
    w_mpl: float
    w_error: float
    w_energy: float
    w_safety: float
    w_novelty: float


def default_objective_vector() -> List[float]:
    """
    Approximate current scalar reward behavior (throughput-focused with mild penalties).
    Tuned to be backwards-compatible when/if plugged in later.
    """
    return [1.0, 0.2, 0.1, 0.05, 0.0]


def build_reward_terms(summary: EpisodeInfoSummary, econ_params: EconParams) -> Dict[str, float]:
    """
    Build basic reward terms from an episode summary (or step-level info if compatible).

    Returns:
        r_mpl: throughput/MPL term
        r_error: negative error rate / collisions
        r_energy: negative energy cost (Wh scaled by price)
        r_safety: penalty from catastrophic/near-miss markers if available
        r_novelty: placeholder (0.0)
    """
    mpl = getattr(summary, "mpl_episode", 0.0)
    err = getattr(summary, "error_rate_episode", 0.0)
    energy_wh = getattr(summary, "energy_Wh", 0.0)
    # Energy penalty scaled by cost if provided
    energy_cost = getattr(econ_params, "electricity_price_kWh", 0.0) * (energy_wh / 1000.0)
    r_mpl = mpl
    r_error = -err
    r_energy = -energy_cost if energy_cost != 0 else -energy_wh
    r_safety = -1.0 if getattr(summary, "termination_reason", "") in ("catastrophic_error", "vase_broken") else 0.0
    r_novelty = 0.0
    return {
        "r_mpl": r_mpl,
        "r_error": r_error,
        "r_energy": r_energy,
        "r_safety": r_safety,
        "r_novelty": r_novelty,
    }


def combine_reward(objective_vector: List[float], reward_terms: Dict[str, float]) -> float:
    """
    Combine reward terms with objective weights.
    """
    w = objective_vector
    return (
        w[0] * reward_terms.get("r_mpl", 0.0)
        + w[1] * reward_terms.get("r_error", 0.0)
        + w[2] * reward_terms.get("r_energy", 0.0)
        + w[3] * reward_terms.get("r_safety", 0.0)
        + w[4] * reward_terms.get("r_novelty", 0.0)
    )

