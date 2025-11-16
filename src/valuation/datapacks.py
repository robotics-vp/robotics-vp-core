from typing import Any, Dict, Optional
from dataclasses import asdict

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


def build_datapack_from_episode(
    episode_info: EpisodeInfoSummary,
    econ_params: EconParams,
    condition_profile: Dict[str, Any],
    agent_profile: Dict[str, Any],
    brick_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a standardized datapack record for valuation across Phase B/C tasks.

    Includes:
    - episode metrics (EpisodeInfoSummary)
    - econ params (for normalization/reference)
    - semantic/condition and agent profiles
    - brick/datapack identifiers
    """
    metrics = asdict(episode_info)
    datapack = {
        "brick_id": brick_id,
        "episode_metrics": metrics,
        "econ_params": {
            "price_per_unit": econ_params.price_per_unit,
            "damage_cost": econ_params.damage_cost,
            "energy_Wh_per_attempt": econ_params.energy_Wh_per_attempt,
            "time_step_s": econ_params.time_step_s,
            "max_steps": econ_params.max_steps,
            "preset": econ_params.preset,
        },
        "condition_profile": condition_profile,
        "agent_profile": agent_profile,
        "tags": condition_profile.get("tags", []),
        "attribution": {
            "delta_mpl": metrics.get("mpl_episode"),
            "delta_error": metrics.get("error_rate_episode"),
            "delta_ep": metrics.get("ep_episode"),
            "novelty": condition_profile.get("novelty", None),
            "trust": metrics.get("wage_parity", None),  # placeholder if trust not logged
            "econ_weight": condition_profile.get("econ_weight", None),
        },
    }
    return datapack
