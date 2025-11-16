from typing import Any, Dict, Optional
from dataclasses import asdict

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.valuation.energy_tags import infer_energy_driver_tags

DATAPACK_SCHEMA_VERSION = "2.0-energy"


def build_datapack_from_episode(
    episode_info: EpisodeInfoSummary,
    econ_params: EconParams,
    condition_profile: Dict[str, Any],
    agent_profile: Dict[str, Any],
    brick_id: Optional[str] = None,
    env_type: str = "dishwashing",
    extra_tags: Optional[list] = None,
    semantic_energy_drivers: Optional[list] = None,
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
    tags = (condition_profile.get("tags", []) if condition_profile else []) + (extra_tags or [])
    datapack = {
        "schema_version": DATAPACK_SCHEMA_VERSION,
        "env_type": env_type,
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
        "tags": tags,
        "attribution": {
            "delta_mpl": metrics.get("mpl_episode"),
            "delta_error": metrics.get("error_rate_episode"),
            "delta_ep": metrics.get("ep_episode"),
            "novelty": condition_profile.get("novelty", None) if condition_profile else None,
            "trust": metrics.get("wage_parity", None),  # placeholder if trust not logged
            "econ_weight": condition_profile.get("econ_weight", None) if condition_profile else None,
        },
        "energy": {
            "total_Wh": metrics.get("energy_Wh", 0.0),
            "Wh_per_unit": metrics.get("energy_Wh_per_unit", 0.0),
            "Wh_per_hour": metrics.get("energy_Wh_per_hour", 0.0),
            "limb_energy_Wh": metrics.get("limb_energy_Wh", {}),
            "skill_energy_Wh": metrics.get("skill_energy_Wh", {}),
            "energy_per_limb": metrics.get("energy_per_limb", {}),
            "energy_per_skill": metrics.get("energy_per_skill", {}),
            "energy_per_joint": metrics.get("energy_per_joint", {}),
            "energy_per_effector": metrics.get("energy_per_effector", {}),
            "coordination_metrics": metrics.get("coordination_metrics", {}),
        },
        "semantic_energy_drivers": semantic_energy_drivers or infer_energy_driver_tags(episode_info, econ_params),
    }
    return datapack
