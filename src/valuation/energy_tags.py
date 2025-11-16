from typing import List

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams


def infer_energy_driver_tags(summary: EpisodeInfoSummary, econ_params: EconParams) -> List[str]:
    """
    Heuristic semantic energy driver tags for attribution/search.
    """
    tags = []
    # Use limb distribution
    limbs = summary.energy_per_limb if hasattr(summary, "energy_per_limb") else {}
    total = sum(v.get("Wh", 0.0) for v in limbs.values()) if limbs else 0.0
    shoulder_frac = limbs.get("shoulder", {}).get("Wh", 0.0) / max(total, 1e-6) if total > 0 else 0.0
    wrist_frac = limbs.get("wrist", {}).get("Wh", 0.0) / max(total, 1e-6) if total > 0 else 0.0

    if shoulder_frac > 0.5:
        tags.append("energy_driver:long_reach")
    if wrist_frac > 0.3:
        tags.append("energy_driver:high_friction")

    # Joint dominance
    joints = summary.energy_per_joint if hasattr(summary, "energy_per_joint") else {}
    for jn, data in joints.items():
        if data.get("Wh", 0.0) > 0 and total > 0:
            tags.append(f"energy_driver:joint_dominated:{jn}")
            break

    # Fragility / cautiousness (low MPL with moderate energy)
    if summary.mpl_episode < 10 and summary.energy_Wh > 0.2:
        tags.append("energy_driver:fragility_cautious")

    # Throughput push (high MPL, high energy)
    if summary.mpl_episode > 80 and summary.energy_Wh_per_unit > econ_params.energy_Wh_per_attempt:
        tags.append("energy_driver:throughput_push")

    # Effectors
    eff = summary.energy_per_effector if hasattr(summary, "energy_per_effector") else {}
    for eff_name, data in eff.items():
        if data.get("Wh", 0.0) > 0:
            tags.append(f"energy_driver:effector:{eff_name}")

    # Coordination metrics
    coord = summary.coordination_metrics if hasattr(summary, "coordination_metrics") else {}
    if coord.get("mean_active_joints", 0.0) > 2:
        tags.append("energy_driver:coordination:high_multi_joint_activity")

    return tags
