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


def make_datapack_feature_vector(
    datapack,
    baseline_delta_j: float = 0.0,
) -> np.ndarray:
    """
    Build a fixed-length feature vector from a DataPackMeta object.

    Features:
        - delta_mpl (normalized)
        - delta_error (normalized)
        - delta_ep (normalized)
        - delta_J (normalized)
        - trust_score
        - w_econ
        - lambda_budget (normalized)
        - bucket_is_positive (binary)
        - has_counterfactual (binary)
        - has_sima (binary)
        - n_skills
        - total_duration (normalized)
        - source_type one-hot (real, synthetic, hybrid)

    Args:
        datapack: DataPackMeta instance
        baseline_delta_j: Baseline ΔJ for normalization

    Returns:
        Feature vector (np.ndarray)
    """
    # Core attribution features (normalized to [-1, 1] range)
    delta_mpl_norm = np.tanh(datapack.attribution.delta_mpl)
    delta_error_norm = np.tanh(datapack.attribution.delta_error * 10)  # Scale up
    delta_ep_norm = np.tanh(datapack.attribution.delta_ep)
    delta_j_norm = np.tanh(datapack.attribution.delta_J - baseline_delta_j)

    # Gating signals (already in [0, 1])
    trust_score = datapack.attribution.trust_score
    w_econ = datapack.attribution.w_econ
    lambda_budget_norm = np.tanh(datapack.attribution.lambda_budget / 100.0)  # Normalize

    # Binary flags
    bucket_is_positive = 1.0 if datapack.bucket == "positive" else 0.0
    has_counterfactual = 1.0 if datapack.counterfactual_plan is not None else 0.0
    has_sima = 1.0 if datapack.sima_annotation is not None else 0.0

    # Skill trace features
    n_skills = len(datapack.skill_trace)
    n_skills_norm = min(n_skills / 10.0, 1.0)  # Normalize to [0, 1]

    total_duration = datapack.get_total_duration()
    total_duration_norm = min(total_duration / 200.0, 1.0)  # Normalize

    # Source type one-hot
    source_type_onehot = [0.0, 0.0, 0.0]  # [real, synthetic, hybrid]
    if datapack.attribution.source_type == "real":
        source_type_onehot[0] = 1.0
    elif datapack.attribution.source_type == "synthetic":
        source_type_onehot[1] = 1.0
    elif datapack.attribution.source_type == "hybrid":
        source_type_onehot[2] = 1.0

    features = np.array([
        delta_mpl_norm,
        delta_error_norm,
        delta_ep_norm,
        delta_j_norm,
        trust_score,
        w_econ,
        lambda_budget_norm,
        bucket_is_positive,
        has_counterfactual,
        has_sima,
        n_skills_norm,
        total_duration_norm,
        *source_type_onehot,
    ], dtype=np.float32)

    return features


def make_condition_feature_vector(condition) -> np.ndarray:
    """
    Build feature vector from ConditionProfile.

    Features:
        - vase_offset (x, y, z)
        - drawer_friction
        - occlusion_level
        - lighting one-hot (normal, low_light, high_contrast)
        - objective_vector (first 4 elements)
        - engine_type one-hot (pybullet, isaac, ue5)

    Args:
        condition: ConditionProfile instance

    Returns:
        Feature vector (np.ndarray)
    """
    # Vase offset (normalized)
    vase_offset = np.array(condition.vase_offset[:3], dtype=np.float32)
    vase_offset_norm = vase_offset / 0.5  # Normalize by max expected offset

    # Friction and occlusion
    friction = condition.drawer_friction
    occlusion = condition.occlusion_level

    # Lighting one-hot
    lighting_onehot = [0.0, 0.0, 0.0]  # [normal, low_light, high_contrast]
    if condition.lighting_profile == "normal":
        lighting_onehot[0] = 1.0
    elif condition.lighting_profile == "low_light":
        lighting_onehot[1] = 1.0
    elif condition.lighting_profile == "high_contrast":
        lighting_onehot[2] = 1.0

    # Objective vector (first 4 weights)
    obj_vec = np.array(condition.objective_vector[:4], dtype=np.float32)
    if len(obj_vec) < 4:
        obj_vec = np.pad(obj_vec, (0, 4 - len(obj_vec)))

    # Engine type one-hot
    engine_onehot = [0.0, 0.0, 0.0]  # [pybullet, isaac, ue5]
    if condition.engine_type == "pybullet":
        engine_onehot[0] = 1.0
    elif condition.engine_type == "isaac":
        engine_onehot[1] = 1.0
    elif condition.engine_type == "ue5":
        engine_onehot[2] = 1.0

    features = np.concatenate([
        vase_offset_norm,
        [friction, occlusion],
        lighting_onehot,
        obj_vec,
        engine_onehot,
    ]).astype(np.float32)

    return features


def make_full_datapack_features(
    datapack,
    baseline_delta_j: float = 0.0,
) -> np.ndarray:
    """
    Build comprehensive feature vector combining datapack and condition features.

    Args:
        datapack: DataPackMeta instance
        baseline_delta_j: Baseline ΔJ for normalization

    Returns:
        Combined feature vector (np.ndarray)
    """
    dp_features = make_datapack_feature_vector(datapack, baseline_delta_j)
    cond_features = make_condition_feature_vector(datapack.condition)

    return np.concatenate([dp_features, cond_features])


def extract_skill_sequence_features(datapack, num_skills: int = 6) -> np.ndarray:
    """
    Extract skill sequence features from datapack.

    Features:
        - Skill usage one-hot (per skill)
        - Skill order encoding
        - Average duration per skill

    Args:
        datapack: DataPackMeta instance
        num_skills: Total number of skills

    Returns:
        Skill sequence feature vector
    """
    # Skill usage counts
    skill_usage = np.zeros(num_skills, dtype=np.float32)
    skill_order = np.zeros(num_skills, dtype=np.float32)
    skill_durations = np.zeros(num_skills, dtype=np.float32)

    for i, entry in enumerate(datapack.skill_trace):
        skill_id = entry.get('skill_id', 0)
        if 0 <= skill_id < num_skills:
            skill_usage[skill_id] += 1
            if skill_order[skill_id] == 0:
                skill_order[skill_id] = i + 1  # First occurrence position
            skill_durations[skill_id] += entry.get('duration', 0)

    # Normalize
    skill_usage = skill_usage / max(len(datapack.skill_trace), 1)
    skill_order = skill_order / max(len(datapack.skill_trace), 1)
    skill_durations = skill_durations / max(datapack.get_total_duration(), 1)

    return np.concatenate([skill_usage, skill_order, skill_durations])
