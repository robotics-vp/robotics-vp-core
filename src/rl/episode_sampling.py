"""
Episode sampling scaffolding (advisory-only).

Provides a minimal descriptor conversion from datapacks to an RL episode
template. Does not alter any training loops.
"""
from typing import Dict, Any, List, Optional

from src.valuation.datapack_schema import DataPackMeta
from src.rl.episode_descriptor_validator import (
    normalize_episode_descriptor,
    validate_episode_descriptor,
)


def datapack_to_rl_episode_descriptor(datapack: DataPackMeta) -> Dict[str, Any]:
    """
    Convert a datapack into a lightweight descriptor suitable for an RL sampler.

    Reads:
    - objective vector (what to optimize for)
    - env_name, task_type (which environment to use)
    - engine/backend (PyBullet, Isaac, etc.)
    - guidance tags (semantic focus areas)
    - tier, trust_score (for prioritization)

    Returns:
    - Minimal episode descriptor that can be passed to RL training loop
    """
    # Extract objective vector
    objective_vector = []
    if datapack.objective_profile:
        objective_vector = datapack.objective_profile.objective_vector
    elif datapack.condition:
        # Fallback to condition profile if objective_profile is missing
        objective_vector = datapack.condition.objective_vector + [0.0, 0.0]  # Pad to 5 dimensions

    # Extract environment info
    env_name = datapack.task_name
    engine_type = datapack.env_type
    backend = "pybullet"  # Default backend

    if datapack.condition:
        env_name = datapack.condition.task_name or env_name
        backend = datapack.condition.engine_type or backend

    # Extract guidance tags
    semantic_tags = []
    focus_areas = []
    priority = "medium"

    if datapack.guidance_profile:
        semantic_tags = datapack.guidance_profile.semantic_tags or []
        focus_areas = [datapack.guidance_profile.main_driver]
        priority = "high" if datapack.guidance_profile.is_good else "medium"

    # Extract tier and trust for sampling weight
    tier = 1
    trust_score = 0.5
    delta_J = 0.0

    if datapack.attribution:
        tier = datapack.attribution.tier
        trust_score = datapack.attribution.trust_score
        delta_J = datapack.attribution.delta_J

    # Compute sampling weight (higher for higher-tier, higher-trust datapacks)
    sampling_weight = trust_score * (1.0 + 0.5 * tier)  # Tier 2 gets 1.5x boost

    # Episode length heuristic (can be overridden by env defaults)
    episode_length = 1000  # Default

    descriptor = {
        # Identification
        "pack_id": datapack.pack_id,
        "datapack_type": "stage1" if "stage1" in datapack.pack_id else "runtime",

        # Environment configuration
        "env_name": env_name,
        "task_type": env_name,
        "backend": backend,
        "engine_type": engine_type,

        # Objective and reward
        "objective_vector": objective_vector,
        "objective_preset": _infer_objective_preset(objective_vector),

        # Guidance
        "semantic_tags": semantic_tags,
        "focus_areas": focus_areas,
        "priority": priority,

        # Quality/sampling signals
        "tier": tier,
        "trust_score": trust_score,
        "delta_J": delta_J,
        "sampling_weight": sampling_weight,

        # Episode parameters
        "episode_length": episode_length,

        # Logging/tracking
        "tags": {
            "is_good": datapack.guidance_profile.is_good if datapack.guidance_profile else False,
            "main_driver": focus_areas[0] if focus_areas else "unknown",
            "source": "stage1_diffusion_vla" if "stage1" in datapack.pack_id else "runtime",
        }
    }
    descriptor = normalize_episode_descriptor(descriptor)
    errors = validate_episode_descriptor(descriptor)
    if errors:
        raise ValueError(f"Episode descriptor validation failed: {errors}")
    return descriptor


def _infer_objective_preset(objective_vector: List[float]) -> str:
    """
    Infer objective preset from objective vector.

    Standard presets:
    - throughput: [2.0, 1.0, 0.5, 1.0, 0.0]
    - safety: [1.0, 1.0, 0.5, 3.0, 0.0]
    - energy_saver: [1.0, 1.0, 2.0, 1.0, 0.0]
    - balanced: [1.0, 1.0, 1.0, 1.0, 0.0]
    """
    if len(objective_vector) < 4:
        return "balanced"

    # Check for throughput (high MPL weight)
    if objective_vector[0] > 1.5:
        return "throughput"

    # Check for safety (high safety weight)
    if len(objective_vector) >= 4 and objective_vector[3] > 2.0:
        return "safety"

    # Check for energy_saver (high energy weight)
    if len(objective_vector) >= 3 and objective_vector[2] > 1.5:
        return "energy_saver"

    return "balanced"


def sampler_stub(datapacks: List[DataPackMeta]) -> Dict[str, Any]:
    """
    Placeholder sampler that returns descriptors only.
    Real sampling logic is intentionally omitted.
    """
    return {dp.pack_id: datapack_to_rl_episode_descriptor(dp) for dp in datapacks}
