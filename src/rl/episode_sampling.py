"""
Episode sampling scaffolding (advisory-only).

Provides a minimal descriptor conversion from datapacks to an RL episode
template. Does not alter any training loops.
"""
from typing import Dict, Any

from src.valuation.datapack_schema import DataPackMeta


def datapack_to_rl_episode_descriptor(datapack: DataPackMeta) -> Dict[str, Any]:
    """
    Convert a datapack into a lightweight descriptor suitable for an RL sampler.
    Contains objective vector, condition profile, and guidance profile summary.
    """
    descriptor = {
        "pack_id": datapack.pack_id,
        "objective_vector": datapack.objective_profile.objective_vector if datapack.objective_profile else [],
        "condition_profile": datapack.condition,
        "guidance_profile": datapack.guidance_profile,
    }
    return descriptor


def sampler_stub(datapacks) -> Dict[str, Any]:
    """
    Placeholder sampler that returns descriptors only.
    Real sampling logic is intentionally omitted.
    """
    return {dp.pack_id: datapack_to_rl_episode_descriptor(dp) for dp in datapacks}
