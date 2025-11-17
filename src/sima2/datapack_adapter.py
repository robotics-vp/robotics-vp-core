"""
SIMA-2 → Datapack bridge stubs.

These are placeholders to establish import paths without integrating SIMA-2.
"""
from typing import Any, Dict, List

from src.valuation.datapack_schema import DataPackMeta


def sima2_episode_to_datapack_meta_stub(rollout: Dict[str, Any]) -> DataPackMeta:
    """
    Convert a SIMA-2 rollout dict into a minimal DataPackMeta placeholder.
    """
    return DataPackMeta(
        task_name=rollout.get("task_name", "sima2_stub_task"),
        env_type=rollout.get("env_type", "sima2_engine"),
        agent_profile={"source": "sima2_stub", "metadata": rollout.get("meta", {})},
        semantic_tags=rollout.get("semantic_tags", []),
    )


def sima2_semantic_tokens_from_rollout_stub(rollout: Dict[str, Any]) -> List[str]:
    """
    Extract semantic tokens from a SIMA-2 rollout (placeholder only).
    """
    tokens = rollout.get("semantic_rollout") or rollout.get("semantic_tokens") or []
    return list(tokens)
