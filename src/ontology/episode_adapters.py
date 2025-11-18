"""
Adapters from Stage 3 episode descriptors into ontology Episodes.
"""
from datetime import datetime
from typing import Any, Dict

from src.ontology.models import Episode


def episode_from_descriptor(descriptor: Dict[str, Any], task_id: str, robot_id: str) -> Episode:
    """
    Convert a Stage 3 episode descriptor into an Episode model.
    """
    episode_id = (
        str(descriptor.get("episode_id"))
        or str(descriptor.get("pack_id"))
        or f"episode_{task_id}_{descriptor.get('backend', 'unknown')}_{descriptor.get('env_name', 'env')}"
    )
    started_at = _safe_datetime(descriptor.get("started_at", descriptor.get("timestamp", 0)))
    ended_at = None
    status = descriptor.get("status", "running")

    metadata = {
        "objective_preset": descriptor.get("objective_preset"),
        "objective_vector": descriptor.get("objective_vector"),
        "tier": descriptor.get("tier"),
        "trust_score": descriptor.get("trust_score"),
        "sampling_metadata": descriptor.get("sampling_metadata", {}),
        "semantic_tags": descriptor.get("semantic_tags"),
        "backend": descriptor.get("backend"),
        "engine_type": descriptor.get("engine_type"),
    }

    return Episode(
        episode_id=episode_id,
        task_id=task_id,
        robot_id=robot_id,
        datapack_id=descriptor.get("pack_id"),
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        metadata=metadata,
    )


def _safe_datetime(value: Any) -> datetime:
    try:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)
    except Exception:
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return datetime.fromtimestamp(0)
