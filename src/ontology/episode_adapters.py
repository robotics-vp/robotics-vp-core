"""
Adapters from Stage 3 episode descriptors into ontology Episodes.
"""
from datetime import datetime
from hashlib import sha256
from typing import Any, Dict

from src.ontology.models import Episode


def _derive_episode_id(descriptor: Dict[str, Any], task_id: str, robot_id: str) -> str:
    if descriptor.get("episode_id"):
        return str(descriptor["episode_id"])
    if descriptor.get("pack_id"):
        return f"ep_{descriptor['pack_id']}"
    payload = f"{task_id}|{robot_id}|{descriptor.get('env_name','env')}|{descriptor.get('backend','')}"
    digest = sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"ep_{digest}"


def episode_from_descriptor(descriptor: Dict[str, Any], task_id: str, robot_id: str) -> Episode:
    """
    Convert a Stage 3 episode descriptor into an Episode model.
    """
    episode_id = _derive_episode_id(descriptor, task_id, robot_id)
    started_at = _safe_datetime(descriptor.get("started_at", descriptor.get("timestamp", 0)))
    ended_at = None
    status = descriptor.get("status", "running")

    sampling_md = descriptor.get("sampling_metadata", {}) or {}
    metadata = {
        "objective_preset": descriptor.get("objective_preset"),
        "objective_vector": descriptor.get("objective_vector"),
        "tier": descriptor.get("tier"),
        "trust_score": descriptor.get("trust_score"),
        "sampling_metadata": sampling_md,
        "semantic_tags": descriptor.get("semantic_tags"),
        "backend": descriptor.get("backend"),
        "engine_type": descriptor.get("engine_type"),
        "curriculum_phase": sampling_md.get("phase"),
        "sampler_strategy": sampling_md.get("strategy"),
    }
    if sampling_md.get("skill_mode"):
        metadata["skill_mode"] = sampling_md.get("skill_mode")
    condition_meta = descriptor.get("condition_metadata") or sampling_md.get("condition_metadata")
    if condition_meta:
        metadata["condition_vector_summary"] = condition_meta

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
