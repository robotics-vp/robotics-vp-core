"""
Adapters to import SIMA-2 rollouts/semantic tags into ontology datapacks.
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List

from src.ontology.models import Datapack


def _deterministic_id(prefix: str, payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _safe_datetime(ts: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        try:
            return datetime.fromtimestamp(float(ts))
        except Exception:
            return datetime.fromtimestamp(0)


def datapack_from_sima2_rollout(rollout: Dict[str, Any], task_id: str) -> Datapack:
    dp_id = rollout.get("datapack_id") or rollout.get("episode_id") or _deterministic_id("dp_sima2", {"task": task_id})
    objects = rollout.get("metadata", {}).get("objects_present", [])
    return Datapack(
        datapack_id=str(dp_id),
        source_type="sima2_rollout",
        task_id=task_id,
        modality="state",
        storage_uri=f"/sima2/rollouts/{dp_id}",
        novelty_score=0.0,
        quality_score=1.0,
        tags={"objects_present": objects, "primitives": rollout.get("primitives", [])},
        created_at=_safe_datetime(rollout.get("timestamp", 0)),
    )


def datapack_from_sima2_tags(tag_record: Dict[str, Any], task_id: str) -> Datapack:
    enrichment = tag_record.get("enrichment", tag_record)
    episode_id = tag_record.get("episode_id", "")
    dp_id = tag_record.get("datapack_id") or _deterministic_id("dp_sima2_tag", {"episode_id": episode_id, "task_id": task_id})
    return Datapack(
        datapack_id=str(dp_id),
        source_type="sima2_tags",
        task_id=task_id,
        modality="state",
        storage_uri=f"/sima2/tags/{dp_id}",
        novelty_score=0.0,
        quality_score=float(enrichment.get("coherence_score", 0.0)),
        tags=enrichment,
        created_at=_safe_datetime(tag_record.get("timestamp", 0)),
    )
