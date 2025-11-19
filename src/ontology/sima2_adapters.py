"""
Adapters to import SIMA-2 rollouts/semantic tags into ontology datapacks.
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List

from src.sima2.config import extract_provenance
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


def _provenance_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    return extract_provenance(record or {})


def datapack_from_sima2_rollout(rollout: Dict[str, Any], task_id: str) -> Datapack:
    dp_id = rollout.get("datapack_id") or rollout.get("episode_id") or _deterministic_id("dp_sima2", {"task": task_id})
    objects = rollout.get("metadata", {}).get("objects_present", [])
    prov = _provenance_fields(rollout)
    metadata = dict(rollout.get("metadata", {}) or {})
    metadata.update(prov)
    tags = {"objects_present": objects, "primitives": rollout.get("primitives", []), "provenance": prov}
    return Datapack(
        datapack_id=str(dp_id),
        source_type="sima2_rollout",
        task_id=task_id,
        modality="state",
        storage_uri=f"/sima2/rollouts/{dp_id}",
        novelty_score=0.0,
        quality_score=1.0,
        tags=tags,
        metadata=metadata,
        sima2_backend_id=prov.get("sima2_backend_id"),
        sima2_model_version=prov.get("sima2_model_version"),
        sima2_task_spec=prov.get("sima2_task_spec", {}),
        created_at=_safe_datetime(rollout.get("timestamp", 0)),
    )


def datapack_from_sima2_tags(tag_record: Dict[str, Any], task_id: str) -> Datapack:
    enrichment = tag_record.get("enrichment", tag_record)
    episode_id = tag_record.get("episode_id", "")
    dp_id = tag_record.get("datapack_id") or _deterministic_id("dp_sima2_tag", {"episode_id": episode_id, "task_id": task_id})
    prov = _provenance_fields(tag_record)
    return Datapack(
        datapack_id=str(dp_id),
        source_type="sima2_tags",
        task_id=task_id,
        modality="state",
        storage_uri=f"/sima2/tags/{dp_id}",
        novelty_score=0.0,
        quality_score=float(enrichment.get("coherence_score", 0.0)),
        tags=enrichment,
        metadata={"provenance": prov},
        sima2_backend_id=prov.get("sima2_backend_id"),
        sima2_model_version=prov.get("sima2_model_version"),
        sima2_task_spec=prov.get("sima2_task_spec", {}),
        created_at=_safe_datetime(tag_record.get("timestamp", 0)),
    )
