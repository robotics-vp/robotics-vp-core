"""
Adapters from Stage 1/2 artifacts into ontology Datapack models.

Purely functional conversions; no side-effects.
"""
import hashlib
import json
from datetime import datetime
from typing import Any, Dict

from src.ontology.models import Datapack


def _deterministic_id(prefix: str, payload: Dict[str, Any]) -> str:
    """Create a deterministic ID from a prefix and JSON payload."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _safe_datetime(ts: Any) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.fromtimestamp(float(ts))
        except Exception:
            return datetime.fromtimestamp(0)


def _extract_auditor_outputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gracefully extract auditor outputs (rating/score/predicted econ) from a raw payload.

    Supports either top-level keys or nested auditor/audit structures. Returns
    None values when no auditor metadata exists.
    """
    if not isinstance(payload, dict):
        return {"auditor_rating": None, "auditor_score": None, "auditor_predicted_econ": None}

    candidates = []
    for key in ("auditor_result", "auditor", "audit", "audit_result"):
        if isinstance(payload.get(key), dict):
            candidates.append(payload.get(key))
    if isinstance(payload.get("metadata"), dict) and isinstance(payload["metadata"].get("auditor"), dict):
        candidates.append(payload["metadata"]["auditor"])
    base = candidates[0] if candidates else payload

    rating = base.get("rating") or payload.get("auditor_rating")
    score = base.get("score") or base.get("auditor_score") or payload.get("auditor_score")
    predicted = base.get("predicted_econ") or payload.get("auditor_predicted_econ")

    if predicted is not None and not isinstance(predicted, dict):
        try:
            predicted = dict(predicted)
        except Exception:
            predicted = None

    try:
        score_val = float(score) if score is not None else None
    except Exception:
        score_val = None

    return {
        "auditor_rating": rating,
        "auditor_score": score_val,
        "auditor_predicted_econ": predicted,
    }


def datapack_from_stage1(raw: Dict[str, Any], task_id: str) -> Datapack:
    """
    Convert a Stage 1 datapack JSON record into a Datapack model.
    """
    datapack_id = (
        str(raw.get("pack_id"))
        or str(raw.get("datapack_id"))
        or _deterministic_id("dp1", {"task_id": task_id, "seed": raw.get("episode_id", "")})
    )
    source_type = raw.get("source_type") or ("human_video" if "real" in str(datapack_id) else "synthetic_video")
    modality = raw.get("modality") or "video"
    storage_uri = raw.get("storage_uri") or raw.get("video_path") or f"/stage1/{datapack_id}"
    novelty_score = float(raw.get("novelty_score", 0.0))
    quality_score = float(raw.get("quality_score", raw.get("semantic_quality", 0.0)))
    created_at_raw = raw.get("created_at") or raw.get("timestamp") or raw.get("time")
    created_at = _safe_datetime(created_at_raw) if created_at_raw is not None else datetime.fromtimestamp(0)

    tags: Dict[str, Any] = {}
    for key in ("semantic_tags", "energy_driver_tags", "econ_semantic_tags"):
        if key in raw and raw[key]:
            tags[key] = raw[key]
    tags["bucket"] = raw.get("bucket", raw.get("source_type", "unknown"))
    auditor = _extract_auditor_outputs(raw)

    return Datapack(
        datapack_id=datapack_id,
        source_type=source_type,
        task_id=task_id,
        modality=modality,
        storage_uri=storage_uri,
        novelty_score=novelty_score,
        quality_score=quality_score,
        tags=tags,
        auditor_rating=auditor.get("auditor_rating"),
        auditor_score=auditor.get("auditor_score"),
        auditor_predicted_econ=auditor.get("auditor_predicted_econ"),
        created_at=created_at,
    )


def datapack_from_stage2_enrichment(enrichment: Dict[str, Any], task_id: str) -> Datapack:
    """
    Convert a Stage 2.4 enrichment JSONL line into a Datapack model.
    """
    base = enrichment.get("enrichment", enrichment)
    episode_id = enrichment.get("episode_id") or enrichment.get("datapack_id") or ""
    datapack_id = enrichment.get("datapack_id") or _deterministic_id("dp2", {"episode_id": episode_id, "task_id": task_id})
    storage_uri = f"/stage2_enrichment/{datapack_id}"
    created_at = _safe_datetime(enrichment.get("timestamp", datetime.fromtimestamp(0).isoformat()))

    tags: Dict[str, Any] = {}
    for key in (
        "fragility_tags",
        "risk_tags",
        "affordance_tags",
        "efficiency_tags",
        "novelty_tags",
        "intervention_tags",
        "semantic_conflicts",
        "supervision_hints",
    ):
        if key in base and base[key]:
            tags[key] = base[key]
    tags["validation_status"] = base.get("validation_status", "unknown")

    novelty_scores = [t.get("novelty_score", 0.0) for t in base.get("novelty_tags", [])] if base.get("novelty_tags") else []
    novelty_score = float(max(novelty_scores)) if novelty_scores else 0.0
    quality_score = float(base.get("coherence_score", 0.0))
    auditor = _extract_auditor_outputs(enrichment)
    if not auditor.get("auditor_rating"):
        auditor = _extract_auditor_outputs(base)

    return Datapack(
        datapack_id=datapack_id,
        source_type=enrichment.get("source_type", "enrichment"),
        task_id=task_id,
        modality=enrichment.get("modality", "video"),
        storage_uri=storage_uri,
        novelty_score=novelty_score,
        quality_score=quality_score,
        tags=tags,
        auditor_rating=auditor.get("auditor_rating"),
        auditor_score=auditor.get("auditor_score"),
        auditor_predicted_econ=auditor.get("auditor_predicted_econ"),
        created_at=created_at,
    )
