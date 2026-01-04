from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.motor_backend.rollout_capture import RolloutBundle
from src.orchestrator.semantic_fusion import SEMANTIC_FUSION_PREFIX, fuse_semantic_evidence_mvp
from src.vision.map_first_supervision.artifacts import MAP_FIRST_PREFIX
from src.vision.map_first_supervision.semantics import parse_vla_semantic_evidence

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _load_trajectory_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "trajectory" not in data:
        return None
    payload = data["trajectory"]
    if hasattr(payload, "item") and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, dict):
        return payload
    return None


def _get_scene_tracks_array(scene_tracks: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if key in scene_tracks:
        return np.asarray(scene_tracks[key])
    prefixed = f"scene_tracks_v1/{key}"
    if prefixed in scene_tracks:
        return np.asarray(scene_tracks[prefixed])
    return None


def _extract_scene_track_ids(payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not payload:
        return None
    scene_tracks = payload.get("scene_tracks_v1") or payload.get("scene_tracks")
    if isinstance(scene_tracks, dict):
        return _get_scene_tracks_array(scene_tracks, "track_ids")
    scene_tracks_path = payload.get("scene_tracks_path") or payload.get("scene_tracks_npz")
    if scene_tracks_path:
        try:
            data = _load_npz(Path(scene_tracks_path))
        except Exception:
            return None
        return _get_scene_tracks_array(data, "track_ids")
    return None


def _find_map_first_path(trajectory_path: Path) -> Optional[Path]:
    candidates = [
        trajectory_path.with_name("map_first_supervision_v1.npz"),
        trajectory_path.with_name(f"{trajectory_path.stem}_map_first_v1.npz"),
        trajectory_path.with_name(f"{trajectory_path.stem}_map_first_supervision_v1.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_vla_path(trajectory_path: Path) -> Optional[Path]:
    candidate = trajectory_path.with_name(f"{trajectory_path.stem}_vla_semantic_evidence_v1.npz")
    if candidate.exists():
        return candidate
    return None


def _get_map_first_array(data: Optional[Dict[str, np.ndarray]], key: str) -> Optional[np.ndarray]:
    if data is None:
        return None
    return data.get(f"{MAP_FIRST_PREFIX}{key}")


def _to_track_list(track_ids: Optional[np.ndarray]) -> Optional[list[str]]:
    if track_ids is None:
        return None
    return [str(tid) for tid in list(track_ids)]


def _write_summary_jsonl(summary_path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _update_episode_metadata(episode_dir: Path, metrics: Dict[str, float], fusion_path: Path) -> None:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return
    existing_metrics = payload.get("metrics", {})
    if not isinstance(existing_metrics, dict):
        existing_metrics = {}
    existing_metrics.update(metrics)
    payload["metrics"] = existing_metrics
    payload["semantic_fusion_path"] = str(fusion_path)
    meta_path.write_text(json.dumps(payload, indent=2))


def _safe_episode_id(episode_id: str) -> str:
    safe = episode_id.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    return safe or "episode"


def run_semantic_fusion_for_rollouts(
    rollout_bundle: RolloutBundle,
    summary_path: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """Run semantic fusion for each episode in a rollout bundle."""
    summaries: list[Dict[str, Any]] = []
    for episode in rollout_bundle.episodes:
        trajectory_path = episode.trajectory_path
        map_first_path = _find_map_first_path(trajectory_path)
        vla_path = _find_vla_path(trajectory_path)

        map_first_data = _load_npz(map_first_path) if map_first_path else None
        vla_payload = _load_npz(vla_path) if vla_path else None

        scene_payload = _load_trajectory_payload(trajectory_path)
        scene_track_ids = _extract_scene_track_ids(scene_payload)

        vla_evidence = parse_vla_semantic_evidence(vla_payload)
        vla_class_probs = vla_evidence.class_probs if vla_evidence is not None else None
        vla_confidence = vla_evidence.confidence if vla_evidence is not None else None
        vla_track_ids = vla_evidence.track_ids if vla_evidence is not None else None

        if vla_class_probs is None:
            vla_class_probs = _get_map_first_array(map_first_data, "vla_class_probs")
        if vla_confidence is None:
            vla_confidence = _get_map_first_array(map_first_data, "vla_confidence")

        map_semantics = _get_map_first_array(map_first_data, "evidence_map_semantics")
        if map_semantics is None:
            map_semantics = _get_map_first_array(map_first_data, "semantics_stable")
        map_stability = _get_map_first_array(map_first_data, "evidence_map_stability")
        if map_stability is None:
            map_stability = _get_map_first_array(map_first_data, "meta_semantics_stability")

        geom_residual = _get_map_first_array(map_first_data, "evidence_geom_residual")
        occlusion = _get_map_first_array(map_first_data, "evidence_occlusion")
        dynamic_evidence = _get_map_first_array(map_first_data, "evidence_dynamics_score")

        if map_semantics is None and vla_class_probs is None:
            continue

        scene_ids = _to_track_list(scene_track_ids)
        vla_ids = _to_track_list(vla_track_ids)
        if scene_ids is not None and vla_ids is not None and scene_ids != vla_ids:
            logger.warning("Semantic fusion skipped: track_ids mismatch for episode %s", episode.metadata.episode_id)
            continue
        if scene_ids is not None and map_semantics is not None and len(scene_ids) != map_semantics.shape[1]:
            logger.warning("Semantic fusion skipped: SceneTracks size mismatch for episode %s", episode.metadata.episode_id)
            continue
        if vla_ids is not None and map_semantics is not None and len(vla_ids) != map_semantics.shape[1]:
            logger.warning("Semantic fusion skipped: VLA size mismatch for episode %s", episode.metadata.episode_id)
            continue
        if vla_class_probs is not None and map_semantics is not None:
            if vla_class_probs.shape != map_semantics.shape:
                logger.warning("Semantic fusion skipped: shape mismatch for episode %s", episode.metadata.episode_id)
                continue

        if scene_ids is None and vla_ids is None:
            logger.warning("Semantic fusion track_ids unavailable for episode %s; proceeding without alignment checks", episode.metadata.episode_id)

        num_classes = 1
        if map_semantics is not None:
            num_classes = int(map_semantics.shape[-1])
        elif vla_class_probs is not None:
            num_classes = int(vla_class_probs.shape[-1])

        fusion_result = fuse_semantic_evidence_mvp(
            vla_class_probs=vla_class_probs,
            vla_confidence=vla_confidence,
            map_semantics=map_semantics,
            map_stability=map_stability,
            geom_residual=geom_residual,
            occlusion=occlusion,
            dynamic_evidence=dynamic_evidence,
            num_classes=num_classes,
        )

        episode_id = getattr(episode.metadata, "episode_id", "") or ""
        fusion_basename = f"{_safe_episode_id(episode_id)}_semantic_fusion_v1.npz"
        fusion_path = trajectory_path.with_name(fusion_basename)
        np.savez_compressed(fusion_path, **fusion_result.to_npz())

        confidence_mean = float(np.mean(fusion_result.fused_confidence))
        disagreement_mean = float(fusion_result.diagnostics.get("disagreement_mean", 0.0)) if fusion_result.diagnostics else 0.0
        summary = {
            "episode_id": episode.metadata.episode_id,
            "semantic_fusion_confidence_mean": confidence_mean,
            "semantic_disagreement_vla_vs_map": disagreement_mean,
            "semantic_fusion_quality_score": confidence_mean,
            "semantic_fusion_path": str(fusion_path),
            "semantic_fusion_keys": list(fusion_result.to_npz().keys()),
            "semantic_fusion_prefix": SEMANTIC_FUSION_PREFIX,
        }
        summaries.append(summary)

        metrics = {
            "semantic_fusion_confidence_mean": confidence_mean,
            "semantic_disagreement_vla_vs_map": disagreement_mean,
            "semantic_fusion_quality_score": confidence_mean,
        }
        _update_episode_metadata(trajectory_path.parent, metrics, fusion_path)
        episode.metrics = dict(episode.metrics, **metrics)

    if summary_path is not None:
        _write_summary_jsonl(summary_path, summaries)

    return summaries
