"""Curated slice selection for epiplexity evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1, SceneTracksLite
from src.process_reward.schemas import MHNSummary


@dataclass
class SliceSelectorConfig:
    occlusion_threshold: float = 0.4
    dynamic_motion_threshold: float = 0.15
    static_motion_threshold: float = 0.05
    max_per_slice: int = 20


def compute_slice_metrics(episode: Dict[str, Any]) -> Dict[str, float]:
    scene_tracks = _extract_scene_tracks_lite(episode)
    occlusion = 0.0
    motion = 0.0
    if scene_tracks is not None:
        occlusion = float(np.mean(scene_tracks.occlusion))
        motion = _mean_speed(scene_tracks.poses_t)
    else:
        occlusion = float(episode.get("occlusion_level", 0.0))

    contact = _extract_contact_coverage(episode)
    mhn_complexity = _extract_mhn_complexity(episode)
    return {
        "occlusion": occlusion,
        "motion": motion,
        "contact_coverage": contact,
        "mhn_complexity": mhn_complexity,
    }


def select_curated_slices(
    episodes: Sequence[Dict[str, Any]],
    config: SliceSelectorConfig | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    cfg = config or SliceSelectorConfig()
    scored: List[Dict[str, Any]] = []
    for ep in episodes:
        metrics = compute_slice_metrics(ep)
        ep = dict(ep)
        ep["slice_metrics"] = metrics
        scored.append(ep)

    occluded = [e for e in scored if e["slice_metrics"]["occlusion"] >= cfg.occlusion_threshold]
    dynamic = [e for e in scored if e["slice_metrics"]["motion"] >= cfg.dynamic_motion_threshold]
    static = [e for e in scored if e["slice_metrics"]["motion"] <= cfg.static_motion_threshold]

    occluded.sort(key=lambda e: e["slice_metrics"]["occlusion"], reverse=True)
    dynamic.sort(key=lambda e: e["slice_metrics"]["motion"], reverse=True)
    static.sort(key=lambda e: e["slice_metrics"]["motion"])

    return {
        "occluded": occluded[: cfg.max_per_slice],
        "dynamic": dynamic[: cfg.max_per_slice],
        "static": static[: cfg.max_per_slice],
    }


def _extract_scene_tracks_lite(episode: Dict[str, Any]) -> SceneTracksLite | None:
    payload = episode.get("scene_tracks") or episode.get("scene_tracks_v1")
    if payload is None:
        return None
    if isinstance(payload, SceneTracksLite):
        return payload
    if isinstance(payload, dict):
        return deserialize_scene_tracks_v1(payload)
    return None


def _mean_speed(poses_t: np.ndarray) -> float:
    if poses_t.ndim != 3 or poses_t.shape[0] < 2:
        return 0.0
    deltas = poses_t[1:] - poses_t[:-1]
    speed = np.linalg.norm(deltas, axis=-1)
    return float(np.mean(speed))


def _extract_contact_coverage(episode: Dict[str, Any]) -> float:
    emb = episode.get("embodiment_profile")
    if emb is None:
        return 0.0
    if isinstance(emb, dict):
        return float(emb.get("contact_coverage_pct", 0.0))
    return float(getattr(emb, "contact_coverage_pct", 0.0) or 0.0)


def _extract_mhn_complexity(episode: Dict[str, Any]) -> float:
    summary = episode.get("mhn_summary")
    if summary is None:
        return 0.0
    if isinstance(summary, dict):
        summary = MHNSummary.from_dict(summary)
    return float(getattr(summary, "structural_difficulty", 0.0))


__all__ = ["SliceSelectorConfig", "select_curated_slices", "compute_slice_metrics"]
