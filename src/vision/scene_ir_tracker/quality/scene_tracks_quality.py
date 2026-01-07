"""
SceneTracks quality metrics and scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1


@dataclass
class SceneTracksQuality:
    """Quality summary for SceneTracks_v1 outputs."""

    quality_score: float
    track_coverage_ratio: Dict[str, float]
    median_track_length: float
    jitter_score: float
    id_switch_proxy: float
    invalid_geometry_count: int
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": float(self.quality_score),
            "track_coverage_ratio": {k: float(v) for k, v in self.track_coverage_ratio.items()},
            "median_track_length": float(self.median_track_length),
            "jitter_score": float(self.jitter_score),
            "id_switch_proxy": float(self.id_switch_proxy),
            "invalid_geometry_count": int(self.invalid_geometry_count),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class SceneTracksQualityConfig:
    """Thresholds and weights for SceneTracks quality."""

    static_displacement_m: float = 0.01
    jitter_tolerance_m: float = 0.005
    min_track_coverage: float = 0.3
    min_median_track_length_ratio: float = 0.3
    max_id_switch_proxy: float = 2.0

    weight_coverage: float = 0.3
    weight_length: float = 0.2
    weight_jitter: float = 0.2
    weight_id_switch: float = 0.2
    weight_invalid: float = 0.1


def compute_scene_tracks_quality(
    scene_tracks: Dict[str, np.ndarray] | Any,
    config: Optional[SceneTracksQualityConfig] = None,
) -> SceneTracksQuality:
    """Compute SceneTracks quality metrics from serialized data."""
    cfg = config or SceneTracksQualityConfig()
    if not hasattr(scene_tracks, "poses_t"):
        scene_tracks = deserialize_scene_tracks_v1(scene_tracks)

    poses_t = np.asarray(scene_tracks.poses_t)
    poses_R = np.asarray(scene_tracks.poses_R)
    visibility = getattr(scene_tracks, "visibility", None)
    if visibility is None:
        visibility = np.ones((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
    else:
        visibility = np.asarray(visibility)

    track_ids = [str(tid) for tid in list(getattr(scene_tracks, "track_ids", []))]
    if poses_t.size == 0 or not track_ids:
        return SceneTracksQuality(
            quality_score=0.0,
            track_coverage_ratio={},
            median_track_length=0.0,
            jitter_score=0.0,
            id_switch_proxy=0.0,
            invalid_geometry_count=int(np.isnan(poses_t).sum()),
            flags=["no_tracks"],
        )

    T, K = poses_t.shape[:2]
    valid_mask = np.isfinite(poses_t).all(axis=-1)
    visible_mask = valid_mask & (visibility > 0.0)

    coverage = {}
    track_lengths = []
    id_switch_proxy = 0.0
    jitter_scores = []
    invalid_geometry_count = int(np.isnan(poses_t).sum() + np.isnan(poses_R).sum())

    for k in range(K):
        tid = track_ids[k] if k < len(track_ids) else f"track_{k}"
        track_visible = visible_mask[:, k]
        coverage_ratio = float(np.sum(track_visible)) / max(T, 1)
        coverage[tid] = coverage_ratio
        track_length = int(np.sum(track_visible))
        track_lengths.append(track_length)

        # ID-switch proxy: count gaps in visibility
        segments = _count_segments(track_visible)
        if segments > 1:
            id_switch_proxy += float(segments - 1)

        # Jitter score for static objects
        positions = poses_t[:, k, :]
        displacement = float(np.linalg.norm(positions[-1] - positions[0])) if T > 1 else 0.0
        if displacement <= cfg.static_displacement_m and track_length > 1:
            jitter = float(np.mean(np.std(positions, axis=0)))
            jitter_scores.append(_score_jitter(jitter, cfg.jitter_tolerance_m))

    coverage_mean = float(np.mean(list(coverage.values()))) if coverage else 0.0
    median_track_length = float(np.median(track_lengths)) if track_lengths else 0.0
    median_track_length_ratio = median_track_length / max(T, 1)
    jitter_score = float(np.mean(jitter_scores)) if jitter_scores else 1.0

    id_switch_score = 1.0 - min(1.0, id_switch_proxy / max(cfg.max_id_switch_proxy, 1e-6))
    invalid_score = 1.0 if invalid_geometry_count == 0 else 0.0

    quality = (
        cfg.weight_coverage * coverage_mean
        + cfg.weight_length * min(1.0, median_track_length_ratio)
        + cfg.weight_jitter * jitter_score
        + cfg.weight_id_switch * id_switch_score
        + cfg.weight_invalid * invalid_score
    )
    quality = float(np.clip(quality, 0.0, 1.0))

    flags = []
    if coverage_mean < cfg.min_track_coverage:
        flags.append("low_coverage")
    if median_track_length_ratio < cfg.min_median_track_length_ratio:
        flags.append("short_tracks")
    if jitter_score < 0.5:
        flags.append("high_jitter")
    if id_switch_proxy > cfg.max_id_switch_proxy:
        flags.append("id_switches")
    if invalid_geometry_count > 0:
        flags.append("invalid_geometry")

    return SceneTracksQuality(
        quality_score=quality,
        track_coverage_ratio=coverage,
        median_track_length=median_track_length,
        jitter_score=jitter_score,
        id_switch_proxy=id_switch_proxy,
        invalid_geometry_count=invalid_geometry_count,
        flags=flags,
    )


def _count_segments(mask: Iterable[bool]) -> int:
    count = 0
    active = False
    for value in mask:
        if value and not active:
            count += 1
            active = True
        elif not value and active:
            active = False
    return count


def _score_jitter(jitter: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return float(max(0.0, 1.0 - min(jitter / tolerance, 1.0)))
