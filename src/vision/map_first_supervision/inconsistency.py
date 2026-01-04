"""Inconsistency and dynamic evidence computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.vision.map_first_supervision.config import MapFirstSupervisionConfig
from src.vision.map_first_supervision.geometry_provider import GeometryProvider
from src.vision.map_first_supervision.static_map import VoxelHashMap


@dataclass
class InconsistencyOutput:
    residual_mean: np.ndarray  # (T, K)
    coverage: np.ndarray  # (T, K)
    visibility_weight: np.ndarray  # (T, K)
    dynamic_evidence: np.ndarray  # (T, K)
    dynamic_mask: np.ndarray  # (T, K)
    confidence: np.ndarray  # (T, K)


def _get_visibility(scene_tracks: object) -> np.ndarray:
    visibility = getattr(scene_tracks, "visibility", None)
    if visibility is None or visibility.size == 0:
        poses_t = getattr(scene_tracks, "poses_t", np.zeros((0, 0, 3)))
        return np.ones((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
    return visibility.astype(np.float32)


def _get_occlusion(scene_tracks: object) -> np.ndarray:
    occlusion = getattr(scene_tracks, "occlusion", None)
    if occlusion is None or occlusion.size == 0:
        poses_t = getattr(scene_tracks, "poses_t", np.zeros((0, 0, 3)))
        return np.zeros((poses_t.shape[0], poses_t.shape[1]), dtype=np.float32)
    return occlusion.astype(np.float32)


def compute_inconsistency(
    scene_tracks: object,
    static_map: VoxelHashMap,
    geometry_provider: GeometryProvider,
    config: MapFirstSupervisionConfig,
    points_per_entity: int = 64,
) -> InconsistencyOutput:
    """Compute per-entity inconsistency residuals and dynamic evidence."""
    if config.residual_method != "voxel_centroid":
        raise NotImplementedError(f"Residual method '{config.residual_method}' is not implemented")
    poses_t = scene_tracks.poses_t
    T, K = poses_t.shape[:2]
    residual_mean = np.full((T, K), np.nan, dtype=np.float32)
    coverage = np.zeros((T, K), dtype=np.float32)

    visibility = _get_visibility(scene_tracks)
    occlusion = _get_occlusion(scene_tracks)
    visibility_weight = visibility * (1.0 - occlusion)

    for t in range(T):
        for k in range(K):
            points = geometry_provider.sample_points_world(t, k, points_per_entity)
            if points.size == 0:
                continue
            centroids = static_map.query_centroids(points)
            valid = np.isfinite(centroids).all(axis=1)
            if valid.size > 0:
                coverage[t, k] = float(np.mean(valid))
            if not np.any(valid):
                continue
            diffs = points[valid] - centroids[valid]
            dist = np.linalg.norm(diffs, axis=1)
            residual_mean[t, k] = float(np.mean(dist))

    valid_residuals = residual_mean[np.isfinite(residual_mean)]
    if valid_residuals.size > 0:
        res_mean = float(np.mean(valid_residuals))
        res_std = float(np.std(valid_residuals))
    else:
        res_mean = 0.0
        res_std = 1.0

    if config.dynamic_use_zscore:
        z = (residual_mean - res_mean) / max(res_std, 1e-6)
        dynamic_evidence = z.astype(np.float32)
    else:
        dynamic_evidence = residual_mean.copy()

    occluded = visibility_weight < 0.2
    confidence = (visibility_weight * coverage).astype(np.float32)

    valid_mask = np.isfinite(dynamic_evidence) & (coverage > 0)
    dynamic_mask = (dynamic_evidence > config.dynamic_threshold) & valid_mask & (~occluded)

    return InconsistencyOutput(
        residual_mean=residual_mean,
        coverage=coverage,
        visibility_weight=visibility_weight.astype(np.float32),
        dynamic_evidence=dynamic_evidence.astype(np.float32),
        dynamic_mask=dynamic_mask.astype(bool),
        confidence=confidence,
    )
