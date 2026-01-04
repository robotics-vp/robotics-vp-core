"""Geometry providers for Map-First pseudo-supervision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

try:
    from src.vision.scene_ir_tracker.serialization import SceneTracksLite
except Exception:  # pragma: no cover - optional import for type hints
    SceneTracksLite = object  # type: ignore


class GeometryProvider(Protocol):
    """Protocol for sampling entity geometry in world coordinates."""

    def sample_points_world(
        self,
        frame_index: int,
        track_index: int,
        n_points: int,
    ) -> np.ndarray:
        """Sample N points on entity geometry in world frame."""


def _halton(index: int, base: int) -> float:
    """Compute Halton sequence value for deterministic sampling."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def _sample_box_surface(n_points: int, half_dims: np.ndarray) -> np.ndarray:
    """Deterministic sampling of box surface points in local frame."""
    faces = (
        (0, -1.0),
        (0, 1.0),
        (1, -1.0),
        (1, 1.0),
        (2, -1.0),
        (2, 1.0),
    )
    pts = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(n_points):
        axis, sign = faces[i % len(faces)]
        u = _halton(i + 1, 2) - 0.5
        v = _halton(i + 1, 3) - 0.5
        coord = np.zeros(3, dtype=np.float32)
        coord[axis] = sign * half_dims[axis]
        other_axes = [ax for ax in range(3) if ax != axis]
        coord[other_axes[0]] = float(u * 2.0 * half_dims[other_axes[0]])
        coord[other_axes[1]] = float(v * 2.0 * half_dims[other_axes[1]])
        pts[i] = coord
    return pts


def _sample_sphere_surface(n_points: int, radius: float) -> np.ndarray:
    """Deterministic sampling of sphere surface points in local frame."""
    if n_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    indices = np.arange(n_points, dtype=np.float32)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - 2.0 * (indices + 0.5) / float(n_points)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1) * float(radius)
    return pts.astype(np.float32)


@dataclass
class PrimitiveGeometryProvider:
    """Primitive geometry provider using boxes/spheres from SceneTracksLite."""

    scene_tracks: SceneTracksLite
    extents: Optional[np.ndarray] = None

    def _get_scale(self, frame_index: int, track_index: int) -> float:
        scales = getattr(self.scene_tracks, "scales", None)
        if scales is None or scales.size == 0:
            return 1.0
        return float(scales[frame_index, track_index])

    def _get_extents(self, frame_index: int, track_index: int) -> Optional[np.ndarray]:
        if self.extents is None:
            return None
        extents = np.asarray(self.extents, dtype=np.float32)
        if extents.ndim == 2:
            return extents[track_index]
        if extents.ndim == 3:
            return extents[frame_index, track_index]
        return None

    def _box_dims(self, frame_index: int, track_index: int) -> np.ndarray:
        scale = self._get_scale(frame_index, track_index)
        extents = self._get_extents(frame_index, track_index)
        if extents is not None:
            dims = extents * scale
        else:
            dims = np.array([scale, scale, scale], dtype=np.float32)
        return np.maximum(dims, 1e-3)

    def sample_points_world(
        self,
        frame_index: int,
        track_index: int,
        n_points: int,
    ) -> np.ndarray:
        if n_points <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        entity_types = getattr(self.scene_tracks, "entity_types", None)
        entity_type = 0
        if entity_types is not None and entity_types.size > 0:
            entity_type = int(entity_types[track_index])
        is_body = entity_type == 1

        if is_body:
            scale = self._get_scale(frame_index, track_index)
            local_pts = _sample_sphere_surface(n_points, radius=0.5 * scale)
        else:
            dims = self._box_dims(frame_index, track_index)
            local_pts = _sample_box_surface(n_points, dims / 2.0)

        poses_R = self.scene_tracks.poses_R
        poses_t = self.scene_tracks.poses_t
        R = poses_R[frame_index, track_index]
        t = poses_t[frame_index, track_index]
        world_pts = (R @ local_pts.T).T + t
        return world_pts.astype(np.float32)

    def box_dims(self, frame_index: int, track_index: int) -> np.ndarray:
        """Get box dimensions for a track (used for 3D boxes output)."""
        return self._box_dims(frame_index, track_index)


@dataclass
class SidecarPointGeometryProvider:
    """Geometry provider backed by precomputed world-space points."""

    points_world: np.ndarray  # (T, K, N, 3)

    def __post_init__(self) -> None:
        self.points_world = np.asarray(self.points_world, dtype=np.float32)

    def sample_points_world(
        self,
        frame_index: int,
        track_index: int,
        n_points: int,
    ) -> np.ndarray:
        pts = self.points_world[frame_index, track_index]
        if pts.shape[0] <= n_points:
            return pts.astype(np.float32)
        if n_points <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        stride = max(1, pts.shape[0] // n_points)
        return pts[::stride][:n_points].astype(np.float32)
