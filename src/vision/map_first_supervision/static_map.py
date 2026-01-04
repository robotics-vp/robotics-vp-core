"""Voxel hash map for static map accumulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _voxel_index(points: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor(points / float(voxel_size)).astype(np.int32)


@dataclass
class VoxelHashMap:
    """Sparse voxel map storing centroids and optional semantics."""

    voxel_size: float
    max_points_per_voxel: int
    semantics_num_classes: int = 0

    def __post_init__(self) -> None:
        self._index: Dict[Tuple[int, int, int], int] = {}
        self._centroids: list[np.ndarray] = []
        self._counts: list[float] = []
        self._semantics: Optional[list[np.ndarray]] = None
        if self.semantics_num_classes > 0:
            self._semantics = []

    @property
    def num_voxels(self) -> int:
        return len(self._centroids)

    def update(
        self,
        points_world: np.ndarray,
        weights: Optional[np.ndarray] = None,
        semantics: Optional[np.ndarray] = None,
        ema_alpha: float = 0.2,
    ) -> None:
        """Update voxel centroids and optional semantics.

        Args:
            points_world: (N, 3) world points.
            weights: Optional (N,) weights per point.
            semantics: Optional (N, C) class probs or (N,) labels.
            ema_alpha: EMA alpha for semantics updates.
        """
        if points_world.size == 0:
            return
        points_world = np.asarray(points_world, dtype=np.float32)
        weights_arr = None if weights is None else np.asarray(weights, dtype=np.float32)
        semantics_arr = None if semantics is None else np.asarray(semantics)

        if semantics_arr is not None and self._semantics is None:
            raise ValueError("Semantics provided but semantics_num_classes is 0")

        voxels = _voxel_index(points_world, self.voxel_size)
        for i, voxel in enumerate(voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            w = 1.0 if weights_arr is None else float(weights_arr[i])
            if key in self._index:
                idx = self._index[key]
                count = self._counts[idx]
                if count < self.max_points_per_voxel:
                    new_count = count + w
                    centroid = self._centroids[idx]
                    centroid = (centroid * count + points_world[i] * w) / max(new_count, 1e-6)
                    self._centroids[idx] = centroid.astype(np.float32)
                    self._counts[idx] = min(float(new_count), float(self.max_points_per_voxel))
                if semantics_arr is not None and self._semantics is not None:
                    self._update_semantics(idx, semantics_arr[i], ema_alpha)
            else:
                idx = len(self._centroids)
                self._index[key] = idx
                self._centroids.append(points_world[i].astype(np.float32))
                self._counts.append(float(min(w, float(self.max_points_per_voxel))))
                if self._semantics is not None:
                    self._semantics.append(self._init_semantics(semantics_arr[i] if semantics_arr is not None else None))

    def _init_semantics(self, value: Optional[np.ndarray]) -> np.ndarray:
        num_classes = int(self.semantics_num_classes)
        if value is None:
            return np.zeros(num_classes, dtype=np.float32)
        value = np.asarray(value)
        if value.ndim == 0:
            vec = np.zeros(num_classes, dtype=np.float32)
            idx = int(value)
            if 0 <= idx < num_classes:
                vec[idx] = 1.0
            return vec
        if value.ndim == 1 and value.shape[0] == num_classes:
            return value.astype(np.float32)
        raise ValueError("Invalid semantics value shape")

    def _update_semantics(self, idx: int, value: np.ndarray, ema_alpha: float) -> None:
        assert self._semantics is not None
        current = self._semantics[idx]
        num_classes = int(self.semantics_num_classes)
        if value.ndim == 0:
            one_hot = np.zeros(num_classes, dtype=np.float32)
            label = int(value)
            if 0 <= label < num_classes:
                one_hot[label] = 1.0
            value_vec = one_hot
        elif value.ndim == 1 and value.shape[0] == num_classes:
            value_vec = value.astype(np.float32)
        else:
            raise ValueError("Invalid semantics value shape")
        updated = (1.0 - ema_alpha) * current + ema_alpha * value_vec
        self._semantics[idx] = updated.astype(np.float32)

    def query_centroids(self, points_world: np.ndarray) -> np.ndarray:
        """Query nearest voxel centroids for points.

        Returns:
            (N, 3) centroids, NaN where voxel is missing.
        """
        if points_world.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        points_world = np.asarray(points_world, dtype=np.float32)
        voxels = _voxel_index(points_world, self.voxel_size)
        centroids = np.full_like(points_world, np.nan, dtype=np.float32)
        for i, voxel in enumerate(voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            idx = self._index.get(key)
            if idx is not None:
                centroids[i] = self._centroids[idx]
        return centroids

    def query_semantics(self, points_world: np.ndarray) -> Optional[np.ndarray]:
        """Query voxel semantics for points.

        Returns:
            (N, C) semantics or None if semantics disabled.
        """
        if self._semantics is None:
            return None
        points_world = np.asarray(points_world, dtype=np.float32)
        voxels = _voxel_index(points_world, self.voxel_size)
        sem = np.zeros((len(points_world), self.semantics_num_classes), dtype=np.float32)
        for i, voxel in enumerate(voxels):
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            idx = self._index.get(key)
            if idx is not None:
                sem[i] = self._semantics[idx]
        return sem

    def voxel_centroids(self) -> np.ndarray:
        if not self._centroids:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(self._centroids, axis=0).astype(np.float32)

    def voxel_counts(self) -> np.ndarray:
        if not self._counts:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self._counts, dtype=np.float32)

    def voxel_semantics(self) -> Optional[np.ndarray]:
        if self._semantics is None or not self._semantics:
            return None
        return np.stack(self._semantics, axis=0).astype(np.float32)

    def project_to_camera(
        self,
        camera_params: object,
        pose_w2c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project voxel centroids into camera space.

        Args:
            camera_params: Object with fx, fy, cx, cy, width, height.
            pose_w2c: (4, 4) world_from_cam transform.

        Returns:
            u: (N,) pixel x
            v: (N,) pixel y
            depth: (N,) depth in camera space
            valid_mask: (N,) bool mask for points in front of camera
        """
        points = self.voxel_centroids()
        if points.size == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )
        pose_w2c = np.asarray(pose_w2c, dtype=np.float32)
        cam_from_world = np.linalg.inv(pose_w2c)
        R = cam_from_world[:3, :3]
        t = cam_from_world[:3, 3]
        pts_cam = (R @ points.T).T + t
        depth = pts_cam[:, 2]
        valid = depth > 1e-6
        fx = float(getattr(camera_params, "fx"))
        fy = float(getattr(camera_params, "fy"))
        cx = float(getattr(camera_params, "cx"))
        cy = float(getattr(camera_params, "cy"))
        u = fx * (pts_cam[:, 0] / depth) + cx
        v = fy * (pts_cam[:, 1] / depth) + cy
        return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32), valid

    def to_npz(self) -> dict[str, np.ndarray]:
        """Serialize voxel map to numpy arrays."""
        data = {
            "voxel_size": np.array([self.voxel_size], dtype=np.float32),
            "centroids": self.voxel_centroids(),
            "counts": self.voxel_counts(),
        }
        sem = self.voxel_semantics()
        if sem is not None:
            data["semantics"] = sem
        return data

    @classmethod
    def from_npz(cls, data: dict[str, np.ndarray]) -> "VoxelHashMap":
        voxel_size = float(data.get("voxel_size", np.array([1.0]))[0])
        centroids = np.asarray(data.get("centroids", np.zeros((0, 3))), dtype=np.float32)
        counts = np.asarray(data.get("counts", np.zeros((0,))), dtype=np.float32)
        semantics = data.get("semantics")
        semantics_num_classes = int(semantics.shape[1]) if semantics is not None and semantics.size > 0 else 0
        instance = cls(
            voxel_size=voxel_size,
            max_points_per_voxel=int(np.max(counts)) if counts.size > 0 else 1,
            semantics_num_classes=semantics_num_classes,
        )
        for idx, centroid in enumerate(centroids):
            key = tuple(_voxel_index(centroid[np.newaxis, :], voxel_size)[0])
            instance._index[key] = idx
            instance._centroids.append(centroid.astype(np.float32))
            instance._counts.append(float(counts[idx]) if counts.size > idx else 1.0)
            if instance._semantics is not None and semantics is not None:
                instance._semantics.append(semantics[idx].astype(np.float32))
        return instance
