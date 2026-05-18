"""CPU-only camera geometry utilities for sim-real consistency work."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def camera_intrinsics_from_fov(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    *,
    principal_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build a 3x3 pinhole intrinsic matrix from horizontal field-of-view."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0.0 < float(horizontal_fov_deg) < 180.0:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")
    focal = (float(width) / 2.0) / math.tan(math.radians(float(horizontal_fov_deg)) / 2.0)
    cx, cy = principal_point or ((float(width) - 1.0) / 2.0, (float(height) - 1.0) / 2.0)
    return np.asarray(
        [
            [focal, 0.0, float(cx)],
            [0.0, focal, float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compose_transforms(*transforms: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Compose homogeneous 4x4 transforms from left to right."""

    if not transforms:
        return np.eye(4, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("all transforms must be 4x4")
        result = result @ matrix
    return result


def invert_transform(transform: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def unproject_depth(
    depth: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: Sequence[Sequence[float]] | np.ndarray,
    *,
    camera_to_world: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Unproject a depth image into HxWx3 camera- or world-space points."""

    depth_image = np.asarray(depth, dtype=np.float64)
    intrinsics_matrix = np.asarray(intrinsics, dtype=np.float64)
    if depth_image.ndim != 2:
        raise ValueError("depth must be a 2D image")
    if intrinsics_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    ys, xs = np.indices(depth_image.shape, dtype=np.float64)
    z = depth_image
    x = (xs - intrinsics_matrix[0, 2]) * z / intrinsics_matrix[0, 0]
    y = (ys - intrinsics_matrix[1, 2]) * z / intrinsics_matrix[1, 1]
    points = np.stack([x, y, z], axis=-1)
    if camera_to_world is None:
        return points
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    homogeneous = np.concatenate(
        [points.reshape(-1, 3), np.ones((points.size // 3, 1), dtype=np.float64)],
        axis=1,
    )
    world = (transform @ homogeneous.T).T[:, :3]
    return world.reshape(points.shape)


def project_points(
    points: Iterable[Sequence[float]] | np.ndarray,
    intrinsics: Sequence[Sequence[float]] | np.ndarray,
    *,
    world_to_camera: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Project Nx3 points to Nx3 `(u, v, depth)` camera coordinates."""

    point_array = np.asarray(points, dtype=np.float64)
    intrinsics_matrix = np.asarray(intrinsics, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if intrinsics_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    camera_points = point_array
    if world_to_camera is not None:
        transform = np.asarray(world_to_camera, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("world_to_camera must be 4x4")
        homogeneous = np.concatenate(
            [point_array, np.ones((point_array.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        camera_points = (transform @ homogeneous.T).T[:, :3]
    z = camera_points[:, 2]
    if np.any(z == 0.0):
        raise ValueError("points with zero depth cannot be projected")
    u = (camera_points[:, 0] * intrinsics_matrix[0, 0] / z) + intrinsics_matrix[0, 2]
    v = (camera_points[:, 1] * intrinsics_matrix[1, 1] / z) + intrinsics_matrix[1, 2]
    return np.stack([u, v, z], axis=1)
