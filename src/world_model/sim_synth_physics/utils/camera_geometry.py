"""CPU-only camera geometry utilities for sim-real consistency work."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

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


def camera_intrinsics_from_mapping(payload: Mapping[str, Any]) -> np.ndarray:
    """Build a 3x3 pinhole intrinsic matrix from common metadata shapes."""

    data = dict(payload or {})
    matrix = data.get("matrix") or data.get("K") or data.get("intrinsics_matrix")
    if matrix is not None:
        intrinsics = np.asarray(matrix, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics matrix must be 3x3")
        return intrinsics

    resolution = data.get("resolution") or data.get("image_size") or data.get("size")
    width = data.get("width")
    height = data.get("height")
    if resolution is not None and (width is None or height is None):
        if not isinstance(resolution, Sequence) or len(resolution) < 2:
            raise ValueError("resolution must contain width and height")
        width = resolution[0]
        height = resolution[1]
    if width is None or height is None:
        raise ValueError("camera intrinsics require width/height or resolution")
    width_i = int(width)
    height_i = int(height)

    fx = data.get("fx")
    fy = data.get("fy")
    if fx is None or fy is None:
        return camera_intrinsics_from_fov(
            width_i,
            height_i,
            float(data.get("fov_deg", data.get("horizontal_fov_deg", 90.0))),
            principal_point=(
                float(data.get("cx", (float(width_i) - 1.0) / 2.0)),
                float(data.get("cy", (float(height_i) - 1.0) / 2.0)),
            ),
        )
    cx = float(data.get("cx", (float(width_i) - 1.0) / 2.0))
    cy = float(data.get("cy", (float(height_i) - 1.0) / 2.0))
    return np.asarray(
        [[float(fx), 0.0, cx], [0.0, float(fy), cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_from_rpy(rotation_rpy: Sequence[float]) -> np.ndarray:
    if len(rotation_rpy) != 3:
        raise ValueError("rotation_rpy must contain roll, pitch, yaw")
    roll, pitch, yaw = [float(value) for value in rotation_rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def transform_from_mapping(payload: Mapping[str, Any] | Sequence[Sequence[float]]) -> np.ndarray:
    """Build a 4x4 camera-to-world transform from common metadata shapes."""

    if isinstance(payload, Mapping):
        data = dict(payload or {})
        matrix = (
            data.get("matrix")
            or data.get("world_from_cam")
            or data.get("camera_to_world")
            or data.get("transform")
        )
        if matrix is not None:
            transform = np.asarray(matrix, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("camera transform must be 4x4")
            return transform
        translation = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError("translation must contain three values")
        rotation = _rotation_from_rpy(data.get("rotation_rpy", [0.0, 0.0, 0.0]))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

    transform = np.asarray(payload, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("camera transform must be 4x4")
    return transform


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


def camera_round_trip_error(
    depth: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: Sequence[Sequence[float]] | np.ndarray,
    *,
    camera_to_world: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> float:
    """Measure pixel-space error after depth unprojection and reprojection."""

    depth_image = np.asarray(depth, dtype=np.float64)
    points = unproject_depth(
        depth_image,
        intrinsics,
        camera_to_world=camera_to_world,
    )
    projected = project_points(
        points.reshape(-1, 3),
        intrinsics,
        world_to_camera=None if camera_to_world is None else invert_transform(camera_to_world),
    )
    ys, xs = np.indices(depth_image.shape, dtype=np.float64)
    expected = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=1)
    return float(np.max(np.abs(projected[:, :2] - expected)))
