"""Densification and reprojection targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.vision.map_first_supervision.static_map import VoxelHashMap


@dataclass
class DensifyOutput:
    depth: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    world_points: Optional[np.ndarray]
    world_mask: Optional[np.ndarray]


def _project_points(points_world: np.ndarray, pose_w2c: np.ndarray) -> np.ndarray:
    pose_w2c = np.asarray(pose_w2c, dtype=np.float32)
    cam_from_world = np.linalg.inv(pose_w2c)
    R = cam_from_world[:3, :3]
    t = cam_from_world[:3, 3]
    return (R @ points_world.T).T + t


def _spherical_bins_depth(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid_z = z > 1e-6
    if not np.any(valid_z):
        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        return depth, mask

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    fov_x = 2.0 * np.arctan(width / (2.0 * fx))
    fov_y = 2.0 * np.arctan(height / (2.0 * fy))

    az = np.arctan2(x, z)
    el = np.arctan2(y, z)

    u = np.floor((az + fov_x / 2.0) / fov_x * width).astype(np.int32)
    v = np.floor((el + fov_y / 2.0) / fov_y * height).astype(np.int32)

    valid = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    for idx in np.where(valid)[0]:
        uu = u[idx]
        vv = v[idx]
        if z[idx] < depth[vv, uu]:
            depth[vv, uu] = z[idx]
    mask = np.isfinite(depth).astype(np.uint8)
    depth[~np.isfinite(depth)] = 0.0
    return depth, mask


def _zbuffer_depth(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid_z = z > 1e-6
    if not np.any(valid_z):
        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        return depth, mask

    u = np.floor(fx * (points_cam[:, 0] / z) + cx).astype(np.int32)
    v = np.floor(fy * (points_cam[:, 1] / z) + cy).astype(np.int32)
    valid = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    for idx in np.where(valid)[0]:
        uu = u[idx]
        vv = v[idx]
        if z[idx] < depth[vv, uu]:
            depth[vv, uu] = z[idx]
    mask = np.isfinite(depth).astype(np.uint8)
    depth[~np.isfinite(depth)] = 0.0
    return depth, mask


def densify_depth_targets(
    static_map: VoxelHashMap,
    camera_params: object,
    num_frames: int,
    occlusion_culling: str = "spherical_bins",
) -> DensifyOutput:
    """Create per-frame depth targets from static map voxels."""
    points_world = static_map.voxel_centroids()
    if points_world.size == 0:
        return DensifyOutput(depth=None, mask=None, world_points=None, world_mask=None)

    height = int(getattr(camera_params, "height"))
    width = int(getattr(camera_params, "width"))
    fx = float(getattr(camera_params, "fx"))
    fy = float(getattr(camera_params, "fy"))
    cx = float(getattr(camera_params, "cx"))
    cy = float(getattr(camera_params, "cy"))

    depth_stack = np.zeros((num_frames, height, width), dtype=np.float32)
    mask_stack = np.zeros((num_frames, height, width), dtype=np.uint8)

    world_from_cam = getattr(camera_params, "world_from_cam")
    world_from_cam = np.asarray(world_from_cam, dtype=np.float32)
    if world_from_cam.ndim == 2:
        world_from_cam = world_from_cam[np.newaxis, ...]

    for t in range(num_frames):
        pose = world_from_cam[min(t, world_from_cam.shape[0] - 1)]
        points_cam = _project_points(points_world, pose)
        if occlusion_culling == "zbuffer":
            depth, mask = _zbuffer_depth(points_cam, fx, fy, cx, cy, width, height)
        else:
            depth, mask = _spherical_bins_depth(points_cam, fx, fy, width, height)
        depth_stack[t] = depth
        mask_stack[t] = mask

    return DensifyOutput(depth=depth_stack, mask=mask_stack, world_points=None, world_mask=None)


def densify_world_points(
    static_map: VoxelHashMap,
    num_frames: int,
) -> DensifyOutput:
    """Create per-frame world point targets from static map voxels."""
    points_world = static_map.voxel_centroids()
    if points_world.size == 0:
        return DensifyOutput(depth=None, mask=None, world_points=None, world_mask=None)

    M = points_world.shape[0]
    world_points = np.zeros((num_frames, M, 3), dtype=np.float32)
    world_mask = np.zeros((num_frames, M), dtype=np.uint8)
    world_points[:] = points_world[np.newaxis, :, :]
    world_mask[:] = 1
    return DensifyOutput(depth=None, mask=None, world_points=world_points, world_mask=world_mask)
