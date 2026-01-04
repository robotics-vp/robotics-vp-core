from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.densify import densify_depth_targets
from src.vision.map_first_supervision.static_map import VoxelHashMap
from src.vision.nag.types import CameraParams

pytestmark = pytest.mark.mapfirst


def test_spherical_bins_keep_nearest_depth() -> None:
    voxel_map = VoxelHashMap(voxel_size=1.0, max_points_per_voxel=10)
    points = np.array([
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    voxel_map.update(points)

    camera = CameraParams(
        fx=50.0,
        fy=50.0,
        cx=2.0,
        cy=2.0,
        height=4,
        width=4,
        world_from_cam=np.eye(4, dtype=np.float32),
    )

    densified = densify_depth_targets(voxel_map, camera, num_frames=1, occlusion_culling="spherical_bins")
    assert densified.depth is not None
    assert densified.mask is not None

    depth = densified.depth[0]
    mask = densified.mask[0]
    assert mask.sum() >= 1
    min_depth = depth[mask.astype(bool)].min()
    assert np.isclose(min_depth, 2.0, atol=1e-3)
