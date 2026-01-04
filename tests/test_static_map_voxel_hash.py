from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.static_map import VoxelHashMap

pytestmark = pytest.mark.mapfirst


def test_voxel_hash_update_query_roundtrip() -> None:
    voxel_map = VoxelHashMap(voxel_size=1.0, max_points_per_voxel=10)
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [1.1, 1.1, 1.1],
    ], dtype=np.float32)
    voxel_map.update(points)

    centroids = voxel_map.query_centroids(points)
    assert centroids.shape == points.shape
    assert np.allclose(centroids[0], centroids[1])
    assert np.allclose(centroids[2], points[2])

    npz = voxel_map.to_npz()
    restored = VoxelHashMap.from_npz(npz)
    restored_centroids = restored.query_centroids(points)
    assert np.allclose(restored_centroids, centroids)
