from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.config import MapFirstSupervisionConfig
from src.vision.map_first_supervision.geometry_provider import PrimitiveGeometryProvider
from src.vision.map_first_supervision.inconsistency import compute_inconsistency
from src.vision.map_first_supervision.static_map import VoxelHashMap

pytestmark = pytest.mark.mapfirst


class MockSceneTracksLite:
    def __init__(self, poses_R: np.ndarray, poses_t: np.ndarray, scales: np.ndarray, visibility: np.ndarray, occlusion: np.ndarray, entity_types: np.ndarray):
        self.poses_R = poses_R
        self.poses_t = poses_t
        self.scales = scales
        self.visibility = visibility
        self.occlusion = occlusion
        self.entity_types = entity_types


def test_dynamic_evidence_higher_for_moving_entity() -> None:
    T, K = 5, 2
    poses_R = np.tile(np.eye(3, dtype=np.float32), (T, K, 1, 1))
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    # Static entity stays at x=2
    poses_t[:, 0, 0] = 2.0
    # Moving entity oscillates within same voxel
    poses_t[:, 1, 0] = np.array([0.0, 0.8, 0.0, 0.8, 0.0], dtype=np.float32)

    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    entity_types = np.zeros((K,), dtype=np.int32)

    scene_tracks = MockSceneTracksLite(poses_R, poses_t, scales, visibility, occlusion, entity_types)
    provider = PrimitiveGeometryProvider(scene_tracks)

    voxel_map = VoxelHashMap(voxel_size=1.0, max_points_per_voxel=50)
    for t in range(T):
        for k in range(K):
            pts = provider.sample_points_world(t, k, n_points=64)
            voxel_map.update(pts)

    config = MapFirstSupervisionConfig(dynamic_threshold=0.3)
    out = compute_inconsistency(scene_tracks, voxel_map, provider, config, points_per_entity=64)

    static_mean = np.nanmean(out.dynamic_evidence[:, 0])
    moving_mean = np.nanmean(out.dynamic_evidence[:, 1])

    assert moving_mean > static_mean
