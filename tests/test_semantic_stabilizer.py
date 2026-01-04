from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.semantics import SemanticStabilizer
from src.vision.map_first_supervision.static_map import VoxelHashMap

pytestmark = pytest.mark.mapfirst


def test_semantic_stabilizer_converges_to_mode() -> None:
    voxel_map = VoxelHashMap(voxel_size=1.0, max_points_per_voxel=10, semantics_num_classes=3)
    stabilizer = SemanticStabilizer(voxel_map, ema_alpha=0.3)

    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
    ], dtype=np.float32)

    # Noisy inputs favor class 1
    noisy_probs = [
        np.array([0.1, 0.8, 0.1], dtype=np.float32),
        np.array([0.2, 0.7, 0.1], dtype=np.float32),
        np.array([0.1, 0.6, 0.3], dtype=np.float32),
    ]

    for probs in noisy_probs:
        stabilizer.update_from_entity_probs(points, probs)

    agg = stabilizer.aggregate_entity_probs(points)
    assert agg.shape == (3,)
    assert int(np.argmax(agg)) == 1
    assert agg[1] > 0.5
