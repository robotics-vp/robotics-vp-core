from __future__ import annotations

import numpy as np
import pytest

from src.vision.map_first_supervision.node import MapFirstPseudoSupervisionNode

pytestmark = pytest.mark.mapfirst


def _make_scene_tracks_dict(T: int = 5, K: int = 2) -> dict[str, np.ndarray]:
    poses_R = np.tile(np.eye(3, dtype=np.float32), (T, K, 1, 1))
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    poses_t[:, 0, 0] = 1.0
    poses_t[:, 1, 1] = 1.5
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.ones((T, K), dtype=bool)

    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(["track_a", "track_b"], dtype="U32"),
        "scene_tracks_v1/entity_types": np.array([0, 0], dtype=np.int32),
        "scene_tracks_v1/class_ids": np.array([-1, -1], dtype=np.int32),
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility,
        "scene_tracks_v1/occlusion": occlusion,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }


def test_end_to_end_map_first_node(tmp_path) -> None:
    scene_tracks = _make_scene_tracks_dict()
    episode_assets = {
        "camera_intrinsics": {"resolution": [4, 4], "fov_deg": 90.0},
        "camera_extrinsics": {"world_from_cam": np.eye(4, dtype=np.float32)},
    }

    node = MapFirstPseudoSupervisionNode()
    output_path = tmp_path / "map_first_output.npz"
    result = node.run(scene_tracks, episode_assets=episode_assets, output_path=str(output_path))

    data = dict(np.load(result.artifact_path, allow_pickle=False))
    prefix = "map_first_supervision_v1/"

    assert f"{prefix}dynamic_evidence" in data
    assert f"{prefix}dynamic_mask" in data
    assert f"{prefix}residual_mean" in data
    assert f"{prefix}boxes3d" in data
    assert f"{prefix}confidence" in data
    assert f"{prefix}densify_depth" in data
    assert f"{prefix}densify_mask" in data
    assert f"{prefix}evidence_dynamics_score" in data
    assert f"{prefix}evidence_geom_residual" in data
    assert f"{prefix}evidence_occlusion" in data

    summary_dict = result.summary.to_dict()
    assert "dynamic_pct" in summary_dict
    assert "map_first_quality_score" in summary_dict
