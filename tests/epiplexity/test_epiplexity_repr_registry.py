import numpy as np

from src.epiplexity.representations import build_default_representation_fns


def _scene_tracks_payload(T: int = 3, K: int = 2) -> dict[str, np.ndarray]:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    for k in range(K):
        poses_t[:, k, 0] = np.linspace(0.0, 0.2, T)
        poses_t[:, k, 1] = 0.1 * k
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.ones((T, K), dtype=bool)
    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": track_ids,
        "scene_tracks_v1/entity_types": entity_types,
        "scene_tracks_v1/class_ids": class_ids,
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility,
        "scene_tracks_v1/occlusion": occlusion,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }


def test_geometry_bev_repr_registered():
    representation_fns = build_default_representation_fns(
        "configs/channel_groups_robotics.json",
        include_geometry_bev=True,
    )
    assert "geometry_bev" in representation_fns
    episode = {"scene_tracks": _scene_tracks_payload()}
    tokens = representation_fns["geometry_bev"]([episode])
    assert tokens.ndim == 3
