import numpy as np

from src.embodiment.core import EmbodimentInputs, compute_embodiment
from src.embodiment.artifacts import EMBODIMENT_PROFILE_PREFIX


def _scene_tracks_payload(
    T=6,
    K=2,
    distance=0.01,
    visibility=1.0,
    occlusion=0.0,
    class_names=None,
):
    track_ids = np.array([f"t{i}" for i in range(K)], dtype="U32")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.arange(K, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    if K > 1:
        poses_t[:, 1, 0] = distance
    scales = np.full((T, K), 0.05, dtype=np.float32)
    visibility_arr = np.full((T, K), visibility, dtype=np.float32)
    occlusion_arr = np.full((T, K), occlusion, dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.ones((T, K), dtype=bool)

    payload = {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": track_ids,
        "scene_tracks_v1/entity_types": entity_types,
        "scene_tracks_v1/class_ids": class_ids,
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility_arr,
        "scene_tracks_v1/occlusion": occlusion_arr,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }
    if class_names is not None:
        payload["scene_tracks_v1/class_names"] = np.array(class_names, dtype="U32")
    return payload


def test_embodiment_profile_npz_schema():
    payload = _scene_tracks_payload()
    inputs = EmbodimentInputs(scene_tracks=payload)
    result = compute_embodiment(inputs)
    npz = result.profile.to_npz(summary=result.summary)

    assert f"{EMBODIMENT_PROFILE_PREFIX}version" in npz
    assert f"{EMBODIMENT_PROFILE_PREFIX}contact_matrix" in npz
    assert npz[f"{EMBODIMENT_PROFILE_PREFIX}contact_matrix"].dtype == bool
    assert npz[f"{EMBODIMENT_PROFILE_PREFIX}contact_confidence"].dtype != object
    for key, arr in npz.items():
        assert arr.dtype != object


def test_embodiment_occlusion_flags_impossible_contacts():
    payload = _scene_tracks_payload(visibility=0.0, occlusion=0.95)
    inputs = EmbodimentInputs(scene_tracks=payload)
    result = compute_embodiment(inputs)

    assert result.summary.physically_impossible_contacts > 0
    assert result.summary.trust_override_candidate is True


def test_embodiment_constraint_drift_detected():
    payload = _scene_tracks_payload(class_names=["cup", "table"])
    inputs = EmbodimentInputs(
        scene_tracks=payload,
        task_constraints={"forbidden_contacts": ["cup|table"]},
    )
    result = compute_embodiment(inputs)

    assert result.drift_report.get("constraint_drift_score", 0.0) > 0.0
