import sys
from pathlib import Path

import numpy as np

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta, ConditionProfile
from scripts import run_epiplexity_curated_slices


def _scene_tracks_payload(T: int = 3, K: int = 2) -> dict[str, np.ndarray]:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    poses_t[:, :, 0] = np.linspace(0.0, 0.2, T)[:, None]
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


def _rgb_features(dim: int = 64, T: int = 3) -> dict:
    features_temporal = np.linspace(0.0, 1.0, T * dim, dtype=np.float32).reshape(T, dim)
    return {
        "encoder": "vision_rgb::deterministic_pool_v1",
        "dim": dim,
        "pooling": "mean",
        "stride_seconds": 1.0,
        "features": features_temporal.mean(axis=0).tolist(),
        "features_temporal": features_temporal.tolist(),
    }


def _make_datapack(pack_id: str, labels: dict) -> DataPackMeta:
    cond = ConditionProfile(task_name="drawer_vase", engine_type="pybullet", occlusion_level=labels["occlusion_level"])
    return DataPackMeta(
        pack_id=pack_id,
        task_name="drawer_vase",
        env_type="drawer_vase",
        condition=cond,
        scene_tracks_v1=_scene_tracks_payload(),
        rgb_features_v1=_rgb_features(),
        slice_labels_v1=labels,
        schema_version="2.1-portable",
    )


def test_curated_slices_portable(tmp_path, monkeypatch):
    repo = DataPackRepo(base_dir=str(tmp_path))
    labels = [
        {
            "occlusion_level": 0.7,
            "is_occluded": True,
            "is_dynamic": False,
            "is_static": False,
        },
        {
            "occlusion_level": 0.1,
            "is_occluded": False,
            "is_dynamic": True,
            "is_static": False,
        },
        {
            "occlusion_level": 0.0,
            "is_occluded": False,
            "is_dynamic": False,
            "is_static": True,
        },
    ]
    datapacks = [
        _make_datapack("portable_occluded", labels[0]),
        _make_datapack("portable_dynamic", labels[1]),
        _make_datapack("portable_static", labels[2]),
    ]
    repo.append_batch(datapacks)

    output_dir = tmp_path / "out"
    argv = [
        "run_epiplexity_curated_slices",
        "--datapack-dir",
        str(tmp_path),
        "--task",
        "drawer_vase",
        "--output-dir",
        str(output_dir),
        "--budget-steps",
        "1",
        "--batch-size",
        "2",
        "--seeds",
        "0",
        "--max-per-slice",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_epiplexity_curated_slices.main()

    assert (output_dir / "curated_occluded.json").exists()
    assert (output_dir / "curated_dynamic.json").exists()
    assert (output_dir / "curated_static.json").exists()
