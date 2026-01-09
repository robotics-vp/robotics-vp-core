"""Test token-only mode for curated slices evaluation."""
import sys
from pathlib import Path

import numpy as np
import pytest

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta, ConditionProfile
from scripts import run_epiplexity_curated_slices


def _scene_tracks_payload(T: int = 3, K: int = 2) -> dict[str, np.ndarray]:
    """Create synthetic scene tracks payload."""
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
    """Create synthetic RGB features."""
    features_temporal = np.linspace(0.0, 1.0, T * dim, dtype=np.float32).reshape(T, dim)
    return {
        "encoder": "vision_rgb::deterministic_pool_v1",
        "dim": dim,
        "pooling": "mean",
        "stride_seconds": 1.0,
        "features": features_temporal.mean(axis=0).tolist(),
        "features_temporal": features_temporal.tolist(),
    }


def _repr_tokens(dim: int = 64) -> dict:
    """Create synthetic repr_tokens payload."""
    return {
        "vision_rgb": {
            "version": "vision_rgb::v1",
            "dim": dim,
            "num_tokens": 3,
            "pooling": "mean",
            "features": np.random.randn(dim).tolist(),
            "metadata": {},
        },
        "geometry_bev": {
            "version": "geometry_bev::v1",
            "dim": dim,
            "num_tokens": 3,
            "pooling": "mean",
            "features": np.random.randn(dim).tolist(),
            "metadata": {},
        },
        "geometry_scene_graph": {
            "version": "geometry_scene_graph::v1",
            "dim": dim,
            "num_tokens": 3,
            "pooling": "mean",
            "features": np.random.randn(dim).tolist(),
            "metadata": {},
        },
    }


def _make_datapack_with_repr_tokens(pack_id: str, labels: dict) -> DataPackMeta:
    """Create datapack with repr_tokens for token-only mode."""
    cond = ConditionProfile(task_name="drawer_vase", engine_type="pybullet", occlusion_level=labels["occlusion_level"])
    return DataPackMeta(
        pack_id=pack_id,
        task_name="drawer_vase",
        env_type="drawer_vase",
        condition=cond,
        scene_tracks_v1=_scene_tracks_payload(),
        rgb_features_v1=_rgb_features(),
        slice_labels_v1=labels,
        repr_tokens=_repr_tokens(),
        schema_version="2.2-repr",
    )


def test_curated_slices_token_only(tmp_path, monkeypatch):
    """Test that token-only mode runs successfully with stored repr_tokens."""
    np.random.seed(42)
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
        _make_datapack_with_repr_tokens("token_only_occluded", labels[0]),
        _make_datapack_with_repr_tokens("token_only_dynamic", labels[1]),
        _make_datapack_with_repr_tokens("token_only_static", labels[2]),
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
        "--token-only",
        "--seeds",
        "0",
        "--max-per-slice",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_epiplexity_curated_slices.main()

    # Verify output files exist
    assert (output_dir / "curated_occluded.json").exists()
    assert (output_dir / "curated_dynamic.json").exists()
    assert (output_dir / "curated_static.json").exists()


def test_curated_slices_token_only_requires_repr_tokens(tmp_path, monkeypatch):
    """Test that token-only mode fails fast when repr_tokens is missing."""
    repo = DataPackRepo(base_dir=str(tmp_path))

    # Create datapack without repr_tokens
    cond = ConditionProfile(task_name="drawer_vase", engine_type="pybullet")
    datapack = DataPackMeta(
        task_name="drawer_vase",
        condition=cond,
        schema_version="2.1-portable",
    )
    repo.append(datapack)

    output_dir = tmp_path / "out"
    argv = [
        "run_epiplexity_curated_slices",
        "--datapack-dir",
        str(tmp_path),
        "--task",
        "drawer_vase",
        "--output-dir",
        str(output_dir),
        "--token-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError) as excinfo:
        run_epiplexity_curated_slices.main()

    msg = str(excinfo.value)
    assert "token-only" in msg.lower() or "repr_tokens" in msg


def test_curated_slices_token_only_synthetic_conflict(tmp_path, monkeypatch):
    """Test that --token-only and --synthetic conflict."""
    argv = [
        "run_epiplexity_curated_slices",
        "--synthetic",
        "--token-only",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(RuntimeError) as excinfo:
        run_epiplexity_curated_slices.main()

    msg = str(excinfo.value)
    assert "token-only" in msg.lower() and "synthetic" in msg.lower()
