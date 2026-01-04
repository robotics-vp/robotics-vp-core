"""
Tests for scene track serialization.

Verifies round-trip serialization, shape consistency, and no object arrays.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
from src.vision.scene_ir_tracker.serialization import (
    serialize_scene_tracks_v1,
    deserialize_scene_tracks_v1,
    compute_scene_ir_quality_score,
    get_scene_ir_summary_dict,
    SCENE_TRACKS_VERSION,
)


@pytest.fixture
def sample_scene_tracks() -> SceneTracks:
    """Create sample SceneTracks for testing."""
    frames = []
    tracks = {}
    
    for t in range(10):  # 10 frames
        frame_entities = []
        
        # Object 1
        entity1 = SceneEntity3D(
            entity_type="object",
            track_id="track_0001",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0 + t * 0.1,
            class_name="box",
            visibility=0.9,
            occlusion_score=0.1,
            ir_loss=0.1 + t * 0.01,
            z_shape=np.random.randn(256).astype(np.float32),
            z_tex=np.random.randn(256).astype(np.float32),
        )
        entity1.pose[:3, 3] = [t * 0.1, 0, 3]
        frame_entities.append(entity1)
        
        # Body 1
        entity2 = SceneEntity3D(
            entity_type="body",
            track_id="track_0002",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
            visibility=0.95,
            occlusion_score=0.05,
            ir_loss=0.08,
            z_shape=np.random.randn(10).astype(np.float32),
            z_tex=np.random.randn(72).astype(np.float32),
            joints_3d={"pelvis": np.array([0, 0, 2], dtype=np.float32)},
        )
        entity2.pose[:3, 3] = [0, t * 0.05, 2]
        frame_entities.append(entity2)
        
        frames.append(frame_entities)
        
        # Build track history
        for e in frame_entities:
            if e.track_id not in tracks:
                tracks[e.track_id] = []
            tracks[e.track_id].append(e)
    
    metrics = SceneTrackerMetrics(
        ir_loss_per_frame=[0.1] * 10,
        id_switch_count=1,
        occlusion_rate=0.1,
        mean_ir_loss=0.1,
        converged_count=9,
        total_frames=10,
        total_tracks=2,
        track_lengths=[10, 10],
    )
    
    return SceneTracks(frames=frames, tracks=tracks, metrics=metrics)


class TestSerializeSceneTracksV1:
    """Tests for serialize_scene_tracks_v1."""

    def test_serialize_returns_dict(self, sample_scene_tracks):
        """Serialization returns a dict."""
        result = serialize_scene_tracks_v1(sample_scene_tracks)
        assert isinstance(result, dict)

    def test_serialize_all_values_are_numpy(self, sample_scene_tracks):
        """All values in serialized dict are numpy arrays."""
        result = serialize_scene_tracks_v1(sample_scene_tracks)
        for key, value in result.items():
            assert isinstance(value, np.ndarray), f"Key {key} is not a numpy array"

    def test_serialize_no_object_arrays(self, sample_scene_tracks):
        """No object dtype arrays (except strings)."""
        result = serialize_scene_tracks_v1(sample_scene_tracks)
        for key, value in result.items():
            if value.dtype.kind == 'O':
                pytest.fail(f"Key {key} has object dtype: {value.dtype}")

    def test_serialize_has_required_keys(self, sample_scene_tracks):
        """All required keys are present."""
        result = serialize_scene_tracks_v1(sample_scene_tracks)
        required = [
            "scene_tracks_v1/version",
            "scene_tracks_v1/track_ids",
            "scene_tracks_v1/entity_types",
            "scene_tracks_v1/class_ids",
            "scene_tracks_v1/poses_R",
            "scene_tracks_v1/poses_t",
            "scene_tracks_v1/scales",
            "scene_tracks_v1/visibility",
            "scene_tracks_v1/occlusion",
            "scene_tracks_v1/ir_loss",
            "scene_tracks_v1/converged",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_serialize_shapes_consistent(self, sample_scene_tracks):
        """Array shapes are consistent with T, K dimensions."""
        result = serialize_scene_tracks_v1(sample_scene_tracks)
        
        T = 10  # expected frames
        K = 2   # expected tracks
        
        assert result["scene_tracks_v1/track_ids"].shape == (K,)
        assert result["scene_tracks_v1/entity_types"].shape == (K,)
        assert result["scene_tracks_v1/class_ids"].shape == (K,)
        assert result["scene_tracks_v1/poses_R"].shape == (T, K, 3, 3)
        assert result["scene_tracks_v1/poses_t"].shape == (T, K, 3)
        assert result["scene_tracks_v1/scales"].shape == (T, K)
        assert result["scene_tracks_v1/visibility"].shape == (T, K)
        assert result["scene_tracks_v1/occlusion"].shape == (T, K)
        assert result["scene_tracks_v1/ir_loss"].shape == (T, K)
        assert result["scene_tracks_v1/converged"].shape == (T, K)

    def test_serialize_with_latents(self, sample_scene_tracks):
        """Latents included when requested."""
        result = serialize_scene_tracks_v1(sample_scene_tracks, include_latents=True)
        
        # Note: latent dims are detected from first entity, which has 256
        assert "scene_tracks_v1/z_shape" in result
        assert "scene_tracks_v1/z_tex" in result
        assert result["scene_tracks_v1/z_shape"].dtype == np.float16


class TestDeserializeSceneTracksV1:
    """Tests for deserialize_scene_tracks_v1."""

    def test_deserialize_returns_lite(self, sample_scene_tracks):
        """Deserialization returns SceneTracksLite."""
        from src.vision.scene_ir_tracker.serialization import SceneTracksLite
        
        serialized = serialize_scene_tracks_v1(sample_scene_tracks)
        result = deserialize_scene_tracks_v1(serialized)
        
        assert isinstance(result, SceneTracksLite)

    def test_round_trip_preserves_track_count(self, sample_scene_tracks):
        """Round-trip preserves number of tracks."""
        serialized = serialize_scene_tracks_v1(sample_scene_tracks)
        result = deserialize_scene_tracks_v1(serialized)
        
        assert result.num_tracks == 2

    def test_round_trip_preserves_frame_count(self, sample_scene_tracks):
        """Round-trip preserves number of frames."""
        serialized = serialize_scene_tracks_v1(sample_scene_tracks)
        result = deserialize_scene_tracks_v1(serialized)
        
        assert result.num_frames == 10


class TestRoundTrip:
    """Tests for full round-trip through npz."""

    def test_save_load_npz(self, sample_scene_tracks):
        """Full save/load through np.savez."""
        serialized = serialize_scene_tracks_v1(sample_scene_tracks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.npz"
            np.savez(path, **serialized)
            
            loaded = dict(np.load(path, allow_pickle=False))
            result = deserialize_scene_tracks_v1(loaded)
            
            assert result.num_tracks == 2
            assert result.num_frames == 10
            assert result.entity_types[0] == 0  # object
            assert result.entity_types[1] == 1  # body


class TestQualityScore:
    """Tests for quality score computation."""

    def test_quality_score_in_range(self, sample_scene_tracks):
        """Quality score is in [0, 1]."""
        score = compute_scene_ir_quality_score(sample_scene_tracks)
        assert 0 <= score <= 1

    def test_summary_dict_has_required_keys(self, sample_scene_tracks):
        """Summary dict has expected keys."""
        summary = get_scene_ir_summary_dict(sample_scene_tracks)
        
        required = [
            "ir_loss_mean",
            "id_switch_count",
            "occlusion_rate",
            "quality_score",
        ]
        for key in required:
            assert key in summary
