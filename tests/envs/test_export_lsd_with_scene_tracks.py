"""
Tests for export_lsd_vector_scene_dataset with scene tracks.

Verifies that scene_tracks_v1 exports have correct structure and dtypes.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestExportWithSceneTracks:
    """Tests for scene track export integration."""

    def test_serialization_no_object_arrays(self):
        """Verify serialized arrays have no object dtypes."""
        from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
        from src.vision.scene_ir_tracker.serialization import (
            serialize_scene_tracks_v1,
            validate_no_object_arrays,
        )
        
        # Create sample tracks
        entity = SceneEntity3D(
            entity_type="object",
            track_id="track_001",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
            class_name="box",
            visibility=0.9,
            occlusion_score=0.1,
            ir_loss=0.05,
        )
        
        metrics = SceneTrackerMetrics(total_frames=5, total_tracks=1)
        tracks = SceneTracks(
            frames=[[entity] for _ in range(5)],
            tracks={"track_001": [entity] * 5},
            metrics=metrics,
        )
        
        serialized = serialize_scene_tracks_v1(tracks)
        
        # Check no object arrays
        for key, arr in serialized.items():
            assert arr.dtype != object, f"Key {key} has object dtype"

    def test_serialized_shapes_consistent(self):
        """Verify serialized arrays have consistent shapes."""
        from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
        from src.vision.scene_ir_tracker.serialization import serialize_scene_tracks_v1
        
        T = 10  # frames
        K = 3   # tracks
        
        frames = []
        tracks = {f"track_{i}": [] for i in range(K)}
        
        for t in range(T):
            frame_entities = []
            for k in range(K):
                entity = SceneEntity3D(
                    entity_type="object" if k < 2 else "body",
                    track_id=f"track_{k}",
                    pose=np.eye(4, dtype=np.float32),
                    scale=1.0 + t * 0.1,
                    visibility=0.9,
                    occlusion_score=0.1,
                    ir_loss=0.05,
                )
                entity.pose[:3, 3] = [t * 0.1, k * 0.5, 3.0]
                frame_entities.append(entity)
                tracks[f"track_{k}"].append(entity)
            frames.append(frame_entities)
        
        metrics = SceneTrackerMetrics(total_frames=T, total_tracks=K)
        scene_tracks = SceneTracks(frames=frames, tracks=tracks, metrics=metrics)
        
        serialized = serialize_scene_tracks_v1(scene_tracks)
        
        # Check shapes
        assert serialized["scene_tracks_v1/track_ids"].shape == (K,)
        assert serialized["scene_tracks_v1/entity_types"].shape == (K,)
        assert serialized["scene_tracks_v1/poses_R"].shape == (T, K, 3, 3)
        assert serialized["scene_tracks_v1/poses_t"].shape == (T, K, 3)
        assert serialized["scene_tracks_v1/scales"].shape == (T, K)
        assert serialized["scene_tracks_v1/visibility"].shape == (T, K)
        assert serialized["scene_tracks_v1/occlusion"].shape == (T, K)
        assert serialized["scene_tracks_v1/ir_loss"].shape == (T, K)
        assert serialized["scene_tracks_v1/converged"].shape == (T, K)

    def test_serialized_dtypes_correct(self):
        """Verify serialized arrays have correct dtypes."""
        from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
        from src.vision.scene_ir_tracker.serialization import serialize_scene_tracks_v1
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="track_001",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
        )
        
        metrics = SceneTrackerMetrics(total_frames=1, total_tracks=1)
        tracks = SceneTracks(
            frames=[[entity]],
            tracks={"track_001": [entity]},
            metrics=metrics,
        )
        
        serialized = serialize_scene_tracks_v1(tracks)
        
        # Check required dtypes
        assert serialized["scene_tracks_v1/entity_types"].dtype == np.int32
        assert serialized["scene_tracks_v1/class_ids"].dtype == np.int32
        assert serialized["scene_tracks_v1/poses_R"].dtype == np.float32
        assert serialized["scene_tracks_v1/poses_t"].dtype == np.float32
        assert serialized["scene_tracks_v1/scales"].dtype == np.float32
        assert serialized["scene_tracks_v1/converged"].dtype == bool

    def test_save_load_npz_round_trip(self):
        """Verify save/load through npz preserves data."""
        from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
        from src.vision.scene_ir_tracker.serialization import (
            serialize_scene_tracks_v1,
            deserialize_scene_tracks_v1,
        )
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="track_001",
            pose=np.eye(4, dtype=np.float32),
            scale=1.5,
            class_name="robot",
        )
        entity.pose[:3, 3] = [1.0, 2.0, 3.0]
        
        metrics = SceneTrackerMetrics(total_frames=2, total_tracks=1)
        tracks = SceneTracks(
            frames=[[entity], [entity]],
            tracks={"track_001": [entity, entity]},
            metrics=metrics,
        )
        
        serialized = serialize_scene_tracks_v1(tracks)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test.npz"
            np.savez(npz_path, **serialized)
            
            # Load without pickle
            loaded = dict(np.load(npz_path, allow_pickle=False))
            
            # Deserialize
            result = deserialize_scene_tracks_v1(loaded)
            
            assert result.num_frames == 2
            assert result.num_tracks == 1
            np.testing.assert_array_almost_equal(
                result.poses_t[0, 0],
                [1.0, 2.0, 3.0],
            )

    def test_json_stored_as_unicode_not_object(self):
        """Verify JSON strings stored as unicode, not object dtype."""
        from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
        from src.vision.scene_ir_tracker.serialization import serialize_scene_tracks_v1
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="track_001",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
        )
        
        metrics = SceneTrackerMetrics(total_frames=1, total_tracks=1)
        tracks = SceneTracks(
            frames=[[entity]],
            tracks={"track_001": [entity]},
            metrics=metrics,
        )
        
        serialized = serialize_scene_tracks_v1(tracks)
        
        # Check summary_json is unicode string, not object
        summary_key = "scene_tracks_v1/summary_json"
        assert summary_key in serialized
        assert serialized[summary_key].dtype.kind in ('U', 'S'), \
            f"summary_json has dtype {serialized[summary_key].dtype}, expected unicode/string"

    def test_export_config_has_scene_ir_fields(self):
        """Verify ExportConfig has scene IR tracker fields."""
        # Import dynamically to avoid module issues
        import sys
        sys.path.insert(0, ".")
        
        from scripts.export_lsd_vector_scene_dataset import ExportConfig
        
        config = ExportConfig()
        assert hasattr(config, "enable_scene_ir_tracker")
        assert hasattr(config, "save_scene_tracks")
        assert hasattr(config, "save_scene_geometry")
        assert hasattr(config, "save_scene_latents")
        assert hasattr(config, "scene_tracks_version")
        
        # Check defaults
        assert config.enable_scene_ir_tracker is False
        assert config.save_scene_tracks is True
        assert config.scene_tracks_version == "v1"
