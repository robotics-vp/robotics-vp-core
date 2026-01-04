"""
Tests for coordinate frame transforms.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics
from src.vision.scene_ir_tracker.transforms import (
    validate_frame_consistency,
    transform_pose_camera_to_world,
    transform_entity_to_world,
    transform_scene_tracks_to_world,
    ensure_world_frame,
    FrameMismatchError,
)


@pytest.fixture
def camera_extrinsics() -> np.ndarray:
    """Simple camera extrinsics: world_from_camera."""
    # Camera at (0, 0, -5) looking at origin
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, 3] = [0, 0, -5]  # Translation
    return extrinsics


@pytest.fixture
def sample_entity() -> SceneEntity3D:
    """Sample entity in camera frame."""
    return SceneEntity3D(
        entity_type="object",
        track_id="test_obj",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        joints_3d={"center": np.array([0, 0, 0], dtype=np.float32)},
    )


class TestValidateFrameConsistency:
    """Tests for validate_frame_consistency."""

    def test_passes_when_frame_matches(self):
        """No error when frame matches."""
        config = {"output_frame": "world"}
        tracks = SceneTracks(config_used=config)
        
        # Should not raise
        validate_frame_consistency(tracks, "world")

    def test_raises_when_frame_mismatches(self):
        """Raises FrameMismatchError when frame doesn't match."""
        config = {"output_frame": "camera"}
        tracks = SceneTracks(config_used=config)
        
        with pytest.raises(FrameMismatchError):
            validate_frame_consistency(tracks, "world")

    def test_warns_when_frame_unspecified(self):
        """Warns but doesn't raise when frame unspecified."""
        tracks = SceneTracks(config_used={})
        
        # Should not raise
        validate_frame_consistency(tracks, "world")


class TestTransformPose:
    """Tests for transform_pose_camera_to_world."""

    def test_transforms_identity_pose(self, camera_extrinsics):
        """Transforms identity pose correctly."""
        pose_camera = np.eye(4, dtype=np.float32)
        
        pose_world = transform_pose_camera_to_world(pose_camera, camera_extrinsics)
        
        assert pose_world.shape == (4, 4)
        # Identity pose at camera origin -> should be at camera position in world
        np.testing.assert_array_almost_equal(pose_world[:3, 3], [0, 0, -5])


class TestTransformEntity:
    """Tests for transform_entity_to_world."""

    def test_transforms_entity_pose(self, camera_extrinsics, sample_entity):
        """Transforms entity pose correctly."""
        world_entity = transform_entity_to_world(sample_entity, camera_extrinsics)
        
        # Position should be offset by camera position
        np.testing.assert_array_almost_equal(world_entity.pose[:3, 3], [0, 0, -5])

    def test_transforms_joints(self, camera_extrinsics, sample_entity):
        """Transforms joints correctly."""
        world_entity = transform_entity_to_world(sample_entity, camera_extrinsics)
        
        assert "center" in world_entity.joints_3d
        np.testing.assert_array_almost_equal(
            world_entity.joints_3d["center"], [0, 0, -5]
        )

    def test_preserves_non_pose_fields(self, camera_extrinsics, sample_entity):
        """Preserves non-pose fields."""
        world_entity = transform_entity_to_world(sample_entity, camera_extrinsics)
        
        assert world_entity.track_id == sample_entity.track_id
        assert world_entity.entity_type == sample_entity.entity_type
        assert world_entity.scale == sample_entity.scale


class TestTransformSceneTracks:
    """Tests for transform_scene_tracks_to_world."""

    def test_transforms_all_frames(self, camera_extrinsics, sample_entity):
        """Transforms all frames."""
        tracks = SceneTracks(
            frames=[[sample_entity], [sample_entity]],
            tracks={"test_obj": [sample_entity, sample_entity]},
            config_used={"output_frame": "camera"},
        )
        
        world_tracks = transform_scene_tracks_to_world(tracks, camera_extrinsics)
        
        assert len(world_tracks.frames) == 2
        assert world_tracks.config_used["output_frame"] == "world"

    def test_updates_config_to_world(self, camera_extrinsics, sample_entity):
        """Updates config to indicate world frame."""
        tracks = SceneTracks(
            frames=[[sample_entity]],
            config_used={"output_frame": "camera"},
        )
        
        world_tracks = transform_scene_tracks_to_world(tracks, camera_extrinsics)
        
        assert world_tracks.config_used["output_frame"] == "world"


class TestEnsureWorldFrame:
    """Tests for ensure_world_frame."""

    def test_returns_same_if_already_world(self, sample_entity):
        """Returns same tracks if already in world frame."""
        tracks = SceneTracks(
            frames=[[sample_entity]],
            config_used={"output_frame": "world"},
        )
        
        result = ensure_world_frame(tracks)
        
        assert result.config_used["output_frame"] == "world"

    def test_transforms_if_camera_with_extrinsics(self, camera_extrinsics, sample_entity):
        """Transforms if camera frame with extrinsics."""
        tracks = SceneTracks(
            frames=[[sample_entity]],
            config_used={"output_frame": "camera"},
        )
        
        result = ensure_world_frame(tracks, camera_extrinsics)
        
        assert result.config_used["output_frame"] == "world"

    def test_raises_if_camera_without_extrinsics(self, sample_entity):
        """Raises if camera frame without extrinsics."""
        tracks = SceneTracks(
            frames=[[sample_entity]],
            config_used={"output_frame": "camera"},
        )
        
        with pytest.raises(FrameMismatchError):
            ensure_world_frame(tracks)
