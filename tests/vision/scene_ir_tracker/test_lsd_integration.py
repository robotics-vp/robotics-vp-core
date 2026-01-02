"""Test LSD backend integration with scene IR tracker."""
import numpy as np
import pytest


def test_scene_tracker_config_in_lsd_config():
    """Test that LSD config has scene tracker fields."""
    from src.config.lsd_vector_scene_config import LSDVectorSceneConfig
    from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig

    config = LSDVectorSceneConfig()

    assert hasattr(config, "enable_scene_ir_tracker")
    assert hasattr(config, "scene_ir_tracker_config")
    assert config.enable_scene_ir_tracker is False  # Default off
    assert isinstance(config.scene_ir_tracker_config, SceneIRTrackerConfig)


def test_lsd_config_serialization_with_tracker():
    """Test that LSD config serializes scene tracker settings."""
    from src.config.lsd_vector_scene_config import LSDVectorSceneConfig
    from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig

    config = LSDVectorSceneConfig(
        enable_scene_ir_tracker=True,
        scene_ir_tracker_config=SceneIRTrackerConfig(
            device="cpu",
            use_stub_adapters=True,
        ),
    )

    data = config.to_dict()
    assert "enable_scene_ir_tracker" in data
    assert data["enable_scene_ir_tracker"] is True
    assert "scene_ir_tracker_config" in data
    assert data["scene_ir_tracker_config"]["device"] == "cpu"

    # Round-trip
    loaded = LSDVectorSceneConfig.from_dict(data)
    assert loaded.enable_scene_ir_tracker is True
    assert loaded.scene_ir_tracker_config.device == "cpu"


def test_episode_result_has_scene_tracks_field():
    """Test that episode result has scene_tracks field."""
    from src.motor_backend.lsd_vector_scene_backend import LSDVectorSceneEpisodeResult

    result = LSDVectorSceneEpisodeResult(
        episode_id="test",
        scene_id="scene",
        steps=10,
        termination_reason="max_steps",
        mpl_units_per_hour=60.0,
        error_rate=0.0,
        energy_wh=0.1,
        reward_sum=10.0,
        difficulty_features={},
        scene_config={},
    )

    assert hasattr(result, "scene_tracks")
    assert result.scene_tracks is None  # Default


def test_scene_tracks_data_model():
    """Test SceneTracks data model with MHN position extraction."""
    from src.vision.scene_ir_tracker.types import SceneEntity3D, SceneTracks, SceneTrackerMetrics

    # Create some test entities
    entity1 = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32),
        scale=1.0,
    )
    entity2 = SceneEntity3D(
        entity_type="body",
        track_id="body_1",
        pose=np.array([[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=np.float32),
        scale=1.0,
        joints_3d={"pelvis": np.array([4, 5, 6], dtype=np.float32)},
    )

    scene_tracks = SceneTracks(
        frames=[[entity1, entity2], [entity1]],
        tracks={
            "obj_1": [entity1, entity1],
            "body_1": [entity2],
        },
        metrics=SceneTrackerMetrics(
            ir_loss_per_frame=[0.1, 0.05],
            id_switch_count=0,
            total_frames=2,
        ),
    )

    # Test MHN position extraction
    positions = scene_tracks.get_positions_for_mhn()
    assert positions.shape[0] == 2  # 2 frames
    assert positions.shape[2] == 3  # 3D positions

    # Test serialization
    data = scene_tracks.to_dict()
    assert "frames" in data
    assert "tracks" in data
    assert "metrics" in data
    assert len(data["frames"]) == 2

    # Test summary
    summary = scene_tracks.summary()
    assert summary["num_frames"] == 2
    assert summary["num_tracks"] == 2


def test_scene_entity_3d_properties():
    """Test SceneEntity3D computed properties."""
    from src.vision.scene_ir_tracker.types import SceneEntity3D

    entity = SceneEntity3D(
        entity_type="object",
        track_id="test",
        pose=np.array([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]], dtype=np.float32),
        scale=1.5,
    )

    assert np.allclose(entity.position, [2, 3, 4])
    assert np.allclose(entity.centroid, [2, 3, 4])
    assert entity.rotation.shape == (3, 3)

    # Body with pelvis
    body = SceneEntity3D(
        entity_type="body",
        track_id="body",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        joints_3d={"pelvis": np.array([1, 2, 3], dtype=np.float32)},
    )
    assert np.allclose(body.centroid, [1, 2, 3])


def test_tracker_metrics_serialization():
    """Test SceneTrackerMetrics serialization."""
    from src.vision.scene_ir_tracker.types import SceneTrackerMetrics

    metrics = SceneTrackerMetrics(
        ir_loss_per_frame=[0.1, 0.2, 0.15],
        id_switch_count=1,
        occlusion_rate=0.3,
        mean_ir_loss=0.15,
        converged_count=2,
        total_frames=3,
        total_tracks=5,
        track_lengths=[3, 2, 1, 2, 1],
    )

    data = metrics.to_dict()
    loaded = SceneTrackerMetrics.from_dict(data)

    assert loaded.id_switch_count == 1
    assert loaded.total_frames == 3
    assert len(loaded.track_lengths) == 5


def test_tracker_config_from_dict():
    """Test SceneIRTrackerConfig from_dict."""
    from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig

    data = {
        "device": "cpu",
        "precision": "float32",
        "use_stub_adapters": True,
        "ir_refiner_config": {
            "num_texture_iters": 10,
        },
        "tracking_config": {
            "max_age": 10,
        },
    }

    config = SceneIRTrackerConfig.from_dict(data)
    assert config.device == "cpu"
    assert config.ir_refiner_config.num_texture_iters == 10
    assert config.tracking_config.max_age == 10


def test_scene_ir_tracker_basic():
    """Basic smoke test for SceneIRTracker."""
    from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig
    from src.vision.nag.types import CameraParams

    config = SceneIRTrackerConfig(
        device="cpu",
        use_stub_adapters=True,
    )

    tracker = SceneIRTracker(config)

    # Create minimal test data
    camera = CameraParams.from_single_pose(
        position=(0, 0, -5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
        fov_deg=60,
        width=32,
        height=32,
    )

    frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
    masks = [{}]

    result = tracker.process_episode(frames, masks, camera)

    assert result.num_frames == 1
    assert hasattr(result, "metrics")
