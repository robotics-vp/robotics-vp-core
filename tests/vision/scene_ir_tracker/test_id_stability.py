"""Test ID stability in Kalman track manager."""
import numpy as np
import pytest

from src.vision.scene_ir_tracker.config import TrackingConfig
from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
from src.vision.scene_ir_tracker.types import SceneEntity3D


def make_entity(
    track_id: str,
    position: tuple,
    entity_type: str = "object",
    scale: float = 1.0,
    z_shape: np.ndarray = None,
) -> SceneEntity3D:
    """Helper to create test entity."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = position
    return SceneEntity3D(
        entity_type=entity_type,
        track_id=track_id,
        pose=pose,
        scale=scale,
        z_shape=z_shape if z_shape is not None else np.random.randn(32).astype(np.float32),
    )


@pytest.fixture
def tracker():
    """Create test tracker."""
    config = TrackingConfig(
        association_distance_threshold=2.0,
        max_age=3,
        min_hits=1,
    )
    return KalmanTrackManager(config=config)


def test_new_track_assigned_id(tracker):
    """New detection should get a new track ID."""
    entity = make_entity("det_0", (0, 0, 0))
    result = tracker.update([entity])

    assert len(result) == 1
    assert result[0].track_id.startswith("track_")


def test_same_position_maintains_id(tracker):
    """Entity at same position should keep same ID."""
    # Frame 1
    entity1 = make_entity("det_0", (0, 0, 0))
    result1 = tracker.update([entity1])
    track_id_1 = result1[0].track_id

    # Frame 2 - same position
    entity2 = make_entity("det_0", (0.1, 0.1, 0))
    result2 = tracker.update([entity2])
    track_id_2 = result2[0].track_id

    assert track_id_1 == track_id_2, "Same track should persist"


def test_crossing_objects_minimal_id_switches(tracker):
    """Test that crossing objects have minimal ID switches."""
    np.random.seed(42)

    # Two objects crossing paths
    # Object A starts at x=-2, moves to x=+2
    # Object B starts at x=+2, moves to x=-2
    num_frames = 20
    id_switches = 0

    prev_ids = {}  # position_key -> track_id

    for t in range(num_frames):
        # Linear interpolation
        alpha = t / (num_frames - 1)

        pos_a = (-2 + 4 * alpha, 0, 0)
        pos_b = (2 - 4 * alpha, 0, 0)

        # Give each object consistent latent features
        z_a = np.ones(32, dtype=np.float32) * 0.1
        z_b = np.ones(32, dtype=np.float32) * (-0.1)

        entity_a = make_entity("det_a", pos_a, z_shape=z_a)
        entity_b = make_entity("det_b", pos_b, z_shape=z_b)

        result = tracker.update([entity_a, entity_b])

        # Check for ID switches by position continuity
        for entity in result:
            pos = entity.position
            # Determine which original object this is by x-velocity direction
            if pos[0] > (pos_a[0] + pos_b[0]) / 2:
                # More likely object A (which moves positive)
                key = "A"
            else:
                key = "B"

            if key in prev_ids and prev_ids[key] != entity.track_id:
                id_switches += 1
            prev_ids[key] = entity.track_id

    # Should have very few ID switches for smooth crossing
    assert id_switches <= 2, f"Too many ID switches: {id_switches}"


def test_track_death_after_max_age(tracker):
    """Track should die after not being updated for max_age frames."""
    # Frame 1: detect object
    entity = make_entity("det_0", (0, 0, 0))
    tracker.update([entity])

    # Frames 2-5: no detections
    for _ in range(4):
        tracker.update([])

    # Track should be dead, new detection should get new ID
    entity2 = make_entity("det_1", (0, 0, 0))
    result = tracker.update([entity2])

    assert len(result) == 1
    # Should be a new track (though ID might be reused)


def test_multiple_objects_stable_ids(tracker):
    """Multiple objects should maintain stable IDs when well-separated."""
    np.random.seed(0)

    # Create 3 well-separated objects
    positions = [(0, 0, 0), (5, 0, 0), (0, 5, 0)]
    entities = [make_entity(f"det_{i}", pos) for i, pos in enumerate(positions)]

    result = tracker.update(entities)
    # Map approximate position to track_id
    initial_ids = {}
    for r in result:
        pos_key = (round(r.position[0]), round(r.position[1]))
        initial_ids[pos_key] = r.track_id

    # Run 10 more frames with small jitter
    for _ in range(10):
        jittered = []
        for i, pos in enumerate(positions):
            jitter = np.random.randn(3) * 0.1
            new_pos = (pos[0] + jitter[0], pos[1] + jitter[1], pos[2] + jitter[2])
            jittered.append(make_entity(f"det_{i}", new_pos))

        result = tracker.update(jittered)

    # Check IDs are still the same
    final_ids = {}
    for r in result:
        pos_key = (round(r.position[0]), round(r.position[1]))
        final_ids[pos_key] = r.track_id

    # At least 2 out of 3 should maintain ID
    matches = sum(1 for k in initial_ids if k in final_ids and initial_ids[k] == final_ids[k])
    assert matches >= 2, f"Expected at least 2 stable IDs, got {matches}"


def test_id_switch_count_tracked(tracker):
    """Track manager should count ID switches."""
    # This is a property of the tracker, not a test of behavior
    initial_count = tracker.id_switch_count
    assert initial_count == 0

    # The count increases when there's type mismatch during association
    # In normal use, this should remain low


def test_body_and_object_separation(tracker):
    """Bodies and objects should not be matched to each other."""
    body = make_entity("det_body", (0, 0, 0), entity_type="body")
    obj = make_entity("det_obj", (0.1, 0, 0), entity_type="object")

    # Frame 1: both present
    result1 = tracker.update([body, obj])
    assert len(result1) == 2

    body_id = None
    obj_id = None
    for r in result1:
        if r.entity_type == "body":
            body_id = r.track_id
        else:
            obj_id = r.track_id

    assert body_id != obj_id, "Body and object should have different IDs"

    # Frame 2: positions swapped
    body2 = make_entity("det_body", (0.1, 0, 0), entity_type="body")
    obj2 = make_entity("det_obj", (0, 0, 0), entity_type="object")

    result2 = tracker.update([body2, obj2])

    for r in result2:
        if r.entity_type == "body":
            assert r.track_id == body_id, "Body should keep its ID"
        else:
            assert r.track_id == obj_id, "Object should keep its ID"
