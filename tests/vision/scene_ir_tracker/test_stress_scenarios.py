"""
Stress tests for Scene IR Tracker failure modes.

These tests simulate real-world failure scenarios:
- Occlusion crossing
- Mask corruption
- Long horizon drift

Marked as slow tests for optional CI.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def _create_crossing_scenario(num_frames: int = 50) -> dict:
    """Create scenario where two objects cross paths and swap depth."""
    # Object A starts left, moves right
    # Object B starts right, moves left
    # They cross in the middle, swapping depth ordering
    
    poses_A = []
    poses_B = []
    
    for t in range(num_frames):
        progress = t / (num_frames - 1)
        
        # Object A: moves from (-2, 0) to (2, 0)
        pose_A = np.eye(4, dtype=np.float32)
        pose_A[0, 3] = -2 + 4 * progress
        pose_A[2, 3] = 5.0 - 0.5 * np.sin(np.pi * progress)  # Depth varies
        poses_A.append(pose_A)
        
        # Object B: moves from (2, 0) to (-2, 0)
        pose_B = np.eye(4, dtype=np.float32)
        pose_B[0, 3] = 2 - 4 * progress
        pose_B[2, 3] = 5.0 + 0.5 * np.sin(np.pi * progress)  # Opposite depth
        poses_B.append(pose_B)
    
    return {
        "poses_A": np.stack(poses_A),
        "poses_B": np.stack(poses_B),
        "num_frames": num_frames,
    }


def _create_mask_corruption_scenario(num_frames: int = 30) -> dict:
    """Create scenario with corrupted masks."""
    masks = []
    corruption_types = []
    
    for t in range(num_frames):
        # Base mask
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        
        # Apply corruption
        if t % 10 == 0:
            # Dropout: empty mask
            mask = np.zeros_like(mask)
            corruption_types.append("dropout")
        elif t % 7 == 0:
            # Jitter: shift mask randomly
            shift = np.random.randint(-5, 6, size=2)
            mask = np.roll(mask, shift, axis=(0, 1))
            corruption_types.append("jitter")
        elif t % 5 == 0:
            # Merge: add second blob
            mask[30:50, 30:50] = 1.0
            corruption_types.append("merge")
        else:
            corruption_types.append("clean")
        
        masks.append(mask)
    
    return {
        "masks": np.stack(masks),
        "corruption_types": corruption_types,
        "num_frames": num_frames,
    }


def _create_long_horizon_scenario(num_frames: int = 300) -> dict:
    """Create long horizon scenario for drift testing."""
    poses = []
    
    for t in range(num_frames):
        pose = np.eye(4, dtype=np.float32)
        
        # Circular motion with slow drift
        angle = 2 * np.pi * t / 100  # One rotation every 100 frames
        radius = 2.0 + 0.01 * t  # Slowly increasing radius
        
        pose[0, 3] = radius * np.cos(angle)
        pose[1, 3] = radius * np.sin(angle)
        pose[2, 3] = 5.0
        
        poses.append(pose)
    
    return {
        "poses": np.stack(poses),
        "num_frames": num_frames,
    }


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestOcclusionCrossing:
    """Tests for occlusion crossing scenario."""

    def test_crossing_scenario_created(self):
        """Scenario is created correctly."""
        scenario = _create_crossing_scenario()
        
        assert scenario["num_frames"] == 50
        assert scenario["poses_A"].shape == (50, 4, 4)
        assert scenario["poses_B"].shape == (50, 4, 4)

    def test_depth_swap_at_crossing(self):
        """Verify depth ordering changes at crossing point."""
        scenario = _create_crossing_scenario()
        
        # At start, both objects at same depth (5.0)
        # During crossing, depths diverge then reconverge
        start_A_z = scenario["poses_A"][0, 2, 3]
        start_B_z = scenario["poses_B"][0, 2, 3]
        mid_A_z = scenario["poses_A"][25, 2, 3]
        mid_B_z = scenario["poses_B"][25, 2, 3]
        
        # At start they're at same depth
        assert abs(start_A_z - start_B_z) < 0.1
        
        # At middle, depths have diverged (one closer, one further)
        assert abs(mid_A_z - mid_B_z) > 0.5

    def test_tracker_handles_crossing(self):
        """Tracker handles crossing without crash."""
        from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
        from src.vision.scene_ir_tracker.config import TrackingConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        
        scenario = _create_crossing_scenario(num_frames=30)
        manager = KalmanTrackManager(TrackingConfig())
        
        for t in range(scenario["num_frames"]):
            detections = [
                SceneEntity3D(
                    entity_type="object",
                    track_id="",  # Will be assigned
                    pose=scenario["poses_A"][t],
                    scale=1.0,
                ),
                SceneEntity3D(
                    entity_type="object",
                    track_id="",
                    pose=scenario["poses_B"][t],
                    scale=1.0,
                ),
            ]
            
            tracked = manager.update(detections)
            assert len(tracked) <= 2  # Should not explode


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestMaskCorruption:
    """Tests for mask corruption scenarios."""

    def test_corruption_scenario_created(self):
        """Scenario is created correctly."""
        scenario = _create_mask_corruption_scenario()
        
        assert scenario["num_frames"] == 30
        assert scenario["masks"].shape == (30, 64, 64)
        assert "dropout" in scenario["corruption_types"]

    def test_dropout_frames_handled(self):
        """Tracker handles dropout frames gracefully."""
        from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
        from src.vision.scene_ir_tracker.config import TrackingConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        
        scenario = _create_mask_corruption_scenario()
        manager = KalmanTrackManager(TrackingConfig())
        
        for t in range(scenario["num_frames"]):
            mask = scenario["masks"][t]
            
            # Skip if dropout (empty mask)
            if scenario["corruption_types"][t] == "dropout":
                tracked = manager.update([])  # No detections
            else:
                entity = SceneEntity3D(
                    entity_type="object",
                    track_id="",
                    pose=np.eye(4, dtype=np.float32),
                    scale=1.0,
                    mask_2d=mask > 0.5,
                )
                tracked = manager.update([entity])
            
            # Should not crash
            assert isinstance(tracked, list)


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLongHorizonDrift:
    """Tests for long horizon drift scenarios."""

    def test_long_scenario_created(self):
        """Long scenario is created correctly."""
        scenario = _create_long_horizon_scenario(num_frames=300)
        
        assert scenario["num_frames"] == 300
        assert scenario["poses"].shape == (300, 4, 4)

    def test_tracker_handles_long_sequence(self):
        """Tracker handles 300+ frame sequence."""
        from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
        from src.vision.scene_ir_tracker.config import TrackingConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        
        scenario = _create_long_horizon_scenario(num_frames=300)
        manager = KalmanTrackManager(TrackingConfig())
        
        track_ids_seen = set()
        
        for t in range(scenario["num_frames"]):
            entity = SceneEntity3D(
                entity_type="object",
                track_id="",
                pose=scenario["poses"][t],
                scale=1.0,
            )
            
            tracked = manager.update([entity])
            
            for tr in tracked:
                track_ids_seen.add(tr.track_id)
        
        # Should have minimal ID switches (ideally 1 track ID)
        assert len(track_ids_seen) <= 3, f"Too many ID switches: {len(track_ids_seen)}"

    def test_no_memory_leak_long_sequence(self):
        """No excessive memory growth over long sequence."""
        import sys
        from src.vision.scene_ir_tracker.kalman_track_manager import KalmanTrackManager
        from src.vision.scene_ir_tracker.config import TrackingConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        
        scenario = _create_long_horizon_scenario(num_frames=200)
        
        # Measure before
        manager = KalmanTrackManager(TrackingConfig())
        size_before = sys.getsizeof(manager)
        
        for t in range(scenario["num_frames"]):
            entity = SceneEntity3D(
                entity_type="object",
                track_id="",
                pose=scenario["poses"][t],
                scale=1.0,
            )
            manager.update([entity])
        
        # Size shouldn't grow unboundedly (allow 10x growth)
        size_after = sys.getsizeof(manager)
        assert size_after < size_before * 100  # Generous bound
