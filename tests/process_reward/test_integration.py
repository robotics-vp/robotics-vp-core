"""
Integration tests for Process Reward module.

Verifies:
1. SceneTracks integration (synthetic T=5, K=2)
2. End-to-end process_reward_episode
3. Serialization stability
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from src.process_reward import (
    ProcessRewardConfig,
    process_reward_episode,
)
from src.process_reward.schemas import MHNSummary


def create_synthetic_scene_tracks_lite(
    T: int = 5,
    K: int = 2,
    include_latents: bool = False,
) -> "SceneTracksLite":
    """Create synthetic SceneTracksLite for testing.

    Args:
        T: Number of frames.
        K: Number of tracks.
        include_latents: Whether to include z_shape/z_tex.

    Returns:
        SceneTracksLite-like object.
    """
    from dataclasses import dataclass
    from typing import Optional

    np.random.seed(42)

    # Smooth trajectory
    t = np.linspace(0, np.pi, T)

    poses_R = np.tile(np.eye(3, dtype=np.float32), (T, K, 1, 1))
    poses_t = np.zeros((T, K, 3), dtype=np.float32)

    for k in range(K):
        poses_t[:, k, 0] = np.sin(t) * (k + 1) * 0.5
        poses_t[:, k, 1] = np.cos(t) * (k + 1) * 0.5
        poses_t[:, k, 2] = 1.0

    @dataclass
    class MockSceneTracksLite:
        track_ids: np.ndarray
        entity_types: np.ndarray
        class_ids: np.ndarray
        poses_R: np.ndarray
        poses_t: np.ndarray
        scales: np.ndarray
        visibility: np.ndarray
        occlusion: np.ndarray
        ir_loss: np.ndarray
        converged: np.ndarray
        z_shape: Optional[np.ndarray] = None
        z_tex: Optional[np.ndarray] = None

        @property
        def num_frames(self) -> int:
            return self.poses_R.shape[0]

        @property
        def num_tracks(self) -> int:
            return len(self.track_ids)

    z_shape = np.random.randn(T, K, 16).astype(np.float16) if include_latents else None
    z_tex = np.random.randn(T, K, 8).astype(np.float16) if include_latents else None

    # Create entity_types and class_ids that match K
    entity_types = np.zeros(K, dtype=np.int32)
    entity_types[0] = 1  # First track is body
    class_ids = np.full(K, -1, dtype=np.int32)
    if K > 1:
        class_ids[1] = 0

    return MockSceneTracksLite(
        track_ids=np.array([f"track_{i}" for i in range(K)], dtype="U32"),
        entity_types=entity_types,
        class_ids=class_ids,
        poses_R=poses_R,
        poses_t=poses_t,
        scales=np.ones((T, K), dtype=np.float32),
        visibility=np.ones((T, K), dtype=np.float32) * 0.9,
        occlusion=np.ones((T, K), dtype=np.float32) * 0.1,
        ir_loss=np.ones((T, K), dtype=np.float32) * 0.05,
        converged=np.ones((T, K), dtype=bool),
        z_shape=z_shape,
        z_tex=z_tex,
    )


class TestSceneTracksIntegration:
    """Test integration with SceneTracks_v1."""

    def test_tiny_scene_tracks_no_latents(self):
        """Run end-to-end with tiny synthetic data (T=5, K=2), no latents."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2, include_latents=False)

        cfg = ProcessRewardConfig(
            gamma=0.99,
            use_latents=False,
            feature_dim=16,
            online_mode=False,  # Offline eval allows hindsight
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test task",
            cfg=cfg,
        )

        # Verify output shapes
        assert result.phi_star.shape == (5,)
        assert result.conf.shape == (5,)
        assert result.r_shape.shape == (4,)  # T-1
        assert result.perspectives.phi_I.shape == (5,)
        assert result.diagnostics.weights.shape == (5, 3)

    def test_tiny_scene_tracks_with_latents(self):
        """Run end-to-end with tiny synthetic data (T=5, K=2), with latents."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2, include_latents=True)

        cfg = ProcessRewardConfig(
            gamma=0.99,
            use_latents=True,
            feature_dim=16,
            online_mode=False,  # Offline eval allows hindsight
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test task with latents",
            cfg=cfg,
        )

        assert result.phi_star.shape == (5,)
        assert result.r_shape.shape == (4,)

    def test_with_mhn_summary(self):
        """Run with MHN summary features."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2)

        mhn = MHNSummary(
            mean_tree_depth=2.5,
            mean_branch_factor=1.5,
            residual_mean=0.1,
            residual_std=0.05,
            structural_difficulty=3.0,
            plausibility_score=0.9,
        )

        cfg = ProcessRewardConfig(
            gamma=0.99,
            use_mhn_features=True,
            online_mode=False,  # Offline eval allows hindsight
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test with mhn",
            cfg=cfg,
            mhn=mhn,
        )

        assert result.phi_star.shape == (5,)
        assert "mhn_summary" in result.metadata

    def test_with_goal_frame(self):
        """Run with explicit goal frame index."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=10, K=2)

        cfg = ProcessRewardConfig(gamma=0.99)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="reach goal at frame 8",
            goal_frame_idx=8,
            cfg=cfg,
        )

        assert result.phi_star.shape == (10,)
        assert result.metadata["goal_frame_idx"] == 8


class TestSerializationStability:
    """Test output serialization stability."""

    def test_to_dict_json_round_trip(self):
        """Output should serialize to JSON without dtype drift."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2)
        cfg = ProcessRewardConfig(gamma=0.99, online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="serialization test",
            cfg=cfg,
            episode_id="test_episode",
        )

        # Serialize to dict
        result_dict = result.to_dict()

        # Serialize to JSON and back
        json_str = json.dumps(result_dict)
        loaded_dict = json.loads(json_str)

        # Verify key fields
        assert loaded_dict["episode_id"] == "test_episode"
        assert len(loaded_dict["phi_star"]) == 5
        assert len(loaded_dict["r_shape"]) == 4

        # Verify values match
        np.testing.assert_allclose(
            loaded_dict["phi_star"],
            result.phi_star.tolist(),
            rtol=1e-5,
        )

    def test_summary_consistent(self):
        """Summary should be consistent with full output."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=10, K=2)
        cfg = ProcessRewardConfig(gamma=0.99, online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="summary test",
            cfg=cfg,
        )

        summary = result.summary()

        assert summary["num_frames"] == 10
        assert summary["phi_star_mean"] == pytest.approx(np.mean(result.phi_star), rel=1e-5)
        assert summary["r_shape_sum"] == pytest.approx(np.sum(result.r_shape), rel=1e-5)

    def test_npz_save_load(self):
        """Output arrays should save/load via npz without issues."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2)
        cfg = ProcessRewardConfig(gamma=0.99, online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="npz test",
            cfg=cfg,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_output.npz"

            # Save key arrays
            np.savez(
                path,
                phi_star=result.phi_star,
                conf=result.conf,
                r_shape=result.r_shape,
                phi_I=result.perspectives.phi_I,
                phi_F=result.perspectives.phi_F,
                phi_B=result.perspectives.phi_B,
                weights=result.diagnostics.weights,
            )

            # Load and verify
            loaded = np.load(path)

            np.testing.assert_array_equal(loaded["phi_star"], result.phi_star)
            np.testing.assert_array_equal(loaded["r_shape"], result.r_shape)
            np.testing.assert_array_equal(loaded["weights"], result.diagnostics.weights)


class TestOnlineMode:
    """Test online_mode guard against hindsight constructs."""

    def test_online_mode_rejects_hindsight_goal(self):
        """online_mode=True should reject goal_frame_idx=None."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2)
        cfg = ProcessRewardConfig(online_mode=True)

        with pytest.raises(ValueError, match="online_mode=True"):
            process_reward_episode(
                scene_tracks=scene_tracks,
                instruction="test",
                goal_frame_idx=None,  # Hindsight!
                cfg=cfg,
            )

    def test_online_mode_accepts_explicit_goal(self):
        """online_mode=True should accept explicit goal_frame_idx."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=5, K=2)
        cfg = ProcessRewardConfig(online_mode=True)

        # Should not raise
        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            goal_frame_idx=4,  # Explicit goal
            cfg=cfg,
        )

        assert result.phi_star.shape == (5,)
        assert not result.metadata.get("goal_is_hindsight", True)


class TestOutputRanges:
    """Test that outputs are in expected ranges."""

    def test_phi_star_in_range(self):
        """Phi_star should be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=20, K=3)
        cfg = ProcessRewardConfig(
            gamma=0.99,
            phi_clip_min=0.0,
            phi_clip_max=1.0,
            online_mode=False,  # Offline eval allows hindsight
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="range test",
            cfg=cfg,
        )

        assert np.all(result.phi_star >= 0.0)
        assert np.all(result.phi_star <= 1.0)

    def test_confidence_in_range(self):
        """Confidence should be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=20, K=3)
        cfg = ProcessRewardConfig(gamma=0.99, online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="conf range test",
            cfg=cfg,
        )

        assert np.all(result.conf >= 0.0)
        assert np.all(result.conf <= 1.0)

    def test_weights_sum_to_one(self):
        """Fusion weights should sum to 1 at each timestep."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=10, K=2)
        cfg = ProcessRewardConfig(gamma=0.99, online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="weight test",
            cfg=cfg,
        )

        weight_sums = np.sum(result.diagnostics.weights, axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-4)

    def test_perspectives_in_range(self):
        """All perspectives should be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks_lite(T=10, K=2)
        cfg = ProcessRewardConfig(
            gamma=0.99,
            phi_clip_min=0.0,
            phi_clip_max=1.0,
            online_mode=False,  # Offline eval allows hindsight
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="perspective range test",
            cfg=cfg,
        )

        for phi in [result.perspectives.phi_I, result.perspectives.phi_F, result.perspectives.phi_B]:
            assert np.all(phi >= 0.0)
            assert np.all(phi <= 1.0)
