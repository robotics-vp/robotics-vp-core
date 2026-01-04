"""
Regression Gates for Process Reward Module.

These tests ensure that process reward metrics stay within expected bounds
across code changes. They serve as CI gates to catch regressions.

Regression gates verify:
1. Output ranges are valid (phi_star in [0,1], weights sum to 1, etc.)
2. PBRS telescoping property holds
3. Confidence gating behaves correctly
4. OrchestratorPolicy applies changes correctly
5. Logging integration produces expected fields
"""
from __future__ import annotations

import numpy as np
import pytest

from src.process_reward import (
    ProcessRewardConfig,
    FusionOverride,
    process_reward_episode,
)
from src.process_reward.logging_utils import (
    OrchestratorPolicy,
    extract_log_entry,
    format_log_for_training,
)
from src.process_reward.shaping import verify_pbrs_telescoping


def create_synthetic_scene_tracks(T: int = 10, K: int = 2):
    """Create synthetic scene tracks for regression testing."""
    from dataclasses import dataclass
    from typing import Optional

    np.random.seed(42)  # Fixed seed for reproducibility

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

    entity_types = np.zeros(K, dtype=np.int32)
    entity_types[0] = 1
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
    )


class TestOutputRangeRegression:
    """Regression gates for output value ranges."""

    def test_phi_star_range(self):
        """GATE: phi_star must be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks(T=20, K=3)
        cfg = ProcessRewardConfig(
            online_mode=False,
            phi_clip_min=0.0,
            phi_clip_max=1.0,
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        assert np.all(result.phi_star >= 0.0), "phi_star has values below 0"
        assert np.all(result.phi_star <= 1.0), "phi_star has values above 1"

    def test_confidence_range(self):
        """GATE: confidence must be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks(T=20, K=3)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        assert np.all(result.conf >= 0.0), "confidence has values below 0"
        assert np.all(result.conf <= 1.0), "confidence has values above 1"

    def test_weights_sum_to_one(self):
        """GATE: fusion weights must sum to 1 at each timestep."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        weight_sums = np.sum(result.diagnostics.weights, axis=1)
        np.testing.assert_allclose(
            weight_sums, 1.0, rtol=1e-4,
            err_msg="Fusion weights do not sum to 1"
        )

    def test_gating_factor_range(self):
        """GATE: gating_factor must be in [0, 1]."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        gating = result.diagnostics.gating_factor
        assert np.all(gating >= 0.0), "gating_factor has values below 0"
        assert np.all(gating <= 1.0), "gating_factor has values above 1"


class TestPBRSRegression:
    """Regression gates for PBRS properties."""

    def test_telescoping_undiscounted(self):
        """GATE: PBRS telescopes correctly for gamma=1."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(
            gamma=1.0,
            online_mode=False,
            use_confidence_gating=False,
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        is_valid, info = verify_pbrs_telescoping(
            result.phi_star, result.r_shape, gamma=1.0
        )
        assert is_valid, f"PBRS telescoping failed: {info}"

    def test_telescoping_discounted(self):
        """GATE: PBRS telescopes correctly for gamma < 1."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(
            gamma=0.99,
            online_mode=False,
            use_confidence_gating=False,
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        # Use ungated r_shape for verification
        r_shape_ungated = cfg.gamma * result.phi_star[1:] - result.phi_star[:-1]
        is_valid, info = verify_pbrs_telescoping(
            result.phi_star, r_shape_ungated, gamma=cfg.gamma
        )
        assert is_valid, f"PBRS telescoping failed: {info}"

    def test_r_shape_length(self):
        """GATE: r_shape must have length T-1."""
        scene_tracks = create_synthetic_scene_tracks(T=15, K=2)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        assert len(result.r_shape) == 14, f"r_shape length {len(result.r_shape)} != T-1 (14)"


class TestOrchestratorPolicyRegression:
    """Regression gates for OrchestratorPolicy behavior."""

    def test_high_occlusion_increases_temperature(self):
        """GATE: High occlusion should increase temperature."""
        policy = OrchestratorPolicy()
        base = FusionOverride(temperature=1.0)

        adjusted, adjustments = policy.apply(
            base,
            mean_occlusion=0.7,  # Above threshold
            ir_loss_mean=0.0,
            mhn_plausibility=1.0,
        )

        assert adjusted.temperature > base.temperature, \
            "Temperature not increased for high occlusion"
        assert "high_occlusion" in adjustments

    def test_low_plausibility_caps_confidence(self):
        """GATE: Low MHN plausibility should cap confidence."""
        policy = OrchestratorPolicy(low_plausibility_conf_cap=0.3)
        base = FusionOverride(confidence_cap=1.0)

        adjusted, adjustments = policy.apply(
            base,
            mean_occlusion=0.0,
            ir_loss_mean=0.0,
            mhn_plausibility=0.3,  # Below threshold
        )

        assert adjusted.confidence_cap == 0.3, \
            f"confidence_cap not applied: {adjusted.confidence_cap}"
        assert "low_plausibility" in adjustments

    def test_confidence_cap_applied_in_fusion(self):
        """GATE: confidence_cap should actually limit confidence values."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)
        override = FusionOverride(confidence_cap=0.5)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
            orchestrator_overrides=override,
        )

        assert np.all(result.conf <= 0.5), \
            f"Confidence exceeds cap: max={result.conf.max()}"


class TestOnlineModeRegression:
    """Regression gates for online_mode behavior."""

    def test_online_mode_rejects_hindsight(self):
        """GATE: online_mode=True must reject goal_frame_idx=None."""
        scene_tracks = create_synthetic_scene_tracks(T=5, K=2)
        cfg = ProcessRewardConfig(online_mode=True)

        with pytest.raises(ValueError, match="online_mode=True"):
            process_reward_episode(
                scene_tracks=scene_tracks,
                instruction="test",
                goal_frame_idx=None,
                cfg=cfg,
            )

    def test_online_mode_accepts_explicit_goal(self):
        """GATE: online_mode=True must accept explicit goal_frame_idx."""
        scene_tracks = create_synthetic_scene_tracks(T=5, K=2)
        cfg = ProcessRewardConfig(online_mode=True)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            goal_frame_idx=4,
            cfg=cfg,
        )

        assert result.phi_star.shape == (5,)
        assert not result.metadata.get("goal_is_hindsight", True)


class TestLoggingIntegrationRegression:
    """Regression gates for logging integration."""

    def test_log_entry_has_required_fields(self):
        """GATE: ProcessRewardLogEntry must have all required fields."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
            episode_id="test_episode",
        )

        log_entry = extract_log_entry(result)
        log_dict = log_entry.to_dict()

        required_fields = [
            "phi_star_mean",
            "phi_star_final",
            "phi_star_delta",
            "conf_mean",
            "r_shape_sum",
            "disagreement_mean",
            "entropy_mean",
            "num_frames",
        ]

        for field in required_fields:
            assert field in log_dict, f"Missing required field: {field}"

    def test_format_log_for_training_integrates_upstream(self):
        """GATE: format_log_for_training must include upstream metrics."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            cfg=cfg,
        )

        log = format_log_for_training(
            result=result,
            scene_ir_quality=0.85,
            motion_quality=0.9,
        )

        assert log["scene_ir_quality"] == 0.85
        assert log["motion_quality"] == 0.9

    def test_fusion_override_serialization_roundtrip(self):
        """GATE: FusionOverride must serialize/deserialize correctly."""
        original = FusionOverride(
            temperature=1.5,
            candidate_mask=(True, False, True),
            risk_tolerance=0.4,
            entropy_penalty=0.2,
            weight_smoothing=0.1,
            min_weight_floor=0.02,
            confidence_cap=0.8,
        )

        serialized = original.to_dict()
        restored = FusionOverride.from_dict(serialized)

        assert restored.temperature == original.temperature
        assert restored.candidate_mask == original.candidate_mask
        assert restored.risk_tolerance == original.risk_tolerance
        assert restored.entropy_penalty == original.entropy_penalty
        assert restored.weight_smoothing == original.weight_smoothing
        assert restored.min_weight_floor == original.min_weight_floor
        assert restored.confidence_cap == original.confidence_cap


class TestMaskBehaviorRegression:
    """Regression gates for candidate mask behavior."""

    def test_masked_candidate_has_zero_weight(self):
        """GATE: Masked candidates must have ~0 weight."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)
        override = FusionOverride(candidate_mask=(True, True, False))  # Mask Phi_B

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            goal_frame_idx=9,  # Explicit goal to avoid auto-masking
            cfg=cfg,
            orchestrator_overrides=override,
        )

        phi_B_weights = result.diagnostics.weights[:, 2]
        assert np.allclose(phi_B_weights, 0.0, atol=1e-6), \
            f"Masked Phi_B has non-zero weights: max={phi_B_weights.max()}"

    def test_floor_cannot_resurrect_masked(self):
        """GATE: min_weight_floor must not resurrect masked candidates."""
        scene_tracks = create_synthetic_scene_tracks(T=10, K=2)
        cfg = ProcessRewardConfig(online_mode=False)
        override = FusionOverride(
            candidate_mask=(True, True, False),
            min_weight_floor=0.1,  # High floor
        )

        result = process_reward_episode(
            scene_tracks=scene_tracks,
            instruction="test",
            goal_frame_idx=9,
            cfg=cfg,
            orchestrator_overrides=override,
        )

        phi_B_weights = result.diagnostics.weights[:, 2]
        assert np.allclose(phi_B_weights, 0.0, atol=1e-6), \
            "min_weight_floor resurrected masked candidate"
