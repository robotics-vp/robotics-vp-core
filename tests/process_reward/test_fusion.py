"""
Tests for Fusion module.

Verifies:
1. Fusion respects candidate_mask
2. Confidence decreases with disagreement
3. FusionOverride parameters work correctly
"""
from __future__ import annotations

import numpy as np
import pytest

from src.process_reward.schemas import (
    ProcessRewardConfig,
    FusionOverride,
    ProgressPerspectives,
    MHNSummary,
)
from src.process_reward.fusion import (
    HeuristicFusion,
    build_context_features,
)
from src.process_reward.progress_perspectives import (
    compute_perspective_disagreement,
    compute_perspective_entropy,
)


@pytest.fixture
def sample_perspectives() -> ProgressPerspectives:
    """Create sample perspectives for testing."""
    T = 10
    return ProgressPerspectives(
        phi_I=np.linspace(0.0, 0.8, T, dtype=np.float32),
        phi_F=np.linspace(0.0, 0.9, T, dtype=np.float32),
        phi_B=np.linspace(0.0, 1.0, T, dtype=np.float32),
        conf_I=np.ones(T, dtype=np.float32) * 0.8,
        conf_F=np.ones(T, dtype=np.float32) * 0.9,
        conf_B=np.ones(T, dtype=np.float32) * 0.7,
    )


@pytest.fixture
def sample_context() -> np.ndarray:
    """Create sample context features."""
    T = 10
    return np.ones((T, 8), dtype=np.float32) * 0.5


class TestCandidateMask:
    """Test that fusion respects candidate_mask."""

    def test_mask_phi_b(self, sample_perspectives, sample_context):
        """Masking Phi_B should give it near-zero weight."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        # Mask out Phi_B
        override = FusionOverride(
            candidate_mask=(True, True, False),
            temperature=1.0,
        )

        phi_star, conf, diagnostics = fusion.fuse_perspectives(
            sample_perspectives,
            sample_context,
            override=override,
        )

        # Phi_B weight should be near min_weight_floor (since mask enforces zeros)
        # After normalization, masked candidate gets min_weight_floor
        max_phi_b_weight = np.max(diagnostics.weights[:, 2])
        assert max_phi_b_weight < 0.1, f"Phi_B weight should be near zero: {max_phi_b_weight}"

    def test_mask_phi_i(self, sample_perspectives, sample_context):
        """Masking Phi_I should give it near-zero weight."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        override = FusionOverride(
            candidate_mask=(False, True, True),
        )

        phi_star, conf, diagnostics = fusion.fuse_perspectives(
            sample_perspectives,
            sample_context,
            override=override,
        )

        max_phi_i_weight = np.max(diagnostics.weights[:, 0])
        assert max_phi_i_weight < 0.1

    def test_mask_two_candidates(self, sample_perspectives, sample_context):
        """Masking two candidates should give remaining one most weight."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        # Only Phi_F enabled
        override = FusionOverride(
            candidate_mask=(False, True, False),
        )

        phi_star, conf, diagnostics = fusion.fuse_perspectives(
            sample_perspectives,
            sample_context,
            override=override,
        )

        # Phi_F should have dominant weight
        mean_phi_f_weight = np.mean(diagnostics.weights[:, 1])
        assert mean_phi_f_weight > 0.8

        # phi_star should closely follow Phi_F
        np.testing.assert_allclose(phi_star, sample_perspectives.phi_F, rtol=0.2)


class TestConfidenceDisagreement:
    """Test that confidence decreases with disagreement."""

    def test_confidence_decreases_with_disagreement(self):
        """Higher disagreement should lead to lower confidence."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)
        T = 10
        context = np.ones((T, 8), dtype=np.float32) * 0.5

        # Case 1: Low disagreement (all perspectives agree)
        perspectives_agree = ProgressPerspectives(
            phi_I=np.linspace(0.0, 1.0, T, dtype=np.float32),
            phi_F=np.linspace(0.0, 1.0, T, dtype=np.float32),
            phi_B=np.linspace(0.0, 1.0, T, dtype=np.float32),
            conf_I=np.ones(T, dtype=np.float32) * 0.9,
            conf_F=np.ones(T, dtype=np.float32) * 0.9,
            conf_B=np.ones(T, dtype=np.float32) * 0.9,
        )

        _, conf_agree, _ = fusion.fuse_perspectives(perspectives_agree, context)

        # Case 2: High disagreement (perspectives diverge)
        perspectives_disagree = ProgressPerspectives(
            phi_I=np.linspace(0.0, 0.3, T, dtype=np.float32),
            phi_F=np.linspace(0.0, 0.6, T, dtype=np.float32),
            phi_B=np.linspace(0.0, 1.0, T, dtype=np.float32),
            conf_I=np.ones(T, dtype=np.float32) * 0.9,
            conf_F=np.ones(T, dtype=np.float32) * 0.9,
            conf_B=np.ones(T, dtype=np.float32) * 0.9,
        )

        _, conf_disagree, _ = fusion.fuse_perspectives(perspectives_disagree, context)

        # Mean confidence should be lower when perspectives disagree
        assert np.mean(conf_agree) > np.mean(conf_disagree), \
            f"Agreement conf {np.mean(conf_agree):.3f} should > disagreement conf {np.mean(conf_disagree):.3f}"

    def test_disagreement_monotonic(self):
        """Confidence should monotonically decrease with increasing disagreement."""
        T = 10
        disagree_levels = [0.0, 0.2, 0.4, 0.6, 0.8]
        mean_confs = []

        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)
        context = np.ones((T, 8), dtype=np.float32) * 0.5

        for d in disagree_levels:
            perspectives = ProgressPerspectives(
                phi_I=np.linspace(0.0, 0.5 - d/2, T, dtype=np.float32),
                phi_F=np.linspace(0.0, 0.5, T, dtype=np.float32),
                phi_B=np.linspace(0.0, 0.5 + d/2, T, dtype=np.float32),
                conf_I=np.ones(T, dtype=np.float32) * 0.9,
                conf_F=np.ones(T, dtype=np.float32) * 0.9,
                conf_B=np.ones(T, dtype=np.float32) * 0.9,
            )

            _, conf, _ = fusion.fuse_perspectives(perspectives, context)
            mean_confs.append(np.mean(conf))

        # Should be monotonically decreasing (or at least non-increasing)
        for i in range(len(mean_confs) - 1):
            assert mean_confs[i] >= mean_confs[i + 1] - 0.01, \
                f"Confidence should decrease: {mean_confs[i]:.3f} -> {mean_confs[i+1]:.3f}"


class TestFusionOverride:
    """Test FusionOverride parameters."""

    def test_temperature_sharpens_weights(self, sample_perspectives, sample_context):
        """Lower temperature should sharpen weight distribution."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        # High temperature (soft)
        override_soft = FusionOverride(temperature=2.0)
        _, _, diag_soft = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_soft
        )

        # Low temperature (sharp)
        override_sharp = FusionOverride(temperature=0.5)
        fusion.reset_temporal_state()
        _, _, diag_sharp = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_sharp
        )

        # Sharp weights should have higher max and lower entropy
        max_weight_soft = np.max(diag_soft.weights, axis=1).mean()
        max_weight_sharp = np.max(diag_sharp.weights, axis=1).mean()

        assert max_weight_sharp > max_weight_soft

    def test_risk_tolerance_affects_gating(self, sample_perspectives, sample_context):
        """Risk tolerance should affect gating factor."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        # High risk tolerance (stricter)
        override_strict = FusionOverride(risk_tolerance=0.8)
        _, conf, diag_strict = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_strict
        )

        # Low risk tolerance (lenient)
        override_lenient = FusionOverride(risk_tolerance=0.2)
        fusion.reset_temporal_state()
        _, _, diag_lenient = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_lenient
        )

        # Strict should have more timesteps with reduced gating
        low_gate_strict = np.sum(diag_strict.gating_factor < 1.0)
        low_gate_lenient = np.sum(diag_lenient.gating_factor < 1.0)

        assert low_gate_strict >= low_gate_lenient

    def test_weight_smoothing(self, sample_perspectives, sample_context):
        """Weight smoothing should reduce temporal variance."""
        config = ProcessRewardConfig()
        fusion = HeuristicFusion(config)

        # No smoothing
        override_no_smooth = FusionOverride(weight_smoothing=0.0)
        _, _, diag_no_smooth = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_no_smooth
        )

        # High smoothing
        override_smooth = FusionOverride(weight_smoothing=0.9)
        fusion.reset_temporal_state()
        _, _, diag_smooth = fusion.fuse_perspectives(
            sample_perspectives, sample_context, override=override_smooth
        )

        # Smoothed weights should have lower temporal variance
        var_no_smooth = np.var(diag_no_smooth.weights)
        var_smooth = np.var(diag_smooth.weights)

        # Note: For the first call, smoothing has limited effect
        # This test verifies the mechanism works
        assert var_smooth <= var_no_smooth + 0.01


class TestPerspectiveMetrics:
    """Test perspective disagreement and entropy computation."""

    def test_disagreement_zero_for_identical(self):
        """Identical perspectives should have zero disagreement."""
        T = 5
        phi = np.linspace(0, 1, T, dtype=np.float32)
        perspectives = ProgressPerspectives(
            phi_I=phi.copy(),
            phi_F=phi.copy(),
            phi_B=phi.copy(),
            conf_I=np.ones(T, dtype=np.float32),
            conf_F=np.ones(T, dtype=np.float32),
            conf_B=np.ones(T, dtype=np.float32),
        )

        disagreement = compute_perspective_disagreement(perspectives)
        np.testing.assert_allclose(disagreement, 0.0, atol=1e-6)

    def test_disagreement_increases_with_spread(self):
        """Disagreement should increase as perspectives spread apart."""
        T = 5
        base = 0.5

        # Small spread
        perspectives_small = ProgressPerspectives(
            phi_I=np.full(T, base - 0.1, dtype=np.float32),
            phi_F=np.full(T, base, dtype=np.float32),
            phi_B=np.full(T, base + 0.1, dtype=np.float32),
            conf_I=np.ones(T, dtype=np.float32),
            conf_F=np.ones(T, dtype=np.float32),
            conf_B=np.ones(T, dtype=np.float32),
        )

        # Large spread
        perspectives_large = ProgressPerspectives(
            phi_I=np.full(T, base - 0.4, dtype=np.float32),
            phi_F=np.full(T, base, dtype=np.float32),
            phi_B=np.full(T, base + 0.4, dtype=np.float32),
            conf_I=np.ones(T, dtype=np.float32),
            conf_F=np.ones(T, dtype=np.float32),
            conf_B=np.ones(T, dtype=np.float32),
        )

        disag_small = compute_perspective_disagreement(perspectives_small)
        disag_large = compute_perspective_disagreement(perspectives_large)

        assert np.mean(disag_large) > np.mean(disag_small)

    def test_entropy_max_for_uniform_confidence(self):
        """Uniform confidences should have maximum entropy."""
        T = 5
        perspectives = ProgressPerspectives(
            phi_I=np.zeros(T, dtype=np.float32),
            phi_F=np.zeros(T, dtype=np.float32),
            phi_B=np.zeros(T, dtype=np.float32),
            conf_I=np.ones(T, dtype=np.float32),
            conf_F=np.ones(T, dtype=np.float32),
            conf_B=np.ones(T, dtype=np.float32),
        )

        entropy = compute_perspective_entropy(perspectives)

        # Max entropy for 3 outcomes = log(3) ≈ 1.099
        expected_max = np.log(3)
        np.testing.assert_allclose(entropy, expected_max, rtol=0.01)
