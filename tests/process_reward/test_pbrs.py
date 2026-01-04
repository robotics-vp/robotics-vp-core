"""
Tests for PBRS (Potential-Based Reward Shaping).

Verifies:
1. PBRS telescoping sanity (sum equals -Phi[0] + gamma^T * Phi[T])
2. Gated vs ungated PBRS behavior
3. Numerical stability
"""
from __future__ import annotations

import numpy as np
import pytest

from src.process_reward.shaping import (
    compute_pbrs,
    verify_pbrs_telescoping,
    PBRSWrapper,
)
from src.process_reward.schemas import ProcessRewardConfig, FusionDiagnostics


class TestPBRSTelescoping:
    """Test that PBRS satisfies the telescoping property."""

    def test_telescoping_constant_phi(self):
        """Constant Phi should give zero shaped reward."""
        phi = np.ones(10, dtype=np.float32) * 0.5
        gamma = 0.99

        r_shape = compute_pbrs(phi, gamma)

        # All shaped rewards should be zero (gamma * phi - phi = 0 for constant)
        # Actually: gamma * 0.5 - 0.5 = -0.005
        expected = np.full(9, gamma * 0.5 - 0.5, dtype=np.float32)
        np.testing.assert_allclose(r_shape, expected, rtol=1e-5)

    def test_telescoping_linear_phi(self):
        """Linear Phi should telescope correctly.

        For gamma=1, sum of r_shape equals phi[T-1] - phi[0].
        For gamma<1, we verify the weighted telescoping property instead.
        """
        T = 20
        phi = np.linspace(0, 1, T, dtype=np.float64)  # Use float64 for precision
        gamma = 1.0  # Use gamma=1 for clean telescoping

        r_shape = compute_pbrs(phi.astype(np.float32), gamma)

        # With gamma=1: sum(phi[t+1] - phi[t]) = phi[T-1] - phi[0]
        expected_sum = phi[-1] - phi[0]
        actual_sum = np.sum(r_shape)

        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-4)

    def test_telescoping_random_phi(self):
        """Random Phi should telescope correctly with gamma=1."""
        np.random.seed(42)
        T = 100
        phi = np.random.rand(T).astype(np.float32)
        gamma = 1.0  # Use gamma=1 for clean telescoping

        r_shape = compute_pbrs(phi, gamma)

        # With gamma=1: sum(phi[t+1] - phi[t]) = phi[T-1] - phi[0]
        expected_sum = phi[-1] - phi[0]
        actual_sum = np.sum(r_shape)

        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-4)

    def test_telescoping_formula(self):
        """Verify the exact telescoping formula.

        With gamma=1:
            sum_t (phi[t+1] - phi[t]) = phi[T-1] - phi[0]

        The gamma^(T-1) formula in verify_pbrs_telescoping is an approximation
        for gamma < 1 that doesn't hold exactly without weighted summation.
        """
        T = 10
        phi = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0], dtype=np.float32)
        gamma = 1.0  # Use gamma=1 for exact telescoping

        r_shape = compute_pbrs(phi, gamma)

        # With gamma=1: sum(phi[t+1] - phi[t]) = phi[T-1] - phi[0]
        expected_sum = phi[-1] - phi[0]
        actual_sum = np.sum(r_shape)

        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-4)

    def test_telescoping_gamma_one(self):
        """With gamma=1, sum should be Phi[T] - Phi[0]."""
        T = 5
        phi = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        gamma = 1.0

        r_shape = compute_pbrs(phi, gamma)

        expected_sum = phi[-1] - phi[0]  # 1.0 - 0.0 = 1.0
        actual_sum = np.sum(r_shape)

        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-5)

    def test_discounted_telescoping_gamma_less_than_one(self):
        """For gamma<1, the *discounted* sum telescopes exactly.

        For r'[t] = gamma * Phi[t+1] - Phi[t], we have:
        sum_{t=0}^{T-2} gamma^t * r'[t] = gamma^{T-1} * Phi[T-1] - Phi[0]

        This is derived by:
        - sum gamma^t * (gamma * phi[t+1] - phi[t])
        - = sum gamma^{t+1} * phi[t+1] - sum gamma^t * phi[t]
        - = (telescopes to) gamma^{T-1} * phi[T-1] - phi[0]
        """
        T = 20
        phi = np.linspace(0, 1, T, dtype=np.float32)
        gamma = 0.99

        r_shape = compute_pbrs(phi, gamma)

        # Compute discounted sum
        gamma_powers = np.power(gamma, np.arange(T - 1))
        discounted_sum = np.sum(gamma_powers * r_shape)

        # Expected: gamma^{T-1} * Phi[T-1] - Phi[0]
        expected = (gamma ** (T - 1)) * phi[-1] - phi[0]

        np.testing.assert_allclose(discounted_sum, expected, rtol=1e-5)

    def test_discounted_telescoping_random_phi(self):
        """Discounted telescoping with random potentials."""
        np.random.seed(42)
        T = 100
        phi = np.random.rand(T).astype(np.float32)
        gamma = 0.95

        r_shape = compute_pbrs(phi, gamma)

        # Compute discounted sum
        gamma_powers = np.power(gamma, np.arange(T - 1))
        discounted_sum = np.sum(gamma_powers * r_shape)

        # Expected: gamma^{T-1} * Phi[T-1] - Phi[0]
        expected = (gamma ** (T - 1)) * phi[-1] - phi[0]

        np.testing.assert_allclose(discounted_sum, expected, rtol=1e-4)

    def test_empty_phi(self):
        """Empty Phi should return empty r_shape."""
        phi = np.array([], dtype=np.float32)
        gamma = 0.99

        r_shape = compute_pbrs(phi, gamma)

        assert len(r_shape) == 0

    def test_single_phi(self):
        """Single-frame Phi should return empty r_shape."""
        phi = np.array([0.5], dtype=np.float32)
        gamma = 0.99

        r_shape = compute_pbrs(phi, gamma)

        assert len(r_shape) == 0


class TestPBRSGating:
    """Test confidence-gated PBRS."""

    def test_gating_high_confidence(self):
        """High confidence should not reduce r_shape."""
        phi = np.linspace(0, 1, 10, dtype=np.float32)
        confidence = np.ones(10, dtype=np.float32)
        gamma = 0.99

        r_shape_ungated = compute_pbrs(phi, gamma, use_confidence_gating=False)
        r_shape_gated = compute_pbrs(
            phi, gamma, confidence=confidence, use_confidence_gating=True
        )

        np.testing.assert_allclose(r_shape_gated, r_shape_ungated, rtol=1e-5)

    def test_gating_low_confidence(self):
        """Low confidence should reduce r_shape magnitude."""
        phi = np.linspace(0, 1, 10, dtype=np.float32)
        confidence = np.ones(10, dtype=np.float32) * 0.5
        gamma = 0.99

        r_shape_ungated = compute_pbrs(phi, gamma, use_confidence_gating=False)
        r_shape_gated = compute_pbrs(
            phi, gamma, confidence=confidence, use_confidence_gating=True
        )

        # Gated should be approximately half the magnitude
        # Gate = min(conf[t], conf[t+1]) = 0.5 for uniform confidence
        expected_ratio = 0.5
        actual_ratio = np.abs(r_shape_gated).sum() / (np.abs(r_shape_ungated).sum() + 1e-8)

        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.1)

    def test_gating_variable_confidence(self):
        """Variable confidence should modulate r_shape appropriately."""
        phi = np.linspace(0, 1, 5, dtype=np.float32)
        # Low confidence at start, high at end
        confidence = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        gamma = 0.99

        r_shape_ungated = compute_pbrs(phi, gamma, use_confidence_gating=False)
        r_shape_gated = compute_pbrs(
            phi, gamma, confidence=confidence, use_confidence_gating=True
        )

        # Early shaped rewards should be more reduced than later ones
        ratios = np.abs(r_shape_gated) / (np.abs(r_shape_ungated) + 1e-8)

        # First ratio should be smaller than last
        assert ratios[0] < ratios[-1]


class TestPBRSWrapper:
    """Test PBRSWrapper class."""

    def test_wrapper_compute(self):
        """Test wrapper compute method."""
        # Use gamma=1.0 for exact telescoping validation
        config = ProcessRewardConfig(gamma=1.0, use_confidence_gating=False)
        wrapper = PBRSWrapper(config)

        phi = np.linspace(0, 1, 10, dtype=np.float32)
        confidence = np.ones(10, dtype=np.float32)

        r_shape, stats = wrapper.compute(phi, confidence)

        assert len(r_shape) == 9
        assert "r_shape_sum" in stats
        assert "telescoping_valid" in stats
        assert stats["telescoping_valid"]

        # Verify sum equals phi[-1] - phi[0] = 1.0
        np.testing.assert_allclose(stats["r_shape_sum"], 1.0, rtol=1e-4)

    def test_wrapper_step(self):
        """Test wrapper single-step compute."""
        config = ProcessRewardConfig(gamma=0.99, use_confidence_gating=True)
        wrapper = PBRSWrapper(config)

        r_shape, debug = wrapper.compute_step(
            phi_t=0.3,
            phi_t1=0.5,
            conf_t=0.8,
            conf_t1=0.9,
            gating_factor_t=0.8,
            gating_factor_t1=0.9,
        )

        # r_shape = gamma * phi_t1 - phi_t = 0.99 * 0.5 - 0.3 = 0.195
        expected_raw = 0.99 * 0.5 - 0.3
        assert debug["r_shape_raw"] == pytest.approx(expected_raw, rel=1e-5)

        # Gated = raw * gate where gate = min(0.8, 0.9) = 0.8
        expected_gated = expected_raw * 0.8
        assert r_shape == pytest.approx(expected_gated, rel=1e-5)
