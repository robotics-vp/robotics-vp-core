"""
Potential-Based Reward Shaping (PBRS) for Process Reward.

Implements the core PBRS formula:
    r_shape[t] = gamma * Phi_star[t+1] - Phi_star[t]

This is guaranteed to preserve optimal policies (Ng et al., 1999).

Key properties:
- Sum of shaped rewards telescopes: sum_t r_shape[t] = gamma^T * Phi[T] - Phi[0]
- Policy-invariant: doesn't change optimal behavior
- Optional confidence gating: scale r_shape by confidence
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    FusionDiagnostics,
)


def compute_pbrs(
    phi_star: np.ndarray,
    gamma: float,
    confidence: Optional[np.ndarray] = None,
    use_confidence_gating: bool = False,
    gating_factor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute PBRS shaped reward.

    Args:
        phi_star: (T,) array of fused potentials.
        gamma: Discount factor in (0, 1].
        confidence: Optional (T,) array of confidences for gating.
        use_confidence_gating: If True, scale r_shape by confidence.
        gating_factor: Optional precomputed gating factor (T,).

    Returns:
        (T-1,) array of shaped rewards.
        r_shape[t] = gamma * phi_star[t+1] - phi_star[t]
    """
    T = len(phi_star)
    if T < 2:
        return np.array([], dtype=np.float32)

    # Core PBRS formula
    r_shape = gamma * phi_star[1:] - phi_star[:-1]

    # Apply confidence gating if requested
    # Use min(conf[t], conf[t+1]) so we don't pay reward when next-state potential is uncertain
    if use_confidence_gating:
        if gating_factor is not None:
            # Use precomputed gating factor (from fusion)
            # Take minimum to be conservative when either state is uncertain
            gate = np.minimum(gating_factor[:-1], gating_factor[1:])
        elif confidence is not None:
            # Use raw confidence as gate
            gate = np.minimum(confidence[:-1], confidence[1:])
        else:
            gate = np.ones(T - 1, dtype=np.float32)

        r_shape = r_shape * gate

    return r_shape.astype(np.float32)


def verify_pbrs_telescoping(
    phi_star: np.ndarray,
    r_shape: np.ndarray,
    gamma: float,
    tolerance: float = 1e-5,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that PBRS satisfies the telescoping property.

    The key identity for PBRS is in the *discounted* return:
        sum_{t=0}^{T-1} gamma^t * r'_t = -Phi[0] + gamma^T * Phi[T]

    Where r'_t = gamma * Phi[t+1] - Phi[t].

    For gamma=1 (undiscounted), this simplifies to:
        sum_t r'_t = Phi[T-1] - Phi[0]

    Args:
        phi_star: (T,) array of potentials.
        r_shape: (T-1,) array of shaped rewards.
        gamma: Discount factor.
        tolerance: Numerical tolerance for equality check.

    Returns:
        (is_valid, diagnostics) tuple.
    """
    T = len(phi_star)
    if T < 2:
        return True, {"message": "trivial case (T < 2)"}

    if gamma == 1.0:
        # Plain sum telescopes: sum = Phi[T-1] - Phi[0]
        actual = float(np.sum(r_shape))
        expected = float(phi_star[-1] - phi_star[0])
        error = abs(actual - expected)
        is_valid = error < tolerance
        diagnostics = {
            "mode": "undiscounted",
            "plain_sum": actual,
            "expected_sum": expected,
            "error": error,
            "tolerance": tolerance,
            "phi_0": float(phi_star[0]),
            "phi_T": float(phi_star[-1]),
        }
    else:
        # Discounted sum telescopes: sum_{t=0}^{T-2} gamma^t * r'_t = gamma^{T-1} * Phi[T-1] - Phi[0]
        T_minus_1 = len(r_shape)
        gamma_powers = np.power(gamma, np.arange(T_minus_1, dtype=np.float64))
        discounted_sum = float(np.sum(gamma_powers * r_shape.astype(np.float64)))
        expected = (gamma ** (T - 1)) * float(phi_star[-1]) - float(phi_star[0])
        error = abs(discounted_sum - expected)
        is_valid = error < tolerance
        diagnostics = {
            "mode": "discounted",
            "discounted_sum": discounted_sum,
            "expected_sum": expected,
            "error": error,
            "tolerance": tolerance,
            "phi_0": float(phi_star[0]),
            "phi_T": float(phi_star[-1]),
            "gamma_power_T_minus_1": gamma ** (T - 1),
        }

    return is_valid, diagnostics


def compute_pbrs_with_baseline(
    phi_star: np.ndarray,
    gamma: float,
    baseline: Optional[np.ndarray] = None,
    baseline_type: str = "mean",
) -> np.ndarray:
    """Compute PBRS with optional baseline subtraction.

    This can help with variance reduction in RL.

    Args:
        phi_star: (T,) array of potentials.
        gamma: Discount factor.
        baseline: Optional (T-1,) baseline to subtract.
        baseline_type: Type of baseline ("mean", "ema", "none").

    Returns:
        (T-1,) array of shaped rewards.
    """
    r_shape = compute_pbrs(phi_star, gamma)

    if baseline_type == "none" or baseline is not None:
        if baseline is not None:
            r_shape = r_shape - baseline
    elif baseline_type == "mean":
        r_shape = r_shape - np.mean(r_shape)
    elif baseline_type == "ema":
        # Exponential moving average baseline
        alpha = 0.1
        ema = np.zeros_like(r_shape)
        ema[0] = r_shape[0]
        for t in range(1, len(r_shape)):
            ema[t] = alpha * r_shape[t] + (1 - alpha) * ema[t - 1]
        r_shape = r_shape - ema

    return r_shape.astype(np.float32)


def compute_discounted_return(
    rewards: np.ndarray,
    gamma: float,
    normalize: bool = False,
) -> np.ndarray:
    """Compute discounted return from rewards.

    G[t] = sum_{k=0}^{T-t-1} gamma^k * r[t+k]

    Args:
        rewards: (T-1,) array of rewards.
        gamma: Discount factor.
        normalize: If True, normalize returns.

    Returns:
        (T-1,) array of discounted returns starting at each timestep.
    """
    T_minus_1 = len(rewards)
    if T_minus_1 == 0:
        return np.array([], dtype=np.float32)

    returns = np.zeros(T_minus_1, dtype=np.float32)

    # Compute returns in reverse order for efficiency
    running_return = 0.0
    for t in range(T_minus_1 - 1, -1, -1):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    if normalize and np.std(returns) > 1e-8:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    return returns


class PBRSWrapper:
    """Wrapper for PBRS computation with additional utilities.

    Handles:
    - Core PBRS computation
    - Confidence gating from FusionDiagnostics
    - Sanity checks (telescoping verification)
    - Episode-level statistics
    """

    def __init__(self, config: ProcessRewardConfig):
        """Initialize.

        Args:
            config: Process reward configuration.
        """
        self.config = config
        self.gamma = config.gamma
        self.use_confidence_gating = config.use_confidence_gating

    def compute(
        self,
        phi_star: np.ndarray,
        confidence: np.ndarray,
        diagnostics: Optional[FusionDiagnostics] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute PBRS shaped reward for an episode.

        Args:
            phi_star: (T,) fused potential.
            confidence: (T,) confidence values.
            diagnostics: Optional fusion diagnostics for gating.

        Returns:
            (r_shape, stats) tuple.
            r_shape is (T-1,) array of shaped rewards.
        """
        gating_factor = None
        if diagnostics is not None:
            gating_factor = diagnostics.gating_factor

        r_shape = compute_pbrs(
            phi_star,
            self.gamma,
            confidence=confidence,
            use_confidence_gating=self.use_confidence_gating,
            gating_factor=gating_factor,
        )

        # Verify telescoping (for ungated case)
        if not self.use_confidence_gating:
            is_valid, telescope_info = verify_pbrs_telescoping(
                phi_star, r_shape, self.gamma
            )
            if not is_valid:
                import warnings
                warnings.warn(
                    f"PBRS telescoping check failed: error={telescope_info['error']:.6f}"
                )
        else:
            is_valid = True
            telescope_info = {"gated": True}

        # Compute statistics
        stats = {
            "r_shape_sum": float(np.sum(r_shape)),
            "r_shape_mean": float(np.mean(r_shape)) if len(r_shape) > 0 else 0.0,
            "r_shape_std": float(np.std(r_shape)) if len(r_shape) > 0 else 0.0,
            "r_shape_min": float(np.min(r_shape)) if len(r_shape) > 0 else 0.0,
            "r_shape_max": float(np.max(r_shape)) if len(r_shape) > 0 else 0.0,
            "phi_star_init": float(phi_star[0]) if len(phi_star) > 0 else 0.0,
            "phi_star_final": float(phi_star[-1]) if len(phi_star) > 0 else 0.0,
            "phi_star_delta": float(phi_star[-1] - phi_star[0]) if len(phi_star) > 0 else 0.0,
            "confidence_mean": float(np.mean(confidence)),
            "confidence_min": float(np.min(confidence)),
            "telescoping_valid": is_valid,
            **{f"telescope_{k}": v for k, v in telescope_info.items() if isinstance(v, (int, float))},
        }

        return r_shape, stats

    def compute_step(
        self,
        phi_t: float,
        phi_t1: float,
        conf_t: float,
        conf_t1: float,
        gating_factor_t: float = 1.0,
        gating_factor_t1: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute PBRS for a single step (for online RL).

        Args:
            phi_t: Potential at t.
            phi_t1: Potential at t+1.
            conf_t: Confidence at t.
            conf_t1: Confidence at t+1.
            gating_factor_t: Gating factor at t.
            gating_factor_t1: Gating factor at t+1.

        Returns:
            (r_shape, debug_info) tuple.
        """
        r_shape = self.gamma * phi_t1 - phi_t

        if self.use_confidence_gating:
            # Use minimum to be conservative when either state is uncertain
            gate = min(gating_factor_t, gating_factor_t1)
            r_shape = r_shape * gate

        debug = {
            "phi_t": phi_t,
            "phi_t1": phi_t1,
            "r_shape_raw": self.gamma * phi_t1 - phi_t,
            "r_shape_gated": r_shape,
            "gate": min(gating_factor_t, gating_factor_t1) if self.use_confidence_gating else 1.0,
        }

        return float(r_shape), debug


def create_pbrs_wrapper(config: ProcessRewardConfig) -> PBRSWrapper:
    """Factory function to create PBRS wrapper.

    Args:
        config: Process reward configuration.

    Returns:
        PBRSWrapper instance.
    """
    return PBRSWrapper(config)
