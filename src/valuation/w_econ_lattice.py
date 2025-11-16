#!/usr/bin/env python3
"""
Deep Lattice-Style Economic Weighting Network

Computes w_econ ∈ [0,1] from episode metrics using:
1. Per-feature monotonic 1D calibrators
2. Brick embeddings
3. Constrained MLP lattice layer (respects monotonicity)
4. Optional objective conditioning

Key insight: Different features have known monotonic relationships with value:
- ΔMPL ↑ → more valuable
- Δerror ↓ (so -Δerror ↑) → more valuable
- ΔEP (energy productivity) ↑ → more valuable
- novelty: non-monotonic (band-pass effect)
- brick_id: categorical embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MonotonicCalibrator(nn.Module):
    """
    1D piecewise-linear monotonic calibrator.

    Maps input to [0, 1] using learned keypoints with monotonicity constraint.

    For monotone increasing: output increases as input increases.
    For monotone decreasing: output decreases as input increases.
    """

    def __init__(self, n_keypoints=16, monotone='increasing', input_min=-1.0, input_max=1.0):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.monotone = monotone
        self.input_min = input_min
        self.input_max = input_max

        # Keypoint x-positions (evenly spaced)
        self.register_buffer(
            'keypoint_x',
            torch.linspace(input_min, input_max, n_keypoints)
        )

        # Keypoint y-values (learned, constrained to be monotonic)
        # Initialize linearly
        if monotone == 'increasing':
            init_y = torch.linspace(0, 1, n_keypoints)
        elif monotone == 'decreasing':
            init_y = torch.linspace(1, 0, n_keypoints)
        else:
            # Non-monotonic: random init
            init_y = torch.rand(n_keypoints)

        self.keypoint_y_raw = nn.Parameter(init_y)

    @property
    def keypoint_y(self):
        """Enforce monotonicity constraint on y values."""
        if self.monotone == 'increasing':
            # Cumulative sum of softplus ensures monotone increasing
            deltas = F.softplus(self.keypoint_y_raw)
            y = torch.cumsum(deltas, dim=0)
            # Normalize to [0, 1]
            y = y / (y[-1] + 1e-6)
            return y
        elif self.monotone == 'decreasing':
            # Reverse of increasing
            deltas = F.softplus(self.keypoint_y_raw)
            y = torch.cumsum(deltas, dim=0)
            y = y / (y[-1] + 1e-6)
            return 1.0 - y
        else:
            # Non-monotonic: just sigmoid to [0, 1]
            return torch.sigmoid(self.keypoint_y_raw)

    def forward(self, x):
        """
        Piecewise-linear interpolation.

        Args:
            x: (...,) input values

        Returns:
            y: (...,) calibrated outputs in [0, 1]
        """
        # Clamp to input range
        x_clamped = torch.clamp(x, self.input_min, self.input_max)

        # Get keypoint y values (with monotonicity constraint)
        kp_y = self.keypoint_y
        kp_x = self.keypoint_x

        # Find which segment each input falls into
        # Use searchsorted to find right boundary
        # Shape: (...)
        flat_x = x_clamped.flatten()
        indices_right = torch.searchsorted(kp_x, flat_x).clamp(1, self.n_keypoints - 1)
        indices_left = indices_right - 1

        # Get left and right keypoint values
        x_left = kp_x[indices_left]
        x_right = kp_x[indices_right]
        y_left = kp_y[indices_left]
        y_right = kp_y[indices_right]

        # Linear interpolation
        t = (flat_x - x_left) / (x_right - x_left + 1e-8)
        y = y_left + t * (y_right - y_left)

        return y.view_as(x_clamped)


class NonMonotonicCalibrator(nn.Module):
    """
    1D piecewise-linear calibrator without monotonicity constraint.

    Used for features like novelty that may have band-pass effects.
    """

    def __init__(self, n_keypoints=16, input_min=0.0, input_max=1.0):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.input_min = input_min
        self.input_max = input_max

        self.register_buffer(
            'keypoint_x',
            torch.linspace(input_min, input_max, n_keypoints)
        )

        # Initialize with bell curve (medium novelty is best)
        init_y = torch.exp(-4 * (torch.linspace(-1, 1, n_keypoints) ** 2))
        self.keypoint_y = nn.Parameter(init_y)

    def forward(self, x):
        """Piecewise-linear interpolation."""
        x_clamped = torch.clamp(x, self.input_min, self.input_max)

        # Constrain y to [0, 1]
        kp_y = torch.sigmoid(self.keypoint_y)
        kp_x = self.keypoint_x

        flat_x = x_clamped.flatten()
        indices_right = torch.searchsorted(kp_x, flat_x).clamp(1, self.n_keypoints - 1)
        indices_left = indices_right - 1

        x_left = kp_x[indices_left]
        x_right = kp_x[indices_right]
        y_left = kp_y[indices_left]
        y_right = kp_y[indices_right]

        t = (flat_x - x_left) / (x_right - x_left + 1e-8)
        y = y_left + t * (y_right - y_left)

        return y.view_as(x_clamped)


class ConstrainedMLP(nn.Module):
    """
    MLP with non-negative weight constraints on certain inputs.

    This approximates the monotonicity preservation in lattice networks:
    if input i is monotonic and weight_i >= 0, monotonicity is preserved.
    """

    def __init__(self, input_dim, hidden_dim=32, n_monotonic_inputs=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_monotonic_inputs = n_monotonic_inputs  # First n inputs are monotonic

        # First layer: constrained for monotonic inputs
        self.weight1_mono = nn.Parameter(torch.randn(hidden_dim, n_monotonic_inputs))
        self.weight1_free = nn.Parameter(torch.randn(hidden_dim, input_dim - n_monotonic_inputs))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))

        # Second layer: fully connected, no constraints
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass with positivity constraint on monotonic weights.

        Args:
            x: (batch, input_dim) - first n_monotonic_inputs are monotonic features

        Returns:
            out: (batch, 1)
        """
        # Split input into monotonic and free parts
        x_mono = x[:, :self.n_monotonic_inputs]
        x_free = x[:, self.n_monotonic_inputs:]

        # Apply positivity constraint to monotonic weights (softplus ensures >= 0)
        w1_mono_pos = F.softplus(self.weight1_mono)

        # First layer
        h = F.linear(x_mono, w1_mono_pos) + F.linear(x_free, self.weight1_free) + self.bias1
        h = F.relu(h)

        # Second layer
        out = self.fc2(h)
        return out


class WEconLattice(nn.Module):
    """
    Deep lattice-style economic weighting network.

    Computes w_econ ∈ [0, 1] from episode metrics with monotonicity constraints.

    Features:
    - ΔMPL: monotone increasing (higher MPL improvement = more valuable)
    - -Δerror: monotone increasing (lower error = more valuable)
    - ΔEP: monotone increasing (better energy productivity = more valuable)
    - novelty: non-monotonic (medium novelty may be optimal)
    - brick_id: categorical embedding
    - objective_vector: optional task conditioning
    """

    def __init__(
        self,
        n_bricks=10,
        brick_emb_dim=8,
        n_keypoints=16,
        hidden_dim=32,
        objective_dim=4,
    ):
        super().__init__()

        # Monotonic calibrators (input → [0, 1])
        self.cal_mpl = MonotonicCalibrator(
            n_keypoints=n_keypoints,
            monotone='increasing',
            input_min=-0.5,  # ΔMPL can be negative
            input_max=2.0,   # Up to 2 units/hr improvement
        )

        self.cal_neg_error = MonotonicCalibrator(
            n_keypoints=n_keypoints,
            monotone='increasing',
            input_min=-0.5,  # -Δerror: if error increased, this is negative
            input_max=0.5,   # Max 50% error reduction
        )

        self.cal_ep = MonotonicCalibrator(
            n_keypoints=n_keypoints,
            monotone='increasing',
            input_min=-0.5,  # ΔEP can be negative
            input_max=2.0,   # Up to 2 units/Wh improvement
        )

        # Non-monotonic calibrator for novelty
        self.cal_novelty = NonMonotonicCalibrator(
            n_keypoints=n_keypoints,
            input_min=0.0,
            input_max=1.0,
        )

        # Brick embedding
        self.brick_embedding = nn.Embedding(n_bricks, brick_emb_dim)
        self.brick_to_scalar = nn.Linear(brick_emb_dim, 1)

        # Objective conditioning (optional)
        self.objective_dim = objective_dim

        # Lattice/MLP combination layer
        # Input: [cal_mpl, cal_neg_error, cal_ep, cal_novelty, brick_scalar, objective_vector]
        # First 3 are monotonic, rest are free
        lattice_input_dim = 4 + 1 + objective_dim  # 3 monotonic cal + 1 novelty + 1 brick + obj
        self.lattice_mlp = ConstrainedMLP(
            input_dim=lattice_input_dim,
            hidden_dim=hidden_dim,
            n_monotonic_inputs=3,  # ΔMPL, -Δerror, ΔEP
        )

    def forward(self, delta_mpl, delta_error, delta_ep, novelty, brick_id, objective_vector=None):
        """
        Compute economic weight.

        Args:
            delta_mpl: (batch,) ΔMPL (improvement in units/hr)
            delta_error: (batch,) Δerror (change in error rate, positive = worse)
            delta_ep: (batch,) ΔEP (change in energy productivity MPL/Wh)
            novelty: (batch,) novelty score in [0, 1]
            brick_id: (batch,) brick indices
            objective_vector: (batch, objective_dim) optional task conditioning

        Returns:
            w_econ: (batch,) weights in [0, 1]
        """
        batch_size = delta_mpl.shape[0]

        # Calibrate monotonic features
        cal_mpl = self.cal_mpl(delta_mpl)
        cal_neg_error = self.cal_neg_error(-delta_error)  # Negate so increasing = better
        cal_ep = self.cal_ep(delta_ep)

        # Calibrate non-monotonic feature
        cal_nov = self.cal_novelty(novelty)

        # Brick embedding → scalar
        brick_emb = self.brick_embedding(brick_id)  # (batch, brick_emb_dim)
        brick_scalar = torch.sigmoid(self.brick_to_scalar(brick_emb).squeeze(-1))  # (batch,)

        # Objective conditioning (default to zeros if not provided)
        if objective_vector is None:
            objective_vector = torch.zeros(batch_size, self.objective_dim, device=delta_mpl.device)

        # Concatenate for lattice layer
        # Order: [monotonic features first, then free features]
        u = torch.stack([cal_mpl, cal_neg_error, cal_ep, cal_nov, brick_scalar], dim=1)
        u = torch.cat([u, objective_vector], dim=1)  # (batch, 5 + objective_dim)

        # Lattice MLP with monotonicity constraints
        logit = self.lattice_mlp(u).squeeze(-1)  # (batch,)

        # Final sigmoid to [0, 1]
        w_econ = torch.sigmoid(logit)

        return w_econ

    def get_calibrator_curves(self, n_points=100):
        """
        Get calibrator curves for visualization.

        Returns:
            dict of (x_values, y_values) for each calibrator
        """
        curves = {}

        # ΔMPL calibrator
        x_mpl = torch.linspace(-0.5, 2.0, n_points)
        with torch.no_grad():
            curves['delta_mpl'] = (x_mpl.numpy(), self.cal_mpl(x_mpl).numpy())

        # -Δerror calibrator (negate x for plotting)
        x_err = torch.linspace(-0.5, 0.5, n_points)
        with torch.no_grad():
            curves['neg_delta_error'] = (x_err.numpy(), self.cal_neg_error(x_err).numpy())

        # ΔEP calibrator
        x_ep = torch.linspace(-0.5, 2.0, n_points)
        with torch.no_grad():
            curves['delta_ep'] = (x_ep.numpy(), self.cal_ep(x_ep).numpy())

        # Novelty calibrator
        x_nov = torch.linspace(0.0, 1.0, n_points)
        with torch.no_grad():
            curves['novelty'] = (x_nov.numpy(), self.cal_novelty(x_nov).numpy())

        return curves


def compute_heuristic_teacher_weight(
    delta_mpl, delta_error, delta_ep, novelty, brick_id,
    w_mpl=0.4, w_error=0.3, w_ep=0.2, w_novelty=0.1
):
    """
    Compute heuristic teacher weight for training the lattice.

    This is a simple weighted combination that the lattice will learn to replicate.

    Args:
        delta_mpl: ΔMPL
        delta_error: Δerror (positive = worse)
        delta_ep: ΔEP (energy productivity improvement)
        novelty: novelty score [0, 1]
        brick_id: brick index (not used in simple heuristic)
        w_*: weight coefficients

    Returns:
        w_teacher: teacher weight in [0, 1]
    """
    # Normalize features to [0, 1] range
    # ΔMPL: assume range [-0.5, 2.0] → [0, 1]
    mpl_norm = np.clip((delta_mpl + 0.5) / 2.5, 0, 1)

    # -Δerror: assume range [-0.5, 0.5] → [0, 1]
    neg_error_norm = np.clip((-delta_error + 0.5) / 1.0, 0, 1)

    # ΔEP: assume range [-0.5, 2.0] → [0, 1]
    ep_norm = np.clip((delta_ep + 0.5) / 2.5, 0, 1)

    # Novelty already in [0, 1], but penalize extremes (band-pass)
    novelty_score = 4 * novelty * (1 - novelty)  # Peak at 0.5

    # Weighted combination
    w_teacher = (
        w_mpl * mpl_norm +
        w_error * neg_error_norm +
        w_ep * ep_norm +
        w_novelty * novelty_score
    )

    # Clip to [0, 1]
    w_teacher = np.clip(w_teacher, 0, 1)

    return w_teacher
