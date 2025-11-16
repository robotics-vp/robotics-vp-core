#!/usr/bin/env python3
"""
Learned Controller for Synthetic Data Share (λ_synth)

This DL controller replaces hardcoded synthetic share knobs by learning
from actual physics/econ outcomes. It takes evaluation window features
and outputs λ_synth ∈ [0, max_synth_share].

The controller is trained via meta-objectives computed from real experimental
results, not heuristic teachers.

Architecture:
- Input: 11 features (objective_vector + normalized metrics + trust stats + progress)
- Hidden: 2 layers, 16 units, ReLU
- Output: sigmoid → multiply by max_synth_share
"""

import torch
import torch.nn as nn
import numpy as np


class SynthLambdaController(nn.Module):
    """
    Tiny MLP that predicts optimal synthetic data share λ_synth.

    Features (11 dims):
    - objective_vector (4 dims): [α_mpl, α_error, α_ep, α_novelty]
    - delta_mpl (1 dim): normalized MPL improvement vs baseline
    - delta_error (1 dim): normalized error change vs baseline
    - delta_ep (1 dim): normalized energy productivity change vs baseline
    - trust_real_mean (1 dim): mean trust score of real data
    - trust_synth_mean (1 dim): mean trust score of synthetic data
    - effective_synth_share (1 dim): current effective synthetic contribution
    - progress (1 dim): training progress (epoch_frac or step_frac)
    """

    def __init__(self, input_dim=11, hidden_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, max_synth_share=1.0):
        """
        Forward pass.

        Args:
            features: (batch, input_dim) or (input_dim,) feature tensor
            max_synth_share: Maximum allowed synthetic share (from internal_profile)

        Returns:
            lambda_synth: (batch,) or scalar, synthetic share in [0, max_synth_share]
        """
        squeeze_output = False
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True

        # MLP forward
        h = self.relu(self.fc1(features))
        h = self.relu(self.fc2(h))
        raw_out = self.fc_out(h)  # (batch, 1)

        # Sigmoid to [0, 1], then scale by max_synth_share
        lambda_synth = self.sigmoid(raw_out) * max_synth_share

        if squeeze_output:
            return lambda_synth.squeeze()
        return lambda_synth.squeeze(-1)

    def predict(self, objective_vector, delta_mpl, delta_error, delta_ep,
                trust_real_mean, trust_synth_mean, effective_synth_share,
                progress, max_synth_share=1.0):
        """
        Predict λ_synth from individual feature components.

        Args:
            objective_vector: list/array of 4 floats
            delta_mpl: normalized MPL change
            delta_error: normalized error change
            delta_ep: normalized EP change
            trust_real_mean: mean trust of real data
            trust_synth_mean: mean trust of synthetic data
            effective_synth_share: current effective synth contribution
            progress: training progress [0, 1]
            max_synth_share: maximum allowed share

        Returns:
            lambda_synth: predicted synthetic share
        """
        # Build feature vector
        features = torch.FloatTensor([
            *objective_vector,
            delta_mpl,
            delta_error,
            delta_ep,
            trust_real_mean,
            trust_synth_mean,
            effective_synth_share,
            progress
        ])

        with torch.no_grad():
            lambda_synth = self.forward(features, max_synth_share)

        return float(lambda_synth)


def build_feature_vector(
    objective_vector,
    current_mpl, baseline_mpl,
    current_error, baseline_error,
    current_ep, baseline_ep,
    trust_real_mean, trust_synth_mean,
    effective_synth_share, progress
):
    """
    Build feature vector for controller from raw metrics.

    Args:
        objective_vector: list/array of 4 floats [α_mpl, α_error, α_ep, α_novelty]
        current_mpl: current MPL
        baseline_mpl: baseline MPL for normalization
        current_error: current error rate
        baseline_error: baseline error rate
        current_ep: current energy productivity
        baseline_ep: baseline energy productivity
        trust_real_mean: mean trust of real data
        trust_synth_mean: mean trust of synthetic data
        effective_synth_share: current effective synthetic share
        progress: training progress [0, 1]

    Returns:
        features: numpy array of shape (11,)
    """
    # Normalize metrics relative to baseline (avoid div by zero)
    delta_mpl = (current_mpl - baseline_mpl) / (baseline_mpl + 1e-6)
    delta_error = (current_error - baseline_error) / (baseline_error + 1e-6)
    delta_ep = (current_ep - baseline_ep) / (baseline_ep + 1e-6)

    features = np.array([
        *objective_vector,
        delta_mpl,
        delta_error,
        delta_ep,
        trust_real_mean,
        trust_synth_mean,
        effective_synth_share,
        progress
    ], dtype=np.float32)

    return features


def compute_meta_objective(
    current_mpl, baseline_mpl,
    current_error, baseline_error,
    current_ep, baseline_ep,
    objective_vector
):
    """
    Compute the economic meta-objective J from physics/econ metrics.

    J = α_mpl * m_mpl - α_error * m_err + α_ep * m_ep

    Where:
    - m_mpl = MPL / baseline_MPL (higher is better)
    - m_err = error / baseline_error (lower is better, hence negative)
    - m_ep = EP / baseline_EP (higher is better)

    Args:
        current_mpl: current MPL
        baseline_mpl: baseline MPL
        current_error: current error rate
        baseline_error: baseline error rate
        current_ep: current energy productivity
        baseline_ep: baseline EP
        objective_vector: [α_mpl, α_error, α_ep, α_novelty]

    Returns:
        J: meta-objective scalar
    """
    # Normalized ratios
    m_mpl = current_mpl / (baseline_mpl + 1e-6)
    m_err = current_error / (baseline_error + 1e-6)
    m_ep = current_ep / (baseline_ep + 1e-6)

    # Extract objective weights
    alpha_mpl = objective_vector[0]
    alpha_error = objective_vector[1]
    alpha_ep = objective_vector[2]
    # alpha_novelty = objective_vector[3]  # Not used in J for now

    # Meta-objective
    J = alpha_mpl * m_mpl - alpha_error * m_err + alpha_ep * m_ep

    return J


def load_controller(checkpoint_path, device='cpu'):
    """
    Load trained controller from checkpoint.

    Args:
        checkpoint_path: path to .pt file
        device: torch device

    Returns:
        controller: SynthLambdaController instance
    """
    import os
    if not os.path.exists(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    controller = SynthLambdaController(
        input_dim=ckpt.get('input_dim', 11),
        hidden_dim=ckpt.get('hidden_dim', 16)
    ).to(device)

    controller.load_state_dict(ckpt['model_state_dict'])
    controller.eval()

    return controller
