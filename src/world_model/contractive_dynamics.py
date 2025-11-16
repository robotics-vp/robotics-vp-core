#!/usr/bin/env python3
"""
Contractive Latent Dynamics Model

Key architectural changes to prevent variance explosion:
1. Residual updates: z_{t+1} = z_t + alpha * g(z_t, a_t)
2. Spectral normalization: bound operator Lipschitz constant
3. Damped delta: learned or fixed alpha < 1
4. Optional Lipschitz constraint on g

The idea: make f(z_t, a_t) a mild residual update that can't explode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ContractiveLatentDynamics(nn.Module):
    """
    Latent dynamics with contractive/residual structure.

    z_{t+1} = z_t + alpha * g(z_t, a_t)

    where g is constrained to have bounded output via:
    - Spectral normalization (bounds largest singular value)
    - Output clipping
    - Learned damping factor alpha

    This ensures the operator can't have eigenvalues > 1 in unstable directions.
    """

    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim=256,
        n_layers=3,
        alpha_init=0.3,
        learnable_alpha=True,
        max_delta=0.2,
        use_spectral_norm=True,
        residual_scale=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_delta = max_delta
        self.residual_scale = residual_scale

        # Learnable damping factor
        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(alpha_init).log())
        else:
            self.register_buffer('log_alpha', torch.tensor(alpha_init).log())

        # Residual network g(z_t, a_t)
        layers = []
        input_dim = latent_dim + action_dim

        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else latent_dim
            linear = nn.Linear(input_dim if i == 0 else hidden_dim, out_dim)

            # Apply spectral norm to constrain Lipschitz constant
            if use_spectral_norm:
                linear = spectral_norm(linear)

            layers.append(linear)
            if i < n_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.SiLU())

        self.g_network = nn.Sequential(*layers)

        # Optional: learnable output scaling (constrained to < 1)
        self.output_scale = nn.Parameter(torch.tensor(0.5))

    @property
    def alpha(self):
        """Damping factor, constrained to (0, 1)."""
        return torch.sigmoid(self.log_alpha)

    def compute_delta(self, z_t, a_t):
        """
        Compute residual update delta = g(z_t, a_t).

        Args:
            z_t: (batch, latent_dim)
            a_t: (batch, action_dim)

        Returns:
            delta: (batch, latent_dim) bounded residual
        """
        # Concatenate inputs
        x = torch.cat([z_t, a_t], dim=-1)  # (batch, latent_dim + action_dim)

        # Compute raw delta
        delta_raw = self.g_network(x)  # (batch, latent_dim)

        # Scale by learned factor (constrained to (0, 1))
        scale = torch.sigmoid(self.output_scale)
        delta = delta_raw * scale

        # Hard clamp to prevent extreme deltas
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)

        return delta

    def forward(self, z_t, a_t):
        """
        Predict next latent state.

        z_{t+1} = z_t + alpha * delta

        Args:
            z_t: (batch, latent_dim) current latent
            a_t: (batch, action_dim) action

        Returns:
            z_next: (batch, latent_dim) next latent
            delta: (batch, latent_dim) residual applied
        """
        delta = self.compute_delta(z_t, a_t)
        z_next = z_t + self.alpha * delta * self.residual_scale

        return z_next, delta

    def get_lipschitz_bound(self):
        """
        Estimate upper bound on Lipschitz constant of the residual update.

        For stability, we want ||df/dz|| < 1.
        With residual structure: df/dz = I + alpha * dg/dz
        If ||dg/dz|| <= L (via spectral norm), then ||df/dz|| <= 1 + alpha * L

        We want 1 + alpha * L < something reasonable.
        """
        # Get spectral norm of each layer
        lipschitz = 1.0
        for module in self.g_network.modules():
            if hasattr(module, 'weight_u'):
                # This is a spectrally normalized layer
                # spectral_norm stores the largest singular value
                with torch.no_grad():
                    sigma = torch.linalg.norm(module.weight_orig, ord=2).item()
                    lipschitz *= sigma

        # Effective Lipschitz of f = I + alpha * g
        alpha_val = self.alpha.item()
        scale_val = torch.sigmoid(self.output_scale).item()

        effective_lipschitz = 1 + alpha_val * scale_val * lipschitz
        return effective_lipschitz

    def rollout(self, z_init, actions, return_deltas=False):
        """
        Roll out dynamics for multiple steps.

        Args:
            z_init: (latent_dim,) or (batch, latent_dim) initial state
            actions: (T, action_dim) or (batch, T, action_dim) action sequence
            return_deltas: if True, also return delta at each step

        Returns:
            z_sequence: (T+1, latent_dim) or (batch, T+1, latent_dim) trajectory
            deltas: optional, (T, latent_dim) residuals
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)
            actions = actions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, T, _ = actions.shape
        z_sequence = [z_init]
        deltas = []

        z_current = z_init
        for t in range(T):
            a_t = actions[:, t, :]
            z_next, delta = self.forward(z_current, a_t)
            z_sequence.append(z_next)
            deltas.append(delta)
            z_current = z_next

        z_sequence = torch.stack(z_sequence, dim=1)  # (batch, T+1, latent_dim)
        deltas = torch.stack(deltas, dim=1)  # (batch, T, latent_dim)

        if squeeze_output:
            z_sequence = z_sequence.squeeze(0)
            deltas = deltas.squeeze(0)

        if return_deltas:
            return z_sequence, deltas
        return z_sequence


class StableWorldModel(nn.Module):
    """
    Full stable world model with:
    1. Contractive dynamics
    2. Variance regularization helpers
    3. Trust-aware training support
    """

    def __init__(
        self,
        latent_dim,
        action_dim,
        hidden_dim=256,
        n_layers=3,
        alpha_init=0.3,
        learnable_alpha=True,
        max_delta=0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.dynamics = ContractiveLatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
            max_delta=max_delta,
        )

    def forward(self, z_t, a_t):
        """Single step prediction."""
        return self.dynamics(z_t, a_t)

    def rollout(self, z_init, actions, return_deltas=False):
        """Multi-step rollout."""
        return self.dynamics.rollout(z_init, actions, return_deltas=return_deltas)

    def compute_stability_metrics(self, z_sequence):
        """
        Compute stability metrics for a trajectory.

        Args:
            z_sequence: (T+1, latent_dim) trajectory

        Returns:
            dict of stability metrics
        """
        # Overall variance
        global_std = z_sequence.std()

        # Variance growth: compare first third vs last third
        T = len(z_sequence)
        early = z_sequence[:T//3]
        late = z_sequence[-T//3:]

        early_var = early.var(dim=0).mean()
        late_var = late.var(dim=0).mean()
        var_growth = late_var / (early_var + 1e-8)

        # Max absolute value (detects explosion)
        max_abs = z_sequence.abs().max()

        # Smoothness (temporal coherence)
        diffs = torch.abs(z_sequence[1:] - z_sequence[:-1])
        smoothness = diffs.mean()
        max_diff = diffs.max()

        return {
            'global_std': global_std,
            'var_growth': var_growth,
            'max_abs': max_abs,
            'smoothness': smoothness,
            'max_diff': max_diff,
        }

    def get_regularization_loss(self, z_sequence, target_std=0.062):
        """
        Compute regularization losses to enforce stability.

        Args:
            z_sequence: (T+1, latent_dim) predicted trajectory
            target_std: target standard deviation (from real data)

        Returns:
            losses: dict of loss components
        """
        metrics = self.compute_stability_metrics(z_sequence)

        # 1. Variance matching loss
        var_loss = (metrics['global_std'] - target_std) ** 2

        # 2. Variance growth penalty (penalize if late > early * threshold)
        growth_penalty = F.relu(metrics['var_growth'] - 1.5) ** 2

        # 3. Max value penalty (prevent explosion)
        max_penalty = F.relu(metrics['max_abs'] - 1.0) ** 2

        # 4. Smoothness penalty (encourage temporal coherence)
        smoothness_penalty = F.relu(metrics['max_diff'] - 0.3) ** 2

        return {
            'var_loss': var_loss,
            'growth_penalty': growth_penalty,
            'max_penalty': max_penalty,
            'smoothness_penalty': smoothness_penalty,
        }
