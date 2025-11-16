"""
Diffusion-based novelty estimation for video-to-policy data valuation.

This module provides differentiable novelty signals that quantify how
"off-manifold" or surprising a latent observation is under a learned
diffusion prior. Used to weight training samples by their informational value.
"""
import torch
import torch.nn.functional as F
from collections import deque


@torch.no_grad()
def mse_noise_gap(denoiser, x0_latent, ts, noise_sampler, reps=2):
    """
    Measure how off-manifold a latent clip is under diffusion prior.

    For each timestep t in ts:
      1. Add noise: x_t = x_0 + t * eps
      2. Denoise: eps_hat = denoiser(x_t, t)
      3. Compute MSE(eps_hat, eps)

    Average over timesteps and reps to get per-sample novelty score.
    Higher MSE → more off-manifold → higher novelty.

    Args:
        denoiser: Callable (x_t, t) -> eps_hat (predicted noise)
        x0_latent: [B, D] tensor of latent observations
        ts: List of timesteps (e.g., [0.1, 0.5, 0.9])
        noise_sampler: Callable (x) -> noise with same shape as x
        reps: Number of noise samples per timestep

    Returns:
        novelty_mse: [B] tensor of MSE-based novelty scores
    """
    gaps = []
    for t in ts:
        cur = []
        for _ in range(reps):
            eps = noise_sampler(x0_latent)
            xt = x0_latent + t * eps
            eps_hat = denoiser(xt, torch.tensor(t).to(x0_latent.device))

            # MSE per sample (reduce over feature dims)
            mse = F.mse_loss(
                eps_hat, eps, reduction="none"
            ).mean(dim=list(range(1, eps.dim())))

            cur.append(mse)

        gaps.append(torch.stack(cur, 0).mean(0))

    # Average over timesteps: [B]
    return torch.stack(gaps, 0).mean(0)


@torch.no_grad()
def recon_gap(short_denoise, x0_latent):
    """
    Compute reconstruction gap via short denoising trajectory.

    Short denoising: add small noise, denoise back, measure error.
    This captures local manifold structure - high error means x0 is
    far from typical samples.

    Args:
        short_denoise: Callable (x_0) -> x_hat (reconstructed)
        x0_latent: [B, D] tensor of latent observations

    Returns:
        novelty_recon: [B] tensor of reconstruction-based novelty
    """
    xhat = short_denoise(x0_latent)

    # L2 reconstruction error per sample
    recon_error = ((x0_latent - xhat) ** 2).mean(
        dim=list(range(1, x0_latent.dim()))
    )

    return recon_error


def combine_novelty(n_mse, n_recon, ema_mu, ema_std, alpha=1.0, beta=1.0):
    """
    Combine MSE and reconstruction novelty signals with EMA normalization.

    Normalizes each signal to z-scores using running statistics, then
    combines with weighted sum and sigmoid squashing to [0, 1].

    Args:
        n_mse: [B] tensor of MSE novelty scores
        n_recon: [B] tensor of reconstruction novelty scores
        ema_mu: dict with keys "mse" and "recon" (running means)
        ema_std: dict with keys "mse" and "recon" (running stds)
        alpha: Weight for MSE component
        beta: Weight for reconstruction component

    Returns:
        novelty: [B] tensor in (0, 1), higher = more novel
    """
    # Z-score normalization
    z1 = (n_mse - ema_mu["mse"]) / (ema_std["mse"] + 1e-6)
    z2 = (n_recon - ema_mu["recon"]) / (ema_std["recon"] + 1e-6)

    # Weighted combination
    raw = alpha * z1 + beta * z2

    # Squash to (0, 1)
    return torch.sigmoid(raw)


class DiffusionNoveltyTracker:
    """
    Tracks EMA statistics and computes differentiable novelty scores.

    This tracker maintains running statistics (mean/std) for MSE and
    reconstruction novelty signals, enabling normalized novelty computation
    that's stable across training.
    """

    def __init__(self, ema_decay=0.99, alpha=1.0, beta=1.0):
        """
        Args:
            ema_decay: Decay rate for exponential moving average
            alpha: Weight for MSE novelty component
            beta: Weight for reconstruction novelty component
        """
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.beta = beta

        # Running statistics
        self.ema_mu = {"mse": 0.0, "recon": 0.0}
        self.ema_std = {"mse": 1.0, "recon": 1.0}

        # History for initialization
        self.mse_history = deque(maxlen=100)
        self.recon_history = deque(maxlen=100)
        self.initialized = False

    def update_ema(self, n_mse, n_recon):
        """
        Update running statistics with new batch.

        Args:
            n_mse: [B] tensor or float
            n_recon: [B] tensor or float
        """
        # Convert to scalars if tensors
        if torch.is_tensor(n_mse):
            n_mse = n_mse.mean().item()
        if torch.is_tensor(n_recon):
            n_recon = n_recon.mean().item()

        # Add to history for initialization
        self.mse_history.append(n_mse)
        self.recon_history.append(n_recon)

        # Initialize from first 100 samples
        if not self.initialized and len(self.mse_history) >= 100:
            self.ema_mu["mse"] = sum(self.mse_history) / len(self.mse_history)
            self.ema_mu["recon"] = sum(self.recon_history) / len(self.recon_history)

            import statistics
            self.ema_std["mse"] = statistics.stdev(self.mse_history)
            self.ema_std["recon"] = statistics.stdev(self.recon_history)

            self.initialized = True
            return

        # EMA update
        if self.initialized:
            self.ema_mu["mse"] = (
                self.ema_decay * self.ema_mu["mse"] +
                (1 - self.ema_decay) * n_mse
            )
            self.ema_mu["recon"] = (
                self.ema_decay * self.ema_mu["recon"] +
                (1 - self.ema_decay) * n_recon
            )

            # Update std (approximate via EMA of squared deviations)
            dev_mse = (n_mse - self.ema_mu["mse"]) ** 2
            dev_recon = (n_recon - self.ema_mu["recon"]) ** 2

            var_mse = (
                self.ema_decay * (self.ema_std["mse"] ** 2) +
                (1 - self.ema_decay) * dev_mse
            )
            var_recon = (
                self.ema_decay * (self.ema_std["recon"] ** 2) +
                (1 - self.ema_decay) * dev_recon
            )

            self.ema_std["mse"] = max(var_mse ** 0.5, 1e-6)
            self.ema_std["recon"] = max(var_recon ** 0.5, 1e-6)

    def compute_novelty(self, n_mse, n_recon, update_stats=True):
        """
        Compute combined novelty score with EMA normalization.

        Args:
            n_mse: [B] tensor of MSE novelty scores
            n_recon: [B] tensor of reconstruction novelty scores
            update_stats: If True, update EMA statistics with this batch

        Returns:
            novelty: [B] tensor in (0, 1)
        """
        if update_stats:
            self.update_ema(n_mse, n_recon)

        # Use current EMA stats for normalization
        novelty = combine_novelty(
            n_mse, n_recon,
            ema_mu=self.ema_mu,
            ema_std=self.ema_std,
            alpha=self.alpha,
            beta=self.beta
        )

        return novelty


# Stub models for testing (will be replaced by real diffusion models)

class StubDenoiser(torch.nn.Module):
    """Stub denoiser for testing. Returns small random noise."""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, xt, t):
        # Return small noise (simulate "good" denoising)
        return torch.randn_like(xt) * 0.1


class StubShortDenoise(torch.nn.Module):
    """Stub short denoiser for testing. Returns input with small noise."""
    def __init__(self):
        super().__init__()

    def forward(self, x0):
        # Simulate reconstruction with small error
        return x0 + torch.randn_like(x0) * 0.05


def gaussian_noise_sampler(x):
    """Standard Gaussian noise sampler."""
    return torch.randn_like(x)


class DiffusionNoveltyEstimator:
    """
    End-to-end novelty estimator operating on latent embeddings.

    This class combines stub diffusion models with the novelty tracker
    to provide a simple .compute() API for novelty estimation.

    Usage:
        novelty_est = DiffusionNoveltyEstimator(latent_dim=128)
        z = encoder(obs)  # (B, latent_dim)
        novelty = novelty_est.compute(z)  # (B,) in [0, 1]
    """

    def __init__(self, latent_dim=128, device='cpu'):
        """
        Args:
            latent_dim: Dimension of latent embeddings
            device: Device for computation
        """
        self.latent_dim = latent_dim
        self.device = device

        # Stub models (will be replaced with real diffusion later)
        self.denoiser = StubDenoiser(latent_dim).to(device)
        self.short_denoise = StubShortDenoise().to(device)

        # Novelty tracker (maintains running statistics)
        self.tracker = DiffusionNoveltyTracker(ema_decay=0.99)

        # Diffusion timesteps for MSE computation
        self.timesteps = [0.1, 0.5, 0.9]

    @torch.no_grad()
    def compute(self, z_latent):
        """
        Compute novelty scores for latent embeddings.

        Args:
            z_latent: Tensor of shape (B, latent_dim) or (latent_dim,)

        Returns:
            novelty: Tensor of shape (B,) or scalar, values in [0, 1]
        """
        # Handle single sample
        single_sample = False
        if z_latent.dim() == 1:
            z_latent = z_latent.unsqueeze(0)
            single_sample = True

        # Ensure on correct device
        z_latent = z_latent.to(self.device)

        # Compute MSE novelty
        n_mse = mse_noise_gap(
            self.denoiser,
            z_latent,
            ts=self.timesteps,
            noise_sampler=gaussian_noise_sampler,
            reps=2
        )

        # Compute reconstruction novelty
        n_recon = recon_gap(self.short_denoise, z_latent)

        # Combine and normalize
        novelty = self.tracker.compute_novelty(n_mse, n_recon, update_stats=True)

        if single_sample:
            return novelty.item()
        return novelty

    def to(self, device):
        """Move models to device."""
        self.device = device
        self.denoiser = self.denoiser.to(device)
        self.short_denoise = self.short_denoise.to(device)
        return self
