"""
Unit tests for diffusion novelty.
"""
import pytest
import torch
from src.data_value.novelty_diffusion import (
    mse_noise_gap,
    recon_gap,
    combine_novelty,
    StubDenoiser,
    StubShortDenoise,
    gaussian_noise_sampler
)


def test_mse_noise_gap_basic():
    """Test MSE noise gap computation."""
    latent_dim = 4
    batch_size = 10

    denoiser = StubDenoiser(latent_dim)

    x0 = torch.randn(batch_size, latent_dim)

    novelty = mse_noise_gap(
        denoiser, x0,
        ts=[0.5],
        noise_sampler=gaussian_noise_sampler,
        reps=2
    )

    # Should return non-negative scores for each sample
    assert novelty.shape == (batch_size,)
    assert (novelty >= 0).all()

    # With multiple timesteps
    novelty_multi = mse_noise_gap(
        denoiser, x0,
        ts=[0.1, 0.5, 0.9],
        noise_sampler=gaussian_noise_sampler,
        reps=2
    )

    assert novelty_multi.shape == (batch_size,)
    assert (novelty_multi >= 0).all()


def test_recon_gap_basic():
    """Test reconstruction gap computation."""
    latent_dim = 4
    batch_size = 10

    short_denoise = StubShortDenoise()

    x0 = torch.randn(batch_size, latent_dim)

    novelty = recon_gap(short_denoise, x0)

    assert novelty.shape == (batch_size,)
    assert (novelty >= 0).all()  # Non-negative


def test_combine_novelty_range():
    """Combined novelty should be in (0, 1)."""
    n_mse = torch.tensor([0.1, 0.5, 1.0])
    n_recon = torch.tensor([0.05, 0.3, 0.8])

    ema_mu = {"mse": 0.5, "recon": 0.4}
    ema_std = {"mse": 0.2, "recon": 0.15}

    novelty = combine_novelty(n_mse, n_recon, ema_mu, ema_std)

    assert (novelty > 0).all()
    assert (novelty < 1).all()


def test_combine_novelty_monotonic():
    """Higher inputs should lead to higher novelty."""
    ema_mu = {"mse": 0.5, "recon": 0.4}
    ema_std = {"mse": 0.2, "recon": 0.15}

    # Low novelty
    n_low = combine_novelty(
        torch.tensor([0.1]), torch.tensor([0.1]),
        ema_mu, ema_std
    )

    # High novelty
    n_high = combine_novelty(
        torch.tensor([1.0]), torch.tensor([1.0]),
        ema_mu, ema_std
    )

    assert n_high.item() > n_low.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
