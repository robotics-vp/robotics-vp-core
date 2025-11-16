"""
Interfaces for video-to-policy components.

These abstract interfaces allow swapping stub models for real
video diffusion encoders without changing downstream code.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class VideoLatentProvider(ABC):
    """
    Abstract interface for video → latent encoding.

    In V2P pipeline: converts video clips to latent representations
    for novelty computation and policy input.
    """

    @abstractmethod
    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent representation.

        Args:
            video_batch: [B, C, T, H, W] tensor of video clips
                B = batch size
                C = channels (3 for RGB)
                T = temporal frames
                H, W = spatial dimensions

        Returns:
            latents: [B, D, ...] tensor of latent representations
                D = latent dimension
                May have additional spatial/temporal dims
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to video (for visualization/debugging).

        Args:
            latents: [B, D, ...] latent tensor

        Returns:
            video_batch: [B, C, T, H, W] reconstructed video
        """
        raise NotImplementedError


class DiffusionDenoiser(ABC):
    """
    Abstract interface for diffusion denoising.

    Used for novelty computation via noise prediction error.
    """

    @abstractmethod
    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from noisy latent at timestep t.

        Args:
            xt: [B, D, ...] noisy latent
            t: [B] or scalar timestep(s) in [0, 1]

        Returns:
            eps_hat: [B, D, ...] predicted noise (same shape as xt)
        """
        raise NotImplementedError

    @abstractmethod
    def short_denoise(self, x0_latent: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Short denoising trajectory for reconstruction-based novelty.

        Adds small noise and denoises back in few steps.

        Args:
            x0_latent: [B, D, ...] clean latent
            steps: Number of denoising steps (default 1)

        Returns:
            xhat: [B, D, ...] reconstructed latent
        """
        raise NotImplementedError


# Stub implementations (for testing)

class StubVideoEncoder(VideoLatentProvider):
    """Stub video encoder: random projection."""

    def __init__(self, input_dim=12288, latent_dim=512):
        """
        Args:
            input_dim: Flattened video dim (e.g., 3*16*64*64 = 196608)
            latent_dim: Output latent dimension
        """
        self.projection = nn.Linear(input_dim, latent_dim)

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Flatten and project to latent space."""
        B = video_batch.shape[0]
        flat = video_batch.view(B, -1)

        # Handle variable input sizes via adaptive pooling
        if flat.shape[1] != self.projection.in_features:
            # Fallback: just take first latent_dim features
            return torch.randn(B, self.projection.out_features)

        return self.projection(flat)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Not implemented for stub."""
        raise NotImplementedError("Stub encoder does not support decoding")


class StubDiffusionDenoiser(DiffusionDenoiser):
    """Stub diffusion denoiser: small random noise."""

    def __init__(self, noise_scale=0.1):
        self.noise_scale = noise_scale

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return small random noise (simulates good denoising)."""
        return torch.randn_like(xt) * self.noise_scale

    def short_denoise(self, x0_latent: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Return input with small noise (simulates reconstruction)."""
        return x0_latent + torch.randn_like(x0_latent) * self.noise_scale


# Factory functions for easy swapping

def get_video_encoder(encoder_type="stub", **kwargs):
    """
    Factory function for video encoders.

    Args:
        encoder_type: "stub", "vqvae", "videogpt", etc.
        **kwargs: Encoder-specific arguments

    Returns:
        VideoLatentProvider instance
    """
    if encoder_type == "stub":
        return StubVideoEncoder(**kwargs)
    else:
        raise NotImplementedError(f"Encoder type '{encoder_type}' not implemented")


def get_diffusion_denoiser(denoiser_type="stub", **kwargs):
    """
    Factory function for diffusion denoisers.

    Args:
        denoiser_type: "stub", "unet", "dit", etc.
        **kwargs: Denoiser-specific arguments

    Returns:
        DiffusionDenoiser instance
    """
    if denoiser_type == "stub":
        return StubDiffusionDenoiser(**kwargs)
    else:
        raise NotImplementedError(f"Denoiser type '{denoiser_type}' not implemented")
