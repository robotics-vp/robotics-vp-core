"""
Unified encoder builder for switching between modalities.

Supports:
- MLP encoder (state-based observations)
- Video encoder (visual observations)
- Aligned video encoder (Phase A.5 - distillation-based)
- Teacher adapter (pretrained backbone for alignment)

Usage:
    encoder = build_encoder(config, obs_dim=10, device=device)
    encoder = build_encoder(config, video_shape=(8,3,64,64), device=device)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from src.encoders.mlp_encoder import MLPEncoder
from src.encoders.video_encoder import VideoEncoder
from src.encoders.student_video_encoder import AlignedVideoEncoder
from src.encoders.teacher_adapter import TeacherAdapter

def build_encoder(
    config: dict,
    obs_dim: Optional[int] = None,
    video_shape: Optional[Tuple[int, int, int, int]] = None,
    device: torch.device = None
) -> nn.Module:
    """
    Build encoder based on config.

    Args:
        config: Dict with encoder configuration
            {
                'type': 'mlp' | 'video',
                'latent_dim': int,
                'mlp': {...},  # MLP-specific config
                'video': {...}  # Video-specific config
            }
        obs_dim: Observation dimension (for MLP encoder)
        video_shape: Video shape (T, C, H, W) for video encoder
        device: Torch device

    Returns:
        Encoder module (MLPEncoder or VideoEncoder)
    """
    if device is None:
        device = torch.device('cpu')

    encoder_type = config.get('type', 'mlp')
    latent_dim = config.get('latent_dim', 128)

    if encoder_type == 'mlp':
        # MLP encoder for state-based observations
        if obs_dim is None:
            raise ValueError("obs_dim must be provided for MLP encoder")

        mlp_config = config.get('mlp', {})
        encoder = MLPEncoder(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=mlp_config.get('hidden_dim', 256),
        )

        print(f"[Encoder] Built MLPEncoder: obs_dim={obs_dim} -> latent_dim={latent_dim}")

    elif encoder_type == 'video':
        # Video encoder for visual observations
        video_config = config.get('video', {})

        encoder = VideoEncoder(
            latent_dim=latent_dim,
            arch=video_config.get('arch', 'simple2dcnn'),
            input_channels=video_config.get('input_channels', 3),
            pretrained=video_config.get('pretrained', False),
        )

        print(f"[Encoder] Built VideoEncoder: arch={video_config.get('arch', 'simple2dcnn')}, latent_dim={latent_dim}")

    elif encoder_type == 'aligned':
        # Aligned video encoder (Phase A.5) for distillation-based learning
        aligned_config = config.get('aligned', {})

        encoder = AlignedVideoEncoder(
            latent_dim=latent_dim,
            arch=aligned_config.get('arch', 'simple2dcnn'),
            input_channels=aligned_config.get('input_channels', 3),
            projection_dim=aligned_config.get('projection_dim', None),
            alignment_type=aligned_config.get('alignment_type', 'mse'),
            temperature=aligned_config.get('temperature', 0.1),
        )

        print(f"[Encoder] Built AlignedVideoEncoder: arch={aligned_config.get('arch', 'simple2dcnn')}, latent_dim={latent_dim}, alignment={aligned_config.get('alignment_type', 'mse')}")

    elif encoder_type == 'teacher':
        # Teacher adapter (pretrained backbone for providing targets)
        teacher_config = config.get('teacher', {})

        encoder = TeacherAdapter(
            teacher_type=teacher_config.get('teacher_type', 'r3d'),
            latent_dim=latent_dim,
            freeze=teacher_config.get('freeze', True),
            input_channels=teacher_config.get('input_channels', 3),
        )

        print(f"[Encoder] Built TeacherAdapter: type={teacher_config.get('teacher_type', 'r3d')}, latent_dim={latent_dim}")

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose 'mlp', 'video', 'aligned', or 'teacher'.")

    # Move to device
    encoder = encoder.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[Encoder] Total parameters: {n_params:,}")

    return encoder

if __name__ == '__main__':
    """Test encoder builder"""
    print("Testing encoder builder...")

    # Test MLP encoder
    print("\n[MLP Encoder Test]")
    mlp_config = {
        'type': 'mlp',
        'latent_dim': 128,
        'mlp': {
            'hidden_dim': 256,
        }
    }
    mlp_encoder = build_encoder(mlp_config, obs_dim=10, device=torch.device('cpu'))

    # Test forward pass
    obs = torch.randn(4, 10)
    z = mlp_encoder(obs)
    print(f"Output shape: {z.shape}")
    assert z.shape == (4, 128), f"Expected (4, 128), got {z.shape}"

    # Test Video encoder
    print("\n[Video Encoder Test]")
    video_config = {
        'type': 'video',
        'latent_dim': 128,
        'video': {
            'arch': 'simple2dcnn',
            'input_channels': 3,
        }
    }
    video_encoder = build_encoder(video_config, video_shape=(8, 3, 64, 64), device=torch.device('cpu'))

    # Test forward pass
    video = torch.randn(4, 8, 3, 64, 64)
    z = video_encoder(video)
    print(f"Output shape: {z.shape}")
    assert z.shape == (4, 128), f"Expected (4, 128), got {z.shape}"

    print("\n✅ Encoder builder working!")
