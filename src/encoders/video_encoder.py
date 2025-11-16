"""
Video Encoder for Video-to-Policy Pipeline

Supports multiple architectures:
- Simple2DCNN: 2D CNN per-frame + temporal pooling
- Simple3DCNN: Lightweight 3D convolutions
- R3D18: ResNet3D-18 (requires GPU)
- TimeSformer: Vision transformer for video (requires GPU)

Input: (B, T, C, H, W) - batch, time/frames, channels, height, width
Output: (B, latent_dim) - latent embedding for policy

Usage:
    encoder = VideoEncoder(latent_dim=128, arch='simple2dcnn', frames=8)
    z = encoder(video)  # (B, 8, 3, 64, 64) -> (B, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Simple2DCNN(nn.Module):
    """
    2D CNN applied per-frame, then temporal pooling.

    Fast, CPU-friendly, good baseline for synthetic video.
    """
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # Per-frame 2D CNN
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # MLP head
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            z: (B, latent_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) for per-frame processing
        x = x.view(B * T, C, H, W)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Adaptive pool
        x = self.adaptive_pool(x)  # (B*T, 128, 4, 4)

        # Flatten spatial (use reshape for non-contiguous tensors)
        x = x.reshape(B * T, -1)  # (B*T, 128*4*4)

        # MLP
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (B*T, latent_dim)

        # Temporal pooling (mean over frames)
        x = x.reshape(B, T, -1)  # (B, T, latent_dim)
        z = x.mean(dim=1)  # (B, latent_dim)

        return z

class Simple3DCNN(nn.Module):
    """
    Lightweight 3D CNN for spatio-temporal features.

    More parameters than 2D CNN, but captures temporal patterns directly.
    """
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # 3D convolutions (spatial + temporal)
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))

        # MLP head
        self.fc1 = nn.Linear(128 * 1 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            z: (B, latent_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape to (B, C, T, H, W) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Adaptive pool
        x = self.adaptive_pool(x)  # (B, 128, 1, 4, 4)

        # Flatten
        x = x.view(B, -1)

        # MLP
        x = F.relu(self.fc1(x))
        z = self.fc2(x)

        return z

class R3D18Encoder(nn.Module):
    """
    ResNet3D-18 pretrained on Kinetics (requires GPU).

    High-quality spatiotemporal features from real video.
    Use this when training on real demonstrations or physics sim.
    """
    def __init__(self, latent_dim=128, pretrained=True):
        super().__init__()
        try:
            import torchvision.models.video as video_models
        except ImportError:
            raise ImportError("torchvision not installed. Install with: pip install torchvision")

        # Load R3D-18
        if pretrained:
            self.backbone = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
        else:
            self.backbone = video_models.r3d_18(weights=None)

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            z: (B, latent_dim)
        """
        # R3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Backbone
        features = self.backbone(x)  # (B, in_features)

        # Project to latent
        z = self.projection(features)

        return z

class VideoEncoder(nn.Module):
    """
    Unified video encoder interface with multiple architecture options.

    Args:
        latent_dim: Output embedding dimension
        arch: Architecture choice ('simple2dcnn', 'simple3dcnn', 'r3d18')
        input_channels: Number of input channels (3 for RGB)
        pretrained: Whether to use pretrained weights (for r3d18)
    """
    ARCHITECTURES = {
        'simple2dcnn': Simple2DCNN,
        'simple3dcnn': Simple3DCNN,
        'r3d18': R3D18Encoder,
    }

    def __init__(
        self,
        latent_dim: int = 128,
        arch: str = 'simple2dcnn',
        input_channels: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.arch = arch

        if arch not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch}. Choose from {list(self.ARCHITECTURES.keys())}")

        # Build encoder
        if arch == 'r3d18':
            self.encoder = R3D18Encoder(latent_dim=latent_dim, pretrained=pretrained)
        else:
            self.encoder = self.ARCHITECTURES[arch](
                input_channels=input_channels,
                latent_dim=latent_dim
            )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim) latent embedding
        """
        return self.encoder(x)

    def encode(self, x):
        """Alias for forward (compatibility with MLPEncoder interface)"""
        return self.forward(x)

def build_video_encoder(config: dict) -> VideoEncoder:
    """
    Build video encoder from config dict.

    Args:
        config: Dict with keys:
            - latent_dim: int
            - arch: str ('simple2dcnn', 'simple3dcnn', 'r3d18')
            - input_channels: int (default 3)
            - pretrained: bool (default False)

    Returns:
        VideoEncoder instance
    """
    return VideoEncoder(
        latent_dim=config.get('latent_dim', 128),
        arch=config.get('arch', 'simple2dcnn'),
        input_channels=config.get('input_channels', 3),
        pretrained=config.get('pretrained', False),
    )

if __name__ == '__main__':
    """Test video encoders"""
    print("Testing video encoders...")

    # Create dummy video input
    B, T, C, H, W = 4, 8, 3, 64, 64
    x = torch.randn(B, T, C, H, W)

    # Test Simple2DCNN
    print("\n[Simple2DCNN]")
    encoder = VideoEncoder(latent_dim=128, arch='simple2dcnn')
    z = encoder(x)
    print(f"Input: {x.shape} -> Output: {z.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test Simple3DCNN
    print("\n[Simple3DCNN]")
    encoder = VideoEncoder(latent_dim=128, arch='simple3dcnn')
    z = encoder(x)
    print(f"Input: {x.shape} -> Output: {z.shape}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test R3D18 (if torchvision available)
    try:
        print("\n[R3D18]")
        encoder = VideoEncoder(latent_dim=128, arch='r3d18', pretrained=False)
        z = encoder(x)
        print(f"Input: {x.shape} -> Output: {z.shape}")
        print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    except ImportError:
        print("\n[R3D18] Skipped (torchvision not installed)")

    print("\n✅ All encoders working!")
