"""
Teacher Adapter for Aligned Visual Backbone (Phase A.5)

Wraps pretrained visual backbones to provide canonical visual representations.
The student encoder learns to align with these representations.

Supported teachers:
- CLIP: OpenAI's vision-language model (strong visual priors)
- DINOv2: Meta's self-supervised vision model
- R3D-18: Video-specific pretrained backbone

The teacher is frozen and provides target representations for the student.

Usage:
    teacher = TeacherAdapter(teacher_type='clip', latent_dim=128)
    z_teacher = teacher(video)  # (B, T, C, H, W) -> (B, latent_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CLIPAdapter(nn.Module):
    """
    CLIP ViT-B/32 adapter for frame-level visual features.

    Processes each frame through CLIP image encoder, then pools temporally.
    Strong visual priors from web-scale image-text pretraining.
    """
    def __init__(self, latent_dim=128, freeze=True):
        super().__init__()
        try:
            import clip
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device='cpu')
            if freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False
            self.clip_dim = 512  # CLIP ViT-B/32 output dim
        except ImportError:
            raise ImportError("OpenAI CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

        # Projection head to match target latent_dim
        self.projection = nn.Linear(self.clip_dim, latent_dim)
        if freeze:
            # Keep projection trainable for downstream task
            pass

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape for per-frame processing
        x = x.view(B * T, C, H, W)

        # CLIP expects images in [0, 1] range
        # Normalize if needed (CLIP has its own normalization)
        with torch.no_grad():
            features = self.clip_model.encode_image(x)  # (B*T, 512)

        # Reshape back to (B, T, dim)
        features = features.view(B, T, -1).float()

        # Temporal pooling (mean over frames)
        features = features.mean(dim=1)  # (B, 512)

        # Project to target latent dim
        z = self.projection(features)  # (B, latent_dim)

        return z


class DINOv2Adapter(nn.Module):
    """
    DINOv2 adapter for frame-level visual features.

    Meta's self-supervised vision model with strong object-centric representations.
    """
    def __init__(self, latent_dim=128, freeze=True):
        super().__init__()
        try:
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            if freeze:
                for param in self.dino_model.parameters():
                    param.requires_grad = False
            self.dino_dim = 384  # DINOv2-S output dim
        except Exception as e:
            raise ImportError(f"DINOv2 loading failed: {e}. Ensure torch hub is accessible.")

        # Projection head
        self.projection = nn.Linear(self.dino_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim)
        """
        B, T, C, H, W = x.shape

        # Reshape for per-frame processing
        x = x.view(B * T, C, H, W)

        # DINOv2 expects 224x224 images
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            features = self.dino_model(x)  # (B*T, 384)

        # Reshape and pool
        features = features.view(B, T, -1).float()
        features = features.mean(dim=1)

        # Project
        z = self.projection(features)

        return z


class R3DAdapter(nn.Module):
    """
    ResNet3D-18 pretrained on Kinetics-400.

    Video-specific temporal features from action recognition pretraining.
    """
    def __init__(self, latent_dim=128, freeze=True):
        super().__init__()
        try:
            import torchvision.models.video as video_models
            self.r3d = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
            if freeze:
                for param in self.r3d.parameters():
                    param.requires_grad = False

            # Remove final FC
            self.r3d_dim = self.r3d.fc.in_features
            self.r3d.fc = nn.Identity()
        except ImportError:
            raise ImportError("torchvision not installed.")

        # Projection head
        self.projection = nn.Linear(self.r3d_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim)
        """
        # R3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            features = self.r3d(x)  # (B, r3d_dim)

        z = self.projection(features.float())

        return z


class RandomProjectionAdapter(nn.Module):
    """
    Random projection teacher for baseline/ablation.

    Uses random (frozen) weights to provide consistent but non-semantic targets.
    Useful for verifying that student actually learns meaningful alignment.
    """
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # Simple 2D CNN with frozen random weights
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

        # Freeze all weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim)
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.reshape(B * T, -1)
            x = self.fc(x)

        # Temporal pooling
        x = x.reshape(B, T, -1).mean(dim=1)

        return x


class TeacherAdapter(nn.Module):
    """
    Unified teacher adapter for Phase A.5 aligned visual backbone.

    The teacher provides canonical visual representations that the student
    learns to align with via distillation loss.

    Args:
        teacher_type: Type of teacher ('clip', 'dino', 'r3d', 'random')
        latent_dim: Output embedding dimension
        freeze: Whether to freeze teacher weights (default True)
    """
    TEACHERS = {
        'clip': CLIPAdapter,
        'dino': DINOv2Adapter,
        'r3d': R3DAdapter,
        'random': RandomProjectionAdapter,
    }

    def __init__(
        self,
        teacher_type: str = 'r3d',
        latent_dim: int = 128,
        freeze: bool = True,
        input_channels: int = 3,
    ):
        super().__init__()

        self.teacher_type = teacher_type
        self.latent_dim = latent_dim

        if teacher_type not in self.TEACHERS:
            raise ValueError(f"Unknown teacher: {teacher_type}. Choose from {list(self.TEACHERS.keys())}")

        # Build teacher
        if teacher_type == 'random':
            self.teacher = RandomProjectionAdapter(input_channels=input_channels, latent_dim=latent_dim)
        elif teacher_type in ['clip', 'dino', 'r3d']:
            self.teacher = self.TEACHERS[teacher_type](latent_dim=latent_dim, freeze=freeze)
        else:
            raise ValueError(f"Unknown teacher type: {teacher_type}")

        print(f"[TeacherAdapter] Built {teacher_type} teacher (frozen={freeze})")

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim) teacher embedding
        """
        return self.teacher(x)

    @torch.no_grad()
    def encode(self, x):
        """Encode without gradients (for distillation targets)"""
        return self.forward(x)


def build_teacher_adapter(config: dict) -> TeacherAdapter:
    """
    Build teacher adapter from config dict.

    Args:
        config: Dict with keys:
            - teacher_type: str ('clip', 'dino', 'r3d', 'random')
            - latent_dim: int
            - freeze: bool (default True)
            - input_channels: int (default 3)

    Returns:
        TeacherAdapter instance
    """
    return TeacherAdapter(
        teacher_type=config.get('teacher_type', 'r3d'),
        latent_dim=config.get('latent_dim', 128),
        freeze=config.get('freeze', True),
        input_channels=config.get('input_channels', 3),
    )


if __name__ == '__main__':
    """Test teacher adapters"""
    print("Testing teacher adapters...")

    # Create dummy video input
    B, T, C, H, W = 2, 4, 3, 64, 64
    x = torch.randn(B, T, C, H, W)

    # Test RandomProjectionAdapter (always available)
    print("\n[RandomProjectionAdapter]")
    teacher = TeacherAdapter(teacher_type='random', latent_dim=128)
    z = teacher(x)
    print(f"Input: {x.shape} -> Output: {z.shape}")
    print(f"Parameters: {sum(p.numel() for p in teacher.parameters()):,}")

    # Test R3D adapter (requires torchvision)
    try:
        print("\n[R3DAdapter]")
        teacher = TeacherAdapter(teacher_type='r3d', latent_dim=128)
        z = teacher(x)
        print(f"Input: {x.shape} -> Output: {z.shape}")
        print(f"Parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    except ImportError as e:
        print(f"\n[R3DAdapter] Skipped: {e}")

    # Test CLIP adapter (requires openai-clip)
    try:
        print("\n[CLIPAdapter]")
        teacher = TeacherAdapter(teacher_type='clip', latent_dim=128)
        z = teacher(x)
        print(f"Input: {x.shape} -> Output: {z.shape}")
    except ImportError as e:
        print(f"\n[CLIPAdapter] Skipped: {e}")

    print("\n✅ Teacher adapter tests complete!")
