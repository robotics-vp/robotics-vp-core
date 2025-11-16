"""
Student Video Encoder with Alignment Loss (Phase A.5)

Learns aligned visual representations by distilling from a pretrained teacher.
The student encoder is trained to match the teacher's representations while
also being useful for downstream policy learning.

Key features:
- Wraps any VideoEncoder architecture
- Computes alignment loss with teacher representations
- Supports multiple distillation objectives (MSE, cosine, contrastive)
- Can be used standalone (after training) for inference

Usage:
    # During training (with teacher)
    student = AlignedVideoEncoder(latent_dim=128, arch='simple2dcnn')
    teacher = TeacherAdapter(teacher_type='r3d', latent_dim=128)

    z_student = student(video)
    z_teacher = teacher.encode(video)
    alignment_loss = student.alignment_loss(z_student, z_teacher)

    # During inference (without teacher)
    z = student(video)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.encoders.video_encoder import VideoEncoder


class AlignedVideoEncoder(nn.Module):
    """
    Student video encoder that learns aligned representations.

    Uses knowledge distillation to align with teacher representations
    while maintaining flexibility for downstream policy learning.

    Args:
        latent_dim: Output embedding dimension
        arch: Base VideoEncoder architecture
        input_channels: Number of input channels
        projection_dim: Intermediate projection dimension for alignment
        alignment_type: Type of alignment loss ('mse', 'cosine', 'contrastive')
        temperature: Temperature for contrastive loss
    """
    def __init__(
        self,
        latent_dim: int = 128,
        arch: str = 'simple2dcnn',
        input_channels: int = 3,
        projection_dim: Optional[int] = None,
        alignment_type: str = 'mse',
        temperature: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.arch = arch
        self.alignment_type = alignment_type
        self.temperature = temperature

        # Base encoder (student backbone)
        self.encoder = VideoEncoder(
            latent_dim=latent_dim,
            arch=arch,
            input_channels=input_channels,
        )

        # Optional projection head for alignment
        # (Allows encoder output to differ from alignment space)
        if projection_dim is not None and projection_dim != latent_dim:
            self.projection = nn.Sequential(
                nn.Linear(latent_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, latent_dim),
            )
        else:
            self.projection = nn.Identity()

        print(f"[AlignedVideoEncoder] Built with arch={arch}, latent_dim={latent_dim}, alignment={alignment_type}")

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim) latent embedding
        """
        return self.encoder(x)

    def encode(self, x):
        """Alias for forward (compatibility)"""
        return self.forward(x)

    def forward_with_projection(self, x):
        """
        Forward pass with projection head for alignment training.

        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            z: (B, latent_dim) encoder output
            z_proj: (B, latent_dim) projected for alignment
        """
        z = self.encoder(x)
        z_proj = self.projection(z)
        return z, z_proj

    def alignment_loss(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute alignment loss between student and teacher representations.

        Args:
            z_student: (B, D) student embeddings
            z_teacher: (B, D) teacher embeddings (targets)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: Alignment loss scalar
        """
        if self.alignment_type == 'mse':
            loss = F.mse_loss(z_student, z_teacher, reduction=reduction)

        elif self.alignment_type == 'cosine':
            # Cosine similarity loss (1 - cos_sim)
            z_s_norm = F.normalize(z_student, dim=-1)
            z_t_norm = F.normalize(z_teacher, dim=-1)
            cos_sim = (z_s_norm * z_t_norm).sum(dim=-1)
            loss = 1 - cos_sim
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()

        elif self.alignment_type == 'contrastive':
            # InfoNCE contrastive loss
            z_s_norm = F.normalize(z_student, dim=-1)
            z_t_norm = F.normalize(z_teacher, dim=-1)

            # Similarity matrix
            sim = torch.mm(z_s_norm, z_t_norm.t()) / self.temperature

            # Labels: diagonal elements are positive pairs
            labels = torch.arange(z_student.shape[0], device=z_student.device)

            # Cross-entropy loss
            loss = F.cross_entropy(sim, labels, reduction=reduction)

        elif self.alignment_type == 'smooth_l1':
            loss = F.smooth_l1_loss(z_student, z_teacher, reduction=reduction)

        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")

        return loss

    def distillation_step(
        self,
        video: torch.Tensor,
        teacher: nn.Module,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Complete distillation step for training.

        Args:
            video: (B, T, C, H, W) video batch
            teacher: Teacher model (TeacherAdapter)
            alpha: Weight for alignment loss

        Returns:
            z_student: Student embeddings
            loss: Alignment loss (weighted)
            metrics: Dict with training metrics
        """
        # Student forward
        z_student, z_proj = self.forward_with_projection(video)

        # Teacher forward (no gradients)
        with torch.no_grad():
            z_teacher = teacher.encode(video)

        # Alignment loss
        align_loss = self.alignment_loss(z_proj, z_teacher)

        # Metrics
        with torch.no_grad():
            # Cosine similarity for monitoring
            z_s_norm = F.normalize(z_proj, dim=-1)
            z_t_norm = F.normalize(z_teacher, dim=-1)
            cos_sim = (z_s_norm * z_t_norm).sum(dim=-1).mean()

            # L2 distance
            l2_dist = (z_proj - z_teacher).pow(2).sum(dim=-1).sqrt().mean()

        metrics = {
            'align_loss': align_loss.item(),
            'cos_sim': cos_sim.item(),
            'l2_dist': l2_dist.item(),
        }

        return z_student, alpha * align_loss, metrics

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pretrained student weights"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=strict)
        print(f"[AlignedVideoEncoder] Loaded weights from {checkpoint_path}")

    def save_pretrained(self, save_path: str):
        """Save student weights"""
        torch.save(self.state_dict(), save_path)
        print(f"[AlignedVideoEncoder] Saved weights to {save_path}")


def build_aligned_encoder(config: dict) -> AlignedVideoEncoder:
    """
    Build aligned video encoder from config dict.

    Args:
        config: Dict with keys:
            - latent_dim: int
            - arch: str (VideoEncoder architecture)
            - input_channels: int
            - projection_dim: Optional[int]
            - alignment_type: str ('mse', 'cosine', 'contrastive')
            - temperature: float (for contrastive)

    Returns:
        AlignedVideoEncoder instance
    """
    return AlignedVideoEncoder(
        latent_dim=config.get('latent_dim', 128),
        arch=config.get('arch', 'simple2dcnn'),
        input_channels=config.get('input_channels', 3),
        projection_dim=config.get('projection_dim', None),
        alignment_type=config.get('alignment_type', 'mse'),
        temperature=config.get('temperature', 0.1),
    )


if __name__ == '__main__':
    """Test aligned video encoder"""
    print("Testing AlignedVideoEncoder...")

    # Create dummy video
    B, T, C, H, W = 4, 4, 3, 64, 64
    x = torch.randn(B, T, C, H, W)

    # Test basic forward
    print("\n[Basic Forward]")
    student = AlignedVideoEncoder(latent_dim=128, arch='simple2dcnn')
    z = student(x)
    print(f"Input: {x.shape} -> Output: {z.shape}")
    print(f"Parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Test with projection
    print("\n[Forward with Projection]")
    z, z_proj = student.forward_with_projection(x)
    print(f"z: {z.shape}, z_proj: {z_proj.shape}")

    # Test alignment losses
    print("\n[Alignment Losses]")
    z_teacher = torch.randn(B, 128)

    # MSE
    student_mse = AlignedVideoEncoder(latent_dim=128, alignment_type='mse')
    loss_mse = student_mse.alignment_loss(z, z_teacher)
    print(f"MSE loss: {loss_mse.item():.4f}")

    # Cosine
    student_cos = AlignedVideoEncoder(latent_dim=128, alignment_type='cosine')
    loss_cos = student_cos.alignment_loss(z, z_teacher)
    print(f"Cosine loss: {loss_cos.item():.4f}")

    # Contrastive
    student_cont = AlignedVideoEncoder(latent_dim=128, alignment_type='contrastive')
    loss_cont = student_cont.alignment_loss(z, z_teacher)
    print(f"Contrastive loss: {loss_cont.item():.4f}")

    # Test distillation step (mock teacher)
    print("\n[Distillation Step]")
    from src.encoders.teacher_adapter import TeacherAdapter
    try:
        teacher = TeacherAdapter(teacher_type='random', latent_dim=128)
        z_out, loss, metrics = student.distillation_step(x, teacher)
        print(f"z_out: {z_out.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"Distillation step test failed: {e}")

    print("\n✅ AlignedVideoEncoder tests complete!")
