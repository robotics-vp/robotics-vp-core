"""
Neural SIMA-2 Segmenter: NN-based boundary and primitive detection.

Replaces heuristic segmentation with learned model.

Architecture:
- Backbone: Lightweight U-Net/FPN with optional frozen encoder
- Heads:
  * Boundary mask (sigmoid): Binary segmentation of segment boundaries
  * Primitive classification (softmax): Classify detected primitives

Losses:
- Boundary: Focal loss (α, γ configurable) + optional Dice
- Primitive: Cross-entropy
- Total = λ_boundary*(focal+dice) + λ_primitive*CE

Accuracy target: ≥85% F1 on boundary mask (held-out)

Bootstrap training:
- Use heuristic segmenter outputs as pseudo-labels
- Gradually transition to neural segmenter as F1 improves
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """Double convolution block for U-Net."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)


    class UNetEncoder(nn.Module):
        """U-Net encoder (downsampling path)."""

        def __init__(self, in_channels: int = 3, features: List[int] = None):
            super().__init__()
            if features is None:
                features = [64, 128, 256, 512]

            self.features = features
            self.layers = nn.ModuleList()

            for feature in features:
                self.layers.append(DoubleConv(in_channels, feature))
                in_channels = feature

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
            """
            Returns skip connections at each level.
            """
            skips = []
            for layer in self.layers:
                x = layer(x)
                skips.append(x)
                x = self.pool(x)

            return skips


    class UNetDecoder(nn.Module):
        """U-Net decoder (upsampling path)."""

        def __init__(self, bottleneck_channels: int, encoder_channels: List[int]):
            """
            Args:
                bottleneck_channels: Number of channels from bottleneck
                encoder_channels: List of encoder feature channels (in encoder order)
            """
            super().__init__()
            self.upconvs = nn.ModuleList()
            self.conv_blocks = nn.ModuleList()

            # Reverse encoder channels for decoder (go from deep to shallow)
            decoder_channels = list(reversed(encoder_channels))

            # First upconv from bottleneck
            in_ch = bottleneck_channels
            for idx, out_ch in enumerate(decoder_channels):
                # Upconv to match encoder level resolution
                self.upconvs.append(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                )
                # Conv after concatenation (skip + upsampled)
                self.conv_blocks.append(DoubleConv(out_ch + out_ch, out_ch))
                in_ch = out_ch

        def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
            """
            Args:
                x: Bottleneck features
                skips: Skip connections from encoder (in reversed order: deep to shallow)

            Returns:
                Decoded features
            """
            for idx in range(len(self.upconvs)):
                x = self.upconvs[idx](x)

                if idx < len(skips):
                    skip = skips[idx]

                    # Handle size mismatch
                    if x.shape[2:] != skip.shape[2:]:
                        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

                    x = torch.cat([skip, x], dim=1)
                    x = self.conv_blocks[idx](x)

            return x


    class NeuralSegmenter(nn.Module):
        """
        Neural segmenter with U-Net backbone + dual heads.

        Heads:
        1. Boundary head: Predicts segment boundary mask
        2. Primitive head: Classifies detected primitives
        """

        def __init__(
            self,
            in_channels: int = 3,
            num_primitives: int = 10,
            features: List[int] = None,
            freeze_encoder: bool = False,
            seed: int = 0,
        ):
            super().__init__()

            # Set seed for deterministic initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            if features is None:
                features = [64, 128, 256, 512]

            self.num_primitives = num_primitives

            # Encoder
            self.encoder = UNetEncoder(in_channels=in_channels, features=features)

            # Freeze encoder if requested
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

            # Bottleneck
            bottleneck_channels = features[-1] * 2
            self.bottleneck = DoubleConv(features[-1], bottleneck_channels)

            # Decoder
            self.decoder = UNetDecoder(
                bottleneck_channels=bottleneck_channels,
                encoder_channels=features[:-1]  # Exclude last level (it's the bottleneck input)
            )

            # Boundary head (sigmoid)
            self.boundary_head = nn.Conv2d(features[0], 1, kernel_size=1)

            # Primitive classification head (softmax)
            # Global average pooling + FC
            self.primitive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.primitive_head = nn.Linear(features[0], num_primitives)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward pass.

            Args:
                x: Input tensor [B, C, H, W]

            Returns:
                Dict with:
                    - boundary_logits: [B, 1, H, W]
                    - primitive_logits: [B, num_primitives]
            """
            # Encode
            skips = self.encoder(x)

            # Bottleneck (operates on pooled last encoder output)
            last_encoder_out = self.encoder.pool(skips[-1])
            bottleneck_out = self.bottleneck(last_encoder_out)

            # Decode (decoder expects features in reverse order: deep to shallow)
            # Skip connections are all encoder outputs except the last one (which fed into bottleneck)
            x = self.decoder(bottleneck_out, skips=list(reversed(skips[:-1])))

            # Boundary head
            boundary_logits = self.boundary_head(x)

            # Primitive head
            prim_features = self.primitive_pool(x).squeeze(-1).squeeze(-1)
            primitive_logits = self.primitive_head(prim_features)

            return {
                "boundary_logits": boundary_logits,
                "primitive_logits": primitive_logits,
            }


def focal_loss(
    inputs: "torch.Tensor",
    targets: "torch.Tensor",
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> "torch.Tensor":
    """
    Focal loss for addressing class imbalance.

    Args:
        inputs: Predicted logits [B, ...]
        targets: Ground truth [B, ...]
        alpha: Weighting factor (0-1)
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Focal loss
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    probs = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma

    loss = alpha * focal_weight * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def dice_loss(
    inputs: "torch.Tensor",
    targets: "torch.Tensor",
    smooth: float = 1.0,
) -> "torch.Tensor":
    """
    Dice loss for segmentation.

    Args:
        inputs: Predicted logits [B, 1, H, W]
        targets: Ground truth masks [B, 1, H, W]
        smooth: Smoothing constant

    Returns:
        Dice loss
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    probs = torch.sigmoid(inputs)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice.mean()

    return loss


def compute_segmentation_loss(
    boundary_logits: "torch.Tensor",
    boundary_targets: "torch.Tensor",
    primitive_logits: Optional["torch.Tensor"] = None,
    primitive_targets: Optional["torch.Tensor"] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lambda_boundary: float = 1.0,
    lambda_primitive: float = 0.5,
    use_dice: bool = True,
) -> Dict[str, "torch.Tensor"]:
    """
    Compute combined segmentation loss.

    Args:
        boundary_logits: Predicted boundary logits [B, 1, H, W]
        boundary_targets: Ground truth masks [B, 1, H, W]
        primitive_logits: Predicted primitive logits [B, num_classes]
        primitive_targets: Ground truth primitive class [B]
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        lambda_boundary: Boundary loss weight
        lambda_primitive: Primitive loss weight
        use_dice: Whether to include Dice loss

    Returns:
        Dict with loss components
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    # Boundary loss
    focal = focal_loss(boundary_logits, boundary_targets, alpha=alpha, gamma=gamma)
    dice = dice_loss(boundary_logits, boundary_targets) if use_dice else torch.tensor(0.0)
    boundary_loss = focal + dice

    # Primitive loss
    if primitive_logits is not None and primitive_targets is not None:
        primitive_loss = F.cross_entropy(primitive_logits, primitive_targets)
    else:
        primitive_loss = torch.tensor(0.0)

    # Total loss
    total_loss = lambda_boundary * boundary_loss + lambda_primitive * primitive_loss

    return {
        "total_loss": total_loss,
        "boundary_loss": boundary_loss,
        "focal_loss": focal,
        "dice_loss": dice,
        "primitive_loss": primitive_loss,
    }


def compute_f1_score(
    predictions: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, F1 for binary segmentation.

    Args:
        predictions: Predicted logits [B, 1, H, W]
        targets: Ground truth masks [B, 1, H, W]
        threshold: Threshold for binarization

    Returns:
        (precision, recall, f1)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    preds = (torch.sigmoid(predictions) > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1


def save_checkpoint(
    model: "NeuralSegmenter",
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    checkpoint_path: Path,
    seed: int,
):
    """Save deterministic checkpoint (JSON-safe)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "trained_steps": epoch,
        "seed": seed,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    return torch.load(checkpoint_path)
