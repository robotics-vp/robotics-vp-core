"""
Deterministic RegNet-style backbone with real convolutional layers.

Implements:
- RegNet-style bottleneck blocks (group convolution)
- Multi-scale feature pyramid extraction (P3/P4/P5 or C3/C4/C5)
- Deterministic initialization and forward pass given seed
- Fallback to hash-based stub when PyTorch unavailable
"""
import hashlib
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.vision.interfaces import VisionFrame

DEFAULT_LEVELS = ("P3", "P4", "P5")
DEFAULT_STRIDES = {"P3": 8, "P4": 16, "P5": 32}

# Try importing PyTorch; fallback to numpy-only mode
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _frame_signature(frame: VisionFrame) -> str:
    try:
        payload = json.dumps(frame.to_dict(), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = f"{frame.backend}|{frame.task_id}|{frame.episode_id}|{frame.timestep}|{frame.state_digest}"
    return payload


def _stable_vector(signature: str, level: str, feature_dim: int) -> np.ndarray:
    """Hash-based deterministic feature vector (fallback mode)."""
    vals = []
    for idx in range(feature_dim):
        digest = hashlib.sha256(f"{signature}|{level}|{idx}".encode("utf-8")).hexdigest()
        vals.append(int(digest[:12], 16) / float(16**12))
    return np.array(vals, dtype=np.float32)


if TORCH_AVAILABLE:
    class RegNetBottleneck(nn.Module):
        """RegNet-style bottleneck block with group convolution."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 8):
            super().__init__()
            mid_channels = out_channels // 2
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=stride, padding=1,
                groups=min(groups, mid_channels), bias=False
            )
            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

            # Downsample if needed
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x

            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
            return out


    class RegNetBackbone(nn.Module):
        """
        RegNet-style backbone for feature pyramid extraction.

        Produces multi-scale features at different strides:
        - P3 (C3): stride 8
        - P4 (C4): stride 16
        - P5 (C5): stride 32
        """

        def __init__(
            self,
            in_channels: int = 3,
            feature_dims: Dict[str, int] = None,
            levels: Sequence[str] = DEFAULT_LEVELS,
            groups: int = 8,
            groups: int = 8,
            seed: int = 0,
            use_checkpointing: bool = False,
        ):
            super().__init__()
            self.use_checkpointing = use_checkpointing

            # Set seed for deterministic initialization
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            self.levels = list(levels)
            self.feature_dims = feature_dims or {level: 256 for level in levels}

            # Stem: initial convolution
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # Build stages for different strides
            current_channels = 64
            self.stages = nn.ModuleDict()

            for level in self.levels:
                stride = DEFAULT_STRIDES.get(level, 8)
                out_channels = self.feature_dims[level]

                # Determine how many stride-2 blocks we need
                # Stem already gives us stride 4, so:
                # P3 (stride 8): 1 stride-2 block
                # P4 (stride 16): 2 stride-2 blocks
                # P5 (stride 32): 3 stride-2 blocks
                num_stride_blocks = {8: 1, 16: 2, 32: 3}.get(stride, 1)

                stage_blocks = []
                for i in range(num_stride_blocks):
                    block_stride = 2 if i == 0 else 1
                    stage_blocks.append(
                        RegNetBottleneck(current_channels, out_channels, stride=block_stride, groups=groups)
                    )
                    current_channels = out_channels

                self.stages[level] = nn.Sequential(*stage_blocks)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward pass producing multi-scale features.

            Args:
                x: Input tensor [B, C, H, W]

            Returns:
                Dict mapping level names to feature tensors
            """
            features = {}

            # Stem
            x = self.stem(x)

            # Progressive stages (each consumes output of previous)
            # Progressive stages (each consumes output of previous)
            for level in self.levels:
                if self.use_checkpointing:
                    from src.utils.training_env import checkpoint_if_enabled
                    # Checkpoint the whole stage
                    x = checkpoint_if_enabled(self.stages[level], x, enabled=True)
                else:
                    x = self.stages[level](x)
                
                # Clamp to prevent NaN/Inf
                x = torch.clamp(x, min=-1e6, max=1e6)
                features[level] = x

            return features


def build_regnet_feature_pyramid(
    frame: VisionFrame,
    feature_dim: int = 8,
    levels: Sequence[str] = DEFAULT_LEVELS,
    use_neural: bool = False,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build feature pyramid from VisionFrame.

    Args:
        frame: Input vision frame
        feature_dim: Dimension of feature vectors per level
        levels: Pyramid levels to extract (e.g., ["P3", "P4", "P5"])
        use_neural: If True and PyTorch available, use neural network; else use hash-based stub
        seed: Random seed for deterministic initialization

    Returns:
        Dict mapping level names to feature vectors (flattened)
    """
    if not use_neural or not TORCH_AVAILABLE:
        # Fallback to hash-based stub
        signature = _frame_signature(frame)
        return {str(level): _stable_vector(signature, str(level), feature_dim) for level in levels}

    # Neural mode: use RegNetBackbone
    # For stub compatibility, we generate dummy input from frame signature
    signature = _frame_signature(frame)

    # Generate deterministic "image" from signature (3x224x224)
    # This is still a stub - in real usage, you'd load actual RGB data
    np.random.seed(hash(signature) % (2**32))
    dummy_img = np.random.randn(3, 224, 224).astype(np.float32)

    # Build model
    feature_dims = {level: feature_dim for level in levels}
    model = RegNetBackbone(in_channels=3, feature_dims=feature_dims, levels=levels, seed=seed)
    model.eval()

    # Forward pass
    with torch.no_grad():
        img_tensor = torch.from_numpy(dummy_img).unsqueeze(0)  # [1, 3, H, W]
        features_dict = model(img_tensor)

    # Flatten to vectors
    result = {}
    for level, feat_tensor in features_dict.items():
        # Global average pooling to get fixed-size vector
        pooled = feat_tensor.mean(dim=[2, 3])  # [1, C]
        vector = pooled.squeeze(0).cpu().numpy()  # [C]

        # Normalize to prevent scale drift
        vector = np.clip(vector, -10.0, 10.0)
        epsilon = 1e-8
        norm = np.linalg.norm(vector) + epsilon
        vector = vector / norm

        result[str(level)] = vector.astype(np.float32)

    return result


def flatten_pyramid(pyramid: Dict[str, np.ndarray]) -> np.ndarray:
    if not pyramid:
        return np.array([], dtype=np.float32)
    ordered = []
    for level in sorted(pyramid.keys()):
        ordered.append(np.asarray(pyramid[level], dtype=np.float32).flatten())
    return np.concatenate(ordered) if ordered else np.array([], dtype=np.float32)


def pyramid_to_json_safe(pyramid: Dict[str, np.ndarray]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for k, v in pyramid.items():
        try:
            safe[str(k)] = [float(x) for x in np.asarray(v, dtype=np.float32).flatten().tolist()]
        except Exception:
            safe[str(k)] = []
    return safe
