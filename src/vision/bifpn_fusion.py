"""
Deterministic BiFPN-style fusion with real weighted bidirectional fusion.

Implements:
- Normalized positive weights with epsilon guard
- Top-down and bottom-up fusion paths (bidirectional)
- Deterministic level ordering
- Edge case handling (single level, mismatched dims, empty input)
"""
from typing import Dict, Optional

import numpy as np

# Try importing PyTorch for neural implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_EPSILON = 1e-4


def fuse_feature_pyramid(
    pyramid: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    epsilon: float = DEFAULT_EPSILON,
    use_neural: bool = False,
    use_checkpointing: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Fuse multi-scale feature pyramid using BiFPN-style weighted fusion.

    Args:
        pyramid: Dict mapping level names to feature arrays
        weights: Optional per-level fusion weights (normalized internally)
        epsilon: Small constant for numerical stability
        use_neural: If True and PyTorch available, use neural BiFPN; else use numpy stub

    Returns:
        Dict mapping level names to fused feature arrays
    """
    if not pyramid:
        return {}

    # Edge case: single level
    if len(pyramid) == 1:
        level, feat = next(iter(pyramid.items()))
        return {level: np.asarray(feat, dtype=np.float32)}

    keys = sorted(pyramid.keys())

    # Normalize weights
    base_weights = weights or {k: 1.0 / float(idx + 1) for idx, k in enumerate(keys)}
    total_weight = sum(base_weights.values()) + epsilon
    normalized_weights = {k: max(epsilon, float(v)) / total_weight for k, v in base_weights.items()}

    # Align dimensions (use minimum feature dim to avoid mismatches)
    feature_dim = min(len(np.asarray(v).flatten()) for v in pyramid.values())

    if not use_neural or not TORCH_AVAILABLE:
        # NumPy stub implementation
        return _fuse_pyramid_numpy(pyramid, keys, normalized_weights, feature_dim, epsilon)

    # Neural implementation (PyTorch)
    if use_checkpointing and TORCH_AVAILABLE:
        from src.utils.training_env import checkpoint_if_enabled
        # Checkpointing requires tensors as input.
        # We need to unpack the pyramid dict into a list of tensors.
        # Note: We assume pyramid values are numpy arrays here (as per type hint), 
        # but _fuse_pyramid_neural converts them to tensors.
        # To checkpoint _fuse_pyramid_neural, we should pass tensors to it.
        # But _fuse_pyramid_neural signature takes dict of numpy arrays.
        # We need a wrapper that takes tensors and returns tensors.
        
        # 1. Convert pyramid to sorted list of tensors
        sorted_keys = sorted(pyramid.keys())
        tensors = []
        for k in sorted_keys:
            vec = np.asarray(pyramid[k], dtype=np.float32).flatten()[:feature_dim]
            t = torch.from_numpy(vec).unsqueeze(0) # [1, C]
            t.requires_grad_(True) # Ensure grad for checkpointing
            tensors.append(t)
            
        # 2. Define wrapper
        def run_fusion(*args):
            # args are tensors corresponding to sorted_keys
            # Reconstruct pyramid dict for _fuse_pyramid_neural (but it expects numpy arrays...)
            # Actually _fuse_pyramid_neural converts numpy to tensor.
            # We should modify _fuse_pyramid_neural or create a tensor-only version.
            # Creating a tensor-only version is cleaner.
            
            # Let's define a tensor-only fusion here
            input_map = {k: v for k, v in zip(sorted_keys, args)}
            
            # Top-down
            top_down = {}
            for i in range(len(sorted_keys) - 1, -1, -1):
                level = sorted_keys[i]
                if i == len(sorted_keys) - 1:
                    top_down[level] = input_map[level]
                else:
                    finer_level = sorted_keys[i + 1]
                    w1 = normalized_weights.get(level, 1.0)
                    w2 = normalized_weights.get(finer_level, 0.5)
                    w_sum = w1 + w2 + epsilon
                    top_down[level] = (w1 * input_map[level] + w2 * top_down[finer_level]) / w_sum
            
            # Bottom-up
            bottom_up = {}
            for i in range(len(sorted_keys)):
                level = sorted_keys[i]
                if i == 0:
                    w1 = normalized_weights.get(level, 1.0)
                    w2 = 0.5
                    w_sum = w1 + w2 + epsilon
                    bottom_up[level] = (w1 * input_map[level] + w2 * top_down[level]) / w_sum
                else:
                    coarser_level = sorted_keys[i - 1]
                    w1 = normalized_weights.get(level, 1.0)
                    w2 = normalized_weights.get(coarser_level, 0.5)
                    w3 = 0.5
                    w_sum = w1 + w2 + w3 + epsilon
                    bottom_up[level] = (
                        w1 * input_map[level] +
                        w2 * bottom_up[coarser_level] +
                        w3 * top_down[level]
                    ) / w_sum
            
            # Return tensors in sorted order
            return tuple(bottom_up[k] for k in sorted_keys)

        # 3. Run checkpointed
        fused_tensors = checkpoint_if_enabled(run_fusion, *tensors, enabled=True)
        
        # 4. Convert back to dict of numpy
        result = {}
        for k, t in zip(sorted_keys, fused_tensors):
            vec = t.squeeze(0).detach().cpu().numpy()
            vec = np.clip(vec, -1e6, 1e6)
            result[k] = vec.astype(np.float32)
        return result

    return _fuse_pyramid_neural(pyramid, keys, normalized_weights, feature_dim, epsilon)


def _fuse_pyramid_numpy(
    pyramid: Dict[str, np.ndarray],
    keys: list,
    weights: Dict[str, float],
    feature_dim: int,
    epsilon: float,
) -> Dict[str, np.ndarray]:
    """NumPy-based BiFPN fusion (stub mode)."""
    fused: Dict[str, np.ndarray] = {}

    for k in keys:
        vec = np.asarray(pyramid[k], dtype=np.float32).flatten()[:feature_dim]

        # Compute neighbor mean
        neighbor_mean = np.zeros(feature_dim, dtype=np.float32)
        if len(keys) > 1:
            neighbors = [np.asarray(pyramid[n], dtype=np.float32).flatten()[:feature_dim] for n in keys if n != k]
            if neighbors:
                neighbor_mean = np.mean(neighbors, axis=0)

        # Weighted fusion
        weight = float(weights.get(k, 1.0))
        neighbor_weight = 0.5
        denom = weight + neighbor_weight + epsilon
        fused_vec = ((weight * vec) + (neighbor_weight * neighbor_mean)) / denom

        # Clamp to prevent NaN/Inf
        fused_vec = np.clip(fused_vec, -1e6, 1e6)
        fused[k] = fused_vec.astype(np.float32)

    return fused


if TORCH_AVAILABLE:
    def _fuse_pyramid_neural(
        pyramid: Dict[str, np.ndarray],
        keys: list,
        weights: Dict[str, float],
        feature_dim: int,
        epsilon: float,
    ) -> Dict[str, np.ndarray]:
        """PyTorch-based BiFPN fusion with top-down and bottom-up paths."""
        # Convert to tensors
        features = {}
        for k in keys:
            vec = np.asarray(pyramid[k], dtype=np.float32).flatten()[:feature_dim]
            features[k] = torch.from_numpy(vec).unsqueeze(0)  # [1, C]

        # BiFPN: Top-down path
        top_down = {}
        for i in range(len(keys) - 1, -1, -1):
            level = keys[i]
            if i == len(keys) - 1:
                # Finest level: no upsampling needed
                top_down[level] = features[level]
            else:
                # Fuse with finer level
                finer_level = keys[i + 1]
                w1 = weights.get(level, 1.0)
                w2 = weights.get(finer_level, 0.5)
                w_sum = w1 + w2 + epsilon

                # Weighted fusion (no spatial upsampling since we're working with vectors)
                top_down[level] = (w1 * features[level] + w2 * top_down[finer_level]) / w_sum

        # BiFPN: Bottom-up path
        bottom_up = {}
        for i in range(len(keys)):
            level = keys[i]
            if i == 0:
                # Coarsest level: combine original + top-down
                w1 = weights.get(level, 1.0)
                w2 = 0.5
                w_sum = w1 + w2 + epsilon
                bottom_up[level] = (w1 * features[level] + w2 * top_down[level]) / w_sum
            else:
                # Fuse with coarser level + top-down
                coarser_level = keys[i - 1]
                w1 = weights.get(level, 1.0)
                w2 = weights.get(coarser_level, 0.5)
                w3 = 0.5  # top-down weight
                w_sum = w1 + w2 + w3 + epsilon

                bottom_up[level] = (
                    w1 * features[level] +
                    w2 * bottom_up[coarser_level] +
                    w3 * top_down[level]
                ) / w_sum

        # Convert back to numpy
        fused = {}
        for level, tensor in bottom_up.items():
            vec = tensor.squeeze(0).detach().cpu().numpy()
            # Clamp to prevent NaN/Inf
            vec = np.clip(vec, -1e6, 1e6)
            fused[level] = vec.astype(np.float32)

        return fused
else:
    def _fuse_pyramid_neural(
        pyramid: Dict[str, np.ndarray],
        keys: list,
        weights: Dict[str, float],
        feature_dim: int,
        epsilon: float,
    ) -> Dict[str, np.ndarray]:
        """Fallback when PyTorch unavailable."""
        return _fuse_pyramid_numpy(pyramid, keys, weights, feature_dim, epsilon)
