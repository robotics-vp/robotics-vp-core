"""
Deterministic BiFPN-style fusion stub.
"""
from typing import Dict, Optional

import numpy as np


def fuse_feature_pyramid(pyramid: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
    if not pyramid:
        return {}
    keys = sorted(pyramid.keys())
    base_weights = weights or {k: 1.0 / float(idx + 1) for idx, k in enumerate(keys)}
    feature_dim = min(len(np.asarray(v).flatten()) for v in pyramid.values())
    fused: Dict[str, np.ndarray] = {}
    for k in keys:
        vec = np.asarray(pyramid[k], dtype=np.float32).flatten()[:feature_dim]
        neighbor_mean = np.zeros(feature_dim, dtype=np.float32)
        if len(keys) > 1:
            neighbors = [np.asarray(pyramid[n], dtype=np.float32).flatten()[:feature_dim] for n in keys if n != k]
            neighbor_mean = np.mean(neighbors, axis=0)
        weight = float(base_weights.get(k, 1.0))
        denom = weight + 0.5
        fused_vec = ((weight * vec) + (0.5 * neighbor_mean)) / denom if denom != 0 else vec
        fused[k] = fused_vec.astype(np.float32)
    return fused
