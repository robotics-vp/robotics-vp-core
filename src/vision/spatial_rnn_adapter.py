"""
Deterministic spatial RNN/ConvGRU-style stub for feature sequences.
"""
from typing import Any, Dict, Sequence

import numpy as np


def run_spatial_rnn(features: Sequence[Any], decay: float = 0.65, input_scale: float = 0.35) -> np.ndarray:
    if not features:
        return np.array([], dtype=np.float32)
    hidden = np.zeros_like(np.asarray(features[0], dtype=np.float32))
    for idx, feat in enumerate(features):
        arr = np.asarray(feat, dtype=np.float32)
        gate = 1.0 / float(idx + 1)
        hidden = decay * hidden + input_scale * arr + gate * arr.mean()
    return hidden.astype(np.float32)


def tensor_to_json_safe(tensor: np.ndarray) -> Dict[str, Any]:
    try:
        return {"values": [float(x) for x in np.asarray(tensor, dtype=np.float32).flatten().tolist()]}
    except Exception:
        return {"values": []}
