"""
Deterministic RegNet-style backbone stub.
"""
import hashlib
import json
from typing import Any, Dict, Sequence

import numpy as np

from src.vision.interfaces import VisionFrame

DEFAULT_LEVELS = ("P3", "P4", "P5")


def _frame_signature(frame: VisionFrame) -> str:
    try:
        payload = json.dumps(frame.to_dict(), sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = f"{frame.backend}|{frame.task_id}|{frame.episode_id}|{frame.timestep}|{frame.state_digest}"
    return payload


def _stable_vector(signature: str, level: str, feature_dim: int) -> np.ndarray:
    vals = []
    for idx in range(feature_dim):
        digest = hashlib.sha256(f"{signature}|{level}|{idx}".encode("utf-8")).hexdigest()
        vals.append(int(digest[:12], 16) / float(16**12))
    return np.array(vals, dtype=np.float32)


def build_regnet_feature_pyramid(
    frame: VisionFrame,
    feature_dim: int = 8,
    levels: Sequence[str] = DEFAULT_LEVELS,
) -> Dict[str, np.ndarray]:
    signature = _frame_signature(frame)
    return {str(level): _stable_vector(signature, str(level), feature_dim) for level in levels}


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
