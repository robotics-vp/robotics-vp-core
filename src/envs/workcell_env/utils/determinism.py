"""
Determinism helpers for workcell environments.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, Iterable, Mapping

import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def deterministic_episode_id(prefix: str, seed: int, extra: Mapping[str, Any] | None = None) -> str:
    """Create a deterministic episode id without UUIDs."""
    payload = {"prefix": prefix, "seed": seed, "extra": dict(extra or {})}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def hash_state(state: Mapping[str, Any]) -> str:
    """Hash a state dict deterministically for trajectory comparisons."""
    payload = json.dumps(_json_safe(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_trajectory(states: Iterable[Mapping[str, Any]]) -> list[str]:
    """Hash each state in a trajectory."""
    return [hash_state(state) for state in states]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
