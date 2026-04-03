"""Shared helpers for perception/grounding world model typed objects."""

from __future__ import annotations

from typing import Any, Mapping


def mapping(value: Any) -> dict[str, Any]:
    """Coerce to dict safely."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def strings(value: Any) -> list[str]:
    """Coerce to list of non-empty strings."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return []


def clip01(value: float) -> float:
    """Clamp a float to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def stable_id(*parts: str) -> str:
    """Build a deterministic ID from parts."""
    import hashlib

    raw = ":".join(str(p) for p in parts if p)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
