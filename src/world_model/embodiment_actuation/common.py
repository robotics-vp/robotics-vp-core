"""Shared helpers for Embodiment / Actuation world-model contracts."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "to_dict"):
        try:
            return mapping(payload.to_dict())
        except Exception:
            return {}
    if isinstance(payload, Mapping):
        return dict(to_json_safe(dict(payload)))
    return {}


def strings(values: Optional[Sequence[Any]]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    try:
        iterable = list(values)
    except Exception:
        return []
    return [str(value) for value in iterable if value not in (None, "")]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def clip01(value: Any) -> float:
    return float(max(0.0, min(1.0, safe_float(value, 0.0))))


def float_mapping(payload: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in mapping(payload).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(mapping(payload))[:16]}"


def truth_status(missing: Sequence[str], degraded: Sequence[str] = ()) -> str:
    if missing:
        return "external_blocked"
    if degraded:
        return "degraded"
    return "available"
