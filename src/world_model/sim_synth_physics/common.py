"""Shared helpers for sim/synth/physics world-model contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or []) if value not in (None, "")]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def clip01(value: Any) -> float:
    return float(max(0.0, min(1.0, safe_float(value, 0.0))))


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(mapping(payload))[:16]}"
