"""
Utility helpers to convert intermediate artifacts into JSON-safe objects.

Keeps downstream consumers (Claude, analytics scripts) aligned without changing
core model behavior.
"""
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import json

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def to_json_safe(obj: Any, include_tensors: bool = False) -> Any:
    """
    Recursively convert objects to JSON-serializable structures.

    Args:
        obj: Arbitrary Python object
        include_tensors: If True, tensors are converted to lists; otherwise they
            are replaced with a placeholder string to avoid large payloads.
    """
    if obj is None:
        return None

    if is_dataclass(obj):
        return {k: to_json_safe(v, include_tensors) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {k: to_json_safe(v, include_tensors) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v, include_tensors) for v in obj]

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()

    if torch is not None and hasattr(torch, "Tensor") and isinstance(obj, torch.Tensor):  # type: ignore
        if not include_tensors:
            return "<tensor omitted>"
        return obj.detach().cpu().tolist()

    if hasattr(obj, "to_dict"):
        try:
            return to_json_safe(obj.to_dict(), include_tensors)
        except Exception:
            # Fall through to best-effort conversion
            pass

    # Simple scalars or already JSON-safe objects
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def json_dumps_safe(obj: Any, include_tensors: bool = False, **kwargs) -> str:
    """Dump an object to JSON string using safe conversion first."""
    return json.dumps(to_json_safe(obj, include_tensors=include_tensors), **kwargs)
