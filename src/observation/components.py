"""
Structured observation components for modular routing/logging.

All fields are optional; arrays are stored as numpy arrays but serialize to
JSON-safe lists for logging or persistence.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.utils.json_safe import to_json_safe


def _to_array(values: Optional[Any]) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        return np.asarray(values, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _array_to_list(arr: Optional[Any]) -> Optional[Any]:
    if arr is None:
        return None
    try:
        return [float(x) for x in np.asarray(arr).flatten().tolist()]
    except Exception:
        return None


@dataclass
class ObservationComponents:
    vision_features: Optional[np.ndarray] = None
    proprio: Optional[np.ndarray] = None
    env_state: Optional[np.ndarray] = None
    econ_slice: Optional[Dict[str, float]] = None
    semantic_slice: Optional[Dict[str, Any]] = None
    condition_vector: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vision_features": _array_to_list(self.vision_features),
            "proprio": _array_to_list(self.proprio),
            "env_state": _array_to_list(self.env_state),
            "econ_slice": to_json_safe(self.econ_slice),
            "semantic_slice": to_json_safe(self.semantic_slice),
            "condition_vector": _array_to_list(self.condition_vector),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservationComponents":
        data = data or {}
        return cls(
            vision_features=_to_array(data.get("vision_features")) if data.get("vision_features") is not None else None,
            proprio=_to_array(data.get("proprio")) if data.get("proprio") is not None else None,
            env_state=_to_array(data.get("env_state")) if data.get("env_state") is not None else None,
            econ_slice=to_json_safe(data.get("econ_slice")) if data.get("econ_slice") is not None else None,
            semantic_slice=to_json_safe(data.get("semantic_slice")) if data.get("semantic_slice") is not None else None,
            condition_vector=_to_array(data.get("condition_vector")) if data.get("condition_vector") is not None else None,
        )
