"""Schema definitions for portable objective tensors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ObjectiveTensorSchema:
    """Schema describing axis semantics, units, and normalization rules."""

    schema_id: str = "objective_tensor_v1"
    axes: Tuple[str, ...] = (
        "throughput",
        "error",
        "safety",
        "energy",
    )
    units: Dict[str, str] = field(
        default_factory=lambda: {
            "throughput": "units_per_hour",
            "error": "fraction",
            "safety": "score",
            "energy": "wh_per_unit",
        }
    )
    normalization: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowed_transforms: Tuple[str, ...] = (
        "identity",
        "scale",
        "clip",
        "affine",
    )

    def axis_index(self, axis: str) -> int:
        if axis not in self.axes:
            raise KeyError(f"Unknown objective axis: {axis}")
        return self.axes.index(axis)

    def shape_signature(self) -> str:
        payload = {
            "schema_id": self.schema_id,
            "axes": list(self.axes),
            "units": {k: self.units.get(k, "") for k in self.axes},
            "allowed_transforms": list(self.allowed_transforms),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def normalize_values(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[-1] != len(self.axes):
            raise ValueError(
                f"Expected last objective dimension {len(self.axes)}, got {arr.shape[-1]}"
            )
        out = arr.copy()
        for i, axis in enumerate(self.axes):
            rule = self.normalization.get(axis, {})
            mode = str(rule.get("mode", "identity")).lower()
            if mode == "identity":
                continue
            if mode == "minmax":
                lo = float(rule.get("min", 0.0))
                hi = float(rule.get("max", 1.0))
                denom = max(hi - lo, 1e-8)
                out[..., i] = (out[..., i] - lo) / denom
            elif mode == "zscore":
                mean = float(rule.get("mean", 0.0))
                std = max(float(rule.get("std", 1.0)), 1e-8)
                out[..., i] = (out[..., i] - mean) / std
            elif mode == "clip":
                lo = float(rule.get("min", -1e9))
                hi = float(rule.get("max", 1e9))
                out[..., i] = np.clip(out[..., i], lo, hi)
            else:
                raise ValueError(f"Unsupported normalization mode '{mode}' for axis '{axis}'")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "axes": list(self.axes),
            "units": dict(self.units),
            "normalization": dict(self.normalization),
            "allowed_transforms": list(self.allowed_transforms),
            "shape_signature": self.shape_signature(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveTensorSchema":
        axes = tuple(payload.get("axes", ()) or ()) if payload else ()
        if not axes:
            axes = cls().axes
        transforms = tuple(payload.get("allowed_transforms", ()) or ()) if payload else ()
        if not transforms:
            transforms = cls().allowed_transforms
        return cls(
            schema_id=str(payload.get("schema_id", "objective_tensor_v1")),
            axes=axes,
            units=dict(payload.get("units", {}) or {}),
            normalization=dict(payload.get("normalization", {}) or {}),
            allowed_transforms=transforms,
        )
