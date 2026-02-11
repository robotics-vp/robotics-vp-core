"""Portable objective tensor container."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np

from src.objectives.schema import ObjectiveTensorSchema


def _to_numpy(values: Any) -> np.ndarray:
    """Convert numpy/torch/list payloads into float32 numpy arrays."""
    if isinstance(values, np.ndarray):
        return values.astype(np.float32, copy=False)
    try:
        import torch  # type: ignore

        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        pass
    return np.asarray(values, dtype=np.float32)


@dataclass
class ObjectiveTensor:
    """Typed objective tensor with schema and provenance metadata."""

    values: Any
    schema: ObjectiveTensorSchema = field(default_factory=ObjectiveTensorSchema)
    context: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: str = "objective_tensor_v1"

    def __post_init__(self) -> None:
        arr = _to_numpy(self.values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[-1] != len(self.schema.axes):
            raise ValueError(
                f"ObjectiveTensor shape mismatch: expected last dim {len(self.schema.axes)}, got {arr.shape[-1]}"
            )
        self.values = arr
        self.context.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.context.setdefault("schema_id", self.schema.schema_id)
        self.context.setdefault("shape_signature", self.schema.shape_signature())

    @property
    def batch_size(self) -> int:
        if self.values.ndim == 1:
            return 1
        return int(np.prod(self.values.shape[:-1]))

    def to_numpy(self, normalize: bool = False) -> np.ndarray:
        arr = _to_numpy(self.values)
        if normalize:
            return self.schema.normalize_values(arr)
        return arr

    def axis(self, axis_name: str, normalize: bool = False) -> np.ndarray:
        idx = self.schema.axis_index(axis_name)
        arr = self.to_numpy(normalize=normalize)
        return arr[..., idx]

    def mean_vector(self, normalize: bool = False) -> np.ndarray:
        arr = self.to_numpy(normalize=normalize)
        if arr.ndim == 1:
            return arr
        return arr.reshape(-1, arr.shape[-1]).mean(axis=0)

    def clone_with(
        self,
        *,
        values: Optional[Any] = None,
        context: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "ObjectiveTensor":
        next_context = dict(self.context)
        next_provenance = dict(self.provenance)
        if context:
            next_context.update(dict(context))
        if provenance:
            next_provenance.update(dict(provenance))
        return ObjectiveTensor(
            values=self.values if values is None else values,
            schema=self.schema,
            context=next_context,
            provenance=next_provenance,
            version=self.version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "schema": self.schema.to_dict(),
            "values": self.to_numpy().tolist(),
            "context": dict(self.context),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveTensor":
        schema_payload = payload.get("schema", {})
        schema = (
            schema_payload
            if isinstance(schema_payload, ObjectiveTensorSchema)
            else ObjectiveTensorSchema.from_dict(schema_payload)
        )
        return cls(
            values=payload.get("values", []),
            schema=schema,
            context=dict(payload.get("context", {}) or {}),
            provenance=dict(payload.get("provenance", {}) or {}),
            version=str(payload.get("version", "objective_tensor_v1")),
        )


def objective_tensor_from_axes(
    axis_values: Mapping[str, float],
    schema: Optional[ObjectiveTensorSchema] = None,
    context: Optional[MutableMapping[str, Any]] = None,
    provenance: Optional[MutableMapping[str, Any]] = None,
) -> ObjectiveTensor:
    """Build an ObjectiveTensor from axis->value mappings."""
    use_schema = schema or ObjectiveTensorSchema()
    vec = [float(axis_values.get(axis, 0.0)) for axis in use_schema.axes]
    return ObjectiveTensor(
        values=np.asarray(vec, dtype=np.float32),
        schema=use_schema,
        context=dict(context or {}),
        provenance=dict(provenance or {}),
    )
