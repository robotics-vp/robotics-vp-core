"""NPZ serialization for ObjectiveTensor artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from src.objectives.tensor import ObjectiveTensor

OBJECTIVE_TENSOR_PREFIX = "objective_tensor_v1/"


def objective_tensor_to_npz_dict(objective_tensor: ObjectiveTensor) -> Dict[str, np.ndarray]:
    """Serialize ObjectiveTensor to a numpy-only dict for np.savez."""
    return {
        f"{OBJECTIVE_TENSOR_PREFIX}version": np.array([objective_tensor.version], dtype="U32"),
        f"{OBJECTIVE_TENSOR_PREFIX}values": objective_tensor.to_numpy().astype(np.float32),
        f"{OBJECTIVE_TENSOR_PREFIX}schema_json": np.array(
            [json.dumps(objective_tensor.schema.to_dict(), sort_keys=True)],
            dtype="U16384",
        ),
        f"{OBJECTIVE_TENSOR_PREFIX}context_json": np.array(
            [json.dumps(objective_tensor.context, sort_keys=True, default=str)],
            dtype="U16384",
        ),
        f"{OBJECTIVE_TENSOR_PREFIX}provenance_json": np.array(
            [json.dumps(objective_tensor.provenance, sort_keys=True, default=str)],
            dtype="U16384",
        ),
    }


def objective_tensor_from_npz_dict(payload: Mapping[str, Any]) -> ObjectiveTensor:
    """Deserialize ObjectiveTensor from an np.load mapping or dict."""

    def _get(name: str) -> Any:
        k = f"{OBJECTIVE_TENSOR_PREFIX}{name}"
        if k in payload:
            return payload[k]
        return payload[name]

    values = np.asarray(_get("values"), dtype=np.float32)
    schema_json = _get("schema_json")
    context_json = _get("context_json")
    provenance_json = _get("provenance_json")
    version_arr = _get("version")

    version = str(version_arr[0]) if hasattr(version_arr, "__len__") and len(version_arr) > 0 else "objective_tensor_v1"
    schema = json.loads(str(schema_json[0]))
    context = json.loads(str(context_json[0]))
    provenance = json.loads(str(provenance_json[0]))

    return ObjectiveTensor.from_dict(
        {
            "version": version,
            "values": values,
            "schema": schema,
            "context": context,
            "provenance": provenance,
        }
    )


def save_objective_tensor_npz(path: Path | str, objective_tensor: ObjectiveTensor) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **objective_tensor_to_npz_dict(objective_tensor))


def load_objective_tensor_npz(path: Path | str) -> ObjectiveTensor:
    data = np.load(path, allow_pickle=False)
    return objective_tensor_from_npz_dict(dict(data))
