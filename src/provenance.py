"""Common NPZ provenance helpers for objective and regal artifacts."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np


def _json_arr(value: Mapping[str, Any]) -> np.ndarray:
    return np.array([json.dumps(dict(value), sort_keys=True, default=str)], dtype="U16384")


def stamp_objective_tensor_metadata(
    payload: MutableMapping[str, np.ndarray],
    *,
    objective_tensor_metadata: Mapping[str, Any],
) -> MutableMapping[str, np.ndarray]:
    """Attach objective tensor metadata into NPZ payloads."""
    payload["provenance/objective_tensor_json"] = _json_arr(objective_tensor_metadata)
    return payload


def stamp_regal_decision_metadata(
    payload: MutableMapping[str, np.ndarray],
    *,
    regal_decision_metadata: Mapping[str, Any],
) -> MutableMapping[str, np.ndarray]:
    """Attach regal decision metadata into NPZ payloads."""
    payload["provenance/regal_decision_json"] = _json_arr(regal_decision_metadata)
    return payload


def extract_stamped_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Recover stamped objective/regal metadata from NPZ payloads."""
    out: Dict[str, Any] = {}
    objective_key = "provenance/objective_tensor_json"
    regal_key = "provenance/regal_decision_json"

    if objective_key in payload:
        raw = payload[objective_key]
        if hasattr(raw, "__len__") and len(raw) > 0:
            out["objective_tensor"] = json.loads(str(raw[0]))
    if regal_key in payload:
        raw = payload[regal_key]
        if hasattr(raw, "__len__") and len(raw) > 0:
            out["regal_decision"] = json.loads(str(raw[0]))
    return out
