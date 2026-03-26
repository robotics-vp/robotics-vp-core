from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.orchestrator.semantic_transformer_bridge import (
    SEMANTIC_WM_FEATURE_DIM,
    SELECTION_META_FEATURE_DIM,
    encode_selection_feedback_features,
    encode_semantic_world_model_features,
)


META_OBJECTIVE_PRESET_LABELS = ["balanced", "safety", "energy_saver", "throughput"]
META_BACKEND_LABELS = ["pybullet", "isaac", "mujoco", "other"]
META_ENERGY_PROFILE_LABELS = ["BASE", "BOOST", "SAVER", "SAFE"]
META_DATA_MIX_LABELS = ["real", "synthetic", "hybrid"]
META_EXPECTED_DELTA_LABELS = [
    "expected_delta_mpl",
    "expected_delta_error",
    "expected_delta_energy_Wh",
]
META_PLANNING_CONTEXT_DIM = SEMANTIC_WM_FEATURE_DIM + 3 + 4 + SELECTION_META_FEATURE_DIM


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def normalize_named_weights(
    weights: Optional[Mapping[str, Any]],
    labels: Sequence[str],
) -> Dict[str, float]:
    clean = {str(label): max(0.0, _safe_float(_mapping(weights).get(label, 0.0))) for label in labels}
    total = float(sum(clean.values()))
    if total <= 0.0:
        return {str(label): 0.0 for label in labels}
    return {str(label): float(value / total) for label, value in clean.items()}


def encode_named_distribution(
    weights: Optional[Mapping[str, Any]],
    labels: Sequence[str],
) -> np.ndarray:
    normalized = normalize_named_weights(weights, labels)
    return np.asarray([normalized.get(str(label), 0.0) for label in labels], dtype=np.float32)


def decode_named_distribution(
    values: Sequence[float] | np.ndarray,
    labels: Sequence[str],
) -> Dict[str, float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < len(labels):
        vector = np.pad(vector, (0, len(labels) - vector.size))
    logits = vector[: len(labels)]
    logits = logits - float(np.max(logits))
    exp = np.exp(logits)
    denom = float(np.sum(exp))
    if denom <= 0.0:
        return {str(label): 0.0 for label in labels}
    probs = exp / denom
    return {str(label): float(probs[idx]) for idx, label in enumerate(labels)}


def encode_objective_preset(label: str) -> int:
    normalized = str(label or "balanced")
    if normalized in META_OBJECTIVE_PRESET_LABELS:
        return META_OBJECTIVE_PRESET_LABELS.index(normalized)
    return 0


def decode_objective_preset(index: int) -> str:
    try:
        return META_OBJECTIVE_PRESET_LABELS[int(index)]
    except Exception:
        return META_OBJECTIVE_PRESET_LABELS[0]


def encode_backend_label(label: str) -> int:
    normalized = str(label or "pybullet")
    if normalized in META_BACKEND_LABELS:
        return META_BACKEND_LABELS.index(normalized)
    return META_BACKEND_LABELS.index("other")


def decode_backend_label(index: int) -> str:
    try:
        return META_BACKEND_LABELS[int(index)]
    except Exception:
        return META_BACKEND_LABELS[0]


def extract_expected_delta_vector(expected_deltas: Optional[Mapping[str, Any]]) -> np.ndarray:
    payload = _mapping(expected_deltas)
    return np.asarray(
        [_safe_float(payload.get(label, 0.0)) for label in META_EXPECTED_DELTA_LABELS],
        dtype=np.float32,
    )


def decode_expected_delta_vector(values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < len(META_EXPECTED_DELTA_LABELS):
        vector = np.pad(vector, (0, len(META_EXPECTED_DELTA_LABELS) - vector.size))
    return {
        label: float(vector[idx])
        for idx, label in enumerate(META_EXPECTED_DELTA_LABELS)
    }


def build_meta_planning_context_vector(
    *,
    semantic_summary: Optional[Mapping[str, Any]] = None,
    econ_signals: Optional[Mapping[str, Any]] = None,
    datapack_signals: Optional[Mapping[str, Any]] = None,
    selection_summary: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    semantic_vector = encode_semantic_world_model_features(_mapping(semantic_summary))
    econ_payload = _mapping(econ_signals)
    datapack_payload = _mapping(datapack_signals)
    econ_vector = np.asarray(
        [
            _safe_float(econ_payload.get("mpl_urgency", 0.0)),
            _safe_float(econ_payload.get("error_urgency", 0.0)),
            _safe_float(econ_payload.get("energy_urgency", 0.0)),
        ],
        dtype=np.float32,
    )
    datapack_vector = np.asarray(
        [
            _safe_float(datapack_payload.get("data_coverage_score", 0.0)),
            _safe_float(datapack_payload.get("embedding_diversity", 0.0)),
            _safe_float(datapack_payload.get("vla_annotation_fraction", 0.0)),
            _safe_float(datapack_payload.get("guidance_annotation_fraction", 0.0)),
        ],
        dtype=np.float32,
    )
    selection_vector = encode_selection_feedback_features(selection_summary)
    vector = np.concatenate(
        [semantic_vector, econ_vector, datapack_vector, selection_vector.astype(np.float32)]
    ).astype(np.float32)
    if vector.size < META_PLANNING_CONTEXT_DIM:
        vector = np.pad(vector, (0, META_PLANNING_CONTEXT_DIM - vector.size))
    return vector[:META_PLANNING_CONTEXT_DIM]


def build_meta_planning_context_from_task_context(
    task_context: Optional[Mapping[str, Any]],
) -> np.ndarray:
    payload = _mapping(task_context)
    return build_meta_planning_context_vector(
        semantic_summary=payload.get("semantic_summary")
        or payload.get("semantic_world_model_summary"),
        econ_signals=payload.get("econ_signals"),
        datapack_signals=payload.get("datapack_signals"),
        selection_summary=payload.get("selection_summary"),
    )


__all__ = [
    "META_BACKEND_LABELS",
    "META_DATA_MIX_LABELS",
    "META_ENERGY_PROFILE_LABELS",
    "META_EXPECTED_DELTA_LABELS",
    "META_OBJECTIVE_PRESET_LABELS",
    "META_PLANNING_CONTEXT_DIM",
    "build_meta_planning_context_from_task_context",
    "build_meta_planning_context_vector",
    "decode_backend_label",
    "decode_expected_delta_vector",
    "decode_named_distribution",
    "decode_objective_preset",
    "encode_backend_label",
    "encode_named_distribution",
    "encode_objective_preset",
    "extract_expected_delta_vector",
    "normalize_named_weights",
]
