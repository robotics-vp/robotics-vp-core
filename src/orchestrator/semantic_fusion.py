"""Semantic fusion result schema and MVP fusion logic."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

SEMANTIC_FUSION_VERSION = "v1"
SEMANTIC_FUSION_PREFIX = "semantic_fusion_v1/"


@dataclass
class SemanticFusionResult:
    """Versioned, numpy-only semantic fusion output."""

    fused_class_probs: np.ndarray
    fused_confidence: np.ndarray
    chosen_policy_id: np.ndarray
    evidence_weights: Optional[np.ndarray] = None
    evidence_used_mask: Optional[np.ndarray] = None
    evidence_keys: Optional[List[str]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def to_npz(self, export_float16: bool = True) -> Dict[str, np.ndarray]:
        probs_dtype = np.float16 if export_float16 else np.float32
        data: Dict[str, np.ndarray] = {
            f"{SEMANTIC_FUSION_PREFIX}version": np.array([SEMANTIC_FUSION_VERSION], dtype="U8"),
            f"{SEMANTIC_FUSION_PREFIX}fused_class_probs": self.fused_class_probs.astype(probs_dtype),
            f"{SEMANTIC_FUSION_PREFIX}fused_confidence": self.fused_confidence.astype(np.float32),
            f"{SEMANTIC_FUSION_PREFIX}chosen_policy_id": self.chosen_policy_id.astype(np.int32),
        }
        if self.evidence_weights is not None:
            weight_dtype = np.float16 if export_float16 else np.float32
            data[f"{SEMANTIC_FUSION_PREFIX}evidence_weights"] = self.evidence_weights.astype(weight_dtype)
        if self.evidence_used_mask is not None:
            data[f"{SEMANTIC_FUSION_PREFIX}evidence_used_mask"] = self.evidence_used_mask.astype(bool)
        if self.evidence_keys is not None:
            data[f"{SEMANTIC_FUSION_PREFIX}evidence_keys"] = np.array(self.evidence_keys, dtype="U64")
        if self.diagnostics is not None:
            data[f"{SEMANTIC_FUSION_PREFIX}diagnostics_json"] = np.array([
                json.dumps(self.diagnostics)
            ], dtype="U4096")
        return data


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 0.0, None)
    denom = np.sum(probs, axis=-1, keepdims=True)
    return probs / (denom + 1e-6)


def _weight_from_residual(residual: Optional[np.ndarray]) -> np.ndarray:
    if residual is None:
        return 1.0
    return 1.0 / (1.0 + np.maximum(residual, 0.0))


def _infer_time_track_shape(*arrays: Optional[np.ndarray]) -> tuple[int, int]:
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    return 1, 1


def fuse_semantic_evidence_mvp(
    vla_class_probs: Optional[np.ndarray],
    vla_confidence: Optional[np.ndarray],
    map_semantics: Optional[np.ndarray],
    map_stability: Optional[np.ndarray],
    geom_residual: Optional[np.ndarray] = None,
    occlusion: Optional[np.ndarray] = None,
    dynamic_evidence: Optional[np.ndarray] = None,
    mhn_plausibility: Optional[float] = None,
    num_classes: int = 1,
) -> SemanticFusionResult:
    """Fuse VLA and map semantics with simple evidence gating."""
    if vla_class_probs is None and map_semantics is None:
        T, K = _infer_time_track_shape(
            map_stability,
            geom_residual,
            occlusion,
            dynamic_evidence,
            vla_confidence,
        )
        C = max(int(num_classes), 1)
        fused = np.full((T, K, C), 1.0 / float(C), dtype=np.float32)
        fused_confidence = np.full((T, K), 0.05, dtype=np.float32)
        chosen_policy_id = np.full((T, K), -1, dtype=np.int32)
        weights = np.zeros((T, K, 1), dtype=np.float32)
        evidence_used_mask = np.zeros((T, K, 1), dtype=bool)
        entropy = -np.sum(fused * np.log(fused + 1e-8), axis=-1)
        diagnostics = {
            "entropy_mean": float(np.mean(entropy)),
            "disagreement_mean": 0.0,
            "coverage_pct": float(np.mean(evidence_used_mask.astype(np.float32))),
            "fallback": True,
        }
        return SemanticFusionResult(
            fused_class_probs=fused,
            fused_confidence=fused_confidence,
            chosen_policy_id=chosen_policy_id,
            evidence_weights=weights,
            evidence_used_mask=evidence_used_mask,
            evidence_keys=["unknown"],
            diagnostics=diagnostics,
        )

    if vla_class_probs is None:
        base_probs = map_semantics
    else:
        base_probs = vla_class_probs
    assert base_probs is not None

    T, K, C = base_probs.shape
    vla_probs = _normalize_probs(vla_class_probs) if vla_class_probs is not None else None
    map_probs = _normalize_probs(map_semantics) if map_semantics is not None else None

    vla_weight = np.ones((T, K), dtype=np.float32)
    map_weight = np.ones((T, K), dtype=np.float32)

    if vla_confidence is not None:
        vla_weight *= np.clip(vla_confidence.astype(np.float32), 0.0, 1.0)
    if map_stability is not None:
        map_weight *= np.clip(map_stability.astype(np.float32), 0.0, 1.0)

    if occlusion is not None:
        occ = np.clip(occlusion.astype(np.float32), 0.0, 1.0)
        vla_weight *= (1.0 - occ)
        map_weight *= (1.0 - occ)

    if geom_residual is not None:
        res_factor = _weight_from_residual(geom_residual.astype(np.float32))
        vla_weight *= res_factor
        map_weight *= np.sqrt(res_factor)

    if dynamic_evidence is not None:
        dyn_factor = _weight_from_residual(dynamic_evidence.astype(np.float32))
        vla_weight *= dyn_factor
        map_weight *= dyn_factor

    if vla_probs is not None and map_probs is not None:
        fused = vla_weight[..., None] * vla_probs + map_weight[..., None] * map_probs
        fused = _normalize_probs(fused)
        weights = np.stack([vla_weight, map_weight], axis=-1)
        chosen_policy_id = (map_weight > vla_weight).astype(np.int32)
        evidence_keys = ["vla", "map"]
        evidence_used_mask = weights > 0.1
        disagreement = np.mean(np.abs(vla_probs - map_probs), axis=-1)
    elif vla_probs is not None:
        fused = vla_probs
        weights = vla_weight[..., None]
        chosen_policy_id = np.zeros((T, K), dtype=np.int32)
        evidence_keys = ["vla"]
        evidence_used_mask = vla_weight > 0.1
        disagreement = np.zeros((T, K), dtype=np.float32)
    else:
        fused = map_probs
        weights = map_weight[..., None]
        chosen_policy_id = np.ones((T, K), dtype=np.int32)
        evidence_keys = ["map"]
        evidence_used_mask = map_weight > 0.1
        disagreement = np.zeros((T, K), dtype=np.float32)

    fused_confidence = np.max(fused, axis=-1)
    fused_confidence *= np.clip(np.mean(weights, axis=-1), 0.0, 1.0)

    if mhn_plausibility is not None:
        fused_confidence *= float(np.clip(mhn_plausibility, 0.0, 1.0))

    entropy = -np.sum(fused * np.log(fused + 1e-8), axis=-1)

    diagnostics = {
        "entropy_mean": float(np.mean(entropy)),
        "disagreement_mean": float(np.mean(disagreement)),
        "coverage_pct": float(np.mean(evidence_used_mask.astype(np.float32))),
    }

    return SemanticFusionResult(
        fused_class_probs=fused.astype(np.float32),
        fused_confidence=fused_confidence.astype(np.float32),
        chosen_policy_id=chosen_policy_id,
        evidence_weights=weights.astype(np.float32),
        evidence_used_mask=evidence_used_mask.astype(bool),
        evidence_keys=evidence_keys,
        diagnostics=diagnostics,
    )
