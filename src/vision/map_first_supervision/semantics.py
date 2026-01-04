"""Semantic stabilization for Map-First pseudo-supervision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.vision.map_first_supervision.static_map import VoxelHashMap


SEMANTIC_EMA_ALPHA = 0.2
VLA_SEMANTIC_EVIDENCE_VERSION = "v1"
VLA_SEMANTIC_EVIDENCE_PREFIX = "vla_semantic_evidence_v1/"


@dataclass
class VLASemanticEvidence:
    """Sidecar semantic evidence produced by VLA."""

    class_probs: Optional[np.ndarray] = None  # (T, K, C)
    confidence: Optional[np.ndarray] = None  # (T, K)
    provenance: Optional[Dict[str, Any]] = None
    embed: Optional[np.ndarray] = None  # (T, K, D)
    track_ids: Optional[np.ndarray] = None
    version: str = VLA_SEMANTIC_EVIDENCE_VERSION


def _normalize_track_ids(track_ids: Optional[np.ndarray]) -> Optional[list[str]]:
    if track_ids is None:
        return None
    return [str(tid) for tid in list(track_ids)]


def _strip_prefix(payload: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {k[len(prefix):]: v for k, v in payload.items() if k.startswith(prefix)}


def _align_track_order(
    arr: np.ndarray,
    evidence_track_ids: Optional[np.ndarray],
    scene_track_ids: Optional[np.ndarray],
) -> np.ndarray:
    if arr is None or evidence_track_ids is None or scene_track_ids is None:
        return arr
    evidence_ids = _normalize_track_ids(evidence_track_ids)
    scene_ids = _normalize_track_ids(scene_track_ids)
    if evidence_ids is None or scene_ids is None:
        return arr
    if len(evidence_ids) != arr.shape[1]:
        return arr
    if evidence_ids == scene_ids:
        return arr
    mapping = {tid: idx for idx, tid in enumerate(evidence_ids)}
    reordered = np.zeros((arr.shape[0], len(scene_ids)) + arr.shape[2:], dtype=arr.dtype)
    for new_idx, tid in enumerate(scene_ids):
        if tid in mapping:
            reordered[:, new_idx] = arr[:, mapping[tid]]
    return reordered


def _build_class_probs_from_instances(
    instances: Any,
    track_ids: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if instances is None:
        return None
    track_list = _normalize_track_ids(track_ids) or []
    track_map = {tid: idx for idx, tid in enumerate(track_list)}
    T = len(instances)
    num_classes = None

    for frame in instances:
        if isinstance(frame, dict) and frame:
            sample = next(iter(frame.values()))
            num_classes = int(np.asarray(sample).shape[-1])
            break
        if isinstance(frame, list) and frame:
            sample = frame[0].get("class_probs") or frame[0].get("probs")
            num_classes = int(np.asarray(sample).shape[-1])
            break

    if num_classes is None or not track_list:
        return None

    probs = np.zeros((T, len(track_list), num_classes), dtype=np.float32)
    for t, frame in enumerate(instances):
        if isinstance(frame, dict):
            for track_id, vec in frame.items():
                idx = track_map.get(str(track_id))
                if idx is None:
                    continue
                probs[t, idx] = np.asarray(vec, dtype=np.float32)
        elif isinstance(frame, list):
            for item in frame:
                track_id = item.get("track_id")
                vec = item.get("class_probs") or item.get("probs")
                if track_id is None or vec is None:
                    continue
                idx = track_map.get(str(track_id))
                if idx is None:
                    continue
                probs[t, idx] = np.asarray(vec, dtype=np.float32)
    return probs


def parse_vla_semantic_evidence(
    payload: Any,
    scene_track_ids: Optional[np.ndarray] = None,
) -> Optional[VLASemanticEvidence]:
    """Parse VLA_SemanticEvidence_v1 payload into a structured object."""
    if payload is None:
        return None
    if isinstance(payload, VLASemanticEvidence):
        return payload
    if isinstance(payload, dict):
        data = payload
        if any(k.startswith(VLA_SEMANTIC_EVIDENCE_PREFIX) for k in payload.keys()):
            data = _strip_prefix(payload, VLA_SEMANTIC_EVIDENCE_PREFIX)
            data["version"] = VLA_SEMANTIC_EVIDENCE_VERSION

        class_probs = data.get("class_probs")
        confidence = data.get("confidence")
        provenance = data.get("provenance")
        if provenance is None and "provenance_json" in data:
            prov_raw = data.get("provenance_json")
            if isinstance(prov_raw, np.ndarray) and prov_raw.size > 0:
                prov_raw = prov_raw.flat[0]
            try:
                provenance = json.loads(str(prov_raw))
            except Exception:
                provenance = None
        embed = data.get("embed")
        track_ids = data.get("track_ids")
        instances = data.get("instances") or data.get("per_frame_instances")

        if class_probs is None and instances is not None:
            class_probs = _build_class_probs_from_instances(instances, track_ids or scene_track_ids)

        class_probs_arr = np.asarray(class_probs, dtype=np.float32) if class_probs is not None else None
        confidence_arr = np.asarray(confidence, dtype=np.float32) if confidence is not None else None
        embed_arr = np.asarray(embed, dtype=np.float32) if embed is not None else None
        track_ids_arr = np.asarray(track_ids) if track_ids is not None else None

        if class_probs_arr is not None:
            class_probs_arr = _align_track_order(class_probs_arr, track_ids_arr, scene_track_ids)
        if confidence_arr is not None:
            confidence_arr = _align_track_order(confidence_arr, track_ids_arr, scene_track_ids)
        if embed_arr is not None:
            embed_arr = _align_track_order(embed_arr, track_ids_arr, scene_track_ids)

        return VLASemanticEvidence(
            class_probs=class_probs_arr,
            confidence=confidence_arr,
            provenance=provenance if isinstance(provenance, dict) else None,
            embed=embed_arr,
            track_ids=track_ids_arr,
            version=str(data.get("version", VLA_SEMANTIC_EVIDENCE_VERSION)),
        )

    return None


@dataclass
class SemanticStabilizer:
    """Maintains per-voxel semantic posteriors and aggregates entity semantics."""

    map_store: VoxelHashMap
    ema_alpha: float = SEMANTIC_EMA_ALPHA

    def update_from_entity_probs(
        self,
        points_world: np.ndarray,
        class_probs: np.ndarray,
        confidence: Optional[float] = None,
    ) -> None:
        if points_world.size == 0:
            return
        class_probs = np.asarray(class_probs, dtype=np.float32)
        if class_probs.ndim != 1:
            raise ValueError("class_probs must be (C,) for entity update")
        if confidence is None:
            ema_alpha = self.ema_alpha
        else:
            ema_alpha = float(np.clip(confidence, 0.0, 1.0)) * self.ema_alpha
        semantics = np.repeat(class_probs[np.newaxis, :], points_world.shape[0], axis=0)
        self.map_store.update(points_world, semantics=semantics, ema_alpha=ema_alpha)

    def update_from_point_labels(self, points_world: np.ndarray, labels: np.ndarray) -> None:
        if points_world.size == 0:
            return
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be (N,) for point updates")
        self.map_store.update(points_world, semantics=labels, ema_alpha=self.ema_alpha)

    def aggregate_entity_probs(self, points_world: np.ndarray) -> np.ndarray:
        sem = self.map_store.query_semantics(points_world)
        if sem is None or sem.size == 0:
            return np.zeros((self.map_store.semantics_num_classes,), dtype=np.float32)
        if sem.ndim != 2 or sem.shape[1] != self.map_store.semantics_num_classes:
            return np.zeros((self.map_store.semantics_num_classes,), dtype=np.float32)
        mean = np.mean(sem, axis=0)
        total = float(np.sum(mean))
        if total > 1e-6:
            mean = mean / total
        return mean.astype(np.float32)
