from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

VLA_SEMANTIC_EVIDENCE_VERSION = "v1"
VLA_SEMANTIC_EVIDENCE_PREFIX = "vla_semantic_evidence_v1/"


def _get_scene_tracks_array(scene_tracks: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if key in scene_tracks:
        return np.asarray(scene_tracks[key])
    prefixed = f"scene_tracks_v1/{key}"
    if prefixed in scene_tracks:
        return np.asarray(scene_tracks[prefixed])
    return None


def _infer_scene_tracks_shape(scene_tracks: Optional[Dict[str, Any]]) -> Tuple[int, int, np.ndarray]:
    if not isinstance(scene_tracks, dict):
        return 1, 1, np.array(["unknown"], dtype="U32")
    poses_t = _get_scene_tracks_array(scene_tracks, "poses_t")
    if poses_t is not None and poses_t.ndim >= 2:
        T, K = poses_t.shape[0], poses_t.shape[1]
    else:
        track_ids = _get_scene_tracks_array(scene_tracks, "track_ids")
        K = int(track_ids.shape[0]) if track_ids is not None else 1
        T = 1
    track_ids = _get_scene_tracks_array(scene_tracks, "track_ids")
    if track_ids is None:
        track_ids = np.array([f"track_{i}" for i in range(K)], dtype="U32")
    return T, K, track_ids.astype("U32")


def build_vla_semantic_evidence_stub(
    scene_tracks: Optional[Dict[str, Any]],
    vla_payload: Optional[Mapping[str, Any]] = None,
    semantic_tags: Optional[list[str]] = None,
    instruction: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Build a stub VLA_SemanticEvidence_v1 payload.

    This emits a minimal numpy-only evidence packet so downstream Map-First
    can consume a consistent sidecar, even when no per-entity logits exist.
    """
    T, K, track_ids = _infer_scene_tracks_shape(scene_tracks)

    class_probs = None
    confidence = None
    embed = None

    if vla_payload is not None:
        class_probs = vla_payload.get("class_probs") if isinstance(vla_payload, Mapping) else None
        confidence = vla_payload.get("confidence") if isinstance(vla_payload, Mapping) else None
        embed = vla_payload.get("embed") if isinstance(vla_payload, Mapping) else None

    if class_probs is None:
        class_probs_arr = np.ones((T, K, 1), dtype=np.float32)
    else:
        class_probs_arr = np.asarray(class_probs, dtype=np.float32)

    if confidence is None:
        vla_available = False
        if isinstance(vla_payload, Mapping):
            vla_available = bool(vla_payload.get("vla_available", False))
        base_conf = 0.2 if vla_available else 0.05
        confidence_arr = np.full((T, K), base_conf, dtype=np.float32)
    else:
        confidence_arr = np.asarray(confidence, dtype=np.float32)

    provenance = {
        "source": "vla_stub",
        "semantic_tags": semantic_tags or [],
        "instruction": instruction or "",
        "vla_available": bool(vla_payload.get("vla_available", False)) if isinstance(vla_payload, Mapping) else False,
    }

    payload = {
        f"{VLA_SEMANTIC_EVIDENCE_PREFIX}version": np.array([VLA_SEMANTIC_EVIDENCE_VERSION], dtype="U8"),
        f"{VLA_SEMANTIC_EVIDENCE_PREFIX}class_probs": class_probs_arr,
        f"{VLA_SEMANTIC_EVIDENCE_PREFIX}confidence": confidence_arr,
        f"{VLA_SEMANTIC_EVIDENCE_PREFIX}track_ids": track_ids,
        f"{VLA_SEMANTIC_EVIDENCE_PREFIX}provenance_json": np.array([json.dumps(provenance)], dtype="U2048"),
    }
    if embed is not None:
        payload[f"{VLA_SEMANTIC_EVIDENCE_PREFIX}embed"] = np.asarray(embed, dtype=np.float32)
    return payload


def save_vla_semantic_evidence_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
