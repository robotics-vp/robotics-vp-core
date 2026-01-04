"""
Scene Track Serialization.

Provides numpy-only serialization format for trajectory.npz compatibility.
No pickled Python objects - only numeric/bool arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.vision.scene_ir_tracker.types import SceneTracks, SceneEntity3D, SceneTrackerMetrics


# Entity type enum
ENTITY_TYPE_OBJECT = 0
ENTITY_TYPE_BODY = 1

# Version string
SCENE_TRACKS_VERSION = "v1"


def validate_no_object_arrays(data: Dict[str, np.ndarray]) -> None:
    """Validate that no arrays have object dtype.
    
    Args:
        data: Dict of numpy arrays to validate.
    
    Raises:
        ValueError: If any array has object dtype.
    """
    for key, arr in data.items():
        if arr.dtype == object:
            raise ValueError(
                f"Array '{key}' has object dtype. "
                "Only numeric, bool, and unicode string dtypes allowed."
            )


# Supported versions for deserialization
SUPPORTED_VERSIONS = {"v1"}


class UnsupportedVersionError(ValueError):
    """Raised when scene_tracks version is not supported."""
    pass


def estimate_size_bytes(
    data: Dict[str, np.ndarray],
    compressed: bool = True,
) -> int:
    """Estimate serialized size in bytes.
    
    Args:
        data: Dict of numpy arrays.
        compressed: Whether compressed format will be used.
    
    Returns:
        Estimated size in bytes.
    """
    total = 0
    for arr in data.values():
        total += arr.nbytes
    
    if compressed:
        # Rough compression ratio estimate
        total = int(total * 0.6)
    
    return total


def check_size_warning(
    data: Dict[str, np.ndarray],
    duration_sec: float,
    max_mb_per_min: float = 10.0,
) -> Optional[str]:
    """Check if export size exceeds threshold.
    
    Args:
        data: Serialized data dict.
        duration_sec: Video duration in seconds.
        max_mb_per_min: Maximum MB per minute threshold.
    
    Returns:
        Warning message if exceeded, None otherwise.
    """
    size_bytes = estimate_size_bytes(data)
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 1.0
    mb_per_min = (size_bytes / 1e6) / duration_min
    
    if mb_per_min > max_mb_per_min:
        return (
            f"Export size {mb_per_min:.2f} MB/min exceeds threshold "
            f"({max_mb_per_min:.2f} MB/min). Consider disabling latents."
        )
    return None


@dataclass
class SceneTracksLite:
    """Lightweight scene tracks representation from numpy arrays.
    
    Used for loading serialized tracks without full SceneEntity3D objects.
    """
    track_ids: np.ndarray  # (K,) string array or int IDs
    entity_types: np.ndarray  # (K,) int, ENTITY_TYPE_OBJECT=0, ENTITY_TYPE_BODY=1
    class_ids: np.ndarray  # (K,) int, -1 for bodies
    poses_R: np.ndarray  # (T, K, 3, 3) rotation matrices
    poses_t: np.ndarray  # (T, K, 3) translations
    scales: np.ndarray  # (T, K) scale factors
    visibility: np.ndarray  # (T, K) visibility [0,1]
    occlusion: np.ndarray  # (T, K) occlusion scores [0,1]
    ir_loss: np.ndarray  # (T, K) IR loss per entity per frame
    converged: np.ndarray  # (T, K) bool, converged flag
    z_shape: Optional[np.ndarray] = None  # (T, K, Zs) float16
    z_tex: Optional[np.ndarray] = None  # (T, K, Zt) float16
    class_names: Optional[List[str]] = None  # Class name mapping for class_ids
    summary: Optional[Dict[str, Any]] = None
    
    @property
    def num_frames(self) -> int:
        return self.poses_R.shape[0]
    
    @property
    def num_tracks(self) -> int:
        return len(self.track_ids)


def serialize_scene_tracks_v1(
    scene_tracks: SceneTracks,
    include_latents: bool = False,
    latent_dtype: np.dtype = np.float16,
) -> Dict[str, np.ndarray]:
    """Serialize SceneTracks to numpy-only format.
    
    Args:
        scene_tracks: SceneTracks to serialize.
        include_latents: If True, include z_shape and z_tex arrays.
        latent_dtype: Dtype for latent arrays (default float16).
    
    Returns:
        Dict of numpy arrays suitable for np.savez.
        Keys are prefixed with 'scene_tracks_v1/' for namespacing.
    """
    if not scene_tracks.frames:
        # Empty case
        return {
            "scene_tracks_v1/version": np.array([SCENE_TRACKS_VERSION], dtype="U8"),
            "scene_tracks_v1/track_ids": np.array([], dtype="U32"),
            "scene_tracks_v1/entity_types": np.array([], dtype=np.int32),
            "scene_tracks_v1/class_ids": np.array([], dtype=np.int32),
            "scene_tracks_v1/poses_R": np.zeros((0, 0, 3, 3), dtype=np.float32),
            "scene_tracks_v1/poses_t": np.zeros((0, 0, 3), dtype=np.float32),
            "scene_tracks_v1/scales": np.zeros((0, 0), dtype=np.float32),
            "scene_tracks_v1/visibility": np.zeros((0, 0), dtype=np.float32),
            "scene_tracks_v1/occlusion": np.zeros((0, 0), dtype=np.float32),
            "scene_tracks_v1/ir_loss": np.zeros((0, 0), dtype=np.float32),
            "scene_tracks_v1/converged": np.zeros((0, 0), dtype=bool),
        }
    
    # Collect all unique track IDs in order of appearance
    track_id_list = list(scene_tracks.tracks.keys())
    track_id_to_idx = {tid: i for i, tid in enumerate(track_id_list)}
    
    T = len(scene_tracks.frames)
    K = len(track_id_list)
    
    # Get info from first appearance of each track
    entity_types = np.zeros(K, dtype=np.int32)
    class_ids = np.full(K, -1, dtype=np.int32)
    class_name_set: Dict[str, int] = {}  # class_name -> class_id
    
    for track_id, track_history in scene_tracks.tracks.items():
        if track_history:
            first_entity = track_history[0]
            k = track_id_to_idx[track_id]
            entity_types[k] = ENTITY_TYPE_BODY if first_entity.entity_type == "body" else ENTITY_TYPE_OBJECT
            if first_entity.class_name:
                if first_entity.class_name not in class_name_set:
                    class_name_set[first_entity.class_name] = len(class_name_set)
                class_ids[k] = class_name_set[first_entity.class_name]
    
    # Allocate arrays
    poses_R = np.zeros((T, K, 3, 3), dtype=np.float32)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.zeros((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.zeros((T, K), dtype=bool)
    
    # Latent dimensions (detect from first entity with latents)
    z_shape_dim = 0
    z_tex_dim = 0
    if include_latents:
        for entities in scene_tracks.frames:
            for e in entities:
                if e.z_shape is not None and z_shape_dim == 0:
                    z_shape_dim = len(e.z_shape)
                if e.z_tex is not None and z_tex_dim == 0:
                    z_tex_dim = len(e.z_tex)
                if z_shape_dim > 0 and z_tex_dim > 0:
                    break
            if z_shape_dim > 0 and z_tex_dim > 0:
                break
    
    z_shape_arr = np.zeros((T, K, z_shape_dim), dtype=latent_dtype) if z_shape_dim > 0 else None
    z_tex_arr = np.zeros((T, K, z_tex_dim), dtype=latent_dtype) if z_tex_dim > 0 else None
    
    # Fill arrays from frames
    for t, frame_entities in enumerate(scene_tracks.frames):
        for entity in frame_entities:
            if entity.track_id not in track_id_to_idx:
                continue
            k = track_id_to_idx[entity.track_id]
            
            poses_R[t, k] = entity.pose[:3, :3]
            poses_t[t, k] = entity.pose[:3, 3]
            scales[t, k] = entity.scale
            visibility[t, k] = entity.visibility
            occlusion[t, k] = entity.occlusion_score
            ir_loss[t, k] = entity.ir_loss
            converged[t, k] = True  # Assume converged if present
            
            if include_latents:
                if z_shape_arr is not None and entity.z_shape is not None:
                    src = entity.z_shape
                    if len(src) >= z_shape_dim:
                        z_shape_arr[t, k] = src[:z_shape_dim].astype(latent_dtype)
                    else:
                        # Pad with zeros if source is smaller
                        z_shape_arr[t, k, :len(src)] = src.astype(latent_dtype)
                if z_tex_arr is not None and entity.z_tex is not None:
                    src = entity.z_tex
                    if len(src) >= z_tex_dim:
                        z_tex_arr[t, k] = src[:z_tex_dim].astype(latent_dtype)
                    else:
                        z_tex_arr[t, k, :len(src)] = src.astype(latent_dtype)
    
    # Build result
    result = {
        "scene_tracks_v1/version": np.array([SCENE_TRACKS_VERSION], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(track_id_list, dtype="U32"),
        "scene_tracks_v1/entity_types": entity_types,
        "scene_tracks_v1/class_ids": class_ids,
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility,
        "scene_tracks_v1/occlusion": occlusion,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }
    
    # Add latents if requested
    if include_latents and z_shape_arr is not None:
        result["scene_tracks_v1/z_shape"] = z_shape_arr
    if include_latents and z_tex_arr is not None:
        result["scene_tracks_v1/z_tex"] = z_tex_arr
    
    # Add class name mapping
    if class_name_set:
        class_names_arr = np.array(
            [name for name, _ in sorted(class_name_set.items(), key=lambda x: x[1])],
            dtype="U64"
        )
        result["scene_tracks_v1/class_names"] = class_names_arr
    
    # Add summary as JSON string
    summary = scene_tracks.summary()
    result["scene_tracks_v1/summary_json"] = np.array([json.dumps(summary)], dtype="U1024")
    
    return result


def deserialize_scene_tracks_v1(data: Dict[str, np.ndarray]) -> SceneTracksLite:
    """Deserialize numpy arrays to SceneTracksLite.
    
    Args:
        data: Dict from np.load() with scene_tracks_v1/* keys.
    
    Returns:
        SceneTracksLite instance.
    
    Raises:
        UnsupportedVersionError: If version is not supported.
    """
    prefix = "scene_tracks_v1/"
    
    # Check version if present
    version_key = f"{prefix}version"
    if version_key in data:
        version = str(data[version_key][0]) if data[version_key].size > 0 else "v1"
        if version not in SUPPORTED_VERSIONS:
            raise UnsupportedVersionError(
                f"Scene tracks version '{version}' is not supported. "
                f"Supported versions: {SUPPORTED_VERSIONS}"
            )
    
    # Strip prefix if present
    def get_key(name: str) -> np.ndarray:
        full_key = f"{prefix}{name}"
        if full_key in data:
            return data[full_key]
        if name in data:
            return data[name]
        raise KeyError(f"Key {name} not found")
    
    track_ids = get_key("track_ids")
    entity_types = get_key("entity_types")
    class_ids = get_key("class_ids")
    poses_R = get_key("poses_R")
    poses_t = get_key("poses_t")
    scales = get_key("scales")
    visibility = get_key("visibility")
    occlusion = get_key("occlusion")
    ir_loss = get_key("ir_loss")
    converged = get_key("converged")
    
    # Optional latents
    z_shape = None
    z_tex = None
    try:
        z_shape = get_key("z_shape")
    except KeyError:
        pass
    try:
        z_tex = get_key("z_tex")
    except KeyError:
        pass
    
    # Optional class names
    class_names = None
    try:
        class_names_arr = get_key("class_names")
        class_names = list(class_names_arr)
    except KeyError:
        pass
    
    # Optional summary
    summary = None
    try:
        summary_json = get_key("summary_json")
        if len(summary_json) > 0:
            summary = json.loads(str(summary_json[0]))
    except (KeyError, json.JSONDecodeError):
        pass
    
    return SceneTracksLite(
        track_ids=track_ids,
        entity_types=entity_types,
        class_ids=class_ids,
        poses_R=poses_R,
        poses_t=poses_t,
        scales=scales,
        visibility=visibility,
        occlusion=occlusion,
        ir_loss=ir_loss,
        converged=converged,
        z_shape=z_shape,
        z_tex=z_tex,
        class_names=class_names,
        summary=summary,
    )


def compute_scene_ir_quality_score(scene_tracks: SceneTracks) -> float:
    """Compute scalar quality score from scene tracks.
    
    Args:
        scene_tracks: SceneTracks to evaluate.
    
    Returns:
        Quality score in [0, 1]. Higher is better.
    """
    metrics = scene_tracks.metrics
    
    # Normalize components
    ir_quality = max(0.0, 1.0 - metrics.mean_ir_loss / 0.5)
    
    convergence_rate = metrics.converged_count / max(metrics.total_frames, 1)
    
    id_switch_rate = metrics.id_switch_count / max(metrics.total_frames, 1) * 100
    id_quality = max(0.0, 1.0 - id_switch_rate / 10)
    
    occlusion_quality = 1.0 - metrics.occlusion_rate
    
    # Weighted combination
    quality = (
        0.4 * ir_quality +
        0.2 * convergence_rate +
        0.2 * id_quality +
        0.2 * occlusion_quality
    )
    
    return float(np.clip(quality, 0.0, 1.0))


def get_scene_ir_summary_dict(scene_tracks: SceneTracks) -> Dict[str, Any]:
    """Get summary dict suitable for storing in npz.
    
    Args:
        scene_tracks: SceneTracks to summarize.
    
    Returns:
        Dict with scalar summary values.
    """
    metrics = scene_tracks.metrics
    
    return {
        "ir_loss_mean": float(metrics.mean_ir_loss),
        "ir_loss_std": float(np.std(metrics.ir_loss_per_frame)) if metrics.ir_loss_per_frame else 0.0,
        "id_switch_count": int(metrics.id_switch_count),
        "id_switch_rate": float(metrics.id_switch_count / max(metrics.total_frames, 1) * 100),
        "occlusion_rate": float(metrics.occlusion_rate),
        "convergence_rate": float(metrics.converged_count / max(metrics.total_frames, 1)),
        "num_tracks": int(metrics.total_tracks),
        "num_frames": int(metrics.total_frames),
        "quality_score": compute_scene_ir_quality_score(scene_tracks),
    }
