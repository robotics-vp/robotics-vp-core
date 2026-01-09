"""Portable datapack artifacts for curated slice evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
from src.representation.token_providers import (
    RGBVisionTokenProvider,
    BaseTokenProvider,
    GeometryBEVProvider,
    SceneGraphTokenProvider,
)
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.valuation.datapack_schema import ConditionProfile, DATAPACK_SCHEMA_VERSION_PORTABLE, DATAPACK_SCHEMA_VERSION_REPR


def load_raw_episode_artifacts(raw_path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    """Load RGB frames and scene_tracks payload from a raw datapack path."""
    meta: Dict[str, Any] = {}
    if raw_path.is_dir():
        meta_path = raw_path / "metadata.json"
        if not meta_path.exists():
            return None, None, meta
        meta = json.loads(meta_path.read_text())
        rgb_path = Path(meta.get("rgb_video_path", ""))
        scene_tracks_path = Path(meta.get("scene_tracks_path", ""))
        rgb_frames = _load_rgb_frames(rgb_path) if rgb_path.exists() else None
        scene_tracks = _load_scene_tracks(scene_tracks_path) if scene_tracks_path.exists() else None
        return rgb_frames, scene_tracks, meta

    if raw_path.suffix != ".npz":
        return None, None, meta

    data = np.load(raw_path)
    rgb_frames = None
    if "rgb_frames" in data or "frames" in data:
        rgb_frames = data.get("rgb_frames") or data.get("frames")
    scene_tracks = _extract_scene_tracks_payload(data)
    return rgb_frames, scene_tracks, meta


def compute_rgb_features_v1(
    rgb_frames: np.ndarray,
    token_dim: int = 64,
    pool_size: tuple[int, int] = (4, 4),
    stride_seconds: Optional[float] = None,
    source_fps: Optional[float] = None,
    store_temporal: bool = False,
) -> Dict[str, Any]:
    """Compute portable RGB features using the vision_rgb encoder."""
    provider = RGBVisionTokenProvider(token_dim=token_dim, pool_size=pool_size, seed=0, allow_synthetic=False)
    tokens = provider.provide({"rgb_frames": rgb_frames}).tokens
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)

    stride_frames = None
    if stride_seconds is not None and source_fps is not None and source_fps > 0:
        stride_frames = max(1, int(round(stride_seconds * source_fps)))
    if stride_frames and stride_frames > 1:
        tokens = tokens[::stride_frames]

    pooled = tokens.mean(dim=0)

    payload: Dict[str, Any] = {
        "encoder": "vision_rgb::deterministic_pool_v1",
        "dim": int(tokens.shape[-1]),
        "pooling": "mean",
        "token_dim": int(token_dim),
        "pool_size": list(pool_size),
        "stride_seconds": float(stride_seconds) if stride_seconds is not None else None,
        "stride_frames": int(stride_frames) if stride_frames is not None else None,
        "source_fps": float(source_fps) if source_fps is not None else None,
        "num_tokens": int(tokens.shape[0]),
        "features": pooled.detach().cpu().tolist(),
    }
    if store_temporal:
        payload["features_temporal"] = tokens.detach().cpu().tolist()
    return payload


def compute_slice_labels_v1(
    condition: ConditionProfile | Dict[str, Any],
    scene_tracks_payload: Optional[Dict[str, Any]],
    occlusion_threshold: float = 0.5,
    dynamic_threshold: float = 0.15,
    static_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Compute portable slice labels from condition + scene tracks."""
    if isinstance(condition, ConditionProfile):
        occlusion_level = float(condition.occlusion_level)
    else:
        occlusion_level = float(condition.get("occlusion_level", 0.0))

    occlusion_bucket = _bucketize(occlusion_level, low=0.33, high=0.66)
    is_occluded = occlusion_level >= occlusion_threshold

    motion_score = None
    if scene_tracks_payload is not None:
        payload = coerce_scene_tracks_payload(scene_tracks_payload)
        scene_tracks = deserialize_scene_tracks_v1(payload)
        motion_score = _mean_speed(scene_tracks.poses_t)

    is_dynamic = motion_score is not None and motion_score >= dynamic_threshold
    is_static = motion_score is not None and motion_score <= static_threshold
    motion_bucket = _bucketize(motion_score, low=static_threshold, high=dynamic_threshold) if motion_score is not None else None

    return {
        "schema_version": DATAPACK_SCHEMA_VERSION_PORTABLE,
        "occlusion_level": occlusion_level,
        "occlusion_threshold": occlusion_threshold,
        "occlusion_bucket": occlusion_bucket,
        "is_occluded": bool(is_occluded),
        "motion_score": float(motion_score) if motion_score is not None else None,
        "motion_thresholds": {"dynamic": dynamic_threshold, "static": static_threshold},
        "motion_bucket": motion_bucket,
        "is_dynamic": bool(is_dynamic) if motion_score is not None else None,
        "is_static": bool(is_static) if motion_score is not None else None,
    }


def compute_repr_tokens_v1(
    episode_artifacts: Dict[str, Any],
    repr_names: List[str],
    providers: Optional[Dict[str, BaseTokenProvider]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute portable repr tokens for specified representations.

    Args:
        episode_artifacts: Dict with keys like 'rgb_frames', 'scene_tracks', 'scene_graphs'
        repr_names: List of representation names to compute (e.g., ['vision_rgb', 'geometry_bev'])
        providers: Optional dict mapping repr names to token providers. If None, defaults are used.

    Returns:
        Dict mapping repr name to versioned payload with 'version', 'dim', 'features', 'metadata'.
    """
    import torch

    if providers is None:
        providers = _default_providers()

    result: Dict[str, Dict[str, Any]] = {}


    for repr_name in repr_names:
        provider = providers.get(repr_name)
        if provider is None:
            continue

        try:
            output = provider.provide(episode_artifacts)
            tokens = output.tokens

            # Handle batch dimension
            if tokens.dim() == 3:
                tokens = tokens.squeeze(0)

            # Pool to single vector
            pooled = tokens.mean(dim=0)

            payload: Dict[str, Any] = {
                "version": f"{repr_name}::v1",
                "dim": int(pooled.shape[-1]),
                "num_tokens": int(tokens.shape[0]),
                "pooling": "mean",
                "features": pooled.detach().cpu().tolist(),
                "metadata": output.metadata,
            }
            result[repr_name] = payload
        except Exception as e:
            # Skip representations that fail (e.g., missing inputs)
            result[repr_name] = {
                "version": f"{repr_name}::v1",
                "error": str(e),
            }

    return result


def _default_providers() -> Dict[str, BaseTokenProvider]:
    """Create default token providers for common representations."""
    return {
        "vision_rgb": RGBVisionTokenProvider(seed=0, allow_synthetic=False),
        "geometry_bev": GeometryBEVProvider(allow_synthetic=False),
        "geometry_scene_graph": SceneGraphTokenProvider(),
    }


def coerce_scene_tracks_payload(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert JSON-loaded scene tracks payload to numpy arrays."""
    coerced: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            coerced[key] = value
        else:
            coerced[key] = np.asarray(value)
    return coerced


def _load_rgb_frames(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = np.load(path)
    if "rgb_frames" in data:
        return data["rgb_frames"]
    if "frames" in data:
        return data["frames"]
    return None


def _load_scene_tracks(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path)
    return _extract_scene_tracks_payload(data)


def _extract_scene_tracks_payload(data: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if any(key.startswith("scene_tracks_v1/") for key in data):
        return {k: data[k] for k in data if k.startswith("scene_tracks_v1/")}
    required = ("scene_tracks_v1/poses_t", "scene_tracks_v1/poses_R")
    if all(key in data for key in required):
        return dict(data)
    return None


def _mean_speed(poses_t: np.ndarray) -> float:
    if poses_t.ndim != 3 or poses_t.shape[0] < 2:
        return 0.0
    deltas = poses_t[1:] - poses_t[:-1]
    speed = np.linalg.norm(deltas, axis=-1)
    return float(np.mean(speed))


def _bucketize(value: Optional[float], low: float, high: float) -> Optional[str]:
    if value is None:
        return None
    if value < low:
        return "low"
    if value < high:
        return "mid"
    return "high"


__all__ = [
    "load_raw_episode_artifacts",
    "compute_rgb_features_v1",
    "compute_slice_labels_v1",
    "compute_repr_tokens_v1",
    "coerce_scene_tracks_payload",
    "DATAPACK_SCHEMA_VERSION_PORTABLE",
    "DATAPACK_SCHEMA_VERSION_REPR",
]
