from __future__ import annotations

"""Sensor bundle I/O helpers for datapack episodes."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

SENSOR_BUNDLE_VERSION = "sensor_bundle_v1"


@dataclass(frozen=True)
class SensorBundleData:
    cameras: Sequence[str]
    rgb: Mapping[str, np.ndarray]
    depth: Mapping[str, np.ndarray]
    seg: Mapping[str, np.ndarray]
    intrinsics: Mapping[str, Mapping[str, Any]]
    extrinsics: Mapping[str, np.ndarray]
    timestamps_s: Sequence[float]
    depth_unit: str = "meters"
    noise_config: Optional[Mapping[str, Any]] = None
    noise_seed: Optional[int] = None


def write_sensor_bundle(episode_dir: Path, bundle: SensorBundleData) -> Dict[str, Any]:
    """Write the canonical sensor bundle layout and return metadata."""
    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"
    seg_dir = episode_dir / "seg"
    intr_dir = episode_dir / "intrinsics"
    extr_dir = episode_dir / "extrinsics"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    intr_dir.mkdir(parents=True, exist_ok=True)
    extr_dir.mkdir(parents=True, exist_ok=True)

    rgb_paths: Dict[str, str] = {}
    depth_paths: Dict[str, str] = {}
    seg_paths: Dict[str, str] = {}
    intr_paths: Dict[str, str] = {}
    extr_paths: Dict[str, str] = {}

    for camera in bundle.cameras:
        if camera in bundle.rgb:
            rgb_path = rgb_dir / f"{camera}.npz"
            np.savez_compressed(rgb_path, frames=np.asarray(bundle.rgb[camera]))
            rgb_paths[camera] = str(rgb_path.relative_to(episode_dir))
        if camera in bundle.depth:
            depth_path = depth_dir / f"{camera}.npz"
            np.savez_compressed(depth_path, frames=np.asarray(bundle.depth[camera], dtype=np.float32))
            depth_paths[camera] = str(depth_path.relative_to(episode_dir))
        if camera in bundle.seg:
            seg_path = seg_dir / f"{camera}.npz"
            np.savez_compressed(seg_path, frames=np.asarray(bundle.seg[camera], dtype=np.int32))
            seg_paths[camera] = str(seg_path.relative_to(episode_dir))
        if camera in bundle.intrinsics:
            intr_path = intr_dir / f"{camera}.json"
            intr_path.write_text(json.dumps(bundle.intrinsics[camera], indent=2))
            intr_paths[camera] = str(intr_path.relative_to(episode_dir))
        if camera in bundle.extrinsics:
            extr_path = extr_dir / f"{camera}.npy"
            np.save(extr_path, np.asarray(bundle.extrinsics[camera], dtype=np.float32))
            extr_paths[camera] = str(extr_path.relative_to(episode_dir))

    timestamps_path = episode_dir / "timestamps_s.npy"
    np.save(timestamps_path, np.asarray(bundle.timestamps_s, dtype=np.float64))

    return {
        "version": SENSOR_BUNDLE_VERSION,
        "cameras": list(bundle.cameras),
        "depth_unit": bundle.depth_unit,
        "rgb": rgb_paths,
        "depth": depth_paths,
        "seg": seg_paths,
        "intrinsics": intr_paths,
        "extrinsics": extr_paths,
        "timestamps_s": str(timestamps_path.relative_to(episode_dir)),
        "noise_config": dict(bundle.noise_config or {}),
        "noise_seed": bundle.noise_seed,
    }
