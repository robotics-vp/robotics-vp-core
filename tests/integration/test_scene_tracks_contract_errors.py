from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.vision.scene_ir_tracker.io.datapack_frame_reader import DatapackFrameError, read_datapack_frames


def _write_rgb_bundle(base: Path, frames: np.ndarray, metadata: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base / "rgb.npz", frames=frames)
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2))


def test_datapack_frame_contract_non_monotonic_timestamps(tmp_path: Path) -> None:
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    metadata = {"frame_timestamps": [0.0, 0.2, 0.1], "camera_name": "front"}
    bundle_dir = tmp_path / "non_monotonic"
    _write_rgb_bundle(bundle_dir, frames, metadata)

    with pytest.raises(DatapackFrameError, match="timestamps"):
        read_datapack_frames(bundle_dir, camera="front", mode="rgb")


def test_datapack_frame_contract_missing_camera(tmp_path: Path) -> None:
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    metadata = {"frame_timestamps": [0.0, 0.1], "camera_name": "front"}
    bundle_dir = tmp_path / "missing_camera"
    _write_rgb_bundle(bundle_dir, frames, metadata)

    with pytest.raises(DatapackFrameError, match="camera"):
        read_datapack_frames(bundle_dir, camera="top", mode="rgb")
