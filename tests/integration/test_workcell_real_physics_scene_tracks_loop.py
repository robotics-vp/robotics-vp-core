from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


@pytest.mark.skipif(not _mujoco_available(), reason="mujoco not installed")
def test_workcell_real_physics_scene_tracks_loop(tmp_path: Path) -> None:
    from scripts.demo_workcell_real_physics_and_tracks import run_demo

    summary = run_demo(out_dir=tmp_path, steps=5, max_frames=5, seed=11)

    assert math.isfinite(summary["scene_tracks_quality"])
    assert math.isfinite(summary["contact_force_delta"])
    assert math.isfinite(summary["constraint_error_delta"])
    assert Path(summary["scene_tracks_path"]).exists()
