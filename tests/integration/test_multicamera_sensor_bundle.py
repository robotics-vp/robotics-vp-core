from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


def _seg_centroid(seg: np.ndarray, seg_id: int) -> tuple[int, int]:
    ys, xs = np.nonzero(seg == seg_id)
    if ys.size == 0:
        return -1, -1
    return int(np.round(xs.mean())), int(np.round(ys.mean()))


@pytest.mark.skipif(not _mujoco_available(), reason="mujoco not installed")
def test_multicamera_sensor_bundle(tmp_path: Path) -> None:
    from src.envs.workcell_env.observations.mujoco_render import render_workcell_frames
    from src.envs.workcell_env.physics.mujoco_adapter import MujocoPhysicsAdapter
    from src.envs.workcell_env.scene.scene_spec import FixtureSpec, PartSpec, WorkcellSceneSpec
    from src.motor_backend.sensor_bundle import SensorBundleData, write_sensor_bundle

    scene_spec = WorkcellSceneSpec(
        workcell_id="multicam_demo",
        fixtures=[
            FixtureSpec(
                id="hole",
                position=(0.0, 0.0, 0.05),
                orientation=(1.0, 0.0, 0.0, 0.0),
                fixture_type="vise",
            )
        ],
        parts=[
            PartSpec(
                id="peg",
                position=(0.0, 0.0, 0.15),
                orientation=(1.0, 0.0, 0.0, 0.0),
                part_type="peg",
                dimensions_mm=(30.0, 30.0, 60.0),
            )
        ],
        spatial_bounds=(1.0, 1.0, 1.0),
    )

    adapter = MujocoPhysicsAdapter(spatial_bounds=scene_spec.spatial_bounds)
    adapter.reset(scene_spec, seed=0)
    states = []
    for _ in range(3):
        adapter.step(0.02)
        states.append(adapter.get_state())

    cameras = ["front", "top"]
    rgb: dict[str, np.ndarray] = {}
    depth: dict[str, np.ndarray] = {}
    seg: dict[str, np.ndarray] = {}
    intrinsics: dict[str, dict[str, float]] = {}
    extrinsics: dict[str, np.ndarray] = {}

    for camera in cameras:
        frames, depth_frames, seg_frames, camera_params = render_workcell_frames(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera,
            width=96,
            height=96,
            max_frames=len(states),
            seed=123,
        )
        rgb[camera] = np.asarray(frames, dtype=np.uint8)
        if depth_frames:
            depth[camera] = np.asarray(depth_frames, dtype=np.float32)
        if seg_frames:
            seg[camera] = np.asarray(seg_frames, dtype=np.int32)
        intrinsics[camera] = {
            "fx": float(camera_params.fx),
            "fy": float(camera_params.fy),
            "cx": float(camera_params.cx),
            "cy": float(camera_params.cy),
            "width": int(camera_params.width),
            "height": int(camera_params.height),
        }
        extrinsics[camera] = np.asarray(camera_params.world_from_cam, dtype=np.float32)

    timestamps = [float(s.get("time_s", idx * 0.02)) for idx, s in enumerate(states)]
    bundle = SensorBundleData(
        cameras=cameras,
        rgb=rgb,
        depth=depth,
        seg=seg,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        timestamps_s=timestamps,
        depth_unit="meters",
    )

    episode_dir = tmp_path / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    write_sensor_bundle(episode_dir, bundle)

    for camera in cameras:
        assert (episode_dir / "rgb" / f"{camera}.npz").exists()
        assert (episode_dir / "depth" / f"{camera}.npz").exists()
        assert (episode_dir / "seg" / f"{camera}.npz").exists()
        assert (episode_dir / "intrinsics" / f"{camera}.json").exists()
        assert (episode_dir / "extrinsics" / f"{camera}.npy").exists()

    seg_front = np.load(episode_dir / "seg" / "front.npz")["frames"]
    seg_top = np.load(episode_dir / "seg" / "top.npz")["frames"]
    assert seg_front.dtype == np.int32
    assert seg_top.dtype == np.int32

    intr_front = json.loads((episode_dir / "intrinsics" / "front.json").read_text())
    intr_top = json.loads((episode_dir / "intrinsics" / "top.json").read_text())
    extr_front = np.load(episode_dir / "extrinsics" / "front.npy")
    extr_top = np.load(episode_dir / "extrinsics" / "top.npy")
    assert extr_front.shape[-2:] == (4, 4)
    assert extr_top.shape[-2:] == (4, 4)

    seg_front0 = seg_front[0]
    seg_top0 = seg_top[0]
    ids_front = set(np.unique(seg_front0)) - {0}
    ids_top = set(np.unique(seg_top0)) - {0}
    common_ids = ids_front & ids_top
    assert common_ids

    overlap_scores = []
    for seg_id in sorted(common_ids):
        count_front = int(np.sum(seg_front0 == seg_id))
        count_top = int(np.sum(seg_top0 == seg_id))
        score = min(count_front, count_top)
        if score > 0:
            overlap_scores.append((score, int(seg_id)))
    assert overlap_scores
    geom_id = max(overlap_scores)[1]

    center_front = _seg_centroid(seg_front0, geom_id)
    center_top = _seg_centroid(seg_top0, geom_id)
    assert center_front != (-1, -1)
    assert center_top != (-1, -1)
    margin = 6
    assert margin <= center_front[0] < intr_front["width"] - margin
    assert margin <= center_front[1] < intr_front["height"] - margin
    assert margin <= center_top[0] < intr_top["width"] - margin
    assert margin <= center_top[1] < intr_top["height"] - margin

    for frame in seg_front:
        assert np.any(frame == geom_id)
    for frame in seg_top:
        assert np.any(frame == geom_id)

    prev_front = center_front
    prev_top = center_top
    for idx in range(1, len(states)):
        curr_front = _seg_centroid(seg_front[idx], geom_id)
        curr_top = _seg_centroid(seg_top[idx], geom_id)
        assert curr_front != (-1, -1)
        assert curr_top != (-1, -1)
        assert np.hypot(curr_front[0] - prev_front[0], curr_front[1] - prev_front[1]) <= 12
        assert np.hypot(curr_top[0] - prev_top[0], curr_top[1] - prev_top[1]) <= 12
        prev_front = curr_front
        prev_top = curr_top
