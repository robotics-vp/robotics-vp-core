#!/usr/bin/env python3
"""Demo: workcell MuJoCo -> datapack -> SceneTracks -> reconstruct -> replay."""
from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.envs.workcell_env import WorkcellEnv
from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.reconstruction.scene_tracks_adapter import SceneTracksAdapter
from src.envs.workcell_env.scene.scene_spec import FixtureSpec, PartSpec, WorkcellSceneSpec
from src.envs.workcell_env.tasks.peg_in_hole import PegInHoleTask
from src.envs.workcell_env.observations.mujoco_render import render_workcell_frames
from src.envs.workcell_env.utils.determinism import deterministic_episode_id
from src.motor_backend.sensor_bundle import SensorBundleData, write_sensor_bundle
from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks


def _ensure_mujoco() -> None:
    if importlib.util.find_spec("mujoco") is None:
        raise RuntimeError("MuJoCo is not installed. Install with `pip install mujoco` to run this demo.")


def _build_scene_spec() -> WorkcellSceneSpec:
    return WorkcellSceneSpec(
        workcell_id="demo_workcell",
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


def _write_episode_bundle(
    *,
    out_dir: Path,
    episode_id: str,
    task_id: str,
    scene_spec: WorkcellSceneSpec,
    trajectory_data: Dict[str, Any],
    rgb_frames: List[np.ndarray],
    seg_frames: Optional[List[np.ndarray]] = None,
    sensor_bundle: Optional[SensorBundleData] = None,
) -> Path:
    episode_dir = out_dir / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = episode_dir / "trajectory.npz"
    np.savez_compressed(trajectory_path, trajectory=trajectory_data)

    rgb_path = episode_dir / "rgb.npz"
    if rgb_frames:
        np.savez_compressed(rgb_path, frames=np.asarray(rgb_frames))

    if seg_frames:
        seg_path = episode_dir / "seg_front.npz"
        np.savez_compressed(seg_path, frames=np.asarray(seg_frames))

    sensor_bundle_meta = None
    if sensor_bundle is not None:
        sensor_bundle_meta = write_sensor_bundle(episode_dir, sensor_bundle)

    metadata = {
        "metadata": {
            "episode_id": episode_id,
            "task_id": task_id,
            "robot_family": "workcell",
            "seed": trajectory_data.get("seed"),
            "env_params": {"scene_spec": scene_spec.to_dict()},
        },
        "trajectory_path": str(trajectory_path),
        "rgb_video_path": str(rgb_path),
        "depth_video_path": None,
        "metrics": {},
    }
    if sensor_bundle_meta:
        metadata["sensor_bundle"] = sensor_bundle_meta
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return episode_dir


def _mean_metric(states: List[Dict[str, Any]], key: str) -> float:
    if not states:
        return 0.0
    return float(sum(float(s.get(key, 0.0)) for s in states) / len(states))


def run_demo(
    *,
    out_dir: Path,
    steps: int = 10,
    max_frames: int = 10,
    seed: int = 7,
) -> Dict[str, Any]:
    _ensure_mujoco()

    scene_spec = _build_scene_spec()
    config = WorkcellEnvConfig(
        physics_mode="MUJOCO",
        max_steps=steps,
        time_step_s=0.02,
        capture_rgb_frames=True,
        render_width=128,
        render_height=128,
        render_fps=10,
        render_max_frames=max_frames,
    )
    task = PegInHoleTask(peg_id="peg", hole_id="hole", tolerance_mm=2.0)
    episode_id = deterministic_episode_id("demo", seed)

    env = WorkcellEnv(config=config, scene_spec=scene_spec, task=task, seed=seed)
    env.reset(seed=seed, episode_id=episode_id)

    actions: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    seg_frames: List[np.ndarray] = []

    for _ in range(steps):
        action = {"object_id": "end_effector", "delta_position": (0.0, 0.0, -0.01)}
        env.step(action)
        actions.append(action)
        states.append(env.physics_adapter.get_state())

    trajectory_data = {
        "scene_spec": scene_spec.to_dict(),
        "states": states,
        "actions": actions,
        "seed": seed,
    }

    frames, depth_frames, seg_frames, camera_params = render_workcell_frames(
        scene_spec=scene_spec,
        states=states,
        camera_name="front",
        width=config.render_width,
        height=config.render_height,
        max_frames=max_frames,
        seed=seed,
    )
    timestamps = [float(s.get("time_s", idx * config.time_step_s)) for idx, s in enumerate(states[:len(frames)])]
    sensor_bundle = SensorBundleData(
        cameras=["front"],
        rgb={"front": np.asarray(frames, dtype=np.uint8)},
        depth={"front": np.asarray(depth_frames, dtype=np.float32)} if depth_frames else {},
        seg={"front": np.asarray(seg_frames, dtype=np.int32)} if seg_frames else {},
        intrinsics={
            "front": {
                "fx": float(camera_params.fx),
                "fy": float(camera_params.fy),
                "cx": float(camera_params.cx),
                "cy": float(camera_params.cy),
                "width": int(camera_params.width),
                "height": int(camera_params.height),
            }
        },
        extrinsics={"front": np.asarray(camera_params.world_from_cam, dtype=np.float32)},
        timestamps_s=timestamps,
        depth_unit="meters",
    )

    episode_dir = _write_episode_bundle(
        out_dir=out_dir,
        episode_id=episode_id,
        task_id="peg_in_hole",
        scene_spec=scene_spec,
        trajectory_data=trajectory_data,
        rgb_frames=frames,
        seg_frames=seg_frames or None,
        sensor_bundle=sensor_bundle,
    )

    tracks_result = run_scene_tracks(
        datapack_path=episode_dir,
        output_path=episode_dir,
        seed=seed,
        max_frames=max_frames,
        camera="front",
        mode="rgb",
        min_quality=0.1,
        allow_low_quality=True,
    )

    adapter = SceneTracksAdapter()
    reconstruction = adapter.reconstruct_from_paths(tracks_result.scene_tracks_path)

    replay_env = WorkcellEnv(config=config, scene_spec=reconstruction.scene_spec, task=task, seed=seed)
    replay_env.reset(seed=seed, episode_id=f"{episode_id}_replay")
    replay_states: List[Dict[str, Any]] = []
    for action in actions[:steps]:
        replay_env.step(action)
        replay_states.append(replay_env.physics_adapter.get_state())

    contact_delta = _mean_metric(states, "contact_force_N") - _mean_metric(replay_states, "contact_force_N")
    constraint_delta = _mean_metric(states, "constraint_error") - _mean_metric(replay_states, "constraint_error")

    summary = {
        "scene_tracks_quality": tracks_result.quality.quality_score,
        "contact_force_delta": contact_delta,
        "constraint_error_delta": constraint_delta,
        "reconstruction_confidence": reconstruction.confidence_score,
        "scene_tracks_path": str(tracks_result.scene_tracks_path),
    }
    summary.update(_confidence_assessment(reconstruction.confidence_score))
    return summary


def _confidence_assessment(confidence: float) -> Dict[str, str]:
    if confidence < 0.3:
        return {
            "reconstruction_assessment": "unusable",
            "reconstruction_action": "Increase max-frames or enable segmentation smoothing.",
        }
    if confidence < 0.6:
        return {
            "reconstruction_assessment": "usable_w_caution",
            "reconstruction_action": "Increase max-frames or improve segmentation stability.",
        }
    if confidence < 0.8:
        return {
            "reconstruction_assessment": "good",
            "reconstruction_action": "Consider more frames if you need tighter alignment.",
        }
    return {
        "reconstruction_assessment": "excellent",
        "reconstruction_action": "No action needed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Workcell real physics + SceneTracks demo")
    parser.add_argument("--out-dir", type=str, default="results/workcell_demo", help="Output directory")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames to render")
    parser.add_argument("--seed", type=int, default=7, help="Seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_demo(out_dir=out_dir, steps=args.steps, max_frames=args.max_frames, seed=args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
