#!/usr/bin/env python3
"""Bootstrap a locally grounded workcell semantic loop end to end."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence.belief_state import BeliefState
from src.envs.workcell_env import WorkcellEnv
from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.observations.mujoco_render import (
    build_segmentation_label_map,
    render_workcell_frames,
)
from src.envs.workcell_env.scene.scene_spec import FixtureSpec, PartSpec, ToolSpec, WorkcellSceneSpec
from src.envs.workcell_env.tasks.peg_in_hole import PegInHoleTask
from src.motor_backend.rollout_capture import EpisodeMetadata, record_episode_rollout, start_rollout_capture
from src.motor_backend.sensor_bundle import SensorBundleData
from src.orchestrator.coverage_loop import run_coverage_loop
from src.orchestrator.semantic_runtime_learning import (
    build_semantic_runtime_learning_corpus,
    write_semantic_runtime_learning_corpus,
)
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.semantic.runtime_backbone import SemanticRuntimeBackbone
from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks
from src.world_model.semantic_world_model import SemanticWorldModelBuilder


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


def _build_scene_spec(seed: int, episode_idx: int) -> WorkcellSceneSpec:
    x_offset = ((seed + episode_idx) % 5 - 2) * 0.01
    y_offset = ((seed + 2 * episode_idx) % 5 - 2) * 0.008
    return WorkcellSceneSpec(
        workcell_id=f"semantic_bootstrap_{episode_idx:03d}",
        fixtures=[
            FixtureSpec(
                id="hole",
                position=(0.03 + x_offset, -0.02 + y_offset, 0.05),
                orientation=(1.0, 0.0, 0.0, 0.0),
                fixture_type="vise",
            )
        ],
        parts=[
            PartSpec(
                id="peg",
                position=(-0.03 + x_offset, 0.01 - y_offset, 0.15),
                orientation=(1.0, 0.0, 0.0, 0.0),
                part_type="peg",
                dimensions_mm=(30.0, 30.0, 60.0),
                material="aluminum",
            )
        ],
        tools=[
            ToolSpec(
                id="end_effector",
                position=(0.0, 0.0, 0.25),
                orientation=(1.0, 0.0, 0.0, 0.0),
                tool_type="gripper",
                precision_mm=0.5,
            )
        ],
        spatial_bounds=(1.0, 1.0, 1.0),
    )


def _action_profiles() -> Sequence[Sequence[Sequence[float]]]:
    return (
        ((0.0, 0.0, -0.01), (0.01, 0.0, -0.01), (0.01, -0.005, -0.01), (0.0, 0.0, -0.005)),
        ((0.01, 0.0, -0.008), (0.01, 0.0, -0.008), (-0.005, 0.008, -0.01), (0.0, 0.0, -0.01)),
        ((-0.008, 0.006, -0.008), (0.012, -0.006, -0.01), (0.0, 0.0, -0.012), (0.0, 0.0, -0.01)),
    )


def _build_scene_object_catalog(scene_spec: WorkcellSceneSpec) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for fixture in list(scene_spec.fixtures):
        catalog.append(
            {
                "object_id": fixture.id,
                "class_name": fixture.fixture_type,
                "category": "fixture",
                "semantic_tags": [fixture.fixture_type, "affordance:align"],
            }
        )
    for part in list(scene_spec.parts):
        catalog.append(
            {
                "object_id": part.id,
                "class_name": part.part_type,
                "category": "workpiece",
                "semantic_tags": [part.part_type, "affordance:insert", "affordance:pick"],
            }
        )
    for tool in list(scene_spec.tools):
        catalog.append(
            {
                "object_id": tool.id,
                "class_name": tool.tool_type,
                "category": "tool",
                "semantic_tags": [tool.tool_type, "affordance:grasp", "affordance:align"],
            }
        )
    return catalog


def _episode_actions(episode_idx: int, steps: int) -> List[Dict[str, Any]]:
    profiles = _action_profiles()
    base_profile = profiles[episode_idx % len(profiles)]
    actions: List[Dict[str, Any]] = []
    for step_idx in range(steps):
        delta = base_profile[min(step_idx, len(base_profile) - 1)]
        actions.append({"object_id": "end_effector", "delta_position": tuple(float(v) for v in delta)})
    return actions


def _run_single_workcell_episode(
    *,
    scenario_id: str,
    episode_idx: int,
    output_root: Path,
    steps: int,
    max_frames: int,
    seed: int,
    camera: str,
    grounding_mode: str,
    backend_policy: str,
) -> Dict[str, Any]:
    scene_spec = _build_scene_spec(seed, episode_idx)
    task = PegInHoleTask(peg_id="peg", hole_id="hole", tolerance_mm=2.0)
    capture_rgb = grounding_mode != "vector_proxy"
    config = WorkcellEnvConfig(
        physics_mode="MUJOCO" if _mujoco_available() else "SIMPLE",
        max_steps=steps,
        time_step_s=0.02,
        capture_rgb_frames=capture_rgb,
        render_width=128,
        render_height=128,
        render_fps=10,
        render_max_frames=max_frames,
    )
    env = WorkcellEnv(config=config, scene_spec=scene_spec, task=task, seed=seed + episode_idx)
    episode_id = f"{scenario_id}_ep_{episode_idx:03d}"
    env.reset(seed=seed + episode_idx, episode_id=episode_id)

    states: List[Dict[str, Any]] = []
    actions = _episode_actions(episode_idx, steps)
    for action in actions:
        env.step(action)
        states.append(env.physics_adapter.get_state())

    frames, depth_frames, seg_frames, camera_params = render_workcell_frames(
        scene_spec=scene_spec,
        states=states,
        camera_name=camera,
        width=config.render_width,
        height=config.render_height,
        max_frames=max_frames,
        seed=seed + episode_idx,
    )
    timestamps = [float(state.get("time_s", idx * config.time_step_s)) for idx, state in enumerate(states[: len(frames)])]
    seg_label_map = build_segmentation_label_map(scene_spec)
    scene_object_catalog = _build_scene_object_catalog(scene_spec)
    sensor_bundle = SensorBundleData(
        cameras=[camera],
        rgb={camera: np.asarray(frames, dtype=np.uint8)} if capture_rgb else {},
        depth={camera: np.asarray(depth_frames, dtype=np.float32)} if depth_frames else {},
        seg={camera: np.asarray(seg_frames, dtype=np.int32)} if seg_frames else {},
        intrinsics={
            camera: {
                "fx": float(camera_params.fx),
                "fy": float(camera_params.fy),
                "cx": float(camera_params.cx),
                "cy": float(camera_params.cy),
                "width": int(camera_params.width),
                "height": int(camera_params.height),
            }
        },
        extrinsics={camera: np.asarray(camera_params.world_from_cam, dtype=np.float32)},
        timestamps_s=timestamps,
        depth_unit="meters",
        segmentation_label_map={
            str(seg_id): {
                "object_id": object_id,
                "class_name": next(
                    (item["class_name"] for item in scene_object_catalog if item["object_id"] == object_id),
                    object_id,
                ),
            }
            for object_id, seg_id in seg_label_map.items()
        },
        scene_object_catalog=scene_object_catalog,
    )

    trajectory_data = {
        "scene_spec": scene_spec.to_dict(),
        "states": states,
        "actions": actions,
        "seed": seed + episode_idx,
    }
    metrics = {
        "reward": float(max(0.0, 5.0 - episode_idx)),
        "quality_score": 0.9,
        "safety_score": 0.95,
        "energy_wh_per_unit": 1.5 + episode_idx * 0.1,
    }

    rollout_root = output_root / "rollouts"
    start_rollout_capture(scenario_id, rollout_root)
    record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=episode_idx,
        metadata=EpisodeMetadata(
            episode_id=episode_id,
            task_id="peg_in_hole",
            robot_family="workcell",
            seed=seed + episode_idx,
            env_params={"config": config.to_dict(), "scene_spec": scene_spec.to_dict()},
        ),
        trajectory_data=trajectory_data,
        rgb_frames=np.asarray(frames, dtype=np.uint8) if capture_rgb else None,
        depth_frames=np.asarray(depth_frames, dtype=np.float32) if depth_frames else None,
        metrics=metrics,
        base_dir=rollout_root,
        sensor_bundle=sensor_bundle,
    )

    episode_dir = rollout_root / scenario_id / f"episode_{episode_idx:03d}"
    scene_tracks_result = run_scene_tracks(
        datapack_path=episode_dir,
        output_path=episode_dir,
        seed=seed + episode_idx,
        max_frames=max_frames,
        camera=camera,
        mode="rgb" if capture_rgb else "vector_proxy",
        min_quality=0.1,
        allow_low_quality=True,
        backend_policy=backend_policy,
    )

    belief_timestamp = datetime.now(timezone.utc).isoformat()
    scene_tracks_summary = dict(scene_tracks_result.frame_metadata.get("semantic_summary", {}) or {})
    belief_state = BeliefState(
        belief_id=f"belief_{episode_id}",
        episode_id=episode_id,
        timestamp=belief_timestamp,
        semantic_tags=sorted(
            {
                "peg",
                "hole",
                "workcell",
                "affordance:align",
                "affordance:insert",
                *[str(tag) for tag in scene_tracks_result.frame_metadata.get("semantic_tags", []) or []],
            }
        ),
        state_vector={
            "geometry_quality": float(scene_tracks_result.scene_ir_quality),
            "semantic_quality": float(scene_tracks_result.quality.quality_score),
            "coverage": float(scene_tracks_summary.get("class_label_coverage", 0.0) or 0.0),
        },
        uncertainty={
            "epistemic": float(max(0.0, 1.0 - scene_tracks_result.quality.quality_score)),
            "coverage_gap": float(max(0.0, 1.0 - scene_tracks_summary.get("class_label_coverage", 0.0))),
        },
        artifact_refs={
            "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
            "datapack_path": str(episode_dir),
        },
        metadata={
            "task_family": "workcell_bootstrap",
            "grounding_mode": "scene_tracks_real_rgb" if capture_rgb else "scene_tracks_vector_proxy",
        },
    )
    world_model_builder = SemanticWorldModelBuilder()
    world_model = world_model_builder.build_from_runtime_fusion(
        episode_id=episode_id,
        task_id="peg_in_hole",
        objective_preset="balanced_contract",
        belief_state=belief_state,
        semantic_tags=belief_state.semantic_tags,
        scene_tracks_payload=dict(np.load(scene_tracks_result.scene_tracks_path, allow_pickle=False)),
        artifact_refs={
            "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
            "datapack_path": str(episode_dir),
        },
        metadata={
            "camera": camera,
            "backend_policy": backend_policy,
            "backend_selected": scene_tracks_result.frame_metadata.get("runner", {})
            .get("run_config", {})
            .get("backend_selected", ""),
        },
    )
    backbone = SemanticRuntimeBackbone()
    backbone_result = backbone.build(
        task_id="peg_in_hole",
        objective_preset="balanced_contract",
        semantic_world_model=world_model,
        runtime_metrics={
            "scene_tracks_quality": float(scene_tracks_result.quality.quality_score),
            "scene_ir_quality": float(scene_tracks_result.scene_ir_quality),
            "expected_delta_mpl": 0.15,
            "expected_delta_error": -0.05,
            "expected_delta_energy_Wh": -0.02,
        },
        metadata={
            "scene_tracks_backend": scene_tracks_result.frame_metadata.get("runner", {})
            .get("run_config", {})
            .get("backend_selected", ""),
            "scene_tracks_non_stub": scene_tracks_result.adapter_status.get("overall_mode") in {"real", "passthrough"},
        },
        backends=[str(scene_tracks_result.adapter_status.get("overall_mode", ""))],
    )
    sidecar_paths = backbone.write_sidecars(
        output_dir=episode_dir,
        file_stem=episode_id,
        result=backbone_result,
    )

    metadata_path = episode_dir / "metadata.json"
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["scene_tracks_non_stub"] = bool(scene_tracks_result.adapter_status.get("overall_mode") in {"real", "passthrough"})
    metadata_payload["semantic_memory_grounded"] = bool(
        world_model.topology.get("grounded_track_object_count", 0) > 0
    )
    metadata_payload["semantic_world_model_path"] = sidecar_paths["semantic_world_model_path"]
    metadata_payload["semantic_snapshot_path"] = sidecar_paths["semantic_snapshot_path"]
    metadata_payload["orchestrator_advisory_path"] = sidecar_paths["orchestrator_advisory_path"]
    metadata_payload["scene_tracks_backend"] = scene_tracks_result.frame_metadata.get("runner", {}).get("run_config", {}).get(
        "backend_selected",
        "",
    )
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return {
        "episode_id": episode_id,
        "episode_dir": str(episode_dir),
        "scene_tracks_path": str(scene_tracks_result.scene_tracks_path),
        "scene_tracks_quality": float(scene_tracks_result.quality.quality_score),
        "scene_ir_quality": float(scene_tracks_result.scene_ir_quality),
        "semantic_world_model_path": sidecar_paths["semantic_world_model_path"],
        "backend_selected": scene_tracks_result.frame_metadata.get("runner", {}).get("run_config", {}).get(
            "backend_selected",
            "",
        ),
        "grounded_track_object_count": int(world_model.topology.get("grounded_track_object_count", 0)),
        "semantic_density_score": float(scene_tracks_summary.get("semantic_density_score", 0.0) or 0.0),
    }


def run_semantic_workcell_bootstrap(
    *,
    output_root: Path,
    episodes: int,
    steps: int,
    max_frames: int,
    seed: int,
    camera: str,
    grounding_mode: str,
    backend_policy: str,
    sim_limit: int,
    diffusion_limit: int,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_id = f"semantic_workcell_{seed}"
    episode_summaries = [
        _run_single_workcell_episode(
            scenario_id=scenario_id,
            episode_idx=episode_idx,
            output_root=output_root,
            steps=steps,
            max_frames=max_frames,
            seed=seed,
            camera=camera,
            grounding_mode=grounding_mode,
            backend_policy=backend_policy,
        )
        for episode_idx in range(episodes)
    ]

    replay_dir = output_root / "replay_dataset"
    replay_bundle = ReplayDatasetBuilder().add_rollout_bundle(
        output_root / "rollouts",
        scenario_id=scenario_id,
        source_domain="workcell_semantic_bootstrap",
    ).write(replay_dir)
    loaded_bundle = load_replay_dataset(replay_dir)

    corpus = build_semantic_runtime_learning_corpus(loaded_bundle, max_counterfactuals=2)
    corpus_paths = write_semantic_runtime_learning_corpus(output_root / "semantic_runtime_corpus", corpus)
    coverage_result = run_coverage_loop(
        [row.to_dict() for row in corpus.rows],
        env_names=["workcell"],
        semantic_world_model=None,
        sim_agenda_limit=sim_limit,
        diffusion_limit=diffusion_limit,
        write_artifacts=False,
        artifact_dir=output_root / "coverage_artifacts",
    )

    coverage_paths = coverage_result.write_artifacts(output_root / "coverage_artifacts")
    summary = {
        "scenario_id": scenario_id,
        "episodes": episode_summaries,
        "replay_summary": replay_bundle.to_summary(),
        "runtime_corpus_summary": corpus.summary,
        "runtime_corpus_paths": corpus_paths,
        "coverage_summary": coverage_result.coverage_summary,
        "coverage_artifact_paths": coverage_paths,
        "mujoco_available": _mujoco_available(),
        "backend_policy": backend_policy,
        "grounding_mode": grounding_mode,
    }
    summary_path = output_root / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Directory for datapacks, replay data, and coverage artifacts")
    parser.add_argument("--episodes", type=int, default=2, help="Number of workcell episodes to capture")
    parser.add_argument("--steps", type=int, default=5, help="Control steps per episode")
    parser.add_argument("--max-frames", type=int, default=5, help="Rendered frames per episode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic scene/task variation")
    parser.add_argument("--camera", default="front", help="Camera name for SceneTracks and rendering")
    parser.add_argument(
        "--grounding-mode",
        choices=("auto", "real_rgb", "vector_proxy"),
        default="auto",
        help="Grounding mode: real_rgb forces rendered RGB; vector_proxy skips RGB and uses geometry/masks only",
    )
    parser.add_argument(
        "--backend-policy",
        choices=("auto", "real", "passthrough", "stub"),
        default="auto",
        help="SceneTracks backend selection policy",
    )
    parser.add_argument("--sim-limit", type=int, default=6, help="Coverage-loop simulation agenda cap")
    parser.add_argument("--diffusion-limit", type=int, default=6, help="Coverage-loop diffusion prompt cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grounding_mode = args.grounding_mode
    if grounding_mode == "auto":
        grounding_mode = "real_rgb" if _mujoco_available() else "vector_proxy"
    summary = run_semantic_workcell_bootstrap(
        output_root=Path(args.output_root),
        episodes=max(args.episodes, 1),
        steps=max(args.steps, 1),
        max_frames=max(args.max_frames, 1),
        seed=args.seed,
        camera=args.camera,
        grounding_mode=grounding_mode,
        backend_policy=args.backend_policy,
        sim_limit=max(args.sim_limit, 1),
        diffusion_limit=max(args.diffusion_limit, 1),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
