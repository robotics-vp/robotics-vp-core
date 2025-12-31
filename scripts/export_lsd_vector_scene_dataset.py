#!/usr/bin/env python3
"""
Export dataset for training LSD Vector Scene world and behaviour models.

This script:
1. Samples scene configs from a parameter space
2. Runs episodes in LSD Vector Scene environments
3. Exports scene graphs, latents, difficulty features, and trajectories
4. Saves in a format suitable for PyTorch dataloaders

Usage:
    python scripts/export_lsd_vector_scene_dataset.py \
        --num-scenes 100 \
        --episodes-per-scene 5 \
        --max-steps 200 \
        --output-path data/lsd_vector_scene_dataset

    # With config template:
    python scripts/export_lsd_vector_scene_dataset.py \
        --config-path configs/lsd_vector_scene/export_template.yaml \
        --num-scenes 50 \
        --output-path data/lsd_dataset_warehouse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DatasetShard:
    """Metadata for a dataset shard."""
    shard_id: str
    num_episodes: int
    num_scenes: int
    file_path: str
    created_at: str


@dataclass
class ExportConfig:
    """Configuration for dataset export."""
    num_scenes: int = 100
    episodes_per_scene: int = 5
    max_steps: int = 200
    output_path: str = "data/lsd_vector_scene_dataset"
    shard_size: int = 100  # Episodes per shard
    save_rgb: bool = False
    save_depth: bool = False
    seed: int = 42
    enable_motion_hierarchy: bool = False

    # Scene parameter ranges for sampling
    topology_types: List[str] = None
    num_nodes_range: Tuple[int, int] = (5, 20)
    num_objects_range: Tuple[int, int] = (5, 30)
    density_range: Tuple[float, float] = (0.3, 0.8)
    route_length_range: Tuple[float, float] = (10.0, 50.0)
    num_humans_range: Tuple[int, int] = (0, 4)
    num_forklifts_range: Tuple[int, int] = (0, 2)
    tilt_range: Tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self):
        if self.topology_types is None:
            self.topology_types = ["WAREHOUSE_AISLES", "KITCHEN_LAYOUT", "RESIDENTIAL_GARAGE"]


def sample_scene_config(rng: random.Random, export_config: ExportConfig) -> Dict[str, Any]:
    """Sample a random scene configuration from the parameter space."""
    return {
        "topology_type": rng.choice(export_config.topology_types),
        "num_nodes": rng.randint(*export_config.num_nodes_range),
        "num_objects": rng.randint(*export_config.num_objects_range),
        "density": rng.uniform(*export_config.density_range),
        "route_length": rng.uniform(*export_config.route_length_range),
        "lighting": rng.choice(["BRIGHT_INDOOR", "DIM_INDOOR", "NIGHT"]),
        "clutter_level": rng.choice(["LOW", "MEDIUM", "HIGH"]),
        "num_humans": rng.randint(*export_config.num_humans_range),
        "num_robots": 1,  # Always one robot (the agent)
        "num_forklifts": rng.randint(*export_config.num_forklifts_range),
        "tilt": rng.uniform(*export_config.tilt_range),
        "use_simple_policy": True,
        "enable_motion_hierarchy": export_config.enable_motion_hierarchy,
        "max_steps": export_config.max_steps,
        "random_seed": rng.randint(0, 2**31),
    }


def _serialize_agent_trajectories(trajectories: List[Any], graph) -> Dict[str, Any]:
    if not trajectories or graph is None:
        return {}

    label_map: Dict[int, str] = {}
    for obj in graph.objects:
        class_name = obj.class_id.name.lower() if hasattr(obj.class_id, "name") else str(obj.class_id).lower()
        agent_index = obj.attributes.get("agent_index") if obj.attributes else None
        suffix = agent_index if agent_index is not None else obj.id
        label_map[obj.id] = f"{class_name}_{suffix}"

    serialized = []
    for traj in trajectories:
        positions = [[float(x), float(y), float(z)] for x, y, z in traj.positions]
        timestamps = [float(t) for t in traj.timestamps]
        serialized.append(
            {
                "agent_id": int(traj.agent_id),
                "label": label_map.get(traj.agent_id, f"agent_{traj.agent_id}"),
                "positions": positions,
                "timestamps": timestamps,
            }
        )

    return {
        "agent_trajectories": serialized,
        "agent_labels": [entry["label"] for entry in serialized],
    }


def run_episode(
    env,
    max_steps: int,
    policy_fn=None,
    *,
    enable_motion_hierarchy: bool = False,
    motion_hierarchy_model=None,
    motion_hierarchy_config=None,
) -> Dict[str, Any]:
    """
    Run a single episode and collect data.

    Returns:
        Dict with trajectory, scene graph, latents, difficulty features, etc.
    """
    obs = env.reset()

    # Collect trajectory data
    trajectory: List[Dict[str, Any]] = []
    info_history: List[Dict[str, Any]] = []
    done = False
    step_count = 0
    total_reward = 0.0

    while not done and step_count < max_steps:
        # Get action from policy (or use simple policy)
        if policy_fn is not None:
            action = policy_fn(obs)
        else:
            # Simple policy: moderate speed, high care
            action = np.array([0.5, 0.7])

        obs, info, done = env.step(action)
        info_history.append(info)
        step_count += 1

        # Compute reward
        reward = info.get("delta_units", 0) - info.get("delta_errors", 0) * 2.0
        total_reward += reward

        # Record trajectory step
        trajectory.append({
            "step": step_count,
            "action": action.tolist() if hasattr(action, "tolist") else list(action),
            "reward": float(reward),
            "done": done,
            "t": obs.get("t", step_count),
            "completed": obs.get("completed", 0),
            "errors": obs.get("errors", 0),
        })

    # Get episode log
    episode_log = env.get_episode_log(info_history)

    episode_data = {
        "episode_id": f"{env.scene_id}_{step_count}_{int(time.time() * 1000) % 10000}",
        "scene_id": env.scene_id,
        "trajectory": trajectory,
        "steps": step_count,
        "total_reward": float(total_reward),
        "termination_reason": episode_log["episode_summary"]["termination_reason"],
        "mpl_metrics": episode_log["mpl_metrics"],
        "difficulty_features": episode_log["difficulty_features"],
    }

    if enable_motion_hierarchy:
        agent_payload = _serialize_agent_trajectories(env.get_agent_trajectories(), env.graph)
        if agent_payload:
            episode_data.update(agent_payload)
        if agent_payload and motion_hierarchy_model and motion_hierarchy_config:
            from src.envs.lsd3d_env.motion_hierarchy_integration import compute_motion_hierarchy_for_lsd_episode

            mh_output = compute_motion_hierarchy_for_lsd_episode(
                {"episode_id": episode_data["episode_id"], "trajectory_data": agent_payload},
                model=motion_hierarchy_model,
                config=motion_hierarchy_config,
            )
            if isinstance(mh_output.get("hierarchy"), np.ndarray):
                mh_output["hierarchy"] = mh_output["hierarchy"].tolist()
            elif hasattr(mh_output.get("hierarchy"), "tolist"):
                mh_output["hierarchy"] = mh_output["hierarchy"].tolist()
            episode_data["motion_hierarchy"] = mh_output

    return episode_data


def serialize_scene_graph(graph) -> Dict[str, Any]:
    """Serialize a SceneGraph to a JSON-compatible dict."""
    nodes = []
    for node in graph.nodes:
        nodes.append({
            "node_id": node.id,
            "polyline": node.polyline.tolist() if hasattr(node.polyline, "tolist") else list(node.polyline),
            "node_type": str(node.node_type.name) if hasattr(node.node_type, "name") else str(node.node_type),
            "attributes": dict(node.attributes) if node.attributes else {},
        })

    edges = []
    for edge in graph.edges:
        edges.append({
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "edge_type": str(edge.edge_type.name) if hasattr(edge.edge_type, "name") else str(edge.edge_type),
        })

    objects = []
    for obj in graph.objects:
        objects.append({
            "object_id": obj.id,
            "class_id": str(obj.class_id.name) if hasattr(obj.class_id, "name") else str(obj.class_id),
            "x": float(obj.x),
            "y": float(obj.y),
            "z": float(obj.z),
            "heading": float(obj.heading),
            "speed": float(obj.speed),
            "length": float(obj.length),
            "width": float(obj.width),
            "height": float(obj.height),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "objects": objects,
        "bounding_box": list(graph.bounding_box()) if hasattr(graph, "bounding_box") else None,
    }


def export_dataset(
    export_config: ExportConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Export the complete dataset.

    Returns:
        Metadata about the exported dataset
    """
    # Late import to avoid circular dependencies
    from src.config.lsd_vector_scene_config import LSDVectorSceneConfig
    from src.envs.lsd_vector_scene_env import (
        LSDVectorSceneEnv,
        LSDVectorSceneEnvConfig,
        SceneGraphConfig,
        VisualStyleConfig,
        BehaviourConfig,
    )

    output_dir = Path(export_config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(export_config.seed)
    np.random.seed(export_config.seed)

    all_episodes: List[Dict[str, Any]] = []
    all_scenes: Dict[str, Dict[str, Any]] = {}  # scene_id -> scene data
    shards: List[DatasetShard] = []

    total_episodes = export_config.num_scenes * export_config.episodes_per_scene

    if verbose:
        print("=" * 60)
        print("LSD Vector Scene Dataset Export")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Num scenes: {export_config.num_scenes}")
        print(f"Episodes per scene: {export_config.episodes_per_scene}")
        print(f"Max steps: {export_config.max_steps}")
        print(f"Total episodes: {total_episodes}")
        print()

    episode_count = 0
    start_time = time.time()

    motion_hierarchy_model = None
    motion_hierarchy_config = None
    if export_config.enable_motion_hierarchy:
        from src.vision.motion_hierarchy.config import MotionHierarchyConfig
        from src.vision.motion_hierarchy.motion_hierarchy_node import MotionHierarchyNode

        motion_hierarchy_config = MotionHierarchyConfig()
        motion_hierarchy_model = MotionHierarchyNode(motion_hierarchy_config)

    for scene_idx in range(export_config.num_scenes):
        # Sample scene config
        scene_config_dict = sample_scene_config(rng, export_config)

        # Build env config
        env_config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type=scene_config_dict["topology_type"],
                num_nodes=scene_config_dict["num_nodes"],
                num_objects=scene_config_dict["num_objects"],
                density=scene_config_dict["density"],
                route_length=scene_config_dict["route_length"],
                seed=scene_config_dict["random_seed"],
            ),
            visual_style_config=VisualStyleConfig(
                lighting=scene_config_dict["lighting"],
                clutter_level=scene_config_dict["clutter_level"],
                voxel_size=0.2,  # Coarse for speed
            ),
            behaviour_config=BehaviourConfig(
                num_humans=scene_config_dict["num_humans"],
                num_robots=scene_config_dict["num_robots"],
                num_forklifts=scene_config_dict["num_forklifts"],
                tilt=scene_config_dict["tilt"],
                use_simple_policy=True,
            ),
            max_steps=scene_config_dict["max_steps"],
        )

        env = LSDVectorSceneEnv(env_config)

        # Run first episode to get scene data
        episode_data = run_episode(
            env,
            export_config.max_steps,
            enable_motion_hierarchy=export_config.enable_motion_hierarchy,
            motion_hierarchy_model=motion_hierarchy_model,
            motion_hierarchy_config=motion_hierarchy_config,
        )
        scene_id = episode_data["scene_id"]

        # Store scene data (only once per scene)
        if scene_id not in all_scenes:
            all_scenes[scene_id] = {
                "scene_id": scene_id,
                "config": scene_config_dict,
                "scene_graph": serialize_scene_graph(env.graph) if env.graph else None,
                "difficulty_features": episode_data["difficulty_features"],
                "voxel_shape": list(env.voxels.shape) if env.voxels else None,
                "num_gaussians": env.gaussian_scene.num_gaussians if env.gaussian_scene else 0,
            }

        episode_data["scene_config"] = scene_config_dict
        all_episodes.append(episode_data)
        episode_count += 1

        # Run additional episodes for this scene
        for ep_idx in range(1, export_config.episodes_per_scene):
            episode_data = run_episode(
                env,
                export_config.max_steps,
                enable_motion_hierarchy=export_config.enable_motion_hierarchy,
                motion_hierarchy_model=motion_hierarchy_model,
                motion_hierarchy_config=motion_hierarchy_config,
            )
            episode_data["scene_config"] = scene_config_dict
            all_episodes.append(episode_data)
            episode_count += 1

        # Progress
        if verbose and (scene_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = episode_count / elapsed
            remaining = (total_episodes - episode_count) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"Scene {scene_idx + 1}/{export_config.num_scenes} | "
                  f"Episodes: {episode_count}/{total_episodes} | "
                  f"Rate: {eps_per_sec:.1f} ep/s | "
                  f"ETA: {remaining:.0f}s")

        # Save shard if needed
        if len(all_episodes) >= export_config.shard_size:
            shard = _save_shard(
                output_dir,
                all_episodes,
                len(shards),
            )
            shards.append(shard)
            all_episodes = []

    # Save remaining episodes
    if all_episodes:
        shard = _save_shard(
            output_dir,
            all_episodes,
            len(shards),
        )
        shards.append(shard)

    # Save scenes file
    scenes_path = output_dir / "scenes.json"
    with open(scenes_path, "w") as f:
        json.dump(all_scenes, f, indent=2)

    # Save index file
    index = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export_config": {
            "num_scenes": export_config.num_scenes,
            "episodes_per_scene": export_config.episodes_per_scene,
            "max_steps": export_config.max_steps,
            "seed": export_config.seed,
        },
        "total_episodes": episode_count,
        "total_scenes": len(all_scenes),
        "shards": [asdict(s) for s in shards],
        "scenes_file": "scenes.json",
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - start_time
    if verbose:
        print()
        print("=" * 60)
        print("Export Complete")
        print("=" * 60)
        print(f"Total episodes: {episode_count}")
        print(f"Total scenes: {len(all_scenes)}")
        print(f"Shards: {len(shards)}")
        print(f"Output directory: {output_dir}")
        print(f"Time: {elapsed:.1f}s")

    return index


def _save_shard(
    output_dir: Path,
    episodes: List[Dict[str, Any]],
    shard_idx: int,
) -> DatasetShard:
    """Save a shard of episodes."""
    shard_id = f"shard_{shard_idx:04d}"
    file_path = output_dir / f"{shard_id}.json"

    # Count unique scenes
    scene_ids = set(ep["scene_id"] for ep in episodes)

    # Save as JSON (could also use .npz for efficiency)
    with open(file_path, "w") as f:
        json.dump(episodes, f)

    return DatasetShard(
        shard_id=shard_id,
        num_episodes=len(episodes),
        num_scenes=len(scene_ids),
        file_path=str(file_path.name),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def load_dataset_shard(shard_path: str) -> List[Dict[str, Any]]:
    """Load a dataset shard."""
    with open(shard_path, "r") as f:
        return json.load(f)


def load_dataset_index(dataset_path: str) -> Dict[str, Any]:
    """Load dataset index."""
    index_path = Path(dataset_path) / "index.json"
    with open(index_path, "r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LSD Vector Scene dataset for training"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="Number of unique scenes to generate",
    )
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=5,
        help="Episodes to run per scene",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/lsd_vector_scene_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Episodes per shard file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--enable-motion-hierarchy",
        action="store_true",
        help="If set, enable Motion Hierarchy Node computation for each LSD episode.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional YAML config with parameter ranges",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Build export config
    export_config = ExportConfig(
        num_scenes=args.num_scenes,
        episodes_per_scene=args.episodes_per_scene,
        max_steps=args.max_steps,
        output_path=args.output_path,
        shard_size=args.shard_size,
        seed=args.seed,
        enable_motion_hierarchy=args.enable_motion_hierarchy,
    )

    # Load config from YAML if provided
    if args.config_path and Path(args.config_path).exists():
        import yaml
        with open(args.config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if "topology_types" in yaml_config:
            export_config.topology_types = yaml_config["topology_types"]
        if "num_nodes_range" in yaml_config:
            export_config.num_nodes_range = tuple(yaml_config["num_nodes_range"])
        if "density_range" in yaml_config:
            export_config.density_range = tuple(yaml_config["density_range"])

    try:
        export_dataset(
            export_config=export_config,
            verbose=not args.quiet,
        )
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
