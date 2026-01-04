#!/usr/bin/env python3
"""
Smoke test for LSD Vector Scene Environment.

Usage:
    python scripts/smoke_lsd_vector_scene_env.py --num-episodes 3 --config-path configs/lsd_vector_scene/smoke_warehouse.yaml
    python scripts/smoke_lsd_vector_scene_env.py --num-episodes 2  # Uses default config
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def run_smoke_test(
    num_episodes: int = 3,
    config_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run smoke test for LSD Vector Scene Environment.

    Args:
        num_episodes: Number of episodes to run
        config_path: Optional path to YAML config
        verbose: Print progress

    Returns:
        Summary of test results
    """
    # Late import to avoid issues if dependencies missing
    from src.envs.lsd_vector_scene_env import (
        LSDVectorSceneEnv,
        LSDVectorSceneEnvConfig,
        SceneGraphConfig,
        VisualStyleConfig,
        BehaviourConfig,
        create_env_from_config_path,
    )

    if verbose:
        print("=" * 60)
        print("LSD Vector Scene Environment Smoke Test")
        print("=" * 60)

    # Create environment
    if config_path and Path(config_path).exists():
        if verbose:
            print(f"Loading config from: {config_path}")
        env = create_env_from_config_path(config_path)
    else:
        if verbose:
            print("Using default configuration")
        config = LSDVectorSceneEnvConfig(
            scene_graph_config=SceneGraphConfig(
                topology_type="WAREHOUSE_AISLES",
                num_nodes=10,
                num_objects=8,
                density=0.5,
                route_length=30.0,
                seed=42,
            ),
            visual_style_config=VisualStyleConfig(
                lighting="DIM_INDOOR",
                clutter_level="MEDIUM",
                voxel_size=0.2,
            ),
            behaviour_config=BehaviourConfig(
                num_humans=2,
                num_robots=1,
                num_forklifts=0,
                use_simple_policy=True,
            ),
            max_steps=50,
        )
        env = LSDVectorSceneEnv(config)

    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for ep in range(num_episodes):
        if verbose:
            print(f"\n--- Episode {ep + 1}/{num_episodes} ---")

        ep_start = time.time()

        # Reset environment
        obs = env.reset()
        if verbose:
            print(f"  Scene ID: {env.scene_id}")
            print(f"  Num nodes: {len(env.graph.nodes) if env.graph else 0}")
            print(f"  Num objects: {len(env.graph.objects) if env.graph else 0}")
            print(f"  Num dynamic agents: {obs.get('num_agents', 0)}")
            if env.voxels:
                print(f"  Voxel grid shape: {env.voxels.shape}")
                print(f"  Occupied voxels: {env.voxels.get_occupied_count()}")
            if env.mesh:
                print(f"  Mesh vertices: {len(env.mesh.vertices)}")
                print(f"  Mesh faces: {len(env.mesh.faces)}")
            if env.gaussian_scene:
                print(f"  Num Gaussians: {env.gaussian_scene.num_gaussians}")

        info_history: List[Dict[str, Any]] = []
        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            # Random action (speed, care)
            action = np.array([
                np.random.uniform(0.4, 0.8),  # speed
                np.random.uniform(0.4, 0.8),  # care
            ])

            obs, info, done = env.step(action)
            info_history.append(info)
            step_count += 1

            # Simple reward approximation
            reward = info.get("delta_units", 0) - info.get("delta_errors", 0) * 0.5
            total_reward += reward

        ep_elapsed = time.time() - ep_start

        # Get episode log
        episode_log = env.get_episode_log(info_history)

        if verbose:
            print(f"\n  Results:")
            print(f"    Steps: {step_count}")
            print(f"    Termination: {episode_log['episode_summary']['termination_reason']}")
            print(f"    MPL (units/hr): {episode_log['mpl_metrics']['mpl_units_per_hour']:.2f}")
            print(f"    Error rate: {episode_log['mpl_metrics']['error_rate']:.3f}")
            print(f"    Energy (Wh): {episode_log['mpl_metrics']['energy_wh']:.4f}")
            print(f"    Elapsed: {ep_elapsed:.2f}s")
            print(f"\n  Difficulty features:")
            for key, value in episode_log['difficulty_features'].items():
                print(f"    {key}: {value}")

        results.append({
            "episode": ep + 1,
            "scene_id": env.scene_id,
            "steps": step_count,
            "termination": episode_log['episode_summary']['termination_reason'],
            "mpl": episode_log['mpl_metrics']['mpl_units_per_hour'],
            "error_rate": episode_log['mpl_metrics']['error_rate'],
            "energy_wh": episode_log['mpl_metrics']['energy_wh'],
            "elapsed_s": ep_elapsed,
            "difficulty_features": episode_log['difficulty_features'],
        })

    total_elapsed = time.time() - total_start

    summary = {
        "num_episodes": num_episodes,
        "total_elapsed_s": total_elapsed,
        "avg_mpl": np.mean([r["mpl"] for r in results]),
        "avg_error_rate": np.mean([r["error_rate"] for r in results]),
        "avg_steps": np.mean([r["steps"] for r in results]),
        "episodes": results,
        "status": "PASSED",
    }

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Episodes: {num_episodes}")
        print(f"Total time: {total_elapsed:.2f}s")
        print(f"Avg MPL: {summary['avg_mpl']:.2f} units/hr")
        print(f"Avg error rate: {summary['avg_error_rate']:.3f}")
        print(f"Avg steps: {summary['avg_steps']:.1f}")
        print(f"\nStatus: {summary['status']}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test for LSD Vector Scene Environment"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    try:
        results = run_smoke_test(
            num_episodes=args.num_episodes,
            config_path=args.config_path,
            verbose=not args.quiet,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            if not args.quiet:
                print(f"\nResults saved to: {args.output}")

        return 0 if results["status"] == "PASSED" else 1

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
