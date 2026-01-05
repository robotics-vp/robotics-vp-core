#!/usr/bin/env python3
"""
Demo: Prompt → Config → Environment → Episodes → Datapack

This script demonstrates the full workcell pipeline from natural language
task specification to datapack export.

Usage:
    python scripts/demo_prompt_to_datapack.py
    python scripts/demo_prompt_to_datapack.py --prompt "Sort 10 widgets by color"
    python scripts/demo_prompt_to_datapack.py --episodes 20 --output data/my_datapack.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_demo(
    prompt: str,
    episodes: int = 10,
    output_path: str | None = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the prompt-to-datapack demo pipeline.

    Args:
        prompt: Natural language task description
        episodes: Number of episodes to collect
        output_path: Path to write datapack JSON (optional)
        verbose: Print progress messages

    Returns:
        Datapack dictionary with episodes, metadata, and analytics
    """
    # Step 1: Compile prompt to config/scene/task
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 1: Compile Prompt → Config")
        print(f"{'='*60}")
        print(f"Prompt: \"{prompt}\"")

    from src.envs.workcell_env.compiler import WorkcellTaskCompiler

    compiler = WorkcellTaskCompiler(default_seed=42)
    result = compiler.compile_from_prompt(prompt)

    if verbose:
        print(f"\nInferred task type: {result.inferred_task_type}")
        print(f"Config: num_parts={result.config.num_parts}, tolerance={result.config.tolerance_mm}mm")
        print(f"Scene: {len(result.scene_spec.parts)} parts, {len(result.scene_spec.fixtures)} fixtures")
        print(f"Task graph: {len(result.task_graph.nodes)} nodes")

    # Step 2: Create environment from config
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 2: Create Environment")
        print(f"{'='*60}")

    from src.envs.workcell_env import WorkcellEnv

    env = WorkcellEnv(config=result.config, seed=42)
    if verbose:
        print(f"Environment created: {env.__class__.__name__}")
        print(f"Max steps: {result.config.max_steps}")

    # Step 3: Collect episodes with random actions
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 3: Collect {episodes} Episodes")
        print(f"{'='*60}")

    episode_logs: List[Dict[str, Any]] = []
    total_rewards = []
    successes = 0

    for ep in range(episodes):
        obs = env.reset(seed=42 + ep)
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < result.config.max_steps:
            # Random action for demo
            action = {
                "target_position": [
                    0.3 + 0.4 * ((steps * 7 + ep) % 100) / 100,
                    0.3 + 0.4 * ((steps * 13 + ep) % 100) / 100,
                    0.05 + 0.1 * ((steps * 17 + ep) % 100) / 100,
                ],
                "gripper_action": steps % 2,
            }
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        # Get episode log
        log = env.get_episode_log()
        episode_data = {
            "episode_id": log.metadata.episode_id,
            "task_type": result.inferred_task_type,
            "success": info.get("success", False),
            "total_reward": episode_reward,
            "steps": steps,
            "items_completed": info.get("items_completed", 0),
            "items_total": info.get("items_total", result.config.num_parts),
        }
        episode_logs.append(episode_data)
        total_rewards.append(episode_reward)
        if episode_data["success"]:
            successes += 1

        if verbose and (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{episodes}: reward={episode_reward:.2f}, steps={steps}")

    if verbose:
        print(f"\nCollection complete:")
        print(f"  Success rate: {100*successes/episodes:.1f}%")
        print(f"  Avg reward: {sum(total_rewards)/len(total_rewards):.2f}")

    # Step 4: Compute analytics
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 4: Compute Analytics")
        print(f"{'='*60}")

    from src.analytics.workcell_analytics import compute_episode_metrics, compute_suite_report

    # Convert episode logs to metrics objects
    metrics_list = []
    for ep_data in episode_logs:
        metrics = compute_episode_metrics(
            episode_id=ep_data["episode_id"],
            task_type=ep_data["task_type"],
            episode_data=ep_data,
        )
        metrics_list.append(metrics)

    report = compute_suite_report(metrics_list)

    if verbose:
        print(f"Suite report:")
        print(f"  Total episodes: {report.num_episodes}")
        print(f"  Success rate: {report.success_rate*100:.1f}%")
        print(f"  Avg reward: {report.mean_reward:.2f}")

    # Step 5: Package datapack
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 5: Package Datapack")
        print(f"{'='*60}")

    datapack = {
        "metadata": {
            "source_prompt": prompt,
            "inferred_task_type": result.inferred_task_type,
            "num_episodes": episodes,
            "config": {
                "num_parts": result.config.num_parts,
                "num_bins": result.config.num_bins,
                "tolerance_mm": result.config.tolerance_mm,
                "max_steps": result.config.max_steps,
            },
        },
        "episodes": episode_logs,
        "analytics": {
            "num_episodes": report.num_episodes,
            "success_rate": report.success_rate,
            "mean_reward": report.mean_reward,
            "mean_steps": report.mean_steps,
            "completion_rate": report.completion_rate,
            "error_rate": report.error_rate,
            "mean_quality_score": report.mean_quality_score,
            "by_task_type": report.by_task_type,
        },
        "task_graph": {
            "num_nodes": len(result.task_graph.nodes),
            "node_ids": [n.step_id for n in result.task_graph.nodes],
        },
    }

    if verbose:
        print(f"Datapack assembled:")
        print(f"  Metadata: {len(datapack['metadata'])} fields")
        print(f"  Episodes: {len(datapack['episodes'])} entries")
        print(f"  Analytics: {len(datapack['analytics'])} metrics")

    # Step 6: Export if path provided
    if output_path:
        if verbose:
            print(f"\n{'='*60}")
            print(f"STEP 6: Export to {output_path}")
            print(f"{'='*60}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(datapack, f, indent=2, default=str)

        if verbose:
            print(f"Datapack written: {output_file.stat().st_size} bytes")

    if verbose:
        print(f"\n{'='*60}")
        print("DEMO COMPLETE")
        print(f"{'='*60}")

    return datapack


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo: Prompt → Config → Env → Episodes → Datapack"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pack 8 bolts into a tray with 2mm tolerance",
        help="Natural language task description",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for datapack JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        datapack = run_demo(
            prompt=args.prompt,
            episodes=args.episodes,
            output_path=args.output,
            verbose=not args.quiet,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
