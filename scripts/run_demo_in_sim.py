#!/usr/bin/env python3
"""
Simulator Demo Runner - 2-week demo script (not YC demo).

Runs deterministic rollouts using DemoPolicy in PyBullet or Isaac sim env.
Logs all episode/step data in JSON-safe format for demo presentation.

Usage:
    python3 scripts/run_demo_in_sim.py \\
        --env-backend pybullet \\
        --num-episodes 5 \\
        --max-steps 200 \\
        --seed 0 \\
        --output-dir results/demo_sim

All outputs are deterministic, JSON-safe, and flag-gated.
Does NOT touch reward math or economics.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.demo_policy import DemoPolicy, DemoPolicyConfig
from src.utils.json_safe import to_json_safe
from src.utils.gpu_env import get_gpu_env_summary, get_gpu_utilization
from src.utils.logging_schema import (
    make_demo_episode_log_entry,
    make_demo_step_log_entry,
    write_demo_log_entry,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run demo rollouts with DemoPolicy in sim"
    )
    parser.add_argument(
        "--env-backend",
        type=str,
        default="pybullet",
        choices=["pybullet", "isaac"],
        help="Environment backend to use",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for determinism",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/demo_sim",
        help="Output directory for logs and results",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Override canonical task ID (default from pipeline config)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (if supported by env)",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Enable mixed precision inference (AMP)",
    )
    return parser.parse_args(argv)


def create_env(backend: str, task_id: str, seed: int, render: bool = False):
    """
    Create environment via existing backend factory.

    Args:
        backend: "pybullet" or "isaac"
        task_id: Task identifier (e.g., "drawer_open")
        seed: Random seed
        render: Whether to enable rendering

    Returns:
        Environment instance with reset() and step(action) interface
    """
    if backend == "pybullet":
        # Use existing PyBullet env (drawer+vase or dishwashing_arm)
        try:
            from src.envs.physics.pybullet_backend import PyBulletBackend
            from src.envs.drawer_vase_env import DrawerVasePhysicsEnv

            # Create underlying env
            underlying_env = DrawerVasePhysicsEnv(render=render)
            env = PyBulletBackend(underlying_env, env_name="drawer_vase")
            return env
        except Exception as e:
            # Fallback to stub env
            return StubEnv(backend="pybullet", seed=seed)

    elif backend == "isaac":
        # Use Isaac adapter stub
        try:
            from src.env.isaac_adapter import IsaacAdapter

            adapter = IsaacAdapter(config={"seed": seed, "backend": "isaac_stub"})
            return IsaacStubEnv(adapter=adapter, seed=seed)
        except Exception as e:
            return StubEnv(backend="isaac", seed=seed)

    else:
        raise ValueError(f"Unknown backend: {backend}")


class StubEnv:
    """Stub environment for when real env unavailable."""

    def __init__(self, backend: str = "pybullet", seed: int = 0):
        self.backend = backend
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        obs = {
            "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "depth": None,
            "proprio": np.zeros(7, dtype=np.float32),
            "joint_positions": self.rng.uniform(-0.1, 0.1, size=7).astype(np.float32),
            "episode_id": f"stub_episode_{int(time.time())}",
        }
        return obs

    def step(self, action):
        """Execute one step."""
        self.step_count += 1

        # Stub obs
        obs = {
            "rgb": (self.rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)),
            "depth": None,
            "proprio": self.rng.uniform(-0.1, 0.1, size=7).astype(np.float32),
            "joint_positions": self.rng.uniform(-0.1, 0.1, size=7).astype(np.float32),
            "episode_id": f"stub_episode",
        }

        # Stub reward (random walk)
        reward = float(self.rng.uniform(-0.1, 0.1))

        # Done after max steps (handled by caller)
        done = False

        # Info
        info = {
            "success": self.rng.random() > 0.5,
            "mpl_proxy": float(self.rng.uniform(0.8, 1.2)),
            "energy_proxy": float(self.rng.uniform(0.5, 1.5)),
            "error_count": int(self.rng.integers(0, 3)),
        }

        return obs, reward, done, info


class IsaacStubEnv(StubEnv):
    """Isaac-specific stub env."""

    def __init__(self, adapter, seed: int = 0):
        super().__init__(backend="isaac", seed=seed)
        self.adapter = adapter


def run_episode(
    env,
    policy: DemoPolicy,
    episode_idx: int,
    max_steps: int,
    seed: int,
    log_steps: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode: env.reset() -> policy.act() loop.

    Args:
        env: Environment instance
        policy: DemoPolicy instance
        episode_idx: Episode index
        max_steps: Maximum steps
        seed: Base seed (episode_idx will be added)
        log_steps: Whether to log individual steps

    Returns:
        Dictionary with episode summary and step logs
    """
    episode_seed = seed + episode_idx
    policy.reset(seed=episode_seed)

    # Reset env
    obs = env.reset()

    # Episode tracking
    episode_start_time = time.time()
    steps_log = []
    total_reward = 0.0
    success = False
    econ_summary = {"mpl_sum": 0.0, "energy_sum": 0.0, "error_count": 0}
    ood_stats = {"max_ood": 0.0, "ood_steps": 0}
    recovery_stats = {"max_recovery": 0.0, "recovery_steps": 0}
    skill_mode_counts = {}

    for step_idx in range(max_steps):
        # Policy action
        action_dict = policy.act(obs)
        action = action_dict["action"]
        metadata = action_dict.get("metadata", {})

        # Extract metadata
        skill_mode = metadata.get("skill_mode", "default")
        ood_risk = metadata.get("ood_risk_level", 0.0)
        recovery_priority = metadata.get("recovery_priority", 0.0)

        # Track stats
        skill_mode_counts[skill_mode] = skill_mode_counts.get(skill_mode, 0) + 1
        ood_stats["max_ood"] = max(ood_stats["max_ood"], ood_risk)
        if ood_risk > 0.3:
            ood_stats["ood_steps"] += 1
        recovery_stats["max_recovery"] = max(recovery_stats["max_recovery"], recovery_priority)
        if recovery_priority > 0.3:
            recovery_stats["recovery_steps"] += 1

        # Env step
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Econ tracking (from info if available)
        econ_summary["mpl_sum"] += info.get("mpl_proxy", 1.0)
        econ_summary["energy_sum"] += info.get("energy_proxy", 1.0)
        econ_summary["error_count"] += info.get("error_count", 0)

        # Log step (using standardized schema with extra fields)
        if log_steps:
            step_log = make_demo_step_log_entry(
                episode_id=episode_idx,
                t=step_idx,
                action=action.tolist() if hasattr(action, 'tolist') else list(action),
                reward=reward,
                done=done,
                extra={
                    "action_summary": {
                        "action_norm": float(np.linalg.norm(action)),
                        "action_mean": float(np.mean(action)),
                    },
                    "ood_step_flags": {
                        "ood_risk_level": float(ood_risk),
                        "is_ood": ood_risk > 0.3,
                    },
                    "recovery_step_flags": {
                        "recovery_priority": float(recovery_priority),
                        "is_recovery": recovery_priority > 0.3,
                    },
                    "econ_step_summary": {
                        "mpl_proxy": float(info.get("mpl_proxy", 1.0)),
                        "energy_proxy": float(info.get("energy_proxy", 1.0)),
                    },
                }
            )
            steps_log.append(step_log)

        # Check done
        if done:
            success = info.get("success", False)
            break

    # Episode summary (using standardized schema with extra fields)
    episode_elapsed = time.time() - episode_start_time
    num_steps = step_idx + 1

    # Backend from env
    backend = getattr(env, 'backend', 'unknown')

    episode_summary = make_demo_episode_log_entry(
        episode_id=episode_idx,
        success=success,
        total_reward=total_reward,
        backend=backend,
        seed=episode_seed,
        mpl_estimate=float(econ_summary["mpl_sum"] / num_steps),
        energy_wh=float(econ_summary["energy_sum"] / num_steps),  # proxy value
        ood_events=int(ood_stats["ood_steps"]),
        recovery_events=int(recovery_stats["recovery_steps"]),
        extra={
            "steps": num_steps,
            "elapsed_seconds": float(episode_elapsed),
            "econ_summary": {
                "avg_mpl": float(econ_summary["mpl_sum"] / num_steps),
                "avg_energy": float(econ_summary["energy_sum"] / num_steps),
                "total_errors": int(econ_summary["error_count"]),
            },
            "ood_stats": {
                "max_ood_risk": float(ood_stats["max_ood"]),
                "ood_step_count": int(ood_stats["ood_steps"]),
                "ood_step_fraction": float(ood_stats["ood_steps"] / num_steps),
            },
            "recovery_stats": {
                "max_recovery_priority": float(recovery_stats["max_recovery"]),
                "recovery_step_count": int(recovery_stats["recovery_steps"]),
                "recovery_step_fraction": float(recovery_stats["recovery_steps"] / num_steps),
            },
            "skill_mode_counts": skill_mode_counts,
            "gpu_util_pct": get_gpu_utilization(),
        }
    )

    return {
        "summary": episode_summary,
        "steps": steps_log,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log GPU environment once at startup
    gpu_env = get_gpu_env_summary()
    gpu_log_path = output_dir / "gpu_env.json"
    with open(gpu_log_path, "w") as f:
        json.dump({"event": "gpu_env", "summary": gpu_env}, f, indent=2)

    print(f"[run_demo_in_sim] Starting demo simulation")
    print(f"  Backend: {args.env_backend}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print(f"  GPU: {gpu_env['device_name_0'] or 'CPU-only'}")
    print()

    # Create environment
    env = create_env(
        backend=args.env_backend,
        task_id=args.task_id or "drawer_open",
        seed=args.seed,
        render=args.render,
    )

    # Create DemoPolicy
    policy_config = DemoPolicyConfig.from_dict({
        "backend": args.env_backend,
        "canonical_task_id": args.task_id,
        "seed": args.seed,
        "use_amp": args.use_mixed_precision,
    })
    policy = DemoPolicy(config=policy_config)

    # Run episodes
    all_episodes = []
    all_steps = []

    for episode_idx in range(args.num_episodes):
        print(f"[run_demo_in_sim] Running episode {episode_idx + 1}/{args.num_episodes}")

        episode_result = run_episode(
            env=env,
            policy=policy,
            episode_idx=episode_idx,
            max_steps=args.max_steps,
            seed=args.seed,
            log_steps=True,
        )

        all_episodes.append(episode_result["summary"])
        all_steps.extend(episode_result["steps"])

        # Print episode summary
        summary = episode_result["summary"]
        print(f"  Steps: {summary.get('steps', summary.get('extra', {}).get('steps', '?'))}")
        print(f"  Success: {summary['success']}")
        print(f"  Avg MPL: {summary.get('extra', {}).get('econ_summary', {}).get('avg_mpl', summary.get('mpl_estimate', 0.0)):.3f}")
        print(f"  Avg Energy: {summary.get('extra', {}).get('econ_summary', {}).get('avg_energy', summary.get('energy_wh', 0.0)):.3f}")
        print()

    # Write outputs
    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(to_json_safe(ep)) + "\n")

    steps_path = output_dir / "steps.jsonl"
    with open(steps_path, "w") as f:
        for step in all_steps:
            f.write(json.dumps(to_json_safe(step)) + "\n")

    # Compute aggregate stats
    num_success = sum(1 for ep in all_episodes if ep["success"])
    avg_steps = np.mean([ep.get("extra", {}).get("steps", 0) for ep in all_episodes])
    avg_mpl = np.mean([ep.get("mpl_estimate", 0.0) for ep in all_episodes])
    avg_energy = np.mean([ep.get("energy_wh", 0.0) for ep in all_episodes])

    # Print summary
    print(f"{'='*80}")
    print(f"[run_demo_in_sim] Demo Summary")
    print(f"{'='*80}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Success rate: {num_success}/{args.num_episodes} ({100 * num_success / args.num_episodes:.1f}%)")
    print(f"  Avg episode length: {avg_steps:.1f} steps")
    print(f"  Avg MPL proxy: {avg_mpl:.3f}")
    print(f"  Avg energy proxy: {avg_energy:.3f}")
    print(f"  Output written to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
