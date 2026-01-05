#!/usr/bin/env python3
"""
Demo: SceneTracks → Reconstruction → Environment Replay

This script demonstrates reconstructing a workcell environment from
SceneTracks_v1 format (video-to-scene) and replaying the trajectory.

Usage:
    python scripts/demo_tracks_to_replay.py
    python scripts/demo_tracks_to_replay.py --tracks data/scene_tracks.npz
    python scripts/demo_tracks_to_replay.py --generate-synthetic --num-tracks 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def generate_synthetic_tracks(
    num_tracks: int = 5,
    num_frames: int = 30,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic SceneTracks_v1 data for demo purposes.

    Args:
        num_tracks: Number of object tracks
        num_frames: Frames per track
        seed: Random seed

    Returns:
        SceneTracks_v1 format dictionary
    """
    rng = np.random.default_rng(seed)

    # Generate positions with small motion
    positions = np.zeros((num_tracks, num_frames, 3))
    for i in range(num_tracks):
        # Base position
        base = rng.uniform([0.2, 0.2, 0.05], [0.8, 0.8, 0.15])
        # Small trajectory motion
        for f in range(num_frames):
            noise = rng.normal(0, 0.005, size=3)
            positions[i, f] = base + noise * (f / num_frames)

    # Generate quaternion orientations (mostly upright)
    orientations = np.zeros((num_tracks, num_frames, 4))
    for i in range(num_tracks):
        base_quat = np.array([0, 0, 0, 1])  # Identity
        for f in range(num_frames):
            # Small rotation perturbation
            axis_noise = rng.normal(0, 0.02, size=3)
            orientations[i, f] = base_quat.copy()
            orientations[i, f, :3] += axis_noise
            # Renormalize
            orientations[i, f] /= np.linalg.norm(orientations[i, f])

    # Track IDs
    track_ids = np.arange(num_tracks)

    return {
        "positions": positions,
        "orientations": orientations,
        "track_ids": track_ids,
    }


def run_demo(
    tracks_path: Optional[str] = None,
    generate_synthetic: bool = False,
    num_tracks: int = 5,
    replay_steps: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the tracks-to-replay demo pipeline.

    Args:
        tracks_path: Path to SceneTracks_v1 .npz file
        generate_synthetic: Generate synthetic tracks if True
        num_tracks: Number of tracks for synthetic generation
        replay_steps: Number of environment steps for replay
        verbose: Print progress messages

    Returns:
        Result dictionary with reconstruction info and replay metrics
    """
    # Step 1: Load or generate SceneTracks
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 1: Load SceneTracks")
        print(f"{'='*60}")

    if tracks_path and Path(tracks_path).exists():
        tracks_data = dict(np.load(tracks_path))
        if verbose:
            print(f"Loaded from: {tracks_path}")
    else:
        if verbose:
            if tracks_path:
                print(f"File not found: {tracks_path}")
            print("Generating synthetic SceneTracks...")
        tracks_data = generate_synthetic_tracks(
            num_tracks=num_tracks,
            num_frames=30,
            seed=42,
        )
        if verbose:
            print("Synthetic tracks generated.")

    if verbose:
        print(f"\nSceneTracks summary:")
        print(f"  Positions shape: {tracks_data['positions'].shape}")
        print(f"  Orientations shape: {tracks_data['orientations'].shape}")
        print(f"  Track IDs: {tracks_data['track_ids']}")

    # Step 2: Reconstruct scene via adapter
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 2: Reconstruct Scene")
        print(f"{'='*60}")

    from src.envs.workcell_env.reconstruction.scene_tracks_adapter import (
        SceneTracksAdapter,
    )

    adapter = SceneTracksAdapter()
    result = adapter.reconstruct_from_tracks(tracks_data)

    if verbose:
        print(f"\nReconstruction result:")
        print(f"  Scene spec: {len(result.scene_spec.parts)} parts")
        print(f"  Confidence score: {result.confidence_score:.3f}")
        print(f"  Track mapping: {result.track_mapping}")

    # Step 3: Create environment from reconstructed scene
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 3: Create Environment")
        print(f"{'='*60}")

    from src.envs.workcell_env import WorkcellEnv
    from src.envs.workcell_env.config import WorkcellEnvConfig

    config = WorkcellEnvConfig(
        num_parts=len(result.scene_spec.parts),
        max_steps=100,
        tolerance_mm=2.0,
    )

    env = WorkcellEnv(config=config, seed=42)
    if verbose:
        print(f"Environment created with {config.num_parts} parts")

    # Step 4: Replay trajectory
    if verbose:
        print(f"\n{'='*60}")
        print(f"STEP 4: Replay Trajectory ({replay_steps} steps)")
        print(f"{'='*60}")

    obs = env.reset(seed=42)
    trajectory_log = []
    total_reward = 0.0

    # Extract first-frame positions for replay targets
    initial_positions = tracks_data["positions"][:, 0, :]  # Shape: (num_tracks, 3)

    for step in range(replay_steps):
        # Use track positions as action targets (cycling through)
        track_idx = step % len(initial_positions)
        target_pos = initial_positions[track_idx].tolist()

        action = {
            "target_position": target_pos,
            "gripper_action": step % 2,
        }

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        trajectory_log.append({
            "step": step,
            "target_position": target_pos,
            "reward": reward,
            "gripper_state": info.get("gripper_state", "unknown"),
        })

        if verbose and (step + 1) % 5 == 0:
            print(f"  Step {step+1}: target={target_pos}, reward={reward:.3f}")

        if terminated or truncated:
            if verbose:
                print(f"  Episode ended at step {step+1}")
            break

    if verbose:
        print(f"\nReplay complete:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps executed: {len(trajectory_log)}")

    # Step 5: Generate replay report
    if verbose:
        print(f"\n{'='*60}")
        print("STEP 5: Replay Report")
        print(f"{'='*60}")

    replay_report = {
        "reconstruction": {
            "num_parts": len(result.scene_spec.parts),
            "confidence_score": result.confidence_score,
            "track_mapping": result.track_mapping,
        },
        "replay": {
            "steps_executed": len(trajectory_log),
            "total_reward": total_reward,
            "trajectory": trajectory_log,
        },
        "source": {
            "tracks_shape": list(tracks_data["positions"].shape),
            "is_synthetic": generate_synthetic or not (tracks_path and Path(tracks_path).exists()),
        },
    }

    if verbose:
        print(f"Report summary:")
        print(f"  Parts reconstructed: {replay_report['reconstruction']['num_parts']}")
        print(f"  Confidence: {replay_report['reconstruction']['confidence_score']:.3f}")
        print(f"  Total replay reward: {replay_report['replay']['total_reward']:.2f}")

    if verbose:
        print(f"\n{'='*60}")
        print("DEMO COMPLETE")
        print(f"{'='*60}")

    return replay_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo: SceneTracks → Reconstruction → Env Replay"
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="Path to SceneTracks_v1 .npz file",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic tracks for demo",
    )
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=5,
        help="Number of tracks for synthetic generation",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=20,
        help="Number of environment steps for replay",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    try:
        report = run_demo(
            tracks_path=args.tracks,
            generate_synthetic=args.generate_synthetic or args.tracks is None,
            num_tracks=args.num_tracks,
            replay_steps=args.replay_steps,
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
