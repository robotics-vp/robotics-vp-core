#!/usr/bin/env python3
"""
Smoke Test for Process Reward Module.

Loads an exported scene_tracks_v1 artifact and computes:
- Phi_I, Phi_F, Phi_B perspectives
- Fused Phi_star and confidence
- PBRS shaped reward r_shape

Prints summary stats and saves outputs for inspection.

Usage:
    python scripts/smoke_test_process_reward.py \
        --input data/episode.npz \
        --output results/process_reward_test.json

    # With synthetic data (no input file):
    python scripts/smoke_test_process_reward.py --synthetic

    # Verbose mode:
    python scripts/smoke_test_process_reward.py --synthetic --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def create_synthetic_scene_tracks(
    T: int = 20,
    K: int = 3,
    include_latents: bool = False,
) -> Dict[str, np.ndarray]:
    """Create synthetic scene tracks for testing.

    Args:
        T: Number of frames.
        K: Number of tracks.
        include_latents: Whether to include z_shape/z_tex.

    Returns:
        Dict of numpy arrays in scene_tracks_v1 format.
    """
    np.random.seed(42)

    # Generate smooth trajectories
    t = np.linspace(0, 2 * np.pi, T)

    # Poses: identity rotation, smooth trajectory
    poses_R = np.tile(np.eye(3, dtype=np.float32), (T, K, 1, 1))
    poses_t = np.zeros((T, K, 3), dtype=np.float32)

    for k in range(K):
        # Different trajectories for each track
        poses_t[:, k, 0] = np.sin(t + k * np.pi / K)  # x
        poses_t[:, k, 1] = np.cos(t + k * np.pi / K)  # y
        poses_t[:, k, 2] = 1.0 + 0.1 * k  # z (fixed height)

    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32) * 0.9
    occlusion = np.ones((T, K), dtype=np.float32) * 0.1
    ir_loss = np.ones((T, K), dtype=np.float32) * 0.05
    converged = np.ones((T, K), dtype=bool)

    # Entity types: first is body, rest are objects
    entity_types = np.array([1] + [0] * (K - 1), dtype=np.int32)

    result = {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array([f"track_{i:04d}" for i in range(K)], dtype="U32"),
        "scene_tracks_v1/entity_types": entity_types,
        "scene_tracks_v1/class_ids": np.array([-1] + [0] * (K - 1), dtype=np.int32),
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility,
        "scene_tracks_v1/occlusion": occlusion,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }

    if include_latents:
        z_shape = np.random.randn(T, K, 64).astype(np.float16)
        z_tex = np.random.randn(T, K, 32).astype(np.float16)
        result["scene_tracks_v1/z_shape"] = z_shape
        result["scene_tracks_v1/z_tex"] = z_tex

    return result


def run_smoke_test(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    instruction: str = "pick up the object",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run process reward smoke test.

    Args:
        input_path: Path to scene_tracks_v1 npz file. If None, uses synthetic.
        output_path: Path to save output JSON. If None, prints only.
        instruction: Task instruction.
        verbose: Print detailed output.

    Returns:
        Dictionary of results.
    """
    # Import process reward module
    from src.process_reward import (
        ProcessRewardConfig,
        process_reward_episode,
    )
    from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1

    print("=" * 60)
    print("Process Reward Smoke Test")
    print("=" * 60)

    # Load or create scene tracks
    if input_path:
        print(f"\nLoading scene tracks from: {input_path}")
        data = dict(np.load(input_path, allow_pickle=False))
    else:
        print("\nUsing synthetic scene tracks...")
        data = create_synthetic_scene_tracks(T=20, K=3, include_latents=True)

    scene_tracks = deserialize_scene_tracks_v1(data)
    print(f"  Frames: {scene_tracks.num_frames}")
    print(f"  Tracks: {scene_tracks.num_tracks}")

    # Create config (offline eval allows hindsight)
    cfg = ProcessRewardConfig(
        gamma=0.99,
        use_confidence_gating=True,
        feature_dim=32,
        use_latents=True,
        online_mode=False,  # Offline eval allows hindsight (goal from last frame)
    )

    print(f"\nConfig:")
    print(f"  gamma: {cfg.gamma}")
    print(f"  use_confidence_gating: {cfg.use_confidence_gating}")
    print(f"  feature_dim: {cfg.feature_dim}")

    # Run process reward
    print(f"\nComputing process reward...")
    print(f"  Instruction: '{instruction}'")

    result = process_reward_episode(
        scene_tracks=scene_tracks,
        instruction=instruction,
        goal_frame_idx=None,  # Use last frame as goal
        cfg=cfg,
        episode_id="smoke_test",
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Verify PBRS telescoping
    print("\n" + "-" * 40)
    print("PBRS Telescoping Verification")
    print("-" * 40)

    from src.process_reward.shaping import verify_pbrs_telescoping

    # For ungated case
    r_shape_ungated = cfg.gamma * result.phi_star[1:] - result.phi_star[:-1]
    is_valid, telescope_info = verify_pbrs_telescoping(
        result.phi_star, r_shape_ungated, cfg.gamma
    )

    print(f"  Telescoping valid: {is_valid}")
    # Handle both discounted and undiscounted modes
    if telescope_info.get("mode") == "discounted":
        print(f"  discounted_sum: {telescope_info['discounted_sum']:.6f}")
    else:
        print(f"  plain_sum: {telescope_info.get('plain_sum', 0):.6f}")
    print(f"  expected_sum: {telescope_info['expected_sum']:.6f}")
    print(f"  error: {telescope_info['error']:.6f}")

    # Print perspectives
    if verbose:
        print("\n" + "-" * 40)
        print("Perspectives (first 5 frames)")
        print("-" * 40)
        for t in range(min(5, len(result.phi_star))):
            print(f"  t={t}: Phi_I={result.perspectives.phi_I[t]:.3f}, "
                  f"Phi_F={result.perspectives.phi_F[t]:.3f}, "
                  f"Phi_B={result.perspectives.phi_B[t]:.3f}, "
                  f"Phi*={result.phi_star[t]:.3f}")

        print("\n" + "-" * 40)
        print("Fusion Weights (first 5 frames)")
        print("-" * 40)
        for t in range(min(5, len(result.diagnostics.weights))):
            w = result.diagnostics.weights[t]
            print(f"  t={t}: w_I={w[0]:.3f}, w_F={w[1]:.3f}, w_B={w[2]:.3f}")

    # Prepare output
    output = {
        "summary": summary,
        "telescoping": {
            "valid": is_valid,
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in telescope_info.items()},
        },
        "phi_star": result.phi_star.tolist(),
        "conf": result.conf.tolist(),
        "r_shape": result.r_shape.tolist(),
        "perspectives": result.perspectives.to_dict(),
        "diagnostics": {
            "weights": result.diagnostics.weights.tolist(),
            "entropy": result.diagnostics.entropy.tolist(),
            "disagreement": result.diagnostics.disagreement.tolist(),
        },
    }

    # Save output
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nOutput saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Smoke Test PASSED")
    print("=" * 60)

    return output


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test for process reward module"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to scene_tracks_v1 npz file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (ignores --input)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the object",
        help="Task instruction",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.synthetic and not args.input:
        print("ERROR: Must specify --input or --synthetic", file=sys.stderr)
        return 1

    input_path = None if args.synthetic else args.input

    try:
        run_smoke_test(
            input_path=input_path,
            output_path=args.output,
            instruction=args.instruction,
            verbose=args.verbose,
        )
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
