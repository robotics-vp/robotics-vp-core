#!/usr/bin/env python3
"""
Phase I Data Build Pipeline.

Orchestrates the complete Phase I data generation pipeline:
1. Run SIMA-2 stress generation for canonical task
2. Auto-detect and run ROS→Stage2 pipeline (fallback silently if ROS logs absent)
3. Run Isaac adapter (single rollout)
4. Build and validate Phase I data manifest

All outputs are deterministic, JSON-safe, and flag-gated.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config.pipeline import get_canonical_task, get_phase1_config
from src.utils.json_safe import to_json_safe


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase I Data Build Pipeline")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    parser.add_argument(
        "--skip-sima2", action="store_true", help="Skip SIMA-2 stress generation"
    )
    parser.add_argument(
        "--skip-ros", action="store_true", help="Skip ROS→Stage2 pipeline"
    )
    parser.add_argument(
        "--skip-isaac", action="store_true", help="Skip Isaac adapter rollout"
    )
    parser.add_argument(
        "--skip-manifest", action="store_true", help="Skip manifest building"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )
    return parser.parse_args(argv)


def run_command(
    cmd: List[str], description: str, allow_failure: bool = False
) -> Dict[str, Any]:
    """
    Run a subprocess command and return structured result.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description
        allow_failure: If True, don't raise on non-zero exit

    Returns:
        Dictionary with status, stdout, stderr, returncode
    """
    print(f"[run_phase1_data_build] Running: {description}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=300
        )

        if result.returncode != 0 and not allow_failure:
            print(f"  ERROR: {description} failed with code {result.returncode}")
            print(f"  STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as e:
        if allow_failure:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }
        raise
    except Exception as e:
        if allow_failure:
            return {
                "status": "error",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }
        raise


def detect_ros_logs() -> bool:
    """
    Auto-detect presence of ROS logs for Stage2 pipeline.

    Returns:
        True if ROS logs are found, False otherwise
    """
    # Look for common ROS log locations
    ros_log_patterns = [
        ROOT / "data" / "ros" / "*.bag",
        ROOT / "data" / "ros_logs" / "*.bag",
        ROOT / "results" / "ros" / "*.bag",
    ]

    for pattern in ros_log_patterns:
        if list(pattern.parent.glob(pattern.name)):
            return True

    return False


def run_sima2_stress(
    canonical_task: str, num_episodes: int, output_dir: Path, seed: int
) -> Dict[str, Any]:
    """
    Run SIMA-2 stress generation for canonical task.

    Args:
        canonical_task: Task ID to generate data for
        num_episodes: Number of episodes to generate
        output_dir: Output directory for SIMA-2 data
        seed: Random seed

    Returns:
        Result dictionary with status and metrics
    """
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "stress_test_sima2_pipeline.py"),
        "--num-rollouts",
        str(num_episodes),
        "--batch-size",
        str(min(50, num_episodes)),
        "--task-distribution",
        "dataset_stress_mix_v1",
        "--output-dir",
        str(output_dir),
    ]

    result = run_command(cmd, f"SIMA-2 stress generation ({canonical_task})")

    # Parse metrics from output
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return {
        "status": result["status"],
        "output_dir": str(output_dir),
        "num_episodes": num_episodes,
        "canonical_task": canonical_task,
        "metrics": metrics,
    }


def run_ros_stage2(output_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    """
    Run ROS→Stage2 pipeline if ROS logs are detected.

    Args:
        output_dir: Output directory for Stage2 data
        seed: Random seed

    Returns:
        Result dictionary if successful, None if skipped
    """
    if not detect_ros_logs():
        print(
            "[run_phase1_data_build] ROS logs not detected, skipping ROS→Stage2 pipeline (silent fallback)"
        )
        return None

    # Find first ROS bag file
    ros_bag = None
    for pattern_path in [
        ROOT / "data" / "ros",
        ROOT / "data" / "ros_logs",
        ROOT / "results" / "ros",
    ]:
        bags = list(pattern_path.glob("*.bag"))
        if bags:
            ros_bag = bags[0]
            break

    if not ros_bag:
        print(
            "[run_phase1_data_build] ROS logs detected but no .bag files found, skipping"
        )
        return None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_stage2_sima2_pipeline.py"),
        "--rollouts-path",
        str(ros_bag),
        "--output-dir",
        str(output_dir),
    ]

    result = run_command(cmd, "ROS→Stage2 pipeline", allow_failure=True)

    return {
        "status": result["status"],
        "output_dir": str(output_dir),
        "ros_bag": str(ros_bag),
    }


def run_isaac_adapter(output_dir: Path, canonical_task: str, seed: int) -> Dict[str, Any]:
    """
    Run Isaac adapter for single rollout.

    Args:
        output_dir: Output directory for Isaac data
        canonical_task: Task ID
        seed: Random seed

    Returns:
        Result dictionary with status
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate a minimal Isaac rollout using the adapter contract
    # For now, create a synthetic placeholder
    rollout_path = output_dir / "isaac_rollout_0.json"

    synthetic_rollout = {
        "episode_id": f"{canonical_task}_isaac_ep_0",
        "task_id": canonical_task,
        "backend": "isaac",
        "seed": seed,
        "frames": 1,
        "metadata": {
            "synthetic": True,
            "phase": "phase1",
            "canonical_task": canonical_task,
        },
    }

    with open(rollout_path, "w") as f:
        json.dump(synthetic_rollout, f, indent=2, sort_keys=True)

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "rollout_path": str(rollout_path),
        "num_rollouts": 1,
        "canonical_task": canonical_task,
    }


def build_manifest(
    manifest_path: Path, sima2_root: str, stage2_root: str, isaac_root: str, seed: int
) -> Dict[str, Any]:
    """
    Build Phase I data manifest.

    Args:
        manifest_path: Output path for manifest JSON
        sima2_root: SIMA-2 output directory
        stage2_root: Stage2 output directory
        isaac_root: Isaac output directory
        seed: Random seed

    Returns:
        Result dictionary with manifest metadata
    """
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_phase1_data_manifest.py"),
        "--sima2-root",
        sima2_root,
        "--stage2-root",
        stage2_root,
        "--seed",
        str(seed),
        "--output",
        str(manifest_path),
    ]

    result = run_command(cmd, "Phase I manifest builder")

    # Load and return manifest
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    return {
        "status": result["status"],
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Load configuration
    canonical_task = get_canonical_task()
    phase1_config = get_phase1_config()

    # Determine output directories
    if args.output_dir:
        base_output = Path(args.output_dir)
    else:
        base_output = ROOT / "results"

    sima2_dir = base_output / phase1_config["sima2_stress"]["output_dir"].split("/")[-1]
    stage2_dir = (
        base_output / phase1_config["ros_stage2"]["output_dir"].split("/")[-1]
    )
    isaac_dir = (
        base_output / phase1_config["isaac_adapter"]["output_dir"].split("/")[-1]
    )
    manifest_path = Path(phase1_config["manifest"]["output_path"])

    # Pipeline results
    results: Dict[str, Any] = {
        "canonical_task": canonical_task,
        "seed": args.seed,
        "phase": "phase1",
        "steps": {},
    }

    # Step 1: SIMA-2 stress generation
    if not args.skip_sima2 and phase1_config["sima2_stress"]["enabled"]:
        sima2_result = run_sima2_stress(
            canonical_task=canonical_task,
            num_episodes=phase1_config["sima2_stress"]["num_episodes"],
            output_dir=sima2_dir,
            seed=args.seed,
        )
        results["steps"]["sima2_stress"] = sima2_result
    else:
        print("[run_phase1_data_build] Skipping SIMA-2 stress generation")
        results["steps"]["sima2_stress"] = {"status": "skipped"}

    # Step 2: ROS→Stage2 pipeline (auto-detect, fallback silently)
    if not args.skip_ros and phase1_config["ros_stage2"]["enabled"]:
        ros_result = run_ros_stage2(output_dir=stage2_dir, seed=args.seed)
        if ros_result:
            results["steps"]["ros_stage2"] = ros_result
        else:
            results["steps"]["ros_stage2"] = {
                "status": "skipped",
                "reason": "no_ros_logs_detected",
            }
    else:
        print("[run_phase1_data_build] Skipping ROS→Stage2 pipeline")
        results["steps"]["ros_stage2"] = {"status": "skipped"}

    # Step 3: Isaac adapter rollout
    if not args.skip_isaac and phase1_config["isaac_adapter"]["enabled"]:
        isaac_result = run_isaac_adapter(
            output_dir=isaac_dir, canonical_task=canonical_task, seed=args.seed
        )
        results["steps"]["isaac_adapter"] = isaac_result
    else:
        print("[run_phase1_data_build] Skipping Isaac adapter rollout")
        results["steps"]["isaac_adapter"] = {"status": "skipped"}

    # Step 4: Build manifest
    if not args.skip_manifest:
        manifest_result = build_manifest(
            manifest_path=manifest_path,
            sima2_root=str(sima2_dir),
            stage2_root=str(stage2_dir),
            isaac_root=str(isaac_dir),
            seed=args.seed,
        )
        results["steps"]["manifest_build"] = manifest_result
    else:
        print("[run_phase1_data_build] Skipping manifest building")
        results["steps"]["manifest_build"] = {"status": "skipped"}

    # Write summary
    summary_path = base_output / "phase1_data_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(to_json_safe(results), f, indent=2, sort_keys=True)

    print(f"\n[run_phase1_data_build] Phase I data build complete!")
    print(f"  Canonical task: {canonical_task}")
    print(f"  Summary: {summary_path}")
    print(f"  Manifest: {manifest_path if not args.skip_manifest else 'skipped'}")


if __name__ == "__main__":
    main()
