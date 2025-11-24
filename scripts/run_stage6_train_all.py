#!/usr/bin/env python3
"""
Stage 6: First Real Model Training Pipeline - Golden Path Runner.

Orchestrates the complete Stage 6 training sequence:
1. Phase I data build (SIMA-2 stress, ROS→Stage2, Isaac adapter, manifest)
2. Train vision backbone (SimCLR contrastive + optional reconstruction)
3. Train spatial RNN (ConvGRU with forward dynamics)
4. Train SIMA-2 neural segmenter (U-Net with focal+dice loss)
5. Train Hydra policy (multi-objective SAC with condition routing)
6. Write Stage 6 success marker

All outputs are deterministic, flag-gated, and JSON-safe.
DO NOT touch economics code.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config.pipeline import (
    get_canonical_task,
    get_phase1_config,
    get_training_config,
    is_neural_mode_enabled,
    get_determinism_config,
)
from src.utils.json_safe import to_json_safe


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: First Real Model Training Pipeline")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    parser.add_argument(
        "--skip-data-build", action="store_true", help="Skip Phase I data build"
    )
    parser.add_argument(
        "--skip-vision", action="store_true", help="Skip vision backbone training"
    )
    parser.add_argument(
        "--skip-spatial-rnn", action="store_true", help="Skip spatial RNN training"
    )
    parser.add_argument(
        "--skip-segmenter", action="store_true", help="Skip SIMA-2 segmenter training"
    )
    parser.add_argument(
        "--skip-hydra", action="store_true", help="Skip Hydra policy training"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    return parser.parse_args(argv)


def run_command(
    cmd: List[str],
    description: str,
    dry_run: bool = False,
    allow_failure: bool = False,
) -> Dict[str, Any]:
    """
    Run a subprocess command and return structured result.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description
        dry_run: If True, print command without executing
        allow_failure: If True, don't raise on non-zero exit

    Returns:
        Dictionary with status, stdout, stderr, returncode
    """
    print(f"\n{'='*80}")
    print(f"[run_stage6_train_all] Step: {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if dry_run:
        return {
            "status": "dry_run",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600
        )
        elapsed = time.time() - start_time

        if result.returncode != 0 and not allow_failure:
            print(f"  ❌ ERROR: {description} failed with code {result.returncode}")
            print(f"  STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )

        status_icon = "✓" if result.returncode == 0 else "⚠"
        print(f"  {status_icon} Completed in {elapsed:.2f}s")

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_seconds": elapsed,
        }
    except subprocess.TimeoutExpired as e:
        if allow_failure:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_seconds": 600,
            }
        raise
    except Exception as e:
        if allow_failure:
            return {
                "status": "error",
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_seconds": 0,
            }
        raise


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Load configuration
    canonical_task = get_canonical_task()
    phase1_config = get_phase1_config()
    determinism_config = get_determinism_config()

    # Determine output directories
    if args.output_dir:
        base_output = Path(args.output_dir)
    else:
        base_output = ROOT / "results"

    stage6_dir = base_output / "stage6"
    checkpoints_dir = ROOT / "checkpoints"

    # Pipeline results
    results: Dict[str, Any] = {
        "canonical_task": canonical_task,
        "seed": args.seed,
        "stage": "stage6",
        "start_time": time.time(),
        "steps": {},
        "checkpoints": {},
    }

    print(f"\n{'#'*80}")
    print(f"# Stage 6: First Real Model Training Pipeline")
    print(f"# Canonical Task: {canonical_task}")
    print(f"# Seed: {args.seed}")
    print(f"# Output Directory: {stage6_dir}")
    print(f"{'#'*80}\n")

    # Step 1: Phase I Data Build
    if not args.skip_data_build:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_phase1_data_build.py"),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(base_output),
        ]
        result = run_command(
            cmd,
            "Phase I Data Build (SIMA-2 stress, ROS→Stage2, Isaac adapter, manifest)",
            dry_run=args.dry_run,
        )
        results["steps"]["phase1_data_build"] = result
    else:
        print("[run_stage6_train_all] Skipping Phase I data build")
        results["steps"]["phase1_data_build"] = {"status": "skipped"}

    # Step 2: Train Vision Backbone
    if not args.skip_vision:
        # Check if neural mode enabled or use real script
        use_real = is_neural_mode_enabled("vision_backbone")
        script_name = "train_vision_backbone_real.py" if use_real else "train_vision_backbone.py"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / script_name),
            "--seed",
            str(args.seed),
            "--checkpoint-dir",
            str(checkpoints_dir),
            "--max-samples",
            "100",
            "--epochs" if use_real else "--max-steps",
            "10" if use_real else "50",
        ]

        if use_real:
            cmd.extend(["--force-neural", "--use-reconstruction"])

        result = run_command(
            cmd,
            f"Train Vision Backbone ({'Real SimCLR' if use_real else 'Stub'})",
            dry_run=args.dry_run,
        )
        results["steps"]["train_vision_backbone"] = result
        results["checkpoints"]["vision_backbone"] = str(
            checkpoints_dir / ("vision_backbone.pt" if use_real else "vision_backbone_phase1.pt")
        )
    else:
        print("[run_stage6_train_all] Skipping vision backbone training")
        results["steps"]["train_vision_backbone"] = {"status": "skipped"}

    # Step 3: Train Spatial RNN
    if not args.skip_spatial_rnn:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_spatial_rnn.py"),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(checkpoints_dir / "spatial_rnn"),
            "--epochs",
            "5",
            "--hidden_dim",
            "64",
            "--mode",
            "convgru",
        ]
        result = run_command(
            cmd,
            "Train Spatial RNN (ConvGRU with forward dynamics)",
            dry_run=args.dry_run,
        )
        results["steps"]["train_spatial_rnn"] = result
        results["checkpoints"]["spatial_rnn"] = str(
            checkpoints_dir / "spatial_rnn" / "checkpoint_epoch_5.pt"
        )
    else:
        print("[run_stage6_train_all] Skipping spatial RNN training")
        results["steps"]["train_spatial_rnn"] = {"status": "skipped"}

    # Step 4: Train SIMA-2 Neural Segmenter
    if not args.skip_segmenter:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_sima2_segmenter.py"),
            "--seed",
            str(args.seed),
            "--output_dir",
            str(checkpoints_dir / "sima2_segmenter"),
            "--epochs",
            "10",
            "--batch_size",
            "8",
            "--use_dice",
            "--f1_threshold",
            "0.85",
        ]
        result = run_command(
            cmd,
            "Train SIMA-2 Neural Segmenter (U-Net with focal+dice loss)",
            dry_run=args.dry_run,
        )
        results["steps"]["train_sima2_segmenter"] = result
        results["checkpoints"]["sima2_segmenter"] = str(
            checkpoints_dir / "sima2_segmenter" / "checkpoint_epoch_10.pt"
        )
    else:
        print("[run_stage6_train_all] Skipping SIMA-2 segmenter training")
        results["steps"]["train_sima2_segmenter"] = {"status": "skipped"}

    # Step 5: Train Hydra Policy
    if not args.skip_hydra:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_hydra_policy.py"),
            "--seed",
            str(args.seed),
            "--checkpoint-dir",
            str(checkpoints_dir),
            "--max-samples",
            "100",
            "--max-steps",
            "100",
        ]
        result = run_command(
            cmd,
            "Train Hydra Policy (Multi-objective with condition routing)",
            dry_run=args.dry_run,
        )
        results["steps"]["train_hydra_policy"] = result
        results["checkpoints"]["hydra_policy"] = str(
            checkpoints_dir / "hydra_policy_phase1.pt"
        )
    else:
        print("[run_stage6_train_all] Skipping Hydra policy training")
        results["steps"]["train_hydra_policy"] = {"status": "skipped"}

    # Compute final statistics
    results["end_time"] = time.time()
    results["total_elapsed_seconds"] = results["end_time"] - results["start_time"]

    # Count successes
    num_steps = len([s for s in results["steps"].values() if s["status"] != "skipped"])
    num_successes = len(
        [s for s in results["steps"].values() if s["status"] == "success"]
    )
    num_failed = len([s for s in results["steps"].values() if s["status"] == "failed"])

    results["summary"] = {
        "total_steps": num_steps,
        "successes": num_successes,
        "failures": num_failed,
        "all_passed": num_failed == 0,
    }

    # Write success marker if all passed
    stage6_dir.mkdir(parents=True, exist_ok=True)
    success_marker = stage6_dir / "success.json"

    if results["summary"]["all_passed"] and not args.dry_run:
        success_data = {
            "stage": "stage6",
            "canonical_task": canonical_task,
            "seed": args.seed,
            "timestamp": time.time(),
            "checkpoints": results["checkpoints"],
            "summary": results["summary"],
        }
        with open(success_marker, "w") as f:
            json.dump(to_json_safe(success_data), f, indent=2, sort_keys=True)
        print(f"\n✓ Stage 6 success marker written to {success_marker}")
    elif args.dry_run:
        print(f"\n[DRY RUN] Would write success marker to {success_marker}")

    # Write detailed results
    results_path = stage6_dir / "stage6_training_results.json"
    if not args.dry_run:
        with open(results_path, "w") as f:
            json.dump(to_json_safe(results), f, indent=2, sort_keys=True)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Stage 6 Training Pipeline Complete!")
    print(f"  Total Steps: {num_steps}")
    print(f"  Successes: {num_successes}")
    print(f"  Failures: {num_failed}")
    print(f"  Total Time: {results['total_elapsed_seconds']:.2f}s")
    print(f"  Results: {results_path}")
    if results["summary"]["all_passed"]:
        print(f"  ✓ Success Marker: {success_marker}")
    else:
        print(f"  ⚠ Some steps failed - no success marker written")
    print(f"{'='*80}\n")

    # Print checkpoints
    if results["checkpoints"]:
        print("Checkpoints created:")
        for name, path in results["checkpoints"].items():
            exists = "✓" if Path(path).exists() else "✗"
            print(f"  {exists} {name}: {path}")

    if not results["summary"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
