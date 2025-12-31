#!/usr/bin/env python3
"""
Master orchestration script for Stage 6 training.
Runs all component training scripts in sequence with appropriate flags.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def run_command(cmd_list):
    print(f"[RunStage6] Executing: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[RunStage6] Command failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run full Stage 6 training pipeline")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--use-mixed-precision", action="store_true", default=True, help="Enable AMP (default: True)")
    parser.add_argument("--no-amp", action="store_false", dest="use_mixed_precision", help="Disable AMP")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision backbone training")
    parser.add_argument("--skip-sima2", action="store_true", help="Skip SIMA-2 training")
    parser.add_argument("--skip-spatial", action="store_true", help="Skip Spatial RNN training")
    parser.add_argument("--skip-hydra", action="store_true", help="Skip Hydra policy training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    args = parser.parse_args()

    base_cmd = [sys.executable]
    common_flags = [f"--seed={args.seed}"]
    if args.use_mixed_precision:
        common_flags.append("--use-mixed-precision")

    # 1. Vision Backbone
    if not args.skip_vision:
        print("\n=== Training Vision Backbone ===")
        cmd = base_cmd + ["scripts/train_vision_backbone_real.py"] + common_flags + [
            f"--epochs={args.epochs}",
            "--batch-size=32"
        ]
        run_command(cmd)

    # 2. SIMA-2 Segmenter
    if not args.skip_sima2:
        print("\n=== Training SIMA-2 Segmenter ===")
        cmd = base_cmd + ["scripts/train_sima2_segmenter.py"] + common_flags + [
            f"--epochs={args.epochs}",
            "--batch-size=32"
        ]
        run_command(cmd)

    # 3. Spatial RNN
    if not args.skip_spatial:
        print("\n=== Training Spatial RNN ===")
        cmd = base_cmd + ["scripts/train_spatial_rnn.py"] + common_flags + [
            f"--epochs={args.epochs}",
            "--sequence_length=16"
        ]
        run_command(cmd)

    # 4. Hydra Policy
    if not args.skip_hydra:
        print("\n=== Training Hydra Policy ===")
        cmd = base_cmd + ["scripts/train_hydra_policy.py"] + common_flags + [
            "--max-steps=1000"
        ]
        run_command(cmd)

    # Write success marker
    output_dir = Path("results/stage6")
    output_dir.mkdir(parents=True, exist_ok=True)
    success_marker = output_dir / "success.json"
    with open(success_marker, "w") as f:
        json.dump({"status": "success", "seed": args.seed, "amp": args.use_mixed_precision}, f)

    print(f"\n[RunStage6] All components trained successfully. Marker written to {success_marker}")

if __name__ == "__main__":
    main()
