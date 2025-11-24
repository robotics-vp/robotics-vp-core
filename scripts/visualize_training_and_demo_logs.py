#!/usr/bin/env python3
"""
Visualization script for training and demo logs.
Generates matplotlib plots from JSONL training logs and demo simulation logs.

Usage:
    python3 scripts/visualize_training_and_demo_logs.py \\
        --training-glob "results/training_logs/*.jsonl" \\
        --demo-dir "results/demo_runs" \\
        --output-dir "results/plots" \\
        --run-filter "vision_backbone"
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize training and demo logs"
    )
    parser.add_argument(
        "--training-glob",
        type=str,
        default="results/training_logs/*.jsonl",
        help="Glob pattern for training log files",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="results/demo_runs",
        help="Directory containing demo run logs (episodes.jsonl, steps.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="Filter runs by name substring (optional)",
    )
    return parser.parse_args()


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    if not filepath.exists():
        return data

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def plot_training_run(
    run_name: str,
    logs: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Generate plots for a single training run.

    Args:
        run_name: Run identifier
        logs: List of log entries
        output_dir: Output directory for plots
    """
    if not logs:
        print(f"  [WARN] No logs for run: {run_name}")
        return

    # Create run output directory
    run_output_dir = output_dir / "training" / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    train_logs = [log for log in logs if log.get("phase") == "train"]
    val_logs = [log for log in logs if log.get("phase") == "val"]

    if not train_logs:
        print(f"  [WARN] No training logs for run: {run_name}")
        return

    # Extract arrays
    train_steps = np.array([log["step"] for log in train_logs])
    train_epochs = np.array([log["epoch"] for log in train_logs])
    train_losses = np.array([log["loss"] for log in train_logs])
    train_lrs = np.array([log["lr"] for log in train_logs])

    val_steps = np.array([log["step"] for log in val_logs]) if val_logs else None
    val_losses = np.array([log["loss"] for log in val_logs]) if val_logs else None

    # Check for GPU stats
    has_gpu_stats = any(log.get("gpu_mem_mb") is not None for log in train_logs)
    if has_gpu_stats:
        gpu_mem = np.array([log.get("gpu_mem_mb", 0) for log in train_logs])
        gpu_util = np.array([log.get("gpu_util_pct", 0) for log in train_logs])

    # Plot: Loss vs Step
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)
    if val_steps is not None:
        plt.plot(val_steps, val_losses, label="Val Loss", alpha=0.7, marker='o', linestyle='--')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Step: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(run_output_dir / "loss_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot: Loss vs Epoch
    plt.figure(figsize=(10, 6))
    # Group by epoch and take mean
    epoch_loss_map = defaultdict(list)
    for epoch, loss in zip(train_epochs, train_losses):
        epoch_loss_map[epoch].append(loss)
    epochs_unique = sorted(epoch_loss_map.keys())
    epoch_losses_mean = [np.mean(epoch_loss_map[e]) for e in epochs_unique]

    plt.plot(epochs_unique, epoch_losses_mean, label="Train Loss (mean)", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(run_output_dir / "loss_vs_epoch.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot: Learning Rate vs Step
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_lrs, label="Learning Rate", color='orange')
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Learning Rate vs Step: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(run_output_dir / "lr_vs_step.png", dpi=150, bbox_inches='tight')
    plt.close()

    # GPU plots (if available)
    if has_gpu_stats:
        # GPU Memory vs Step
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, gpu_mem, label="GPU Memory (MB)", color='green')
        plt.xlabel("Step")
        plt.ylabel("GPU Memory (MB)")
        plt.title(f"GPU Memory vs Step: {run_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(run_output_dir / "gpu_mem_vs_step.png", dpi=150, bbox_inches='tight')
        plt.close()

        # GPU Utilization vs Step
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, gpu_util, label="GPU Utilization (%)", color='red')
        plt.xlabel("Step")
        plt.ylabel("GPU Utilization (%)")
        plt.title(f"GPU Utilization vs Step: {run_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.savefig(run_output_dir / "gpu_util_vs_step.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  ✓ Plots saved to: {run_output_dir}")


def plot_demo_runs(
    demo_dir: Path,
    output_dir: Path,
) -> None:
    """
    Generate plots for demo simulation runs.

    Args:
        demo_dir: Directory containing demo logs
        output_dir: Output directory for plots
    """
    # Find all episodes.jsonl files in subdirectories
    episode_files = list(demo_dir.glob("**/episodes.jsonl"))

    if not episode_files:
        print(f"  [WARN] No demo episode logs found in: {demo_dir}")
        return

    # Create demo output directory
    demo_output_dir = output_dir / "demo"
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    # Load all episodes
    all_episodes = []
    for episode_file in episode_files:
        episodes = load_jsonl(episode_file)
        all_episodes.extend(episodes)

    if not all_episodes:
        print(f"  [WARN] No demo episodes loaded")
        return

    # Extract data
    returns = np.array([ep.get("total_reward", 0.0) for ep in all_episodes])
    lengths = np.array([ep.get("extra", {}).get("steps", 0) for ep in all_episodes])
    successes = np.array([ep.get("success", False) for ep in all_episodes])
    mpl_estimates = np.array([ep.get("mpl_estimate", 0.0) for ep in all_episodes if ep.get("mpl_estimate") is not None])

    # Plot: Histogram of Episode Returns
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Total Return")
    plt.ylabel("Count")
    plt.title(f"Episode Returns Distribution (n={len(returns)})")
    plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(demo_output_dir / "episode_returns_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot: Episode Length vs Return Scatter
    plt.figure(figsize=(10, 6))
    colors = ['green' if s else 'red' for s in successes]
    plt.scatter(lengths, returns, c=colors, alpha=0.6, edgecolors='black')
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Total Return")
    plt.title(f"Episode Length vs Return (green=success, red=failure)")
    plt.grid(True, alpha=0.3)
    plt.savefig(demo_output_dir / "episode_length_vs_return.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot: MPL Estimates (if available)
    if len(mpl_estimates) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(mpl_estimates, bins=20, alpha=0.7, edgecolor='black', color='purple')
        plt.xlabel("MPL Estimate")
        plt.ylabel("Count")
        plt.title(f"MPL Estimates Distribution (n={len(mpl_estimates)})")
        plt.axvline(np.mean(mpl_estimates), color='red', linestyle='--', label=f'Mean: {np.mean(mpl_estimates):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(demo_output_dir / "mpl_estimates_histogram.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot: Success Rate Bar Chart
    success_rate = np.mean(successes) * 100
    plt.figure(figsize=(8, 6))
    plt.bar(["Success", "Failure"], [np.sum(successes), len(successes) - np.sum(successes)], color=['green', 'red'], alpha=0.7)
    plt.ylabel("Count")
    plt.title(f"Success Rate: {success_rate:.1f}% ({np.sum(successes)}/{len(successes)})")
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(demo_output_dir / "success_rate.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Demo plots saved to: {demo_output_dir}")


def print_training_summary(run_name: str, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Print summary statistics for a training run.

    Args:
        run_name: Run identifier
        logs: List of log entries

    Returns:
        Dictionary with summary statistics
    """
    if not logs:
        return {}

    train_logs = [log for log in logs if log.get("phase") == "train"]
    if not train_logs:
        return {}

    losses = [log["loss"] for log in train_logs]
    epochs = [log["epoch"] for log in train_logs]
    steps = [log["step"] for log in train_logs]

    # Find best epoch (min validation loss if available, otherwise min train loss)
    val_logs = [log for log in logs if log.get("phase") == "val"]
    if val_logs:
        val_losses = [log["loss"] for log in val_logs]
        val_epochs = [log["epoch"] for log in val_logs]
        best_epoch_idx = np.argmin(val_losses)
        best_epoch = val_epochs[best_epoch_idx]
    else:
        best_epoch_idx = np.argmin(losses)
        best_epoch = epochs[best_epoch_idx]

    has_gpu_stats = any(log.get("gpu_mem_mb") is not None for log in train_logs)

    summary = {
        "run_name": run_name,
        "min_loss": float(np.min(losses)),
        "final_loss": float(losses[-1]),
        "best_epoch": int(best_epoch),
        "steps": int(steps[-1]),
        "has_gpu_stats": has_gpu_stats,
    }

    return summary


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("Training and Demo Log Visualization")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process training logs
    print("\n[1/2] Processing training logs...")
    training_files = glob.glob(args.training_glob)

    if not training_files:
        print(f"  [WARN] No training log files found matching: {args.training_glob}")
    else:
        print(f"  Found {len(training_files)} training log file(s)")

        training_summaries = []

        for train_file in training_files:
            train_path = Path(train_file)
            run_name = train_path.stem

            # Apply filter if specified
            if args.run_filter and args.run_filter not in run_name:
                continue

            print(f"\n  Processing run: {run_name}")

            # Load logs
            logs = load_jsonl(train_path)
            if not logs:
                print(f"    [WARN] No valid logs in file: {train_file}")
                continue

            # Generate plots
            plot_training_run(run_name, logs, output_dir)

            # Collect summary
            summary = print_training_summary(run_name, logs)
            if summary:
                training_summaries.append(summary)

        # Print summary table
        if training_summaries:
            print("\n" + "=" * 80)
            print("Training Run Summary")
            print("=" * 80)
            print(f"{'run_name':<40} {'min_loss':>10} {'final_loss':>12} {'best_epoch':>12} {'steps':>10} {'has_gpu_stats':>15}")
            print("-" * 80)
            for summary in training_summaries:
                print(
                    f"{summary['run_name']:<40} "
                    f"{summary['min_loss']:>10.4f} "
                    f"{summary['final_loss']:>12.4f} "
                    f"{summary['best_epoch']:>12} "
                    f"{summary['steps']:>10} "
                    f"{'Yes' if summary['has_gpu_stats'] else 'No':>15}"
                )
            print("=" * 80)

    # Process demo logs
    print("\n[2/2] Processing demo logs...")
    demo_dir = Path(args.demo_dir)

    if not demo_dir.exists():
        print(f"  [WARN] Demo directory does not exist: {demo_dir}")
    else:
        plot_demo_runs(demo_dir, output_dir)

    print("\n" + "=" * 80)
    print(f"✓ All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
