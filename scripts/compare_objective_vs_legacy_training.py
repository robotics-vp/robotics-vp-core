#!/usr/bin/env python3
"""
Compare objective-conditioned vs legacy reward training runs.

Loads training history from results/, plots:
- MPL/Wh/error trajectories
- Whether objective-based reward pushes towards different Pareto points
- Reward decomposition analysis

No Phase B/RL math changes - analysis only.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_run_history(run_dir: str) -> dict:
    """Load training history and config from run directory."""
    config_path = os.path.join(run_dir, "config.json")
    history_path = os.path.join(run_dir, "training_history.json")
    summary_path = os.path.join(run_dir, "summary.json")

    result = {"run_dir": run_dir}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            result["config"] = json.load(f)
    else:
        result["config"] = {}

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            result["history"] = json.load(f)
    else:
        result["history"] = []

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            result["summary"] = json.load(f)
    else:
        result["summary"] = {}

    return result


def compute_trajectory_stats(history: list) -> dict:
    """Compute statistics for MPL/error/energy trajectories."""
    if not history:
        return {}

    mpls = [h.get("mpl_episode", 0) for h in history]
    errors = [h.get("error_rate", 0) for h in history]
    energies = [h.get("energy_Wh", 0) for h in history]

    return {
        "mpl": {
            "initial": mpls[0],
            "final": mpls[-1],
            "mean": np.mean(mpls),
            "std": np.std(mpls),
            "max": max(mpls),
            "min": min(mpls),
            "trend": mpls[-1] - mpls[0],
        },
        "error_rate": {
            "initial": errors[0],
            "final": errors[-1],
            "mean": np.mean(errors),
            "std": np.std(errors),
            "max": max(errors),
            "min": min(errors),
            "trend": errors[-1] - errors[0],
        },
        "energy_Wh": {
            "initial": energies[0],
            "final": energies[-1],
            "mean": np.mean(energies),
            "std": np.std(energies),
            "max": max(energies),
            "min": min(energies),
            "trend": energies[-1] - energies[0],
        },
    }


def compute_pareto_position(stats: dict) -> str:
    """
    Classify Pareto position based on final metrics.

    Returns: "high_mpl", "low_error", "low_energy", "balanced"
    """
    mpl_final = stats["mpl"]["final"]
    error_final = stats["error_rate"]["final"]
    energy_final = stats["energy_Wh"]["final"]

    # Simple classification based on relative priorities
    # Higher MPL = throughput focused
    # Lower error = safety focused
    # Lower energy = energy efficient

    if mpl_final > 50 and error_final < 0.05:
        return "high_mpl_low_error"
    elif mpl_final > 50 and energy_final < 10:
        return "high_mpl_low_energy"
    elif error_final < 0.03:
        return "low_error_focus"
    elif energy_final < 5:
        return "low_energy_focus"
    else:
        return "balanced"


def print_run_summary(run_data: dict, label: str):
    """Print summary for a single run."""
    config = run_data.get("config", {})
    summary = run_data.get("summary", {})
    history = run_data.get("history", [])

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    print(f"Config:")
    print(f"  Episodes: {config.get('episodes', 'N/A')}")
    print(f"  Objective reward: {config.get('use_objective_reward', False)}")
    print(f"  Objective preset: {config.get('objective_preset', 'N/A')}")
    print(f"  Objective vector: {config.get('objective_vector', 'N/A')}")

    if history:
        stats = compute_trajectory_stats(history)
        pareto_pos = compute_pareto_position(stats)

        print(f"\nTrajectory Statistics:")
        print(f"  MPL: {stats['mpl']['initial']:.2f} -> {stats['mpl']['final']:.2f} (trend: {stats['mpl']['trend']:+.2f})")
        print(f"  Error: {stats['error_rate']['initial']:.4f} -> {stats['error_rate']['final']:.4f} (trend: {stats['error_rate']['trend']:+.4f})")
        print(f"  Energy: {stats['energy_Wh']['initial']:.2f} -> {stats['energy_Wh']['final']:.2f} (trend: {stats['energy_Wh']['trend']:+.2f})")

        print(f"\nPareto Position: {pareto_pos}")

        # Show reward decomposition if available
        if history and "r_mpl_total" in history[-1]:
            print(f"\nFinal Episode Reward Decomposition:")
            last = history[-1]
            print(f"  r_mpl: {last.get('r_mpl_total', 0):.3f}")
            print(f"  r_error: {last.get('r_error_total', 0):.3f}")
            print(f"  r_energy: {last.get('r_energy_total', 0):.3f}")
            print(f"  r_safety: {last.get('r_safety_total', 0):.3f}")
            print(f"  r_novelty: {last.get('r_novelty_total', 0):.3f}")

    if summary:
        print(f"\nSummary Stats:")
        print(f"  Mean legacy reward: {summary.get('mean_legacy_reward', 0):.3f}")
        print(f"  Mean objective reward: {summary.get('mean_objective_reward', 0):.3f}")


def compare_runs(legacy_run: dict, objective_run: dict):
    """Compare legacy vs objective-based training runs."""
    print("\n" + "=" * 60)
    print("COMPARISON: Legacy vs Objective-Based Training")
    print("=" * 60)

    legacy_stats = compute_trajectory_stats(legacy_run.get("history", []))
    objective_stats = compute_trajectory_stats(objective_run.get("history", []))

    if not legacy_stats or not objective_stats:
        print("Cannot compare: missing history data")
        return

    # MPL comparison
    mpl_diff = objective_stats["mpl"]["final"] - legacy_stats["mpl"]["final"]
    print(f"\nMPL Final:")
    print(f"  Legacy: {legacy_stats['mpl']['final']:.2f}")
    print(f"  Objective: {objective_stats['mpl']['final']:.2f}")
    print(f"  Difference: {mpl_diff:+.2f}")

    # Error comparison
    error_diff = objective_stats["error_rate"]["final"] - legacy_stats["error_rate"]["final"]
    print(f"\nError Rate Final:")
    print(f"  Legacy: {legacy_stats['error_rate']['final']:.4f}")
    print(f"  Objective: {objective_stats['error_rate']['final']:.4f}")
    print(f"  Difference: {error_diff:+.4f}")

    # Energy comparison
    energy_diff = objective_stats["energy_Wh"]["final"] - legacy_stats["energy_Wh"]["final"]
    print(f"\nEnergy (Wh) Final:")
    print(f"  Legacy: {legacy_stats['energy_Wh']['final']:.2f}")
    print(f"  Objective: {objective_stats['energy_Wh']['final']:.2f}")
    print(f"  Difference: {energy_diff:+.2f}")

    # Pareto position comparison
    legacy_pareto = compute_pareto_position(legacy_stats)
    objective_pareto = compute_pareto_position(objective_stats)
    print(f"\nPareto Position:")
    print(f"  Legacy: {legacy_pareto}")
    print(f"  Objective: {objective_pareto}")

    if legacy_pareto != objective_pareto:
        print(f"  -> Objective-based reward pushes towards DIFFERENT Pareto point!")
    else:
        print(f"  -> Both converge to similar Pareto region")

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    improvements = []
    regressions = []

    if mpl_diff > 1.0:
        improvements.append(f"MPL +{mpl_diff:.2f}")
    elif mpl_diff < -1.0:
        regressions.append(f"MPL {mpl_diff:.2f}")

    if error_diff < -0.01:
        improvements.append(f"Error {error_diff:.4f}")
    elif error_diff > 0.01:
        regressions.append(f"Error +{error_diff:.4f}")

    if energy_diff < -1.0:
        improvements.append(f"Energy {energy_diff:.2f}")
    elif energy_diff > 1.0:
        regressions.append(f"Energy +{energy_diff:.2f}")

    if improvements:
        print(f"Improvements: {', '.join(improvements)}")
    if regressions:
        print(f"Regressions: {', '.join(regressions)}")

    if not improvements and not regressions:
        print("No significant differences observed")


def main():
    parser = argparse.ArgumentParser(description="Compare objective vs legacy training runs")
    parser.add_argument("--results-dir", type=str, default="results/sac_objective_training",
                        help="Directory containing training results")
    parser.add_argument("--legacy-run", type=str, default=None,
                        help="Specific legacy run directory name (if not specified, uses most recent)")
    parser.add_argument("--objective-run", type=str, default=None,
                        help="Specific objective run directory name (if not specified, uses most recent)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run some training first:")
        print("  python train_sac_objective.py --episodes 10")
        print("  python train_sac_objective.py --episodes 10 --use-objective-reward")
        return

    # Find available runs
    runs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not runs:
        print(f"No runs found in {results_dir}")
        return

    print(f"Found {len(runs)} runs in {results_dir}")

    # Categorize runs
    legacy_runs = []
    objective_runs = []

    for run_dir in runs:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if config.get("use_objective_reward", False):
                objective_runs.append(run_dir)
            else:
                legacy_runs.append(run_dir)

    print(f"  Legacy runs: {len(legacy_runs)}")
    print(f"  Objective runs: {len(objective_runs)}")

    # Select runs to compare
    legacy_run_data = None
    objective_run_data = None

    if args.legacy_run:
        legacy_path = results_dir / args.legacy_run
        if legacy_path.exists():
            legacy_run_data = load_run_history(str(legacy_path))
    elif legacy_runs:
        # Use most recent legacy run
        legacy_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        legacy_run_data = load_run_history(str(legacy_runs[0]))

    if args.objective_run:
        objective_path = results_dir / args.objective_run
        if objective_path.exists():
            objective_run_data = load_run_history(str(objective_path))
    elif objective_runs:
        # Use most recent objective run
        objective_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        objective_run_data = load_run_history(str(objective_runs[0]))

    # Print summaries
    if legacy_run_data:
        print_run_summary(legacy_run_data, "LEGACY REWARD RUN")

    if objective_run_data:
        print_run_summary(objective_run_data, "OBJECTIVE-BASED REWARD RUN")

    # Compare if both available
    if legacy_run_data and objective_run_data:
        compare_runs(legacy_run_data, objective_run_data)
    else:
        print("\nCannot compare: need both legacy and objective runs")
        if not legacy_run_data:
            print("  Missing: legacy run")
            print("  Run: python train_sac_objective.py --episodes 10")
        if not objective_run_data:
            print("  Missing: objective run")
            print("  Run: python train_sac_objective.py --episodes 10 --use-objective-reward")


if __name__ == "__main__":
    main()
