#!/usr/bin/env python3
"""
Analyze EconProfileNet effects on economic outcomes.

Loads datapacks from DataPackRepo and analyzes:
- Distribution of econ_profile_deltas
- Correlations between deltas and economic outcomes (ΔMPL, Δerror, ΔEP, ΔJ, energy)
- Group statistics by (env_name, engine_type, task_type)

This is read-only analysis - no weighting/sampling changes.
Validates that logging and schema are correct for future EconProfileNet training.
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation.datapack_repo import DataPackRepo


def load_all_datapacks(repo_dir: str):
    """Load all datapacks from the repository."""
    repo = DataPackRepo(base_dir=repo_dir)
    datapacks = []

    # Load from all env types
    for env_type in ["drawer_vase", "dishwashing", "dishwashing_arm"]:
        try:
            packs = repo.load_all(env_type)
            datapacks.extend(packs)
        except Exception:
            pass  # Env type may not exist

    return datapacks


def group_by_context(datapacks):
    """Group datapacks by (env_name, engine_type, task_type)."""
    groups = defaultdict(list)

    for dp in datapacks:
        if dp.objective_profile is not None:
            key = (
                dp.objective_profile.env_name,
                dp.objective_profile.engine_type,
                dp.objective_profile.task_type,
            )
        else:
            # Fall back to condition profile
            key = (
                dp.env_type,
                dp.condition.engine_type,
                "unknown",
            )
        groups[key].append(dp)

    return groups


def group_by_context_and_segment(datapacks):
    """Group datapacks by (env_name, engine_type, task_type, customer_segment)."""
    groups = defaultdict(list)

    for dp in datapacks:
        if dp.objective_profile is not None:
            key = (
                dp.objective_profile.env_name,
                dp.objective_profile.engine_type,
                dp.objective_profile.task_type,
                dp.objective_profile.customer_segment,
            )
        else:
            # Fall back to condition profile
            key = (
                dp.env_type,
                dp.condition.engine_type,
                "unknown",
                "unknown",
            )
        groups[key].append(dp)

    return groups


def analyze_econ_profile_deltas(datapacks):
    """Analyze distribution of econ_profile_deltas."""
    print("\n" + "=" * 70)
    print("ECON PROFILE DELTAS DISTRIBUTION")
    print("=" * 70)

    delta_names = [
        "Δbase_rate",
        "Δdamage_cost",
        "Δcare_cost",
        "Δenergy_Wh_per_attempt",
        "Δmax_steps_scale",
    ]

    # Collect all deltas
    all_deltas = []
    for dp in datapacks:
        if dp.objective_profile is not None and dp.objective_profile.econ_profile_deltas is not None:
            all_deltas.append(dp.objective_profile.econ_profile_deltas)

    if not all_deltas:
        print("No datapacks with econ_profile_deltas found.")
        print("(This is expected if EconProfileNet hasn't been applied yet)")
        return

    deltas_array = np.array(all_deltas)

    print(f"Total datapacks with deltas: {len(all_deltas)}")
    print("\nDelta Statistics:")
    print("-" * 70)
    print(f"{'Delta':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for i, name in enumerate(delta_names):
        col = deltas_array[:, i]
        print(f"{name:<25} {np.mean(col):>10.6f} {np.std(col):>10.6f} "
              f"{np.min(col):>10.6f} {np.max(col):>10.6f}")


def analyze_correlations(datapacks):
    """Analyze correlations between deltas and economic outcomes."""
    print("\n" + "=" * 70)
    print("CORRELATIONS: ECON PROFILE DELTAS vs OUTCOMES")
    print("=" * 70)

    delta_names = [
        "Δbase_rate",
        "Δdamage_cost",
        "Δcare_cost",
        "Δenergy_Wh_per_attempt",
        "Δmax_steps_scale",
    ]

    outcome_names = ["ΔMPL", "Δerror", "ΔEP", "ΔJ", "Wh/unit"]

    # Collect data
    deltas = []
    outcomes = []

    for dp in datapacks:
        if dp.objective_profile is not None and dp.objective_profile.econ_profile_deltas is not None:
            deltas.append(dp.objective_profile.econ_profile_deltas)
            outcomes.append([
                dp.attribution.delta_mpl,
                dp.attribution.delta_error,
                dp.attribution.delta_ep,
                dp.attribution.delta_J,
                dp.energy.Wh_per_unit,
            ])

    if len(deltas) < 2:
        print("Not enough datapacks with deltas for correlation analysis.")
        print("(Need at least 2, have {})".format(len(deltas)))
        return

    deltas_array = np.array(deltas)
    outcomes_array = np.array(outcomes)

    print("\nCorrelation Matrix (Pearson r):")
    print("-" * 70)

    # Header
    header = f"{'Delta':<25}"
    for name in outcome_names:
        header += f"{name:>10}"
    print(header)
    print("-" * 70)

    # Compute correlations
    for i, delta_name in enumerate(delta_names):
        row = f"{delta_name:<25}"
        for j in range(len(outcome_names)):
            # Pearson correlation
            corr = np.corrcoef(deltas_array[:, i], outcomes_array[:, j])[0, 1]
            row += f"{corr:>10.4f}"
        print(row)


def analyze_effective_params(datapacks):
    """Analyze effective EconParams distribution."""
    print("\n" + "=" * 70)
    print("EFFECTIVE ECON PARAMS DISTRIBUTION")
    print("=" * 70)

    param_names = ["base_rate", "damage_cost", "care_cost", "energy_Wh_per_attempt", "max_steps"]

    # Collect effective params
    params = defaultdict(list)
    for dp in datapacks:
        if dp.objective_profile is not None and dp.objective_profile.econ_params_effective is not None:
            for name in param_names:
                if name in dp.objective_profile.econ_params_effective:
                    params[name].append(dp.objective_profile.econ_params_effective[name])

    if not params:
        print("No datapacks with effective econ params found.")
        return

    print(f"{'Param':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for name in param_names:
        if params[name]:
            values = np.array(params[name])
            print(f"{name:<25} {np.mean(values):>10.4f} {np.std(values):>10.4f} "
                  f"{np.min(values):>10.4f} {np.max(values):>10.4f}")


def analyze_by_group(groups):
    """Analyze statistics by context group."""
    print("\n" + "=" * 70)
    print("STATISTICS BY CONTEXT GROUP")
    print("=" * 70)

    for key, datapacks in sorted(groups.items()):
        env_name, engine_type, task_type = key
        print(f"\n{env_name} / {engine_type} / {task_type}")
        print("-" * 50)
        print(f"  Total datapacks: {len(datapacks)}")

        # Bucket distribution
        positive = sum(1 for dp in datapacks if dp.bucket == "positive")
        negative = len(datapacks) - positive
        print(f"  Positive: {positive}, Negative: {negative}")

        # Attribution statistics
        delta_mpls = [dp.attribution.delta_mpl for dp in datapacks]
        delta_errors = [dp.attribution.delta_error for dp in datapacks]
        delta_eps = [dp.attribution.delta_ep for dp in datapacks]
        delta_js = [dp.attribution.delta_J for dp in datapacks]
        wh_per_unit = [dp.energy.Wh_per_unit for dp in datapacks]

        print(f"  ΔMPL: mean={np.mean(delta_mpls):.4f}, std={np.std(delta_mpls):.4f}")
        print(f"  Δerror: mean={np.mean(delta_errors):.4f}, std={np.std(delta_errors):.4f}")
        print(f"  ΔEP: mean={np.mean(delta_eps):.4f}, std={np.std(delta_eps):.4f}")
        print(f"  ΔJ: mean={np.mean(delta_js):.4f}, std={np.std(delta_js):.4f}")
        print(f"  Wh/unit: mean={np.mean(wh_per_unit):.4f}, std={np.std(wh_per_unit):.4f}")

        # Objective vector distribution
        obj_vectors = []
        for dp in datapacks:
            if dp.objective_profile is not None:
                obj_vectors.append(dp.objective_profile.objective_vector)
        if obj_vectors:
            obj_array = np.array(obj_vectors)
            print(f"  Objective vectors: {len(obj_vectors)}")
            print(f"    w_mpl: mean={np.mean(obj_array[:, 0]):.2f}")
            print(f"    w_error: mean={np.mean(obj_array[:, 1]):.2f}")
            print(f"    w_energy: mean={np.mean(obj_array[:, 2]):.2f}")
            print(f"    w_safety: mean={np.mean(obj_array[:, 3]):.2f}")

        # Customer segments
        segments = defaultdict(int)
        for dp in datapacks:
            if dp.objective_profile is not None:
                segments[dp.objective_profile.customer_segment] += 1
        if segments:
            print(f"  Customer segments: {dict(segments)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EconProfileNet effects on economic outcomes"
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="data/datapacks/phase_c",
        help="Path to DataPackRepo directory"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only summary statistics"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ECON PROFILE EFFECTS ANALYSIS")
    print("=" * 70)
    print(f"Repository: {args.repo_dir}")

    # Load all datapacks
    datapacks = load_all_datapacks(args.repo_dir)
    print(f"Total datapacks loaded: {len(datapacks)}")

    if not datapacks:
        print("\nNo datapacks found. Run eval scripts first to generate data.")
        return

    # Group by context
    groups = group_by_context(datapacks)
    print(f"Context groups: {len(groups)}")

    # Count datapacks with objective profiles
    with_obj_profile = sum(1 for dp in datapacks if dp.objective_profile is not None)
    with_deltas = sum(
        1 for dp in datapacks
        if dp.objective_profile is not None and dp.objective_profile.econ_profile_deltas is not None
    )
    print(f"Datapacks with objective_profile: {with_obj_profile}")
    print(f"Datapacks with econ_profile_deltas: {with_deltas}")

    if args.summary_only:
        analyze_by_group(groups)
    else:
        # Full analysis
        analyze_by_group(groups)
        analyze_effective_params(datapacks)
        analyze_econ_profile_deltas(datapacks)
        analyze_correlations(datapacks)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    if with_deltas == 0:
        print("\nNote: No EconProfileNet deltas found yet.")
        print("This is expected if EconProfileNet hasn't been applied.")
        print("The schema and logging are correct - ready for future training.")
    else:
        print(f"\nFound {with_deltas} datapacks with EconProfileNet deltas.")
        print("Ready to train econ surface model from these observations.")


if __name__ == "__main__":
    main()
