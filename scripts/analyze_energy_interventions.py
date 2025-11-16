#!/usr/bin/env python3
"""
Analyze energy interventions to check causal patterns across profiles.
"""
import argparse
import json
import numpy as np
from collections import defaultdict


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def summarize(records):
    by_profile = defaultdict(list)
    for r in records:
        by_profile[r["profile"]].append(r["summary"])

    for profile, summaries in by_profile.items():
        mpl = np.array([s.get("mpl_episode", 0.0) for s in summaries])
        err = np.array([s.get("error_rate_episode", 0.0) for s in summaries])
        energy = np.array([s.get("energy_Wh", 0.0) for s in summaries])
        per_unit = np.array([s.get("energy_Wh_per_unit", 0.0) for s in summaries])

        print(f"=== {profile} ===")
        print(f"Count: {len(summaries)}")
        print(f"MPL: mean={mpl.mean():.4f} std={mpl.std():.4f}")
        print(f"Error: mean={err.mean():.4f} std={err.std():.4f}")
        print(f"Energy Wh: mean={energy.mean():.4f} std={energy.std():.4f}")
        print(f"Energy Wh/unit: mean={per_unit.mean():.4f} std={per_unit.std():.4f}")

        # Limb fractions if available
        shoulder_frac = []
        for s in summaries:
            limb_energy = s.get("energy_per_limb", {})
            total = sum(v.get("Wh", 0.0) for v in limb_energy.values()) if limb_energy else 0.0
            if total > 0:
                shoulder_frac.append(limb_energy.get("shoulder", {}).get("Wh", 0.0) / total)
        if shoulder_frac:
            arr = np.array(shoulder_frac)
            print(f"Shoulder energy fraction mean={arr.mean():.4f} std={arr.std():.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze energy interventions")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    args = parser.parse_args()

    records = load_records(args.interventions)
    summarize(records)


if __name__ == "__main__":
    main()
