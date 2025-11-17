#!/usr/bin/env python3
"""
Inspect orchestration transformer training data distribution.

This is read-only and should help spot tool/urgency/objective skews.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import Counter, defaultdict


def load_dataset(path: str):
    if not os.path.exists(path):
        print(f"[inspect] Dataset not found at {path}")
        return []
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="results/orchestrator_training_dataset.jsonl")
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    if not samples:
        print("[inspect] No samples loaded; run dataset builder first.")
        return

    tool_counts = Counter()
    objective_counts = Counter()
    urgency_buckets = defaultdict(int)

    for sample in samples:
        for tc in sample.get("target_tool_sequence", []):
            tool_name = tc.get("tool") if isinstance(tc, dict) else str(tc)
            tool_counts[tool_name] += 1
        ctx = sample.get("context", {}) or {}
        objective = ctx.get("objective_preset") or ctx.get("objective_profile", {}).get("objective_preset")
        if objective:
            objective_counts[objective] += 1
        econ = ctx.get("econ_signals", {}) or {}
        energy_urgency = econ.get("energy_urgency")
        if energy_urgency is not None:
            if energy_urgency > 0.66:
                urgency_buckets["high"] += 1
            elif energy_urgency > 0.33:
                urgency_buckets["medium"] += 1
            else:
                urgency_buckets["low"] += 1

    print("\nTool distribution:")
    for tool, count in tool_counts.most_common():
        print(f"  {tool}: {count}")

    print("\nObjective preset distribution:")
    for obj, count in objective_counts.most_common():
        print(f"  {obj}: {count}")

    print("\nEnergy urgency distribution:")
    for bucket, count in urgency_buckets.items():
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()
