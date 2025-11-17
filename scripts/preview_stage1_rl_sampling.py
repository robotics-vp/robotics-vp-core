#!/usr/bin/env python3
"""
Preview script for Stage 1 → RL episode sampling.

Loads datapacks from Stage 1 pipeline and shows how they would be converted
to RL episode descriptors for downstream training.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.valuation.datapack_schema import DataPackMeta
from src.rl.episode_sampling import datapack_to_rl_episode_descriptor


def load_datapacks(datapacks_path: str) -> List[DataPackMeta]:
    """Load datapacks from JSON file."""
    with open(datapacks_path, "r") as f:
        datapacks_data = json.load(f)

    datapacks = []
    for dp_dict in datapacks_data:
        try:
            dp = DataPackMeta.from_dict(dp_dict)
            datapacks.append(dp)
        except Exception as e:
            print(f"Warning: Failed to load datapack: {e}")

    return datapacks


def preview_rl_sampling(datapacks_path: str, output_json: str = None):
    """Preview RL episode sampling from Stage 1 datapacks."""
    print("=" * 70)
    print("Stage 1 → RL Episode Sampling Preview")
    print("=" * 70)

    # Load datapacks
    print(f"\n[1/3] Loading datapacks from {datapacks_path}...")
    datapacks = load_datapacks(datapacks_path)
    print(f"✓ Loaded {len(datapacks)} datapacks")

    # Convert to episode descriptors
    print("\n[2/3] Converting to RL episode descriptors...")
    descriptors = []
    for dp in datapacks:
        descriptor = datapack_to_rl_episode_descriptor(dp)
        descriptors.append(descriptor)

    print(f"✓ Created {len(descriptors)} episode descriptors")

    # Display summary table
    print("\n[3/3] Episode Descriptor Summary:")
    print("-" * 70)
    print(f"{'Pack ID':<25} {'Env':<15} {'Preset':<12} {'Tier':<6} {'Trust':<7} {'Weight':<7}")
    print("-" * 70)

    for desc in descriptors:
        pack_id_short = desc['pack_id'][:23] + "..."
        env_name = desc['env_name'][:13]
        preset = desc['objective_preset'][:10]
        tier = desc['tier']
        trust = desc['trust_score']
        weight = desc['sampling_weight']

        print(f"{pack_id_short:<25} {env_name:<15} {preset:<12} {tier:<6} {trust:<7.3f} {weight:<7.3f}")

    print("-" * 70)

    # Compute statistics
    total_weight = sum(d['sampling_weight'] for d in descriptors)
    tier_dist = {}
    preset_dist = {}
    backend_dist = {}

    for desc in descriptors:
        tier_dist[desc['tier']] = tier_dist.get(desc['tier'], 0) + 1
        preset_dist[desc['objective_preset']] = preset_dist.get(desc['objective_preset'], 0) + 1
        backend_dist[desc['backend']] = backend_dist.get(desc['backend'], 0) + 1

    print("\nStatistics:")
    print(f"  Total descriptors: {len(descriptors)}")
    print(f"  Total sampling weight: {total_weight:.3f}")
    print(f"  Avg sampling weight: {total_weight/len(descriptors):.3f}")
    print(f"  Tier distribution: {tier_dist}")
    print(f"  Objective preset distribution: {preset_dist}")
    print(f"  Backend distribution: {backend_dist}")

    # Show detailed view of first descriptor
    if descriptors:
        print("\nDetailed View (First Descriptor):")
        print("-" * 70)
        first = descriptors[0]
        for key, value in first.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

    # Optionally save to JSON
    if output_json:
        with open(output_json, "w") as f:
            json.dump(descriptors, f, indent=2)
        print(f"\n✓ Saved descriptors to {output_json}")

    print("\n" + "=" * 70)
    print("Preview Complete")
    print("=" * 70)
    print("\nNext Steps:")
    print("  - These descriptors can be used to configure RL training episodes")
    print("  - Backend-agnostic: works with PyBullet, Isaac, or other engines")
    print("  - Sampling weight determines episode prioritization in curriculum")


def main():
    parser = argparse.ArgumentParser(
        description="Preview RL episode sampling from Stage 1 datapacks"
    )
    parser.add_argument(
        "--datapacks",
        type=str,
        default="results/stage1_pipeline/datapacks.json",
        help="Path to Stage 1 datapacks JSON file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional: save episode descriptors to JSON file",
    )
    args = parser.parse_args()

    preview_rl_sampling(args.datapacks, args.output_json)


if __name__ == "__main__":
    main()
