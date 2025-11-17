#!/usr/bin/env python3
"""
Analyze Stage 1 datapacks for economic semantic summaries.

This script:
1. Reads Stage 1 datapacks
2. Derives implied objective presets and econ-relevant tags
3. Generates EconSemanticDecisionSummary-like structures
4. Writes JSONL for potential orchestrator training supervision

This is offline analysis only - no reward changes or training modifications.
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


def analyze_datapack_econ_semantics(datapack: DataPackMeta) -> Dict[str, Any]:
    """
    Analyze a datapack for economic semantic content.

    Returns:
        Dict with:
        - implied_objective_preset
        - econ_relevant_tags
        - semantic_summary (EconSemanticDecisionSummary-like)
    """
    # Get episode descriptor for convenience
    descriptor = datapack_to_rl_episode_descriptor(datapack)

    # Derive objective preset
    objective_preset = descriptor['objective_preset']
    objective_vector = descriptor['objective_vector']

    # Extract econ-relevant tags from semantic tags
    econ_relevant_tags = []
    if datapack.guidance_profile:
        for tag in datapack.guidance_profile.semantic_tags:
            # Safety-related
            if any(kw in tag.lower() for kw in ["safety", "fragile", "avoid", "careful"]):
                econ_relevant_tags.append(f"safety:{tag}")

            # Energy-related
            if any(kw in tag.lower() for kw in ["energy", "efficient", "slow"]):
                econ_relevant_tags.append(f"energy:{tag}")

            # Throughput-related
            if any(kw in tag.lower() for kw in ["speed", "fast", "throughput", "high_speed"]):
                econ_relevant_tags.append(f"throughput:{tag}")

            # Error recovery
            if any(kw in tag.lower() for kw in ["error", "recovery", "retry"]):
                econ_relevant_tags.append(f"error_recovery:{tag}")

    # Create semantic summary (EconSemanticDecisionSummary-like)
    semantic_summary = {
        # Identification
        "datapack_id": datapack.pack_id,
        "task_name": datapack.task_name,
        "env_name": descriptor['env_name'],
        "backend": descriptor['backend'],

        # Objective context
        "objective_preset": objective_preset,
        "objective_vector": objective_vector,

        # Economic drivers (from guidance profile)
        "main_driver": descriptor['focus_areas'][0] if descriptor['focus_areas'] else "unknown",
        "is_good_datapack": descriptor['tags']['is_good'],

        # Economic deltas (from attribution)
        "delta_mpl": datapack.attribution.delta_mpl if datapack.attribution else 0.0,
        "delta_error": datapack.attribution.delta_error if datapack.attribution else 0.0,
        "delta_J": datapack.attribution.delta_J if datapack.attribution else 0.0,

        # Quality signals
        "tier": descriptor['tier'],
        "trust_score": descriptor['trust_score'],
        "sampling_weight": descriptor['sampling_weight'],

        # Semantic context
        "econ_relevant_tags": econ_relevant_tags,
        "semantic_tags": descriptor['semantic_tags'],

        # Advisory fields (for orchestrator training)
        "suggested_action": _suggest_orchestrator_action(datapack, descriptor),
        "suggested_focus": _suggest_focus_areas(datapack, descriptor),
    }

    return semantic_summary


def _suggest_orchestrator_action(datapack: DataPackMeta, descriptor: Dict[str, Any]) -> str:
    """
    Suggest an orchestrator action based on datapack characteristics.

    This is advisory-only for potential orchestrator training.
    """
    tier = descriptor['tier']
    is_good = descriptor['tags']['is_good']
    main_driver = descriptor['tags']['main_driver']

    # High-tier good datapacks → sample more
    if tier >= 2 and is_good:
        return "prioritize_sampling"

    # Safety-focused → increase safety weight
    if main_driver == "safety_margin":
        return "increase_safety_weight"

    # Energy-focused → increase energy weight
    if main_driver == "energy_efficiency":
        return "increase_energy_weight"

    # Throughput-focused → increase throughput weight
    if main_driver == "throughput_gain":
        return "increase_throughput_weight"

    # Default
    return "sample_normally"


def _suggest_focus_areas(datapack: DataPackMeta, descriptor: Dict[str, Any]) -> List[str]:
    """
    Suggest focus areas for curriculum learning based on datapack.
    """
    focus_areas = []
    main_driver = descriptor['tags']['main_driver']

    # Main driver is always a focus
    focus_areas.append(main_driver)

    # Safety-critical → add safety focus
    if "safety" in descriptor['semantic_tags'] or "fragile" in descriptor['semantic_tags']:
        focus_areas.append("safety_critical")

    # Energy-related → add energy focus
    if "energy" in descriptor['semantic_tags'] or "efficient" in descriptor['semantic_tags']:
        focus_areas.append("energy_optimization")

    # Error recovery → add robustness focus
    if "error" in descriptor['semantic_tags'] or "recovery" in descriptor['semantic_tags']:
        focus_areas.append("robustness")

    return list(set(focus_areas))  # Remove duplicates


def analyze_datapacks(
    datapacks_path: str,
    output_jsonl: str = None,
    output_summary: str = None,
):
    """Analyze Stage 1 datapacks for econ semantic content."""
    print("=" * 70)
    print("Stage 1 Datapacks → Economic Semantic Analysis")
    print("=" * 70)

    # Load datapacks
    print(f"\n[1/4] Loading datapacks from {datapacks_path}...")
    with open(datapacks_path, "r") as f:
        datapacks_data = json.load(f)

    datapacks = []
    for dp_dict in datapacks_data:
        try:
            dp = DataPackMeta.from_dict(dp_dict)
            datapacks.append(dp)
        except Exception as e:
            print(f"Warning: Failed to load datapack: {e}")

    print(f"✓ Loaded {len(datapacks)} datapacks")

    # Analyze each datapack
    print("\n[2/4] Analyzing datapacks for econ semantics...")
    semantic_summaries = []
    for dp in datapacks:
        summary = analyze_datapack_econ_semantics(dp)
        semantic_summaries.append(summary)

    print(f"✓ Generated {len(semantic_summaries)} semantic summaries")

    # Compute aggregate statistics
    print("\n[3/4] Computing aggregate statistics...")
    stats = {
        "total_datapacks": len(datapacks),
        "objective_preset_distribution": {},
        "main_driver_distribution": {},
        "tier_distribution": {},
        "suggested_action_distribution": {},
        "avg_delta_mpl": 0.0,
        "avg_delta_error": 0.0,
        "avg_delta_J": 0.0,
        "avg_trust_score": 0.0,
    }

    delta_mpl_sum = 0.0
    delta_error_sum = 0.0
    delta_J_sum = 0.0
    trust_sum = 0.0

    for summary in semantic_summaries:
        # Distributions
        preset = summary['objective_preset']
        stats['objective_preset_distribution'][preset] = \
            stats['objective_preset_distribution'].get(preset, 0) + 1

        driver = summary['main_driver']
        stats['main_driver_distribution'][driver] = \
            stats['main_driver_distribution'].get(driver, 0) + 1

        tier = summary['tier']
        stats['tier_distribution'][tier] = \
            stats['tier_distribution'].get(tier, 0) + 1

        action = summary['suggested_action']
        stats['suggested_action_distribution'][action] = \
            stats['suggested_action_distribution'].get(action, 0) + 1

        # Aggregates
        delta_mpl_sum += summary['delta_mpl']
        delta_error_sum += summary['delta_error']
        delta_J_sum += summary['delta_J']
        trust_sum += summary['trust_score']

    n = len(semantic_summaries)
    stats['avg_delta_mpl'] = delta_mpl_sum / n if n > 0 else 0.0
    stats['avg_delta_error'] = delta_error_sum / n if n > 0 else 0.0
    stats['avg_delta_J'] = delta_J_sum / n if n > 0 else 0.0
    stats['avg_trust_score'] = trust_sum / n if n > 0 else 0.0

    print("✓ Statistics computed")

    # Display statistics
    print("\n" + "=" * 70)
    print("Economic Semantic Analysis Results")
    print("=" * 70)
    print(f"Total datapacks: {stats['total_datapacks']}")
    print(f"\nObjective preset distribution:")
    for preset, count in stats['objective_preset_distribution'].items():
        print(f"  {preset}: {count}")

    print(f"\nMain driver distribution:")
    for driver, count in stats['main_driver_distribution'].items():
        print(f"  {driver}: {count}")

    print(f"\nTier distribution:")
    for tier, count in stats['tier_distribution'].items():
        print(f"  Tier {tier}: {count}")

    print(f"\nSuggested action distribution:")
    for action, count in stats['suggested_action_distribution'].items():
        print(f"  {action}: {count}")

    print(f"\nAverage economic deltas:")
    print(f"  ΔMPL: {stats['avg_delta_mpl']:.3f}")
    print(f"  Δerror: {stats['avg_delta_error']:.3f}")
    print(f"  ΔJ: {stats['avg_delta_J']:.3f}")
    print(f"  Trust score: {stats['avg_trust_score']:.3f}")

    # Save outputs
    print("\n[4/4] Saving outputs...")
    if output_jsonl:
        with open(output_jsonl, "w") as f:
            for summary in semantic_summaries:
                f.write(json.dumps(summary) + "\n")
        print(f"✓ Saved semantic summaries to {output_jsonl}")

    if output_summary:
        with open(output_summary, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved aggregate statistics to {output_summary}")

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print("\nUsage Notes:")
    print("  - Semantic summaries can be used for orchestrator training")
    print("  - Suggested actions are advisory-only (no reward changes)")
    print("  - Main drivers indicate dominant economic focus areas")
    print("  - This is offline analysis; does not affect RL training")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Stage 1 datapacks for economic semantic content"
    )
    parser.add_argument(
        "--datapacks",
        type=str,
        default="results/stage1_pipeline/datapacks.json",
        help="Path to Stage 1 datapacks JSON file",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="results/stage1_pipeline/econ_semantic_summaries.jsonl",
        help="Output path for semantic summaries JSONL",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="results/stage1_pipeline/econ_semantic_stats.json",
        help="Output path for aggregate statistics JSON",
    )
    args = parser.parse_args()

    analyze_datapacks(
        args.datapacks,
        args.output_jsonl,
        args.output_summary,
    )


if __name__ == "__main__":
    main()
