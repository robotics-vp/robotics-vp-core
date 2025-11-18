#!/usr/bin/env python3
"""
Pricing/performance report for a task from the ontology store.
"""
import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.analytics.econ_reports import (
    compute_task_econ_summary,
    compute_datapack_mix_summary,
    compute_pricing_snapshot,
)
from src.ontology.store import OntologyStore


def main():
    parser = argparse.ArgumentParser(description="Report task pricing and performance from ontology")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    task_summary = compute_task_econ_summary(store, args.task_id)
    dp_summary = compute_datapack_mix_summary(store, args.task_id)
    pricing = compute_pricing_snapshot(store, args.task_id)

    print(f"[pricing_report] Task: {task_summary.get('task', {}).get('name', '')} (task_id={args.task_id})")
    task = task_summary.get("task", {})
    print(f"  Human MPL: {task.get('human_mpl_units_per_hour', 0)} units/hr @ ${task.get('human_wage_per_hour', 0):.2f}/hr")
    mpl = task_summary.get("mpl", {})
    print(f"  Robot MPL: mean={mpl.get('mean',0):.2f}, p10={mpl.get('p10',0):.2f}, p90={mpl.get('p90',0):.2f}")
    wp = task_summary.get("wage_parity", {})
    print(f"  Wage parity: mean={wp.get('mean',0):.2f}, p10={wp.get('p10',0):.2f}, p90={wp.get('p90',0):.2f}")
    print(f"  Reward scalar sum mean={task_summary.get('reward_scalar_sum',{}).get('mean',0):.2f}")
    print()
    print("[datapack_mix]")
    sources = dp_summary.get("sources", {})
    for src, stats in sources.items():
        print(f"  {src}: count={stats.get('count',0)}, avg_novelty={stats.get('avg_novelty',0):.2f}, avg_quality={stats.get('avg_quality',0):.2f}")
    if dp_summary.get("recent"):
        print("  Recent datapacks:")
        for dp in dp_summary["recent"]:
            print(f"    - {dp['datapack_id']} ({dp['source_type']}, {dp['modality']}) @ {dp['created_at']}")
    print()
    print("[pricing_snapshot]")
    print(f"  Human unit cost: ${pricing.get('human_unit_cost',0):.4f}")
    print(f"  Robot unit cost: ${pricing.get('robot_unit_cost',0):.4f}")
    print(f"  Implied spread: ${pricing.get('implied_spread_per_unit',0):.4f} per unit")
    print(f"  Rough datapack price floor: ${pricing.get('datapack_price_floor',0):.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "task_summary": task_summary,
                    "datapack_mix": dp_summary,
                    "pricing_snapshot": pricing,
                },
                f,
                indent=2,
            )
        print(f"[pricing_report] Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
