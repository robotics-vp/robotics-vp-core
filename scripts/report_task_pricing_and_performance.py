#!/usr/bin/env python3
"""
Pricing/performance report for a task from the ontology store.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.analytics.econ_reports import (
    compute_task_econ_summary,
    compute_datapack_mix_summary,
    compute_pricing_snapshot,
)
from src.ontology.store import OntologyStore


def _load_recap_scores(path: Path) -> List[Dict]:
    if not path or not path.exists():
        return []
    scores = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scores.append(json.loads(line))
    return scores


def _load_jsonl_map(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    records: Dict[str, Any] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ep_id = rec.get("episode_id") or rec.get("episode")
                if ep_id:
                    records[str(ep_id)] = rec
            except Exception:
                continue
    return records


def _bucket_scores(scores: List[Dict]) -> Dict[str, List[Dict]]:
    buckets = {"low": [], "mid": [], "high": []}
    for s in scores:
        val = float(s.get("recap_goodness_score", 0.0))
        if val >= 1.0:
            buckets["high"].append(s)
        elif val <= -1.0:
            buckets["low"].append(s)
        else:
            buckets["mid"].append(s)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Report task pricing and performance from ontology")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--recap-scores", type=str, default="", help="Optional path to RECAP episode_scores.jsonl")
    parser.add_argument("--reward-model-scores", type=str, default="", help="Optional reward model scores JSONL")
    parser.add_argument("--segmentation-tags", type=str, default="", help="Optional segmentation tags JSONL")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    rm_scores = _load_jsonl_map(Path(args.reward_model_scores)) if args.reward_model_scores else {}
    seg_tags = _load_jsonl_map(Path(args.segmentation_tags)) if args.segmentation_tags else {}
    task_summary = compute_task_econ_summary(store, args.task_id, reward_model_scores=rm_scores, segmentation_tags=seg_tags)
    dp_summary = compute_datapack_mix_summary(store, args.task_id)
    pricing = compute_pricing_snapshot(store, args.task_id, reward_model_scores=rm_scores, segmentation_tags=seg_tags)
    recap_scores = _load_recap_scores(Path(args.recap_scores)) if args.recap_scores else []

    print(f"[pricing_report] Task: {task_summary.get('task', {}).get('name', '')} (task_id={args.task_id})")
    task = task_summary.get("task", {})
    print(f"  Human MPL: {task.get('human_mpl_units_per_hour', 0)} units/hr @ ${task.get('human_wage_per_hour', 0):.2f}/hr")
    mpl = task_summary.get("mpl", {})
    print(f"  Robot MPL: mean={mpl.get('mean',0):.2f}, p10={mpl.get('p10',0):.2f}, p90={mpl.get('p90',0):.2f}")
    wp = task_summary.get("wage_parity", {})
    print(f"  Wage parity: mean={wp.get('mean',0):.2f}, p10={wp.get('p10',0):.2f}, p90={wp.get('p90',0):.2f}")
    print(f"  Reward scalar sum mean={task_summary.get('reward_scalar_sum',{}).get('mean',0):.2f}")
    qa_mpl = task_summary.get("quality_adjusted_mpl", {})
    if qa_mpl:
        print(f"  Quality-adjusted MPL: mean={qa_mpl.get('mean',0):.2f}")
    grades = task_summary.get("quality_grades", {})
    if grades:
        print(f"  Quality grade buckets: {grades}")
    if task_summary.get("mobility_penalty"):
        print(f"  Mobility penalty mean={task_summary['mobility_penalty'].get('mean',0):.3f}")
    if task_summary.get("precision_bonus"):
        print(f"  Precision bonus mean={task_summary['precision_bonus'].get('mean',0):.3f}")
    if task_summary.get("stability_risk_score"):
        print(f"  Stability risk mean={task_summary['stability_risk_score'].get('mean',0):.3f}")
    recovery = task_summary.get("recovery_segments", {})
    if recovery:
        print(
            f"  Recovery fraction={recovery.get('mean_recovery_fraction',0):.3f}, "
            f"fraction_with_recovery={recovery.get('fraction_with_recovery',0):.3f}"
        )
    auditor = task_summary.get("auditor", {})
    ratings = auditor.get("datapack_ratings", {})
    if ratings:
        print(f"  Auditor ratings (counts): {ratings.get('counts', {})}")
        print(f"  Auditor ratings (shares): {ratings.get('shares', {})}")
    econ_by_rating = auditor.get("econ_by_rating", {})
    if econ_by_rating:
        print("  Econ by auditor rating:")
        for rating, stats in econ_by_rating.items():
            print(
                f"    {rating}: count={stats.get('count',0)}, "
                f"mpl={stats.get('mean_mpl',0):.3f}, "
                f"energy={stats.get('mean_energy_cost',0):.3f}, "
                f"damage={stats.get('mean_damage_cost',0):.3f}"
            )
    print()
    print("[datapack_mix]")
    sources = dp_summary.get("sources", {})
    for src, stats in sources.items():
        print(f"  {src}: count={stats.get('count',0)}, avg_novelty={stats.get('avg_novelty',0):.2f}, avg_quality={stats.get('avg_quality',0):.2f}")
    if dp_summary.get("recent"):
        print("  Recent datapacks:")
        for dp in dp_summary["recent"]:
            print(f"    - {dp['datapack_id']} ({dp['source_type']}, {dp['modality']}) @ {dp['created_at']}")
    if dp_summary.get("auditor_ratings"):
        print(f"  Auditor rating mix: {dp_summary['auditor_ratings'].get('counts', {})}")
    print()
    print("[pricing_snapshot]")
    print(f"  Human unit cost: ${pricing.get('human_unit_cost',0):.4f}")
    print(f"  Robot unit cost: ${pricing.get('robot_unit_cost',0):.4f}")
    print(f"  Implied spread: ${pricing.get('implied_spread_per_unit',0):.4f} per unit")
    print(f"  Rough datapack price floor: ${pricing.get('datapack_price_floor',0):.4f}")
    if recap_scores:
        buckets = _bucket_scores(recap_scores)
        print("\n[recap_alignment]")
        for name, rows in buckets.items():
            if not rows:
                continue
            mean_score = sum(float(r.get("recap_goodness_score", 0.0)) for r in rows) / len(rows)
            print(f"  bucket={name}: count={len(rows)}, mean_recap_goodness={mean_score:.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "task_summary": task_summary,
                    "datapack_mix": dp_summary,
                    "pricing_snapshot": pricing,
                    "recap_scores": recap_scores,
                    "reward_model_scores": rm_scores,
                    "segmentation_tags": seg_tags,
                },
                f,
                indent=2,
            )
        print(f"[pricing_report] Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
