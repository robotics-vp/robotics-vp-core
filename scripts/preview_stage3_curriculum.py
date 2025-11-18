#!/usr/bin/env python3
"""
Preview Stage 3 curriculum sampling across phases.

Loads Stage 1/2 artifacts (or a tiny synthetic pool) and shows which sampler
strategies are used at different steps. Advisory-only; does not touch reward
math or training loops.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.rl.episode_sampling import (
    DataPackRLSampler,
    load_enrichments_from_jsonl,
    load_episode_descriptors_from_jsonl,
)
from src.rl.curriculum import DataPackCurriculum
from src.valuation.datapack_schema import DataPackMeta
from src.orchestrator.semantic_orchestrator_v2 import load_latest_advisory


def _load_datapacks(path: Path) -> List[DataPackMeta]:
    datapacks: List[DataPackMeta] = []
    if not path.exists():
        return datapacks
    if path.suffix == ".jsonl":
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                datapacks.append(DataPackMeta.from_dict(json.loads(line)))
    else:
        with path.open("r") as f:
            data = json.load(f)
            for dp in data:
                datapacks.append(DataPackMeta.from_dict(dp))
    return datapacks


def _format_row(desc):
    meta = desc.get("sampling_metadata", {})
    return {
        "pack_id": str(desc.get("pack_id", ""))[:22],
        "tier": desc.get("tier", 0),
        "strategy": meta.get("strategy", ""),
        "phase": meta.get("phase", ""),
        "frontier": meta.get("frontier_score", 0.0),
        "econ": meta.get("econ_urgency_score", 0.0),
        "novelty": meta.get("novelty_score", 0.0),
    }


def _print_batch(step: int, phase: str, batch):
    print(f"\nStep {step} | Phase: {phase} | Batch size: {len(batch)}")
    counts = {}
    for item in batch:
        strat = item.get("sampling_metadata", {}).get("strategy")
        counts[strat] = counts.get(strat, 0) + 1
    print(f"Strategy mix: {counts}")
    print(f"{'pack_id':<24} {'tier':<4} {'strategy':<20} {'frontier':<9} {'econ':<7} {'novelty':<7}")
    for desc in batch:
        row = _format_row(desc)
        print(
            f"{row['pack_id']:<24} {row['tier']:<4} {row['strategy']:<20} "
            f"{row['frontier']:<9.2f} {row['econ']:<7.2f} {row['novelty']:<7.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Preview Stage 3 curriculum sampling")
    parser.add_argument("--stage1-datapacks", type=str, default="results/stage1_pipeline/datapacks.json")
    parser.add_argument("--stage2-enrichments", type=str, default="results/stage2_semantic/semantic_enrichments.jsonl")
    parser.add_argument("--existing-descriptors", type=str, default="")
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for curriculum sampling")
    parser.add_argument("--task-id", type=str, default="task_curriculum")
    parser.add_argument("--use-orchestrator-advisories", action="store_true")
    args = parser.parse_args()

    datapack_path = Path(args.stage1_datapacks)
    enrichment_path = Path(args.stage2_enrichments)
    existing_desc_path = Path(args.existing_descriptors) if args.existing_descriptors else None

    datapacks = _load_datapacks(datapack_path)
    enrichments = load_enrichments_from_jsonl(enrichment_path) if enrichment_path.exists() else []
    existing_descriptors = (
        load_episode_descriptors_from_jsonl(existing_desc_path) if existing_desc_path and existing_desc_path.exists() else []
    )

    if not datapacks and not existing_descriptors:
        print("[preview_stage3_curriculum] No datapacks/descriptors found; generating a tiny synthetic pool.")
        datapacks = [DataPackMeta()]

    advisory = load_latest_advisory(args.task_id) if args.use_orchestrator_advisories else None
    sampler = DataPackRLSampler(datapacks=datapacks, enrichments=enrichments, existing_descriptors=existing_descriptors, advisory=advisory)
    curriculum = DataPackCurriculum(
        sampler=sampler,
        total_steps=args.total_steps,
        config={"base_seed": args.seed},
        advisory=advisory,
    )

    print("[preview_stage3_curriculum] Pool summary:", sampler.pool_summary())

    default_steps = [
        0,
        int(0.15 * args.total_steps),
        int(0.30 * args.total_steps),
        int(0.55 * args.total_steps),
        int(0.80 * args.total_steps),
        int(0.95 * args.total_steps),
    ]

    for step in default_steps:
        phase = curriculum.get_phase(step)
        batch = curriculum.sample_batch(step=step, batch_size=args.batch_size)
        _print_batch(step, phase, batch)

    print("\nPreview complete. Curriculum selects sampler strategies only; reward math and PPO/SAC remain unchanged.")


if __name__ == "__main__":
    main()
