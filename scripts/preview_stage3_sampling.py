#!/usr/bin/env python3
"""
Preview Stage 3 sampler output using Stage 1/2 artifacts.

Loads:
- Stage 1 datapacks (JSON or JSONL) and converts to RL descriptors
- Stage 2 enrichments (JSONL)
- Optional existing episode descriptors (JSONL)

Produces deterministic sampler batches for balanced / frontier / econ_urgency
strategies without modifying reward math or training loops.
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
from src.valuation.datapack_schema import DataPackMeta


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


def _print_samples(label: str, samples):
    print(f"\n{label} ({len(samples)} episodes)")
    print("-" * 80)
    header = f"{'pack_id':<24} {'tier':<4} {'trust':<6} {'strategy':<20} {'frontier':<8} {'econ':<7}"
    print(header)
    for desc in samples:
        meta = desc.get("sampling_metadata", {})
        pack_id = str(desc.get("pack_id", ""))[:22] + ("..." if len(str(desc.get("pack_id", ""))) > 22 else "")
        print(
            f"{pack_id:<24} "
            f"{desc.get('tier', 0):<4} "
            f"{desc.get('trust_score', 0.0):<6.2f} "
            f"{meta.get('strategy',''): <20} "
            f"{meta.get('frontier_score',0.0):<8.2f} "
            f"{meta.get('econ_urgency_score',0.0):<7.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Preview Stage 3 RL sampling")
    parser.add_argument(
        "--stage1-datapacks",
        type=str,
        default="results/stage1_pipeline/datapacks.json",
        help="Path to Stage 1 datapacks JSON/JSONL",
    )
    parser.add_argument(
        "--stage2-enrichments",
        type=str,
        default="results/stage2_semantic/semantic_enrichments.jsonl",
        help="Path to Stage 2 semantic enrichment JSONL",
    )
    parser.add_argument(
        "--existing-descriptors",
        type=str,
        default="",
        help="Optional JSONL of existing RL episode descriptors",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
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
        print("[preview_stage3_sampling] No datapacks/descriptors found; generating a tiny synthetic pool.")
        synthetic_dp = DataPackMeta()
        datapacks = [synthetic_dp]

    sampler = DataPackRLSampler(
        datapacks=datapacks,
        enrichments=enrichments,
        existing_descriptors=existing_descriptors,
    )

    print("[preview_stage3_sampling] Pool summary:", sampler.pool_summary())

    balanced = sampler.sample_batch(args.batch_size, seed=args.seed, strategy="balanced")
    frontier = sampler.sample_batch(args.batch_size, seed=args.seed, strategy="frontier_prioritized")
    econ = sampler.sample_batch(args.batch_size, seed=args.seed, strategy="econ_urgency")

    _print_samples("Balanced", balanced)
    _print_samples("Frontier Prioritized", frontier)
    _print_samples("Econ Urgency", econ)

    print("\nPreview complete. Objective vectors/presets are left untouched; sampler is advisory-only.")


if __name__ == "__main__":
    main()
