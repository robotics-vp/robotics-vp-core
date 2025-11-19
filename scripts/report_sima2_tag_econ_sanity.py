#!/usr/bin/env python3
"""
Quick SIMA-2 tag ↔ econ sanity report.

Reads semantic tags and econ vectors from a stress run directory and prints
per-tag averages of damage, energy, and MPL for eyeballing TrustMatrix sanity.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _tag_type_from_key(key: str) -> str:
    # Normalize list keys to tag type labels
    mapping = {
        "risk_tags": "RiskTag",
        "fragility_tags": "FragilityTag",
        "affordance_tags": "AffordanceTag",
        "efficiency_tags": "EfficiencyTag",
        "novelty_tags": "NoveltyTag",
        "intervention_tags": "InterventionTag",
        "segment_boundary_tags": "SegmentBoundaryTag",
        "subtask_tags": "SubtaskTag",
        "recovery_pattern_tags": "RecoveryPatternTag",
        "mobility_risk_tags": "MobilityRiskTag",
        "contact_quality_tags": "ContactQualityTag",
        "precision_tolerance_tags": "PrecisionToleranceTag",
    }
    return mapping.get(key, key)


def _tag_types(record: Dict[str, Any]) -> Iterable[str]:
    # Derive tag types from SemanticEnrichmentProposal-like dict
    for key, value in record.items():
        if key.endswith("_tags") and isinstance(value, list) and value:
            yield _tag_type_from_key(key)


def _aggregate(tags_records: List[Dict[str, Any]], econ_map: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    sums = defaultdict(lambda: {"count": 0, "damage": 0.0, "energy": 0.0, "mpl": 0.0})
    for rec in tags_records:
        episode_id = rec.get("episode_id") or rec.get("video_id") or ""
        econ = econ_map.get(episode_id)
        if not econ:
            continue
        dmg = float(econ.get("damage_cost", 0.0))
        energy = float(econ.get("energy_cost", 0.0))
        mpl = float(econ.get("mpl_units_per_hour", 0.0))
        for tag_type in _tag_types(rec):
            bucket = sums[tag_type]
            bucket["count"] += 1
            bucket["damage"] += dmg
            bucket["energy"] += energy
            bucket["mpl"] += mpl
    return sums


def _format_table(agg: Dict[str, Dict[str, float]]) -> str:
    lines = ["tag_type\tcount\tavg_damage\tavg_energy\tavg_mpl"]
    for tag_type in sorted(agg.keys()):
        bucket = agg[tag_type]
        count = max(bucket["count"], 1)
        lines.append(
            f"{tag_type}\t{bucket['count']}\t{bucket['damage']/count:.3f}\t{bucket['energy']/count:.3f}\t{bucket['mpl']/count:.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Report SIMA-2 tag → econ sanity stats.")
    parser.add_argument("--stress-dir", type=str, default="results/sima2_stress", help="Directory containing stress run outputs.")
    parser.add_argument("--tags-path", type=str, default="", help="Optional override for semantic_tags.jsonl path.")
    parser.add_argument("--econ-path", type=str, default="", help="Optional override for econ_vectors.jsonl path.")
    args = parser.parse_args()

    stress_dir = Path(args.stress_dir)
    tags_path = Path(args.tags_path) if args.tags_path else stress_dir / "semantic_tags.jsonl"
    econ_path = Path(args.econ_path) if args.econ_path else stress_dir / "ontology_store" / "econ_vectors.jsonl"

    tags_records = _load_jsonl(tags_path)
    econ_vectors = _load_jsonl(econ_path)
    econ_map = {rec.get("episode_id", ""): rec for rec in econ_vectors}

    agg = _aggregate(tags_records, econ_map)
    print(_format_table(agg))


if __name__ == "__main__":
    main()
