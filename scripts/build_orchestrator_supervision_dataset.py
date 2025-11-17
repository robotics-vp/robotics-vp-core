#!/usr/bin/env python3
"""
Build orchestrator supervision dataset (advisory only).

Consumes:
- econ_semantic_summaries.jsonl (from analyze_stage1_datapacks_for_econ_semantics.py)
- stage1_econ_semantic_tags.jsonl (from Stage 1 pipeline)
- episode_descriptors.jsonl (Stage 1 → RL bridge)

Produces:
- orchestrator_supervision_v0.jsonl with combined supervision hints
"""
import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[warn] Missing file: {path} (skipping)")
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[warn] Could not parse line in {path}: {e}")
    return rows


def _load_descriptors(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[warn] Missing descriptor file: {path} (skipping)")
        return []
    with open(path, "r") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            try:
                return json.loads(content)
            except Exception:
                pass
        # Default: JSONL
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def index_by_key(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if key in row:
            indexed[row[key]] = row
    return indexed


def build_record(
    pack_id: str,
    summary: Dict[str, Any],
    tags: Dict[str, Any],
    descriptor: Dict[str, Any],
) -> Dict[str, Any]:
    semantic_tags = summary.get("semantic_tags") or descriptor.get("semantic_tags") or []
    econ_semantic_tags = tags.get("econ_semantic_tags") or []
    semantic_quality = tags.get("semantic_quality")
    if semantic_quality is None:
        semantic_quality = summary.get("trust_score") or descriptor.get("trust_score")

    try:
        tier_val = int(summary.get("tier", descriptor.get("tier", 0)) or 0)
    except Exception:
        tier_val = 0
    tier = tier_val

    try:
        trust = float(summary.get("trust_score", descriptor.get("trust_score", 0.0)) or 0.0)
    except Exception:
        trust = 0.0

    sampling_weight = descriptor.get("sampling_weight", summary.get("sampling_weight", 0.0)) or 0.0

    ctx_features = {
        "env_name": descriptor.get("env_name"),
        "backend": descriptor.get("backend"),
        "objective_vector": descriptor.get("objective_vector"),
        "semantic_tags": semantic_tags,
        "econ_semantic_tags": econ_semantic_tags,
        "focus": summary.get("suggested_focus", []),
    }

    record = {
        "pack_id": pack_id,
        "ctx_features": ctx_features,
        "chosen_preset": descriptor.get("objective_preset") or summary.get("objective_preset") or "balanced",
        "recommended_profile": summary.get("suggested_action", "sample_normally"),
        "econ_priority": summary.get("main_driver", "balanced"),
        "sampling_override": sampling_weight,
        "semantic_quality": semantic_quality,
        "tier": tier,
        "trust_score": trust,
        "novelty_flag": bool(summary.get("delta_J", 0.0) > 0.0 or tier >= 2),
        "is_good": summary.get("is_good_datapack", False),
    }
    return record


def build_dataset(
    summaries_path: str,
    econ_tags_path: str,
    descriptors_path: str,
    output_path: str,
) -> List[Dict[str, Any]]:
    summaries = _load_jsonl(summaries_path)
    econ_tags = _load_jsonl(econ_tags_path)
    descriptors = _load_descriptors(descriptors_path)

    summary_index = index_by_key(summaries, "datapack_id")
    tags_index = index_by_key(econ_tags, "pack_id")
    desc_index = index_by_key(descriptors, "pack_id")

    all_ids = set(summary_index.keys()) | set(tags_index.keys()) | set(desc_index.keys())
    records: List[Dict[str, Any]] = []

    for pack_id in sorted(all_ids):
        record = build_record(
            pack_id,
            summary_index.get(pack_id, {}),
            tags_index.get(pack_id, {}),
            desc_index.get(pack_id, {}),
        )
        records.append(record)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[done] Wrote {len(records)} supervision rows to {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Build orchestrator supervision dataset (advisory only)")
    parser.add_argument(
        "--econ-semantic-summaries",
        type=str,
        default="results/stage1_pipeline/econ_semantic_summaries.jsonl",
        help="Path to econ_semantic_summaries.jsonl",
    )
    parser.add_argument(
        "--econ-tags",
        type=str,
        default="results/stage1_pipeline/stage1_econ_semantic_tags.jsonl",
        help="Path to stage1_econ_semantic_tags.jsonl",
    )
    parser.add_argument(
        "--episode-descriptors",
        type=str,
        default="results/stage1_pipeline/episode_descriptors.jsonl",
        help="Path to episode_descriptors.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/stage1_pipeline/orchestrator_supervision_v0.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    build_dataset(
        summaries_path=args.econ_semantic_summaries,
        econ_tags_path=args.econ_tags,
        descriptors_path=args.episode_descriptors,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
