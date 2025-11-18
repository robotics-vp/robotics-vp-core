#!/usr/bin/env python3
"""
Offline scorer using the RewardModel policy.

Consumes ontology episodes, econ vectors, semantic tags, and optional RECAP
scores to emit RewardModelEpisodeScores JSONL.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

repo_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.policies.registry import build_all_policies
from src.policies.reward_model_types import RewardModelEpisodeScores
from src.vla.recap_inference import RecapEpisodeScores


def _load_jsonl_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    items: Dict[str, Dict[str, Any]] = {}
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ep_id = rec.get("episode_id") or rec.get("episode")
            if not ep_id:
                continue
            items[str(ep_id)] = rec
    return items


def _load_recap_map(path: Optional[str]) -> Dict[str, RecapEpisodeScores]:
    raw = _load_jsonl_map(path)
    out: Dict[str, RecapEpisodeScores] = {}
    for eid, rec in raw.items():
        try:
            out[eid] = RecapEpisodeScores(**rec)
        except Exception:
            continue
    return out


def _score(store: OntologyStore, tags_map: Dict[str, Dict[str, Any]], recap_map: Dict[str, RecapEpisodeScores]) -> Dict[str, RewardModelEpisodeScores]:
    policies = build_all_policies()
    rm_policy = getattr(policies, "reward_model", None)
    if rm_policy is None:
        raise RuntimeError("reward_model policy not available")

    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    episodes = sorted(store.list_episodes(), key=lambda e: e.episode_id)
    scores: Dict[str, RewardModelEpisodeScores] = {}
    for ep in episodes:
        econ = econ_map.get(ep.episode_id)
        if not econ:
            continue
        tags = tags_map.get(ep.episode_id, {})
        recap = recap_map.get(ep.episode_id)
        scores[ep.episode_id] = rm_policy.score_episode(ep, econ, tags=tags, recap_scores=recap)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score episodes with RewardModel.")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--semantic-tags", type=str, required=True, help="Path to Stage2.4 semantic tags JSONL.")
    parser.add_argument("--recap-scores", type=str, help="Optional RECAP scores JSONL.")
    parser.add_argument("--output-jsonl", type=str, default="results/reward_model/episode_scores.jsonl")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    tags_map = _load_jsonl_map(args.semantic_tags)
    recap_map = _load_recap_map(args.recap_scores)
    scores = _score(store, tags_map, recap_map)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ep_id in sorted(scores.keys()):
            f.write(json.dumps(scores[ep_id].to_dict(), sort_keys=True))
            f.write("\n")
    print(f"[score_episodes_with_reward_model] wrote {len(scores)} records to {out_path}")


if __name__ == "__main__":
    main()
