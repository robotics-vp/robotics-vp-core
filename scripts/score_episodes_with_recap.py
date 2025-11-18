#!/usr/bin/env python3
"""
Score ontology episodes with RECAP heads (inference-only).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

repo_root = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.vla.recap_inference import load_recap_heads, compute_recap_scores


def _load_recap_scores(store: OntologyStore, bundle, task_id: Optional[str] = None) -> Dict[str, Dict]:
    episodes = store.list_episodes(task_id=task_id)
    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    scores = {}
    for ep in sorted(episodes, key=lambda e: e.episode_id):
        events = store.get_events(ep.episode_id)
        econ = econ_map.get(ep.episode_id)
        if not events or not econ:
            continue
        score = compute_recap_scores(bundle, events, econ, metadata=getattr(ep, "metadata", {}) or {})
        scores[ep.episode_id] = score.to_dict()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Score ontology episodes with RECAP heads.")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RECAP checkpoint (.pt)")
    parser.add_argument("--task-id", type=str, help="Optional task filter.")
    parser.add_argument("--output-jsonl", type=str, default="results/recap/episode_scores.jsonl")
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    bundle = load_recap_heads(args.checkpoint, device="cpu")
    scores = _load_recap_scores(store, bundle, task_id=args.task_id)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ep_id in sorted(scores.keys()):
            f.write(json.dumps(scores[ep_id], sort_keys=True))
            f.write("\n")
    goodness = [s.get("recap_goodness_score", 0.0) for s in scores.values()]
    count = len(goodness)
    mean_goodness = sum(goodness) / count if count else 0.0
    print(f"[score_episodes_with_recap] Scored {count} episodes. Mean recap_goodness_score={mean_goodness:.4f}")


if __name__ == "__main__":
    main()
