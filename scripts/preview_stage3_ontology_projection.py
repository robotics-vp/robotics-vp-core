#!/usr/bin/env python3
"""
Preview projection of Stage 3 episode descriptors into the ontology store.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.ontology.episode_adapters import episode_from_descriptor
from src.ontology.store import OntologyStore


def _load_descriptors(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        records = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    with path.open("r") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def main():
    parser = argparse.ArgumentParser(description="Preview Stage 3 ontology projection")
    parser.add_argument("--episode-descriptors-path", type=str, required=True)
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--robot-id", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    descriptors = _load_descriptors(Path(args.episode_descriptors_path))
    store = OntologyStore(root_dir=args.ontology_root)

    count = 0
    for desc in descriptors:
        if count >= args.limit:
            break
        episode = episode_from_descriptor(desc, task_id=args.task_id, robot_id=args.robot_id)
        store.upsert_episode(episode)
        count += 1

    episodes = store.list_episodes(task_id=args.task_id)
    print(f"[preview_stage3_ontology_projection] Projected {count} episodes into ontology for task {args.task_id}")
    print(f"{'episode_id':<24} {'task':<12} {'robot':<12} {'status':<10} {'pack_id':<18}")
    for ep in episodes[: args.limit]:
        print(
            f"{ep.episode_id[:22]:<24} {ep.task_id:<12} {ep.robot_id:<12} "
            f"{ep.status:<10} {str(ep.datapack_id or '')[:16]:<18}"
        )


if __name__ == "__main__":
    main()
