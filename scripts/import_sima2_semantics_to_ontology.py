#!/usr/bin/env python3
"""
Import SIMA-2 rollouts/tags into the OntologyStore as datapacks.
"""
import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.ontology.models import Task
from src.ontology.sima2_adapters import datapack_from_sima2_rollout, datapack_from_sima2_tags


def _load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Import SIMA-2 semantics into OntologyStore.")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--rollouts-path", type=str, required=True)
    parser.add_argument("--tags-path", type=str, required=True)
    args = parser.parse_args()

    store = OntologyStore(root_dir=args.ontology_root)
    task = store.get_task(args.task_id) or Task(task_id=args.task_id, name=args.task_id)
    store.upsert_task(task)

    rollouts = _load_jsonl(Path(args.rollouts_path))
    tags = _load_jsonl(Path(args.tags_path))

    datapacks = []
    for r in rollouts:
        datapacks.append(datapack_from_sima2_rollout(r, args.task_id))
    for t in tags:
        datapacks.append(datapack_from_sima2_tags(t, args.task_id))
    if datapacks:
        store.append_datapacks(datapacks)
    print(f"[import_sima2_semantics_to_ontology] Imported {len(datapacks)} datapacks into {args.ontology_root}")


if __name__ == "__main__":
    main()
