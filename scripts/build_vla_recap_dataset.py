#!/usr/bin/env python3
"""
CLI wrapper to build RECAP dataset from ontology store.
"""
import argparse
from pathlib import Path

from src.vla.recap_dataset_builder import build_recap_dataset
from src.ontology.store import OntologyStore


def main():
    parser = argparse.ArgumentParser(description="Build VLA RECAP dataset from ontology")
    parser.add_argument("--ontology-root", type=str, default="data/ontology")
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--max-episodes", type=int, default=1000)
    args = parser.parse_args()

    output_path = args.output_path or f"results/recap/recap_dataset_{args.task_id}.jsonl"
    store = OntologyStore(root_dir=args.ontology_root)
    build_recap_dataset(store, task_id=args.task_id, output_path=output_path, max_episodes=args.max_episodes)
    print(f"[build_vla_recap_dataset] Wrote dataset to {output_path}")


if __name__ == "__main__":
    main()
