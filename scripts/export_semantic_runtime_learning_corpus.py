#!/usr/bin/env python3
"""Export replay-backed semantic runtime learning artifacts for transformer training/inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator.meta_transformer_training import save_meta_transformer_dataset
from src.orchestrator.semantic_runtime_learning import (
    build_meta_transformer_runtime_dataset,
    build_orchestration_runtime_dataset,
    build_semantic_runtime_learning_corpus,
    write_semantic_runtime_learning_corpus,
)
from src.orchestrator.training_dataset import save_dataset
from src.replay.dataset import load_replay_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dataset", required=True, help="Path to canonical replay dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write semantic runtime learning artifacts")
    parser.add_argument("--max-counterfactuals", type=int, default=3, help="Maximum shadow counterfactuals per row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_bundle = load_replay_dataset(args.replay_dataset)
    corpus = build_semantic_runtime_learning_corpus(
        replay_bundle,
        max_counterfactuals=max(args.max_counterfactuals, 1),
    )
    written = write_semantic_runtime_learning_corpus(args.output_dir, corpus)

    output_root = Path(args.output_dir)
    meta_samples = build_meta_transformer_runtime_dataset(corpus.rows)
    meta_dataset_path = output_root / "meta_transformer_runtime_dataset.json"
    save_meta_transformer_dataset(meta_samples, str(meta_dataset_path))

    orchestration_samples = build_orchestration_runtime_dataset(corpus.rows)
    orchestration_dataset_path = output_root / "orchestration_runtime_dataset.json"
    save_dataset(orchestration_samples, str(orchestration_dataset_path))

    summary = {
        **corpus.summary,
        "rows_path": written["rows_path"],
        "summary_path": written["summary_path"],
        "meta_transformer_dataset_path": str(meta_dataset_path),
        "orchestration_dataset_path": str(orchestration_dataset_path),
    }
    summary_path = output_root / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
