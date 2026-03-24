#!/usr/bin/env python3
"""Train semantic runtime scorers from replay-backed semantic runtime rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.orchestrator.semantic_runtime_learning import build_semantic_runtime_learning_corpus
from src.orchestrator.semantic_runtime_scorer_training import (
    TORCH_AVAILABLE,
    build_semantic_runtime_scorer_training_dataset,
    save_semantic_runtime_scorer_checkpoint,
    train_semantic_runtime_scorer_net,
    write_semantic_runtime_scorer_training_dataset,
)
from src.orchestrator.semantic_runtime_scorers import (
    score_semantic_runtime_learning_row,
    train_semantic_runtime_scorer_package,
    write_semantic_runtime_scorer_package,
)
from src.replay.dataset import load_replay_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dataset", required=True, help="Path to canonical replay dataset directory")
    parser.add_argument("--output-dir", required=True, help="Directory to write scorer artifacts")
    parser.add_argument("--max-counterfactuals", type=int, default=3, help="Maximum shadow counterfactuals per row")
    parser.add_argument(
        "--trainer",
        choices=["linear", "torch", "both"],
        default="both",
        help="Which training artifacts to emit",
    )
    parser.add_argument("--epochs", type=int, default=24, help="Epochs for the torch scorer trainer")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for the torch scorer trainer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_bundle = load_replay_dataset(args.replay_dataset)
    corpus = build_semantic_runtime_learning_corpus(
        replay_bundle,
        max_counterfactuals=max(args.max_counterfactuals, 1),
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    training_dataset = build_semantic_runtime_scorer_training_dataset(corpus)
    training_dataset_path = write_semantic_runtime_scorer_training_dataset(
        output_root / "semantic_runtime_scorer_training_dataset.json",
        training_dataset,
    )

    package_path = ""
    scorer_package = None
    if args.trainer in {"linear", "both"}:
        scorer_package = train_semantic_runtime_scorer_package(corpus)
        package_path = write_semantic_runtime_scorer_package(
            output_root / "semantic_runtime_scorer_package.json",
            scorer_package,
        )

    scores_path = output_root / "semantic_runtime_shadow_scores.jsonl"
    if scorer_package is not None:
        with scores_path.open("w", encoding="utf-8") as handle:
            for row in corpus.rows:
                handle.write(json.dumps(score_semantic_runtime_learning_row(scorer_package, row).to_dict(), sort_keys=True) + "\n")

    torch_summary = {
        "torch_available": TORCH_AVAILABLE,
        "trained": False,
    }
    torch_checkpoint_path = None
    if args.trainer in {"torch", "both"}:
        torch_result = train_semantic_runtime_scorer_net(
            training_dataset,
            epochs=max(args.epochs, 1),
            hidden_dim=max(args.hidden_dim, 8),
        )
        torch_summary = {
            key: value
            for key, value in dict(torch_result.get("summary", torch_result)).items()
            if key != "model"
        }
        torch_checkpoint_path = save_semantic_runtime_scorer_checkpoint(
            output_root / "semantic_runtime_scorer_model.pt",
            torch_result,
        )

    summary = {
        **corpus.summary,
        "scorer_package_path": package_path,
        "shadow_scores_path": str(scores_path) if scorer_package is not None else "",
        "training_dataset_path": training_dataset_path,
        "scorer_summary": scorer_package.summary if scorer_package is not None else {},
        "torch_trainer_summary": torch_summary,
        "torch_checkpoint_path": torch_checkpoint_path,
    }
    summary_path = output_root / "semantic_runtime_scorer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
