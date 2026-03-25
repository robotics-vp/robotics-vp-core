#!/usr/bin/env python3
"""Train the learned semantic WM refiner from coverage-loop artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.training.wrap_training_entrypoint import regal_training
from src.world_model.semantic_wm_refiner import (
    build_semantic_wm_refinement_dataset_from_artifact_dirs,
    train_semantic_wm_refiner_package,
)


@regal_training(env_type="workcell")
def main(runner=None):
    if runner:
        runner.start_training()

    parser = argparse.ArgumentParser(description="Train semantic WM refiner")
    parser.add_argument(
        "--artifact-dir",
        action="append",
        required=True,
        help="Path to a coverage-loop artifact directory. Repeat to train on multiple runs.",
    )
    parser.add_argument("--epochs", type=int, default=32, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/semantic_wm_refiner",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_semantic_wm_refinement_dataset_from_artifact_dirs(args.artifact_dir)
    dataset_path = output_dir / "semantic_wm_refiner_dataset.json"
    dataset_path.write_text(json.dumps(dataset.to_dict(), indent=2), encoding="utf-8")

    package = train_semantic_wm_refiner_package(
        dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    checkpoint_path = output_dir / "semantic_wm_refiner_package.pt"
    torch.save(package.to_checkpoint(), checkpoint_path)

    summary_path = output_dir / "semantic_wm_refiner_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "artifact_dirs": list(args.artifact_dir),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "object_rows": len(dataset.object_features),
                "relation_rows": len(dataset.relation_features),
                "capability_rows": len(dataset.capability_features),
                "proposal_rows": len(dataset.proposal_features),
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if runner:
        runner.update_step(
            int(args.epochs) * max(
                len(dataset.object_features)
                + len(dataset.relation_features)
                + len(dataset.proposal_features),
                1,
            )
        )


if __name__ == "__main__":
    main()
