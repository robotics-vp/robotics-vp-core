#!/usr/bin/env python3
"""Train learned trust/econ/readiness/correction overlays from coverage graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from src.training.wrap_training_entrypoint import regal_training
from src.world_model.feedback_topology_adapters import (
    build_feedback_topology_dataset,
    train_semantic_feedback_adapter_package,
)
from src.world_model.semantic_coverage_graph import SemanticCoverageGraph


def _load_graph(path: str) -> SemanticCoverageGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SemanticCoverageGraph.from_dict(payload)


@regal_training(env_type="workcell")
def main(runner=None):
    if runner:
        runner.start_training()

    parser = argparse.ArgumentParser(description="Train semantic feedback topology adapters")
    parser.add_argument(
        "--coverage-graph",
        action="append",
        required=True,
        help="Path to a coverage_graph.json artifact. Repeat to train on multiple graphs.",
    )
    parser.add_argument("--epochs", type=int, default=32, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/semantic_feedback_adapters",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_features: List[List[float]] = []
    combined_trust: List[float] = []
    combined_econ: List[float] = []
    combined_readiness: List[float] = []
    combined_correction: List[float] = []
    feature_names: List[str] = []

    for path in args.coverage_graph:
        dataset = build_feedback_topology_dataset(_load_graph(path))
        feature_names = dataset.feature_names
        combined_features.extend(dataset.features)
        combined_trust.extend(dataset.trust_targets)
        combined_econ.extend(dataset.econ_targets)
        combined_readiness.extend(dataset.readiness_targets)
        combined_correction.extend(dataset.correction_targets)

    merged_dataset = {
        "feature_names": feature_names,
        "features": combined_features,
        "trust_targets": combined_trust,
        "econ_targets": combined_econ,
        "readiness_targets": combined_readiness,
        "correction_targets": combined_correction,
        "metadata": {"graph_count": len(args.coverage_graph)},
    }
    dataset_path = output_dir / "semantic_feedback_adapter_dataset.json"
    dataset_path.write_text(json.dumps(merged_dataset, indent=2), encoding="utf-8")

    from src.world_model.feedback_topology_adapters import FeedbackTopologyDataset

    package = train_semantic_feedback_adapter_package(
        FeedbackTopologyDataset(
            feature_names=feature_names,
            features=combined_features,
            trust_targets=combined_trust,
            econ_targets=combined_econ,
            readiness_targets=combined_readiness,
            correction_targets=combined_correction,
            metadata={"graph_count": len(args.coverage_graph)},
        ),
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    checkpoint_path = output_dir / "semantic_feedback_adapter_package.pt"
    torch.save(package.to_checkpoint(), checkpoint_path)
    summary_path = output_dir / "semantic_feedback_adapter_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "coverage_graphs": list(args.coverage_graph),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "dataset_rows": len(combined_features),
                "feature_names": feature_names,
                "checkpoint_path": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if runner:
        runner.update_step(args.epochs * max(len(combined_features), 1))


if __name__ == "__main__":
    main()
