#!/usr/bin/env python3
"""Train semantic feedback topology adapters with canonical runtime artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit
from src.world_model.feedback_topology_adapters import (
    FEATURE_NAMES,
    FeedbackTopologyDataset,
    build_feedback_topology_dataset,
    train_semantic_feedback_adapter_package,
)
from src.world_model.semantic_coverage_graph import SemanticCoverageGraph


FEEDBACK_ADAPTER_MIN_ROWS = 96
FEEDBACK_ADAPTER_MIN_GRAPHS = 4
FEEDBACK_ADAPTER_MIN_CORRECTION_ROWS = 16


def _load_graph(path: str) -> SemanticCoverageGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SemanticCoverageGraph.from_dict(payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    parser.add_argument("--run-name", type=str, default="semantic_feedback_adapters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_dataset(paths: Sequence[str]) -> tuple[FeedbackTopologyDataset, list[SemanticCoverageGraph]]:
    combined_features: list[list[float]] = []
    combined_trust: list[float] = []
    combined_econ: list[float] = []
    combined_readiness: list[float] = []
    combined_correction: list[float] = []
    graphs: list[SemanticCoverageGraph] = []
    feature_names: list[str] = []

    for path in paths:
        graph = _load_graph(path)
        graphs.append(graph)
        dataset = build_feedback_topology_dataset(graph)
        feature_names = dataset.feature_names
        combined_features.extend(dataset.features)
        combined_trust.extend(dataset.trust_targets)
        combined_econ.extend(dataset.econ_targets)
        combined_readiness.extend(dataset.readiness_targets)
        combined_correction.extend(dataset.correction_targets)

    merged = FeedbackTopologyDataset(
        feature_names=feature_names,
        features=combined_features,
        trust_targets=combined_trust,
        econ_targets=combined_econ,
        readiness_targets=combined_readiness,
        correction_targets=combined_correction,
        metadata={"graph_count": len(paths)},
    )
    return merged, graphs


def _build_dataset_summary(dataset: FeedbackTopologyDataset, paths: Sequence[str]) -> dict[str, Any]:
    correction_rows = sum(1 for value in dataset.correction_targets if float(value) >= 0.1)
    benchmark_gate_ready = (
        len(dataset.features) >= FEEDBACK_ADAPTER_MIN_ROWS
        and len(paths) >= FEEDBACK_ADAPTER_MIN_GRAPHS
        and correction_rows >= FEEDBACK_ADAPTER_MIN_CORRECTION_ROWS
    )
    return {
        "schema_version": "semantic_feedback_adapter_dataset_summary_v1",
        "coverage_graph_paths": [str(Path(path).resolve()) for path in paths],
        "dataset_digest": sha256_json(
            {
                "coverage_graph_paths": [str(Path(path).resolve()) for path in paths],
                "row_count": len(dataset.features),
                "graph_count": len(paths),
                "correction_rows": correction_rows,
            }
        ),
        "row_count": len(dataset.features),
        "graph_count": len(paths),
        "correction_rows": correction_rows,
        "feature_dim": len(dataset.feature_names),
        "feature_names": list(dataset.feature_names),
        "mean_targets": {
            "trust": float(np.mean(dataset.trust_targets) if dataset.trust_targets else 0.0),
            "econ": float(np.mean(dataset.econ_targets) if dataset.econ_targets else 0.0),
            "readiness": float(np.mean(dataset.readiness_targets) if dataset.readiness_targets else 0.0),
            "correction": float(np.mean(dataset.correction_targets) if dataset.correction_targets else 0.0),
        },
        "benchmark_gate": {
            "name": "semantic_feedback_adapter_coverage_density",
            "ready": benchmark_gate_ready,
            "required_rows": FEEDBACK_ADAPTER_MIN_ROWS,
            "required_graphs": FEEDBACK_ADAPTER_MIN_GRAPHS,
            "required_correction_rows": FEEDBACK_ADAPTER_MIN_CORRECTION_ROWS,
            "observed_rows": len(dataset.features),
            "observed_graphs": len(paths),
            "observed_correction_rows": correction_rows,
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::coverage_graph_paths": int(bool(dataset_summary.get("coverage_graph_paths"))),
        "dataset::non_empty": int(int(dataset_summary.get("row_count", 0)) > 0),
        "dataset::correction_signal_support": int(
            int(dataset_summary.get("correction_rows", 0))
            >= int(benchmark_gate.get("required_correction_rows", 0))
        ),
        "dataset::graph_count_support": int(
            int(dataset_summary.get("graph_count", 0))
            >= int(benchmark_gate.get("required_graphs", 0))
        ),
        "benchmark::semantic_feedback_adapter_coverage_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "semantic_feedback_adapter_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "semantic_feedback_adapter_model_config_v1",
        "input_dim": len(FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "feature_names": list(FEATURE_NAMES),
        "targets": [
            "trust_priority",
            "economic_priority",
            "promotion_readiness",
            "wm_correction_pressure",
        ],
    }


def _evaluate_train_mae(package, dataset: FeedbackTopologyDataset) -> dict[str, float]:
    inputs = torch.tensor(np.asarray(dataset.features, dtype=np.float32))
    with torch.no_grad():
        predictions = package.model(inputs)
    return {
        "trust_mae": float(np.mean(np.abs(predictions["trust"].cpu().numpy() - np.asarray(dataset.trust_targets, dtype=np.float32)))),
        "econ_mae": float(np.mean(np.abs(predictions["econ"].cpu().numpy() - np.asarray(dataset.econ_targets, dtype=np.float32)))),
        "readiness_mae": float(
            np.mean(np.abs(predictions["readiness"].cpu().numpy() - np.asarray(dataset.readiness_targets, dtype=np.float32)))
        ),
        "correction_mae": float(
            np.mean(np.abs(predictions["correction"].cpu().numpy() - np.asarray(dataset.correction_targets, dtype=np.float32)))
        ),
    }


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_summary_path: Path,
    dataset_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "semantic_feedback_adapter_runtime_package_v1",
        "package_id": f"semantic_feedback_adapter_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary_path": str(dataset_summary_path),
        "dataset_path": str(dataset_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "target_contract": "semantic_feedback_adapter_v1",
            "allowed_modes": ["disabled", "auto", "required"],
            "shadow_candidate_helper_weight": 0.18,
            "promoted_helper_weight": 0.42,
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "semantic_feedback_topology_overlay_v1",
            "routing_targets": ["coverage_loop_edge_priorities", "coverage_loop_wm_pressure"],
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    train_mae: Mapping[str, float],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "semantic_feedback_adapter_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "row_count": int(dataset_summary.get("row_count", 0)),
        "graph_count": int(dataset_summary.get("graph_count", 0)),
        "train_mae": dict(train_mae),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(graphs: Sequence[SemanticCoverageGraph]) -> list[Any]:
    audits: list[Any] = []
    for index, graph in enumerate(graphs):
        rewards = [float(getattr(edge, "promotion_readiness", 0.0)) for edge in graph.edges[:32]]
        reward_components = {
            "economic_priority": [float(getattr(edge, "economic_priority", 0.0)) for edge in graph.edges[:32]],
            "trust_priority": [float(getattr(edge, "trust_priority", 0.0)) for edge in graph.edges[:32]],
        }
        events = [f"{edge.source_id}->{edge.target_id}" for edge in graph.edges[:16]]
        audits.append(
            create_trajectory_audit(
                episode_id=f"semantic_feedback_graph_{index:03d}",
                num_steps=max(len(rewards), 1),
                rewards=rewards or [0.0],
                reward_components=reward_components,
                events=events,
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset, graphs = _build_dataset(args.coverage_graph)
    if not dataset.features:
        raise ValueError("no feedback topology samples available")

    dataset_summary = _build_dataset_summary(dataset, args.coverage_graph)
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "coverage_graphs": [str(Path(path).resolve()) for path in args.coverage_graph],
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )

    dataset_path = output_dir / "semantic_feedback_adapter_dataset.json"
    dataset_summary_path = output_dir / "semantic_feedback_adapter_dataset_summary.json"
    model_config_path = output_dir / "semantic_feedback_adapter_model_config.json"
    preconditions_path = output_dir / "semantic_feedback_adapter_execution_preconditions.json"
    training_summary_path = output_dir / "semantic_feedback_adapter_training_summary.json"
    runtime_package_path = output_dir / "semantic_feedback_adapter_runtime_package.json"
    checkpoint_path = output_dir / "semantic_feedback_adapter_package.pt"
    training_job_result_path = output_dir / "training_job_result.json"

    package = train_semantic_feedback_adapter_package(
        dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
    )
    torch.save(package.to_checkpoint(), checkpoint_path)
    train_mae = _evaluate_train_mae(package, dataset)

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "dataset": str(dataset_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "runtime_package": str(runtime_package_path),
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        train_mae=train_mae,
        artifacts=artifacts,
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_summary_path=dataset_summary_path,
        dataset_path=dataset_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )

    _write_json(dataset_path, dataset.to_dict())
    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)
    _write_json(
        training_job_result_path,
        {
            "training_kind": "semantic_feedback_adapters",
            "result": {
                "checkpoint": str(checkpoint_path),
                "dataset_summary": str(dataset_summary_path),
                "runtime_package": str(runtime_package_path),
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
            },
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = [f"coverage_graph_{index:03d}" for index in range(len(graphs))]
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for index, path in enumerate(args.coverage_graph):
            runner.record_sample("semantic_feedback_adapter", datapack_id=datapack_ids[index], slice_id=str(path))
        for audit in _build_trajectory_audits(graphs):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "row_count": int(dataset_summary.get("row_count", 0)),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="semantic_feedback_adapters",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "semantic_feedback_adapter"},
            promotion_policy_snapshot={"benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})},
            source_domain_coverage={"coverage_graph_count": int(dataset_summary.get("graph_count", 0))},
            receipt_label_coverage={"correction_rows": int(dataset_summary.get("correction_rows", 0))},
            metadata={
                "trajectory_audit_kind": "semantic_feedback_graph_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "target_contract": "semantic_feedback_adapter_v1",
            },
        )
        runner.register_artifact("semantic_feedback_adapter_dataset", dataset_path)
        runner.register_artifact("semantic_feedback_adapter_dataset_summary", dataset_summary_path)
        runner.register_artifact("semantic_feedback_adapter_model_config", model_config_path)
        runner.register_artifact("semantic_feedback_adapter_preconditions", preconditions_path)
        runner.register_artifact("semantic_feedback_adapter_training_summary", training_summary_path)
        runner.register_artifact("semantic_feedback_adapter_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="semantic_feedback_adapter_latest",
                model_family="semantic_feedback_adapter",
                model_version="semantic_feedback_adapter_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_mae": dict(train_mae),
                },
            )
        )

    return {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "dataset": str(dataset_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(runtime_package_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "semantic_feedback_adapters",
            "coverage_graphs": [str(Path(path).resolve()) for path in args.coverage_graph],
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "run_name": args.run_name,
            "seed": args.seed,
        }
    )

    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder["payload"] = _run_training(args, runner)

    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            num_episodes=max(1, len(args.coverage_graph)),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="semantic_feedback_adapters",
    )
    print(
        json.dumps(
            {
                "training_run": result.to_dict(),
                "job": holder.get("payload", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
