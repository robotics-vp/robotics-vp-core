#!/usr/bin/env python3
"""Train the learned fill-path policy with canonical runtime/package artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit
from src.world_model.fill_outcome_store import FillOutcomeRecord, FillOutcomeStore
from src.world_model.fill_path_policy import FILL_METHODS, train_fill_path_policy
from src.world_model.gap_ranker import GapFeatureExtractor

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - explicit failure below
    torch = None
    F = None


FILL_PATH_BENCHMARK_MIN_RECORDS = 200
FILL_PATH_BENCHMARK_MIN_LABELED_EDGES = 50
FILL_PATH_BENCHMARK_MIN_POSITIVE_DELTA = 50
FILL_PATH_BENCHMARK_MIN_METHODS = 3


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcome-store",
        type=str,
        default="data/fill_outcomes.jsonl",
        help="Path to fill-outcome JSONL store",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints/fill_path_policy")
    parser.add_argument("--run-name", type=str, default="fill_path_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _group_records_by_edge(records: Sequence[FillOutcomeRecord]) -> dict[str, list[FillOutcomeRecord]]:
    grouped: dict[str, list[FillOutcomeRecord]] = {}
    for record in records:
        grouped.setdefault(str(record.edge_key), []).append(record)
    return grouped


def _best_method_for_records(records: Sequence[FillOutcomeRecord]) -> str:
    by_method: dict[str, list[float]] = {}
    for record in records:
        by_method.setdefault(str(record.fill_method), []).append(float(record.marginal_value))
    best_method = "blocked"
    best_value = -float("inf")
    for method, values in by_method.items():
        mean_value = sum(values) / max(len(values), 1)
        if mean_value > best_value:
            best_value = mean_value
            best_method = method
    return best_method


def _build_dataset_summary(records: Sequence[FillOutcomeRecord], store: FillOutcomeStore) -> dict[str, Any]:
    store_summary = store.summary()
    grouped = _group_records_by_edge(records)
    positive_delta_count = sum(1 for record in records if float(record.coverage_delta) > 0.0)
    method_counts = Counter(record.fill_method for record in records)
    label_method_counts = Counter(
        _best_method_for_records(edge_records)
        for edge_records in grouped.values()
        if _best_method_for_records(edge_records) in FILL_METHODS
    )
    benchmark_gate_ready = (
        len(records) >= FILL_PATH_BENCHMARK_MIN_RECORDS
        and len(grouped) >= FILL_PATH_BENCHMARK_MIN_LABELED_EDGES
        and positive_delta_count >= FILL_PATH_BENCHMARK_MIN_POSITIVE_DELTA
        and len(label_method_counts) >= FILL_PATH_BENCHMARK_MIN_METHODS
    )
    return {
        "schema_version": "fill_path_policy_dataset_summary_v1",
        "outcome_store_path": str(store.path.resolve()),
        "dataset_digest": sha256_json(
            {
                "outcome_store": str(store.path.resolve()),
                "total_records": len(records),
                "labeled_edges": len(grouped),
                "positive_delta_count": positive_delta_count,
                "method_counts": dict(sorted(method_counts.items())),
                "label_method_counts": dict(sorted(label_method_counts.items())),
            }
        ),
        "total_records": len(records),
        "labeled_edges": len(grouped),
        "positive_delta_count": positive_delta_count,
        "method_counts": dict(sorted(method_counts.items())),
        "label_method_counts": dict(sorted(label_method_counts.items())),
        "store_summary": store_summary,
        "benchmark_gate": {
            "name": "fill_path_policy_fill_outcome_density",
            "ready": benchmark_gate_ready,
            "required_records": FILL_PATH_BENCHMARK_MIN_RECORDS,
            "required_labeled_edges": FILL_PATH_BENCHMARK_MIN_LABELED_EDGES,
            "required_positive_delta_count": FILL_PATH_BENCHMARK_MIN_POSITIVE_DELTA,
            "required_distinct_label_methods": FILL_PATH_BENCHMARK_MIN_METHODS,
            "observed_records": len(records),
            "observed_labeled_edges": len(grouped),
            "observed_positive_delta_count": positive_delta_count,
            "observed_distinct_label_methods": len(label_method_counts),
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::outcome_store_present": int(bool(dataset_summary.get("outcome_store_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("total_records", 0)) > 0),
        "dataset::labeled_edges_present": int(int(dataset_summary.get("labeled_edges", 0)) > 0),
        "dataset::positive_delta_support": int(
            int(dataset_summary.get("positive_delta_count", 0))
            >= int(benchmark_gate.get("required_positive_delta_count", 0))
        ),
        "dataset::label_method_diversity": int(
            int(benchmark_gate.get("observed_distinct_label_methods", 0))
            >= int(benchmark_gate.get("required_distinct_label_methods", 0))
        ),
        "benchmark::fill_path_policy_fill_outcome_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "fill_path_policy_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "fill_path_policy_model_config_v1",
        "input_dim": GapFeatureExtractor.FEATURE_DIM,
        "hidden_dim": int(hidden_dim),
        "fill_methods": list(FILL_METHODS),
        "target_contract": "best_fill_method_classification_v1",
    }


def _evaluate_train_accuracy(model: Any, records: Sequence[FillOutcomeRecord]) -> float:
    if torch is None or F is None:
        return 0.0
    grouped = _group_records_by_edge(records)
    extractor = GapFeatureExtractor()
    features = []
    targets = []
    for edge_records in grouped.values():
        label = _best_method_for_records(edge_records)
        if label not in FILL_METHODS:
            continue
        features.append(extractor.from_outcome_record(edge_records[-1]).raw)
        targets.append(FILL_METHODS.index(label))
    if not features:
        return 0.0
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(np.array(features, dtype=np.float32)))
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1).tolist()
    correct = sum(1 for pred, target in zip(predictions, targets) if int(pred) == int(target))
    return float(correct / max(len(targets), 1))


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_summary_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "fill_path_policy_runtime_package_v1",
        "package_id": f"fill_path_policy_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary_path": str(dataset_summary_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "helper_blend_policy": "bounded_fill_path_helper_v1",
            "allowed_modes": ["disabled", "auto", "required"],
            "conditioning_contract": "fill_path_routing_trace_v1",
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "best_fill_method_classification_v1",
            "routing_targets": ["coverage_loop_fill_decisions", "fill_outcome_runtime_traces"],
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    train_accuracy: float,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "fill_path_policy_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "total_records": int(dataset_summary.get("total_records", 0)),
        "labeled_edges": int(dataset_summary.get("labeled_edges", 0)),
        "train_accuracy": float(train_accuracy),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(records: Sequence[FillOutcomeRecord]) -> list[Any]:
    audits: list[Any] = []
    for edge_key, edge_records in sorted(_group_records_by_edge(records).items()):
        best_method = _best_method_for_records(edge_records)
        rewards = [float(record.marginal_value) for record in edge_records]
        reward_components = {
            "coverage_delta": [float(record.coverage_delta) for record in edge_records],
            "quality_score": [float(record.quality_score) for record in edge_records],
            "wall_time_s": [float(record.wall_time_s) for record in edge_records],
        }
        events = [f"{record.fill_method}:{best_method}" for record in edge_records]
        audits.append(
            create_trajectory_audit(
                episode_id=edge_key,
                num_steps=len(edge_records),
                rewards=rewards,
                reward_components=reward_components,
                events=events,
            )
        )
    return audits


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to train the fill-path policy")

    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    store = FillOutcomeStore(args.outcome_store)
    records = store.load_all()
    if not records:
        raise ValueError(f"No records in {args.outcome_store}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_summary = _build_dataset_summary(records, store)
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "run_name": args.run_name,
        }
    )

    checkpoint_path = output_root / "fill_path_policy.pt"
    dataset_summary_path = output_root / "fill_path_policy_dataset_summary.json"
    model_config_path = output_root / "fill_path_policy_model_config.json"
    preconditions_path = output_root / "fill_path_policy_execution_preconditions.json"
    training_summary_path = output_root / "fill_path_policy_training_summary.json"
    runtime_package_path = output_root / "fill_path_policy_package.json"
    training_job_result_path = output_root / "training_job_result.json"

    model = train_fill_path_policy(
        records,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=str(checkpoint_path),
    )
    train_accuracy = _evaluate_train_accuracy(model, records)

    artifacts = {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "runtime_package": str(runtime_package_path),
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        train_accuracy=train_accuracy,
        artifacts=artifacts,
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_summary_path=dataset_summary_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(runtime_package_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "fill_path_policy",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = [record.edge_key for record in records]
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for record in records:
            runner.record_sample(
                record.fill_method,
                datapack_id=str(record.edge_key),
                slice_id=f"{record.fill_method}:{record.edge_key}",
            )
        for audit in _build_trajectory_audits(records):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "total_records": len(records),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="fill_path_policy",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "fill_path_policy"},
            promotion_policy_snapshot={},
            source_domain_coverage={"fill_methods": dict(dataset_summary.get("method_counts", {}) or {})},
            receipt_label_coverage={"labeled_edges": int(dataset_summary.get("labeled_edges", 0))},
            metadata={
                "trajectory_audit_kind": "fill_path_policy_fill_outcome_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "target_contract": "best_fill_method_classification_v1",
            },
        )
        runner.register_artifact("fill_path_policy_dataset_summary", dataset_summary_path)
        runner.register_artifact("fill_path_policy_model_config", model_config_path)
        runner.register_artifact("fill_path_policy_preconditions", preconditions_path)
        runner.register_artifact("fill_path_policy_training_summary", training_summary_path)
        runner.register_artifact("fill_path_policy_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="fill_path_policy_latest",
                model_family="fill_path_policy",
                model_version="fill_path_policy_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_accuracy": train_accuracy,
                },
            )
        )
    return result


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "fill_path_policy",
            "outcome_store": args.outcome_store,
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
            output_dir=args.save_dir,
            seed=args.seed,
            num_episodes=max(1, FillOutcomeStore(args.outcome_store).record_count()),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="fill_path_policy",
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
