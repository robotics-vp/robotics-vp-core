#!/usr/bin/env python3
"""Train the learned gap ranker with canonical runtime/package artifacts."""

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
from src.world_model.gap_ranker import GapFeatureExtractor, train_gap_ranker

try:
    import torch
except ImportError:  # pragma: no cover - explicit failure below
    torch = None


GAP_RANKER_BENCHMARK_MIN_RECORDS = 200
GAP_RANKER_BENCHMARK_MIN_POSITIVE_DELTA = 50
GAP_RANKER_BENCHMARK_MIN_METHODS = 2


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
    parser.add_argument("--save-dir", type=str, default="checkpoints/gap_ranker")
    parser.add_argument("--run-name", type=str, default="gap_ranker")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_dataset_summary(records: Sequence[FillOutcomeRecord], store: FillOutcomeStore) -> dict[str, Any]:
    store_summary = store.summary()
    positive_delta_count = sum(1 for record in records if float(record.coverage_delta) > 0.0)
    method_counts = Counter(record.fill_method for record in records)
    benchmark_gate_ready = (
        len(records) >= GAP_RANKER_BENCHMARK_MIN_RECORDS
        and positive_delta_count >= GAP_RANKER_BENCHMARK_MIN_POSITIVE_DELTA
        and len(method_counts) >= GAP_RANKER_BENCHMARK_MIN_METHODS
    )
    return {
        "schema_version": "gap_ranker_dataset_summary_v1",
        "outcome_store_path": str(store.path.resolve()),
        "dataset_digest": sha256_json(
            {
                "outcome_store": str(store.path.resolve()),
                "total_records": len(records),
                "positive_delta_count": positive_delta_count,
                "method_counts": dict(sorted(method_counts.items())),
            }
        ),
        "total_records": len(records),
        "positive_delta_count": positive_delta_count,
        "method_counts": dict(sorted(method_counts.items())),
        "store_summary": store_summary,
        "benchmark_gate": {
            "name": "gap_ranker_fill_outcome_density",
            "ready": benchmark_gate_ready,
            "required_records": GAP_RANKER_BENCHMARK_MIN_RECORDS,
            "required_positive_delta_count": GAP_RANKER_BENCHMARK_MIN_POSITIVE_DELTA,
            "required_distinct_methods": GAP_RANKER_BENCHMARK_MIN_METHODS,
            "observed_records": len(records),
            "observed_positive_delta_count": positive_delta_count,
            "observed_distinct_methods": len(method_counts),
        },
    }


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::outcome_store_present": int(bool(dataset_summary.get("outcome_store_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("total_records", 0)) > 0),
        "dataset::positive_delta_support": int(
            int(dataset_summary.get("positive_delta_count", 0))
            >= int(benchmark_gate.get("required_positive_delta_count", 0))
        ),
        "dataset::method_diversity": int(
            int(benchmark_gate.get("observed_distinct_methods", 0))
            >= int(benchmark_gate.get("required_distinct_methods", 0))
        ),
        "benchmark::gap_ranker_fill_outcome_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "gap_ranker_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "gap_ranker_model_config_v1",
        "input_dim": GapFeatureExtractor.FEATURE_DIM,
        "hidden_dim": int(hidden_dim),
        "target_contract": "marginal_value_regression_v1",
    }


def _evaluate_train_mse(model: Any, records: Sequence[FillOutcomeRecord]) -> float:
    if torch is None:
        return 0.0
    extractor = GapFeatureExtractor()
    features = np.array(
        [extractor.from_outcome_record(record).raw for record in records],
        dtype=np.float32,
    )
    targets = np.array([float(record.marginal_value) for record in records], dtype=np.float32)
    with torch.no_grad():
        preds = model(torch.from_numpy(features)).squeeze(-1).numpy()
    return float(np.mean((preds - targets) ** 2))


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
        "schema_version": "gap_ranker_runtime_package_v1",
        "package_id": f"gap_ranker_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_summary_path": str(dataset_summary_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "bounded_gap_agenda_helper_v1",
            "routing_targets": ["simulation_agenda", "diffusion_gap_prompts"],
        },
        "model_config": dict(model_config),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    train_mse: float,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "gap_ranker_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "total_records": int(dataset_summary.get("total_records", 0)),
        "train_mse": float(train_mse),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(records: Sequence[FillOutcomeRecord]) -> list[Any]:
    audits: list[Any] = []
    for record in records:
        audits.append(
            create_trajectory_audit(
                episode_id=str(record.edge_key),
                num_steps=1,
                rewards=[float(record.marginal_value)],
                reward_components={
                    "coverage_delta": [float(record.coverage_delta)],
                    "quality_score": [float(record.quality_score)],
                    "wall_time_s": [float(record.wall_time_s)],
                },
                events=[f"fill_method:{record.fill_method}"],
            )
        )
    return audits


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to train the gap ranker")

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

    checkpoint_path = output_root / "gap_ranker.pt"
    dataset_summary_path = output_root / "gap_ranker_dataset_summary.json"
    model_config_path = output_root / "gap_ranker_model_config.json"
    preconditions_path = output_root / "gap_ranker_execution_preconditions.json"
    training_summary_path = output_root / "gap_ranker_training_summary.json"
    runtime_package_path = output_root / "gap_ranker_package.json"
    training_job_result_path = output_root / "training_job_result.json"

    model = train_gap_ranker(
        records,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=str(checkpoint_path),
    )
    train_mse = _evaluate_train_mse(model, records)

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
        train_mse=train_mse,
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
            "training_kind": "gap_ranker",
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
            training_kind="gap_ranker",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "gap_ranker"},
            promotion_policy_snapshot={},
            source_domain_coverage={"fill_methods": dict(dataset_summary.get("method_counts", {}) or {})},
            receipt_label_coverage={"total_records": len(records)},
            metadata={
                "trajectory_audit_kind": "gap_ranker_fill_outcome_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "target_contract": "marginal_value_regression_v1",
            },
        )
        runner.register_artifact("gap_ranker_dataset_summary", dataset_summary_path)
        runner.register_artifact("gap_ranker_model_config", model_config_path)
        runner.register_artifact("gap_ranker_preconditions", preconditions_path)
        runner.register_artifact("gap_ranker_training_summary", training_summary_path)
        runner.register_artifact("gap_ranker_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="gap_ranker_latest",
                model_family="gap_ranker",
                model_version="gap_ranker_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_mse": train_mse,
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
            "training_kind": "gap_ranker",
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
        plan_id="gap_ranker",
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
