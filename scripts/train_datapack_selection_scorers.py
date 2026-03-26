#!/usr/bin/env python3
"""Train semantic datapack-selection scorer packages from semantic run logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator.datapack_selection_training import (
    DatapackSelectionTrainingDataset,
    DatapackSelectionTrainingExample,
    build_datapack_selection_training_dataset,
    load_selection_run_logs,
    train_datapack_selection_scorer_package,
    write_datapack_selection_scorer_package,
    write_datapack_selection_training_dataset,
)
from src.orchestrator.semantic_policy import (
    DatapackSelectionContext,
    DatapackSelectionFeatures,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-log",
        dest="run_logs",
        action="append",
        default=[],
        help="Path to a semantic run log JSONL file; may be provided multiple times",
    )
    parser.add_argument(
        "--run-log-dir",
        dest="run_log_dirs",
        action="append",
        default=[],
        help="Directory containing semantic run log JSONL files; may be provided multiple times",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/semantic_selection",
        help="Directory for scorer artifacts",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="semantic_datapack_selection",
        help="Training run name prefix",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_log_paths(args: argparse.Namespace) -> list[Path]:
    explicit_paths = [Path(path) for path in list(args.run_logs or []) if path]
    dir_paths = [Path(path) for path in list(args.run_log_dirs or []) if path]
    resolved: list[Path] = []
    for path in explicit_paths:
        if path.is_dir():
            resolved.extend(sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file()))
        else:
            resolved.append(path)
    for directory in dir_paths:
        if not directory.exists():
            raise FileNotFoundError(f"selection run log directory not found: {directory}")
        resolved.extend(sorted(candidate for candidate in directory.glob("*.jsonl") if candidate.is_file()))
    if not resolved:
        default_path = Path("data/logs/semantic_runs.jsonl")
        resolved = [default_path]
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def _build_dataset_summary(
    *,
    log_paths: Sequence[Path],
    dataset: DatapackSelectionTrainingDataset,
) -> dict[str, Any]:
    feature_names = sorted(DatapackSelectionFeatures().to_dict().keys())
    context_feature_names = sorted(DatapackSelectionContext().to_dict().keys())
    datapack_counts = Counter(example.datapack_id for example in dataset.examples)
    selected_count = sum(1 for example in dataset.examples if example.selected)
    pairwise_count = sum(
        1
        for example in dataset.examples
        if example.supervision_kind == "selected_vs_alternative_pairwise"
    )
    regression_count = sum(
        1
        for example in dataset.examples
        if example.supervision_kind == "selected_outcome_regression"
    )
    return {
        "schema_version": "datapack_selection_dataset_summary_v1",
        "dataset_kind": "semantic_selection_run_logs",
        "run_log_paths": [str(path.resolve()) for path in log_paths],
        "dataset_digest": dataset.summary.get("dataset_digest"),
        "num_logs": len(log_paths),
        "num_runs": int(dataset.summary.get("num_runs", 0)),
        "num_examples": len(dataset.examples),
        "num_selected_examples": selected_count,
        "num_pairwise_examples": pairwise_count,
        "num_regression_examples": regression_count,
        "feature_contract": {
            "feature_names": feature_names,
            "context_feature_names": context_feature_names,
            "scoring_contract": "neural_feature_mlp_with_context_conditioned_adjustment_v2",
        },
        "selection_policy_counts": dict(dataset.summary.get("selection_policy_counts", {}) or {}),
        "promotion_stage_counts": dict(dataset.summary.get("promotion_stage_counts", {}) or {}),
        "outcome_score_summary": dict(dataset.summary.get("outcome_score_summary", {}) or {}),
        "top_datapacks": [
            {"datapack_id": datapack_id, "count": int(count)}
            for datapack_id, count in datapack_counts.most_common(10)
        ],
        "benchmark_gate": dict(dataset.summary.get("benchmark_gate", {}) or {}),
    }


def _build_model_config(dataset: DatapackSelectionTrainingDataset) -> dict[str, Any]:
    return {
        "schema_version": "datapack_selection_model_config_v1",
        "feature_names": sorted(DatapackSelectionFeatures().to_dict().keys()),
        "context_feature_names": sorted(DatapackSelectionContext().to_dict().keys()),
        "supervision_mode": dataset.summary.get("supervision_mode"),
        "selection_context_contract": dict(dataset.summary.get("selection_context_contract", {}) or {}),
        "scoring_contract": "neural_feature_mlp_with_context_conditioned_adjustment_v2",
    }


def _build_execution_preconditions(
    *,
    log_paths: Sequence[Path],
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    pairwise_examples = int(dataset_summary.get("num_pairwise_examples", 0))
    selected_examples = int(dataset_summary.get("num_selected_examples", 0))
    satisfied = {
        "artifact::run_logs_present": int(all(path.exists() for path in log_paths)),
        "dataset::non_empty": int(int(dataset_summary.get("num_examples", 0)) > 0),
        "dataset::selected_examples_present": int(selected_examples > 0),
        "dataset::pairwise_examples_present": int(pairwise_examples > 0),
        "benchmark::datapack_selection_min_runs": int(
            bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready"))
        ),
    }
    return {
        "schema_version": "datapack_selection_execution_preconditions_v1",
        "run_log_paths": [str(path.resolve()) for path in log_paths],
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [
            key for key, value in sorted(satisfied.items()) if not value
        ],
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    scorer_package: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(scorer_package.get("metadata", {}) or {})
    return {
        "schema_version": "datapack_selection_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "num_logs": int(dataset_summary.get("num_logs", 0)),
        "num_runs": int(dataset_summary.get("num_runs", 0)),
        "num_examples": int(dataset_summary.get("num_examples", 0)),
        "selection_policy_counts": dict(dataset_summary.get("selection_policy_counts", {}) or {}),
        "promotion_stage_counts": dict(dataset_summary.get("promotion_stage_counts", {}) or {}),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "package_summary": {
            "package_id": scorer_package.get("package_id"),
            "schema_version": scorer_package.get("schema_version"),
            "model_kind": scorer_package.get("model_kind"),
            "max_adjustment": float(scorer_package.get("max_adjustment", 0.0) or 0.0),
            "min_adjustment": float(scorer_package.get("min_adjustment", 0.0) or 0.0),
            "conditioning_contract": metadata.get("conditioning_contract"),
            "future_conditioning_path": metadata.get("future_conditioning_path"),
            "neural_training_summary": dict(metadata.get("neural_training_summary", {}) or {}),
        },
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(
    examples: Sequence[DatapackSelectionTrainingExample],
) -> list[Any]:
    rows_by_run: dict[str, list[DatapackSelectionTrainingExample]] = defaultdict(list)
    for example in examples:
        rows_by_run[str(example.run_id)].append(example)
    audits: list[Any] = []
    for run_id, run_examples in sorted(rows_by_run.items()):
        rewards = [float(example.target_score) for example in run_examples]
        reward_components = {
            "selected": [1.0 if example.selected else 0.0 for example in run_examples],
            "outcome_score": [float(example.outcome_score) for example in run_examples],
        }
        events = [
            f"{example.supervision_kind}:{example.datapack_id}"
            for example in run_examples
        ]
        audits.append(
            create_trajectory_audit(
                episode_id=run_id,
                num_steps=len(run_examples),
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
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log_paths = _resolve_log_paths(args)
    run_rows = load_selection_run_logs(log_paths)
    dataset = build_datapack_selection_training_dataset(run_rows)
    scorer_package = train_datapack_selection_scorer_package(dataset)

    dataset_path = output_root / "datapack_selection_training_dataset.json"
    dataset_summary_path = output_root / "datapack_selection_dataset_summary.json"
    model_config_path = output_root / "datapack_selection_model_config.json"
    preconditions_path = output_root / "datapack_selection_execution_preconditions.json"
    scorer_package_path = output_root / "datapack_selection_scorer_package.json"
    training_summary_path = output_root / "datapack_selection_training_summary.json"
    training_job_result_path = output_root / "training_job_result.json"

    dataset_summary = _build_dataset_summary(log_paths=log_paths, dataset=dataset)
    model_config = _build_model_config(dataset)
    execution_preconditions = _build_execution_preconditions(
        log_paths=log_paths,
        dataset_summary=dataset_summary,
    )
    config_digest = sha256_json(
        {
            "run_logs": [str(path.resolve()) for path in log_paths],
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "seed": int(args.seed),
            "run_name": str(args.run_name),
            "scoring_contract": model_config.get("scoring_contract"),
        }
    )

    dataset_ref = write_datapack_selection_training_dataset(dataset_path, dataset)
    scorer_package_ref = write_datapack_selection_scorer_package(
        scorer_package_path,
        scorer_package,
    )
    artifacts = {
        "training_dataset": dataset_ref,
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "execution_preconditions": str(preconditions_path),
        "scorer_package": scorer_package_ref,
        "run_logs": [str(path.resolve()) for path in log_paths],
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        scorer_package=scorer_package.to_dict(),
        artifacts=artifacts,
    )

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)

    result = {
        "dataset": dataset_ref,
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "scorer_package": scorer_package_ref,
        "training_summary": str(training_summary_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "semantic_datapack_selection",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = sorted(
            {str(example.datapack_id) for example in dataset.examples if str(example.datapack_id)}
        )
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for example in dataset.examples:
            runner.record_sample(
                "semantic_datapack_selection",
                datapack_id=str(example.datapack_id),
                slice_id=f"{example.run_id}:{example.datapack_id}:{example.supervision_kind}",
            )
        for audit in _build_trajectory_audits(dataset.examples):
            runner.add_trajectory_audit(audit)
        runner.update_step(len(dataset.examples))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "num_examples": len(dataset.examples),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="semantic_datapack_selection",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "semantic_datapack_selection"},
            promotion_policy_snapshot={},
            source_domain_coverage={
                "dataset_kind": "semantic_selection_run_logs",
                "run_log_paths": [str(path.resolve()) for path in log_paths],
            },
            receipt_label_coverage={
                "selection_runs": int(dataset.summary.get("num_runs", 0)),
                "selection_examples": len(dataset.examples),
            },
            metadata={
                "trajectory_audit_kind": "selection_receipt_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "selection_policy_counts": dict(dataset.summary.get("selection_policy_counts", {}) or {}),
                "promotion_stage_counts": dict(dataset.summary.get("promotion_stage_counts", {}) or {}),
            },
        )
        runner.register_artifact("datapack_selection_training_dataset", dataset_path)
        runner.register_artifact("datapack_selection_dataset_summary", dataset_summary_path)
        runner.register_artifact("datapack_selection_model_config", model_config_path)
        runner.register_artifact("datapack_selection_preconditions", preconditions_path)
        runner.register_artifact("datapack_selection_scorer_package", scorer_package_path)
        runner.register_artifact("datapack_selection_training_summary", training_summary_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="semantic_datapack_selection_helper",
                model_family="semantic_datapack_selection",
                model_version="datapack_selection_helper_v1",
                path=scorer_package_path,
                step=len(dataset.examples),
                epoch=1,
                is_best=True,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                },
            )
        )
    return result


def _run_training(
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "semantic_datapack_selection",
            "run_logs": list(args.run_logs or []),
            "run_log_dirs": list(args.run_log_dirs or []),
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
            num_episodes=max(1, len(args.run_logs or []) + len(args.run_log_dirs or [])),
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="semantic_datapack_selection",
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
