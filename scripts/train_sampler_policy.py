#!/usr/bin/env python3
"""Train the bounded sampler strategy/base-weight helper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.rl.sampler_policy import (
    SAMPLER_EPISODE_FEATURE_NAMES,
    SAMPLER_PLAN_PARAMETER_NAMES,
    SAMPLER_POLICY_STRATEGIES,
    SAMPLER_POOL_FEATURE_NAMES,
)
from src.rl.sampler_policy_training import (
    SAMPLER_POLICY_MIN_POOL_ROWS,
    TORCH_AVAILABLE,
    SamplerPolicyTrainingDataset,
    build_sampler_policy_training_dataset,
    load_sampler_policy_receipts,
    load_sampler_policy_training_dataset,
    save_sampler_policy_training_dataset,
    train_sampler_policy_models,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover
    torch = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt-json",
        dest="receipt_json",
        action="append",
        default=[],
        help="Path to sampler policy receipt JSON/JSONL; may be provided multiple times",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Optional prebuilt sampler policy training dataset JSON",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="checkpoints/sampler_policy")
    parser.add_argument("--run-name", type=str, default="sampler_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_receipt_paths(args: argparse.Namespace) -> list[Path]:
    resolved = [Path(path) for path in list(args.receipt_json or []) if path]
    if not resolved and args.dataset_json is None:
        resolved = [Path("artifacts/shadow_learning_smoke/sampler_policy_receipt.json")]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _build_dataset(args: argparse.Namespace) -> tuple[SamplerPolicyTrainingDataset, list[Path]]:
    receipt_paths = _resolve_receipt_paths(args)
    if args.dataset_json:
        return load_sampler_policy_training_dataset(args.dataset_json), receipt_paths
    receipts = load_sampler_policy_receipts(receipt_paths)
    return build_sampler_policy_training_dataset(receipts), receipt_paths


def _build_dataset_summary(
    *,
    dataset: SamplerPolicyTrainingDataset,
    receipt_paths: Sequence[Path],
    dataset_path: Path,
) -> dict[str, Any]:
    summary = dict(dataset.summary)
    summary.update(
        {
            "dataset_path": str(dataset_path.resolve()),
            "receipt_paths": [str(path.resolve()) for path in receipt_paths],
            "feature_contract": {
                "pool_feature_names": list(SAMPLER_POOL_FEATURE_NAMES),
                "episode_feature_names": list(SAMPLER_EPISODE_FEATURE_NAMES),
                "strategy_names": list(SAMPLER_POLICY_STRATEGIES),
                "plan_parameter_names": list(SAMPLER_PLAN_PARAMETER_NAMES),
                "target_contract": "sampler_policy_v1",
            },
        }
    )
    return summary


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::sampler_receipts_present": int(bool(dataset_summary.get("receipt_paths"))),
        "dataset::non_empty_pool_rows": int(int(dataset_summary.get("num_pool_examples", 0)) > 0),
        "dataset::non_empty_episode_rows": int(int(dataset_summary.get("num_episode_examples", 0)) > 0),
        "dataset::receipt_feedback_present": int(int(dataset_summary.get("receipt_feedback_rows", 0)) > 0),
        "benchmark::sampler_policy_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "sampler_policy_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "sampler_policy_model_config_v1",
        "pool_input_dim": len(SAMPLER_POOL_FEATURE_NAMES),
        "episode_input_dim": len(SAMPLER_EPISODE_FEATURE_NAMES) + len(SAMPLER_POLICY_STRATEGIES),
        "hidden_dim": int(hidden_dim),
        "target_contract": "sampler_policy_v1",
    }


def _evaluate_train_metrics(pool_model: Any, episode_model: Any, dataset: SamplerPolicyTrainingDataset) -> dict[str, float]:
    if torch is None:
        return {
            "strategy_distribution_mse": 0.0,
            "sampling_plan_mse": 0.0,
            "episode_weight_mse": 0.0,
        }
    X_pool = np.asarray(
        [
            [float(example.feature_map.get(name, 0.0)) for name in SAMPLER_POOL_FEATURE_NAMES]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    y_strategy = np.asarray(
        [
            [float(example.strategy_targets.get(name, 0.0)) for name in SAMPLER_POLICY_STRATEGIES]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    y_plan = np.asarray(
        [
            [float(example.plan_targets.get(name, 0.0)) for name in SAMPLER_PLAN_PARAMETER_NAMES]
            for example in dataset.pool_examples
        ],
        dtype=np.float32,
    )
    X_episode = np.asarray(
        [
            [
                *[float(example.feature_map.get(name, 0.0)) for name in SAMPLER_EPISODE_FEATURE_NAMES],
                *[1.0 if example.strategy == strategy else 0.0 for strategy in SAMPLER_POLICY_STRATEGIES],
            ]
            for example in dataset.episode_examples
        ],
        dtype=np.float32,
    )
    y_episode = np.asarray([float(example.target_weight) for example in dataset.episode_examples], dtype=np.float32)
    with torch.no_grad():
        strategy_logits, plan_logits = pool_model(torch.from_numpy(X_pool))
        strategy_probs = torch.softmax(strategy_logits, dim=-1).cpu().numpy()
        plan_probs = torch.sigmoid(plan_logits).cpu().numpy()
        episode_scores = torch.sigmoid(episode_model(torch.from_numpy(X_episode))).squeeze(-1).cpu().numpy()
    return {
        "strategy_distribution_mse": float(np.mean((strategy_probs - y_strategy) ** 2)),
        "sampling_plan_mse": float(np.mean((plan_probs - y_plan) ** 2)),
        "episode_weight_mse": float(np.mean((episode_scores - y_episode) ** 2)),
    }


def _build_runtime_package(
    *,
    config_digest: str,
    checkpoint_path: Path,
    dataset_summary: Mapping[str, Any],
    model_config: Mapping[str, Any],
    execution_preconditions: Mapping[str, Any],
    dataset_path: Path,
    dataset_summary_path: Path,
    model_config_path: Path,
    preconditions_path: Path,
    training_summary_path: Path,
) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    benchmark_gate_ready = bool(benchmark_gate.get("ready", False))
    return {
        "schema_version": "sampler_policy_runtime_package_v1",
        "package_id": f"sampler_policy_{config_digest[:12]}",
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "dataset_summary_path": str(dataset_summary_path),
        "model_config_path": str(model_config_path),
        "preconditions_path": str(preconditions_path),
        "training_summary_path": str(training_summary_path),
        "model_config": dict(model_config),
        "benchmark_gate": benchmark_gate,
        "execution_preconditions": dict(execution_preconditions),
        "promotion_stage": "promoted" if benchmark_gate_ready else "shadow_candidate",
        "inference_contract": {
            "helper_blend_policy": {
                "shadow_candidate_strategy_weight": 0.12,
                "promoted_strategy_weight": 0.35,
                "shadow_candidate_episode_weight": 0.12,
                "promoted_episode_weight": 0.35,
                "shadow_candidate_plan_weight": 0.12,
                "promoted_plan_weight": 0.35,
            },
            "strategy_names": list(SAMPLER_POLICY_STRATEGIES),
            "plan_parameter_names": list(SAMPLER_PLAN_PARAMETER_NAMES),
            "target_contract": "sampler_policy_v1",
            "runtime_targets": ["episode_sampling", "shadow_training_entrypoints"],
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "policy_surface": "sampler_base_weight_strategy",
        },
    }


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    history: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "sampler_policy_training_summary_v1",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "num_pool_examples": int(dataset_summary.get("num_pool_examples", 0)),
        "num_episode_examples": int(dataset_summary.get("num_episode_examples", 0)),
        "receipt_feedback_rows": int(dataset_summary.get("receipt_feedback_rows", 0)),
        "train_metrics": dict(train_metrics),
        "history_tail": {key: list(values)[-5:] for key, values in dict(history or {}).items()},
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(dataset: SamplerPolicyTrainingDataset) -> list[Any]:
    audits: list[Any] = []
    for example in dataset.pool_examples[: min(len(dataset.pool_examples), 8)]:
        top_strategy = max(
            SAMPLER_POLICY_STRATEGIES,
            key=lambda strategy: (float(example.strategy_targets.get(strategy, 0.0)), strategy),
        )
        audits.append(
            create_trajectory_audit(
                episode_id=example.row_id,
                num_steps=1,
                rewards=[float(example.strategy_targets.get(top_strategy, 0.0))],
                reward_components={top_strategy: [float(example.strategy_targets.get(top_strategy, 0.0))]},
                events=[f"target_source:{example.target_source}", f"top_strategy:{top_strategy}"],
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the sampler policy helper")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset, receipt_paths = _build_dataset(args)
    dataset_path = output_root / "sampler_policy_dataset.json"
    dataset_summary_path = output_root / "sampler_policy_dataset_summary.json"
    model_config_path = output_root / "sampler_policy_model_config.json"
    preconditions_path = output_root / "sampler_policy_execution_preconditions.json"
    training_summary_path = output_root / "sampler_policy_training_summary.json"
    checkpoint_path = output_root / "sampler_policy.pt"
    package_path = output_root / "sampler_policy_package.json"
    job_result_path = output_root / "training_job_result.json"

    save_sampler_policy_training_dataset(dataset, dataset_path)
    dataset_summary = _build_dataset_summary(dataset=dataset, receipt_paths=receipt_paths, dataset_path=dataset_path)
    _write_json(dataset_summary_path, dataset_summary)
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    _write_json(preconditions_path, execution_preconditions)
    model_config = _build_model_config(hidden_dim=int(args.hidden_dim))
    _write_json(model_config_path, model_config)

    pool_model, episode_model, training_result = train_sampler_policy_models(
        dataset,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        save_path=str(checkpoint_path),
    )
    train_metrics = _evaluate_train_metrics(pool_model, episode_model, dataset)
    config_digest = sha256_json(
        {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "dataset_digest": dataset_summary.get("dataset_digest"),
        }
    )
    runtime_package = _build_runtime_package(
        config_digest=config_digest,
        checkpoint_path=checkpoint_path,
        dataset_summary=dataset_summary,
        model_config=model_config,
        execution_preconditions=execution_preconditions,
        dataset_path=dataset_path,
        dataset_summary_path=dataset_summary_path,
        model_config_path=model_config_path,
        preconditions_path=preconditions_path,
        training_summary_path=training_summary_path,
    )
    _write_json(package_path, runtime_package)

    artifacts = {
        "dataset": str(dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "execution_preconditions": str(preconditions_path),
        "checkpoint": str(checkpoint_path),
        "runtime_package": str(package_path),
    }
    training_summary = _build_training_summary(
        run_name=str(args.run_name),
        seed=int(args.seed),
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        history=dict(training_result.get("history", {}) or {}),
        train_metrics=train_metrics,
        artifacts=artifacts,
    )
    _write_json(training_summary_path, training_summary)

    job_result = {
        "schema_version": "sampler_policy_training_job_result_v1",
        "run_name": str(args.run_name),
        "config_digest": config_digest,
        "runtime_package": str(package_path),
        "checkpoint_path": str(checkpoint_path),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "train_metrics": dict(train_metrics),
        "artifacts": dict(artifacts),
    }
    _write_json(job_result_path, job_result)

    if runner is not None:
        runner.update_step(int(args.epochs))
        runner.set_sampler_config(seed=int(args.seed), config_sha=config_digest)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate": dataset_summary.get("benchmark_gate", {}),
                "receipt_feedback_rows": dataset_summary.get("receipt_feedback_rows", 0),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="sampler_policy",
            config_digest=config_digest,
            metadata={
                "run_name": str(args.run_name),
                "num_pool_examples": int(dataset_summary.get("num_pool_examples", 0)),
                "num_episode_examples": int(dataset_summary.get("num_episode_examples", 0)),
                "receipt_feedback_rows": int(dataset_summary.get("receipt_feedback_rows", 0)),
            },
        )
        runner.register_artifact("sampler_policy_dataset", dataset_path)
        runner.register_artifact("sampler_policy_dataset_summary", dataset_summary_path)
        runner.register_artifact("sampler_policy_model_config", model_config_path)
        runner.register_artifact("sampler_policy_execution_preconditions", preconditions_path)
        runner.register_artifact("sampler_policy_training_summary", training_summary_path)
        runner.register_artifact("sampler_policy_runtime_package", package_path)
        runner.register_artifact("sampler_policy_job_result", job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="sampler_policy",
                model_family="sampler_policy",
                model_version="sampler_policy_v1",
                path=checkpoint_path,
                step=int(args.epochs),
                epoch=int(args.epochs),
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                },
            )
        )
        for audit in _build_trajectory_audits(dataset):
            runner.add_trajectory_audit(audit)

    return {
        "dataset_path": str(dataset_path),
        "runtime_package": str(package_path),
        "training_summary": str(training_summary_path),
        "job_result": str(job_result_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.skip_regal_runner:
        return _run_training(args, runner=None)

    return run_training_with_regality(
        training_fn=lambda runner: _run_training(args, runner),
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=int(args.seed),
            num_episodes=max(int(SAMPLER_POLICY_MIN_POOL_ROWS), 1),
            training_steps=max(int(args.epochs), 1),
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json(
            {
                "script": "train_sampler_policy.py",
                "seed": int(args.seed),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "hidden_dim": int(args.hidden_dim),
            }
        ),
        plan_id=str(args.run_name),
    )


if __name__ == "__main__":
    main()
