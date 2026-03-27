#!/usr/bin/env python3
"""Train the bounded semantic-orchestrator shell policy helper."""

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

from src.orchestrator.orchestrator_shell_policy import (
    SHELL_POLICY_FEATURE_NAMES,
    SHELL_POLICY_PRESET_LABELS,
    SHELL_POLICY_STRATEGY_KEYS,
)
from src.orchestrator.orchestrator_shell_policy_training import (
    TORCH_AVAILABLE,
    OrchestratorShellTrainingDataset,
    build_orchestrator_shell_training_dataset,
    load_orchestrator_shell_training_dataset,
    load_runtime_rows,
    save_orchestrator_shell_training_dataset,
    train_orchestrator_shell_policy_model,
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
        "--runtime-export-dir",
        type=str,
        default=None,
        help="Directory containing semantic_runtime_learning_rows.jsonl",
    )
    parser.add_argument(
        "--rows-json",
        dest="rows_json",
        action="append",
        default=[],
        help="Path to semantic_runtime_learning_rows.jsonl; may be provided multiple times",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Optional prebuilt orchestrator shell training dataset JSON",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="checkpoints/orchestrator_shell_policy")
    parser.add_argument("--run-name", type=str, default="orchestrator_shell_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_row_paths(args: argparse.Namespace) -> list[Path]:
    resolved = [Path(path) for path in list(args.rows_json or []) if path]
    if args.runtime_export_dir:
        resolved.append(Path(args.runtime_export_dir) / "semantic_runtime_learning_rows.jsonl")
    if not resolved and args.dataset_json is None:
        resolved = [Path("artifacts/semantic_runtime_corpus/semantic_runtime_learning_rows.jsonl")]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _build_dataset(args: argparse.Namespace) -> tuple[OrchestratorShellTrainingDataset, list[Path]]:
    row_paths = _resolve_row_paths(args)
    if args.dataset_json:
        return load_orchestrator_shell_training_dataset(args.dataset_json), row_paths
    runtime_rows = load_runtime_rows(row_paths)
    return build_orchestrator_shell_training_dataset(runtime_rows), row_paths


def _build_dataset_summary(
    *,
    dataset: OrchestratorShellTrainingDataset,
    row_paths: Sequence[Path],
    dataset_path: Path,
) -> dict[str, Any]:
    summary = dict(dataset.summary)
    summary.update(
        {
            "dataset_path": str(dataset_path.resolve()),
            "runtime_row_paths": [str(path.resolve()) for path in row_paths],
            "feature_contract": {
                "feature_names": list(SHELL_POLICY_FEATURE_NAMES),
                "preset_labels": list(SHELL_POLICY_PRESET_LABELS),
                "strategy_keys": list(SHELL_POLICY_STRATEGY_KEYS),
                "target_contract": "semantic_orchestrator_shell_policy_v1",
            },
        }
    )
    return summary


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::runtime_rows_present": int(bool(dataset_summary.get("runtime_row_paths"))),
        "dataset::non_empty": int(int(dataset_summary.get("num_examples", 0)) > 0),
        "dataset::activated_rows_present": int(int(dataset_summary.get("activated_rows", 0)) > 0),
        "benchmark::orchestrator_shell_policy_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "orchestrator_shell_policy_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "orchestrator_shell_policy_model_config_v1",
        "input_dim": len(SHELL_POLICY_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "preset_labels": list(SHELL_POLICY_PRESET_LABELS),
        "strategy_keys": list(SHELL_POLICY_STRATEGY_KEYS),
        "target_contract": "semantic_orchestrator_shell_policy_v1",
    }


def _evaluate_train_metrics(model: Any, dataset: OrchestratorShellTrainingDataset) -> dict[str, float]:
    if torch is None:
        return {
            "preset_mse": 0.0,
            "strategy_mse": 0.0,
            "safety_mse": 0.0,
            "activation_mse": 0.0,
        }
    X = np.asarray(
        [
            [float(example.feature_map.get(name, 0.0)) for name in SHELL_POLICY_FEATURE_NAMES]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_preset = np.asarray(
        [
            [float(example.preset_distribution.get(label, 0.0)) for label in SHELL_POLICY_PRESET_LABELS]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_strategy = np.asarray(
        [
            [float(example.strategy_distribution.get(label, 0.0)) for label in SHELL_POLICY_STRATEGY_KEYS]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y_safety = np.asarray([float(example.safety_emphasis) for example in dataset.examples], dtype=np.float32)
    y_activation = np.asarray([float(example.activation_label) for example in dataset.examples], dtype=np.float32)
    with torch.no_grad():
        preset_logits, strategy_logits, safety_logits, activation_logits = model(
            torch.from_numpy(X)
        )
        preset_probs = torch.softmax(preset_logits, dim=-1).cpu().numpy()
        strategy_probs = torch.softmax(strategy_logits, dim=-1).cpu().numpy()
        safety = torch.sigmoid(safety_logits).squeeze(-1).cpu().numpy()
        activation = torch.sigmoid(activation_logits).squeeze(-1).cpu().numpy()
    return {
        "preset_mse": float(np.mean((preset_probs - y_preset) ** 2)),
        "strategy_mse": float(np.mean((strategy_probs - y_strategy) ** 2)),
        "safety_mse": float(np.mean((safety - y_safety) ** 2)),
        "activation_mse": float(np.mean((activation - y_activation) ** 2)),
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
        "schema_version": "orchestrator_shell_policy_runtime_package_v1",
        "package_id": f"orchestrator_shell_policy_{config_digest[:12]}",
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
                "shadow_candidate_helper_weight": 0.15,
                "promoted_helper_weight": 0.4,
                "shadow_candidate_max_safety_delta": 0.1,
                "promoted_max_safety_delta": 0.25,
            },
            "feature_names": list(SHELL_POLICY_FEATURE_NAMES),
            "target_contract": "semantic_orchestrator_shell_policy_v1",
            "runtime_targets": ["semantic_orchestrator_v2"],
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "training_contract": "semantic_orchestrator_shell_policy_v1",
            "policy_surface": "higher_order_semantic_orchestrator_shell",
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
        "schema_version": "orchestrator_shell_policy_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "num_examples": int(dataset_summary.get("num_examples", 0)),
        "train_loss": float(list(history.get("loss", [0.0]))[-1] if history.get("loss") else 0.0),
        "train_metrics": dict(train_metrics),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(dataset: OrchestratorShellTrainingDataset) -> list[Any]:
    audits: list[Any] = []
    for example in dataset.examples:
        audits.append(
            create_trajectory_audit(
                episode_id=example.row_id,
                num_steps=1,
                rewards=[float(example.safety_emphasis + example.activation_label)],
                reward_components={
                    "safety_emphasis": [float(example.safety_emphasis)],
                    "activation_label": [float(example.activation_label)],
                },
                events=[
                    f"target_source:{example.target_source}",
                    f"policy_source:{example.policy_source}",
                    f"promotion_stage:{example.promotion_stage}",
                ],
            )
        )
    return audits


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to train the orchestrator shell policy")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset, row_paths = _build_dataset(args)
    if not dataset.examples:
        raise ValueError("No orchestrator shell training examples were built")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = output_root / "orchestrator_shell_policy_dataset.json"
    dataset_summary_path = output_root / "orchestrator_shell_policy_dataset_summary.json"
    model_config_path = output_root / "orchestrator_shell_policy_model_config.json"
    preconditions_path = output_root / "orchestrator_shell_policy_execution_preconditions.json"
    training_summary_path = output_root / "orchestrator_shell_policy_training_summary.json"
    runtime_package_path = output_root / "orchestrator_shell_policy_package.json"
    checkpoint_path = output_root / "orchestrator_shell_policy.pt"
    training_job_result_path = output_root / "training_job_result.json"

    save_orchestrator_shell_training_dataset(dataset, dataset_path)
    dataset_summary = _build_dataset_summary(
        dataset=dataset,
        row_paths=row_paths,
        dataset_path=dataset_path,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    model_config = _build_model_config(hidden_dim=args.hidden_dim)
    config_digest = sha256_json(
        {
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "run_name": args.run_name,
        }
    )

    model, history = train_orchestrator_shell_policy_model(
        dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        save_path=str(checkpoint_path),
    )
    train_metrics = _evaluate_train_metrics(model, dataset)
    artifacts = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
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
        history=history,
        train_metrics=train_metrics,
        artifacts=artifacts,
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

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)
    _write_json(runtime_package_path, runtime_package)

    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
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
            "training_kind": "orchestrator_shell_policy",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        runner.set_eligible_datapacks([example.row_id for example in dataset.examples])
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for example in dataset.examples:
            runner.record_sample(
                example.target_source,
                datapack_id=example.row_id,
                slice_id=f"{example.target_source}:{example.row_id}",
            )
        for audit in _build_trajectory_audits(dataset):
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, args.epochs))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "num_examples": len(dataset.examples),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="orchestrator_shell_policy",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "semantic_orchestrator_shell_policy"},
            promotion_policy_snapshot={"benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})},
            source_domain_coverage={"target_source_counts": dict(dataset_summary.get("target_source_counts", {}) or {})},
            receipt_label_coverage={"policy_source_counts": dict(dataset_summary.get("policy_source_counts", {}) or {})},
            metadata={
                "trajectory_audit_kind": "semantic_orchestrator_shell_policy_projection",
                "target_contract": "semantic_orchestrator_shell_policy_v1",
            },
        )
        runner.register_artifact("orchestrator_shell_policy_dataset", dataset_path)
        runner.register_artifact("orchestrator_shell_policy_dataset_summary", dataset_summary_path)
        runner.register_artifact("orchestrator_shell_policy_model_config", model_config_path)
        runner.register_artifact("orchestrator_shell_policy_preconditions", preconditions_path)
        runner.register_artifact("orchestrator_shell_policy_training_summary", training_summary_path)
        runner.register_artifact("orchestrator_shell_policy_runtime_package", runtime_package_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="orchestrator_shell_policy_latest",
                model_family="orchestrator_shell_policy",
                model_version="orchestrator_shell_policy_v1",
                path=checkpoint_path,
                step=max(1, args.epochs),
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_metrics": dict(train_metrics),
                },
            )
        )
    return result


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dataset, _ = _build_dataset(args)
    plan_sha = sha256_json(
        {
            "training_kind": "orchestrator_shell_policy",
            "num_examples": len(dataset.examples),
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
            num_episodes=max(1, len(dataset.examples)),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="orchestrator_shell_policy",
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
