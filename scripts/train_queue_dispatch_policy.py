#!/usr/bin/env python3
"""Train the bounded queue-dispatch policy helper."""

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

from src.orchestrator.queue_dispatch_policy import QUEUE_DISPATCH_FEATURE_NAMES
from src.orchestrator.queue_dispatch_policy_training import (
    TORCH_AVAILABLE,
    QueueDispatchTrainingDataset,
    build_queue_dispatch_training_dataset,
    load_queue_dispatch_training_dataset,
    load_queue_selection_payloads,
    save_queue_dispatch_training_dataset,
    train_queue_dispatch_policy_model,
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
        "--queue-json",
        dest="queue_json",
        action="append",
        default=[],
        help="Path to live_queue_selection JSON/JSONL or advisory JSON payload; may be provided multiple times",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Optional prebuilt queue dispatch training dataset JSON",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="checkpoints/queue_dispatch_policy")
    parser.add_argument("--run-name", type=str, default="queue_dispatch_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_queue_paths(args: argparse.Namespace) -> list[Path]:
    resolved = [Path(path) for path in list(args.queue_json or []) if path]
    if not resolved and args.dataset_json is None:
        resolved = [Path("artifacts/shadow_learning_smoke/live_queue_selection.json")]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _build_dataset(args: argparse.Namespace) -> tuple[QueueDispatchTrainingDataset, list[Path]]:
    queue_paths = _resolve_queue_paths(args)
    if args.dataset_json:
        return load_queue_dispatch_training_dataset(args.dataset_json), queue_paths
    payloads = load_queue_selection_payloads(queue_paths)
    return build_queue_dispatch_training_dataset(payloads), queue_paths


def _build_dataset_summary(
    *,
    dataset: QueueDispatchTrainingDataset,
    queue_paths: Sequence[Path],
    dataset_path: Path,
) -> dict[str, Any]:
    summary = dict(dataset.summary)
    summary.update(
        {
            "dataset_path": str(dataset_path.resolve()),
            "queue_paths": [str(path.resolve()) for path in queue_paths],
            "feature_contract": {
                "feature_names": list(QUEUE_DISPATCH_FEATURE_NAMES),
                "target_contract": "queue_dispatch_policy_v1",
            },
        }
    )
    return summary


def _build_execution_preconditions(dataset_summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_gate = dict(dataset_summary.get("benchmark_gate", {}) or {})
    satisfied = {
        "artifact::queue_payloads_present": int(bool(dataset_summary.get("queue_paths"))),
        "dataset::non_empty": int(int(dataset_summary.get("num_examples", 0)) > 0),
        "dataset::receipt_feedback_present": int(int(dataset_summary.get("receipt_feedback_rows", 0)) > 0),
        "benchmark::queue_dispatch_policy_density": int(bool(benchmark_gate.get("ready", False))),
    }
    return {
        "schema_version": "queue_dispatch_policy_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
    }


def _build_model_config(*, hidden_dim: int) -> dict[str, Any]:
    return {
        "schema_version": "queue_dispatch_policy_model_config_v1",
        "input_dim": len(QUEUE_DISPATCH_FEATURE_NAMES),
        "hidden_dim": int(hidden_dim),
        "target_contract": "queue_dispatch_policy_v1",
    }


def _evaluate_train_metrics(model: Any, dataset: QueueDispatchTrainingDataset) -> dict[str, float]:
    if torch is None:
        return {"dispatch_score_mse": 0.0}
    X = np.asarray(
        [
            [float(example.feature_map.get(name, 0.0)) for name in QUEUE_DISPATCH_FEATURE_NAMES]
            for example in dataset.examples
        ],
        dtype=np.float32,
    )
    y = np.asarray([float(example.dispatch_score) for example in dataset.examples], dtype=np.float32)
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.from_numpy(X))).squeeze(-1).cpu().numpy()
    return {"dispatch_score_mse": float(np.mean((preds - y) ** 2))}


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
        "schema_version": "queue_dispatch_policy_runtime_package_v1",
        "package_id": f"queue_dispatch_policy_{config_digest[:12]}",
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
                "shadow_candidate_helper_weight": 0.12,
                "promoted_helper_weight": 0.35,
            },
            "feature_names": list(QUEUE_DISPATCH_FEATURE_NAMES),
            "target_contract": "queue_dispatch_policy_v1",
            "runtime_targets": ["queue_selection", "episode_sampling"],
        },
        "metadata": {
            "config_digest": config_digest,
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "policy_surface": "queue_dispatch_reweighting",
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
        "schema_version": "queue_dispatch_policy_training_summary_v1",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "num_examples": int(dataset_summary.get("num_examples", 0)),
        "receipt_feedback_rows": int(dataset_summary.get("receipt_feedback_rows", 0)),
        "train_metrics": dict(train_metrics),
        "history_tail": {
            key: list(values)[-5:]
            for key, values in dict(history or {}).items()
        },
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _build_trajectory_audits(dataset: QueueDispatchTrainingDataset) -> list[Any]:
    audits: list[Any] = []
    for example in dataset.examples[: min(len(dataset.examples), 8)]:
        audits.append(
            create_trajectory_audit(
                episode_id=example.row_id,
                num_steps=1,
                rewards=[float(example.dispatch_score)],
                reward_components={"dispatch_score": [float(example.dispatch_score)]},
                events=[f"target_source:{example.target_source}"],
            )
        )
    return audits


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the queue dispatch policy")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset, queue_paths = _build_dataset(args)
    dataset_path = output_root / "queue_dispatch_policy_dataset.json"
    dataset_summary_path = output_root / "queue_dispatch_policy_dataset_summary.json"
    model_config_path = output_root / "queue_dispatch_policy_model_config.json"
    preconditions_path = output_root / "queue_dispatch_policy_execution_preconditions.json"
    training_summary_path = output_root / "queue_dispatch_policy_training_summary.json"
    checkpoint_path = output_root / "queue_dispatch_policy.pt"
    package_path = output_root / "queue_dispatch_policy_package.json"
    job_result_path = output_root / "training_job_result.json"

    save_queue_dispatch_training_dataset(dataset, dataset_path)
    dataset_summary = _build_dataset_summary(dataset=dataset, queue_paths=queue_paths, dataset_path=dataset_path)
    _write_json(dataset_summary_path, dataset_summary)
    execution_preconditions = _build_execution_preconditions(dataset_summary)
    _write_json(preconditions_path, execution_preconditions)
    model_config = _build_model_config(hidden_dim=int(args.hidden_dim))
    _write_json(model_config_path, model_config)

    model, training_result = train_queue_dispatch_policy_model(
        dataset,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        save_path=str(checkpoint_path),
    )
    train_metrics = _evaluate_train_metrics(model, dataset)
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

    trajectory_audits = _build_trajectory_audits(dataset)
    job_result = {
        "schema_version": "queue_dispatch_policy_training_job_result_v1",
        "run_name": str(args.run_name),
        "checkpoint_path": str(checkpoint_path),
        "training_summary": str(training_summary_path),
        "runtime_package": str(package_path),
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "trajectory_audits": [audit.model_dump(mode="json") for audit in trajectory_audits],
    }
    _write_json(job_result_path, job_result)

    if runner is not None:
        runner.set_eligible_datapacks([example.row_id for example in dataset.examples])
        runner.set_sampler_config(seed=int(args.seed), config_sha=config_digest)
        for example in dataset.examples:
            runner.record_sample(
                example.target_source,
                datapack_id=example.row_id,
                slice_id=f"{example.target_source}:{example.row_id}",
            )
        for audit in trajectory_audits:
            runner.add_trajectory_audit(audit)
        runner.update_step(max(1, int(args.epochs)))
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "num_rows": int(dataset_summary.get("num_examples", 0)),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="queue_dispatch_policy",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "queue_dispatch_reweighting"},
            promotion_policy_snapshot={"benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {})},
            source_domain_coverage={"target_source_counts": dict(dataset_summary.get("target_source_counts", {}) or {})},
            receipt_label_coverage={"target_source_counts": dict(dataset_summary.get("target_source_counts", {}) or {})},
            metadata={
                "trajectory_audit_kind": "queue_dispatch_policy_projection",
                "target_contract": "queue_dispatch_policy_v1",
                "queue_paths": [str(path.resolve()) for path in queue_paths],
            },
        )
        runner.register_artifact("queue_dispatch_policy_dataset", dataset_path)
        runner.register_artifact("queue_dispatch_policy_dataset_summary", dataset_summary_path)
        runner.register_artifact("queue_dispatch_policy_model_config", model_config_path)
        runner.register_artifact("queue_dispatch_policy_execution_preconditions", preconditions_path)
        runner.register_artifact("queue_dispatch_policy_training_summary", training_summary_path)
        runner.register_artifact("queue_dispatch_policy_runtime_package", package_path)
        runner.register_artifact("training_job_result", job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="queue_dispatch_policy_latest",
                model_family="queue_dispatch_policy",
                model_version="queue_dispatch_policy_v1",
                path=checkpoint_path,
                step=max(1, int(args.epochs)),
                epoch=int(args.epochs),
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                    "train_metrics": dict(train_metrics),
                },
            )
        )

    return {
        "dataset_path": str(dataset_path),
        "runtime_package": str(package_path),
        "training_summary": str(training_summary_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.skip_regal_runner:
        print(json.dumps(_run_training(args, runner=None), indent=2, sort_keys=True))
        return 0

    holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        holder["payload"] = _run_training(args, runner)

    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(args.output_dir),
            seed=int(args.seed),
            num_episodes=max(1, len(list(args.queue_json or [])) or 1),
            training_steps=max(1, int(args.epochs)),
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"run_name": str(args.run_name), "seed": int(args.seed)}),
        plan_id="queue_dispatch_policy",
    )
    print(
        json.dumps(
            {"training_run": result.to_dict(), "job": holder.get("payload", {})},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
