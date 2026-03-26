#!/usr/bin/env python3
"""Train the orchestration transformer with runtime-backed datasets when available."""

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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.orchestrator.orchestration_transformer import OrchestrationTransformer
from src.orchestrator.training_dataset import (
    OrchestrationSample,
    build_mixed_training_dataset,
    build_training_dataset,
    dataset_to_model_tensors,
    load_dataset_samples,
    save_dataset,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit


ORCHESTRATION_BENCHMARK_MIN_RUNTIME_SAMPLES = 1000


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-export-dir",
        type=str,
        default=None,
        help="Directory containing orchestration_runtime_dataset.json from export_semantic_runtime_learning_corpus.py",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Path to a saved orchestration dataset JSON",
    )
    parser.add_argument(
        "--runtime-summary-json",
        type=str,
        default=None,
        help="Optional semantic_runtime_learning_summary.json to join with --dataset-json",
    )
    parser.add_argument("--num-samples", type=int, default=1000, help="Synthetic fallback sample count")
    parser.add_argument("--use-mixed-dataset", action="store_true", help="Use mixed heuristic + econ/semantic synthetic dataset")
    parser.add_argument("--econ-semantic-ratio", type=float, default=0.5, help="Fraction of samples from econ/semantic synthetic generation when using mixed fallback")
    parser.add_argument("--epochs", type=int, default=24, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=96, help="Hidden dimension")
    parser.add_argument("--ctx-dim", type=int, default=0, help="Optional explicit context dimension; 0 means infer from data")
    parser.add_argument("--vocab-size", type=int, default=256, help="Instruction vocabulary size")
    parser.add_argument("--instruction-seq-len", type=int, default=16, help="Instruction token sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--save-dir", type=str, default="checkpoints/orchestrator", help="Directory for checkpoints and runtime artifacts")
    parser.add_argument("--run-name", type=str, default="orchestration_transformer", help="Training run name prefix")
    parser.add_argument("--skip-regal-runner", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_runtime_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")) or {})


def _resolve_samples(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[OrchestrationSample], dict[str, Any]]:
    dataset_path: Path | None = None
    runtime_summary_path: Path | None = None
    dataset_source = ""

    if args.runtime_export_dir:
        export_dir = Path(args.runtime_export_dir)
        dataset_path = export_dir / "orchestration_runtime_dataset.json"
        runtime_summary_path = export_dir / "semantic_runtime_learning_summary.json"
        dataset_source = "semantic_runtime_export"
    elif args.dataset_json:
        dataset_path = Path(args.dataset_json)
        runtime_summary_path = Path(args.runtime_summary_json) if args.runtime_summary_json else None
        dataset_source = "dataset_json"
    elif args.use_mixed_dataset:
        num_econ_semantic = int(args.num_samples * max(0.0, min(1.0, args.econ_semantic_ratio)))
        num_heuristic = max(0, int(args.num_samples) - num_econ_semantic)
        samples, dataset_stats = build_mixed_training_dataset(
            num_heuristic=num_heuristic,
            num_econ_semantic=num_econ_semantic,
        )
        dataset_path = output_root / "generated_orchestration_dataset.json"
        save_dataset(samples, str(dataset_path))
        return samples, {
            "dataset_source": "synthetic_mixed_fallback",
            "dataset_path": str(dataset_path.resolve()),
            "runtime_summary_path": None,
            "runtime_summary": {},
            "dataset_stats": dataset_stats,
        }
    else:
        samples = build_training_dataset(num_samples=args.num_samples)
        dataset_path = output_root / "generated_orchestration_dataset.json"
        save_dataset(samples, str(dataset_path))
        return samples, {
            "dataset_source": "heuristic_synthetic_fallback",
            "dataset_path": str(dataset_path.resolve()),
            "runtime_summary_path": None,
            "runtime_summary": {},
            "dataset_stats": {},
        }

    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Orchestration dataset not found: {dataset_path}")

    samples = load_dataset_samples(str(dataset_path))
    return samples, {
        "dataset_source": dataset_source,
        "dataset_path": str(dataset_path.resolve()),
        "runtime_summary_path": (
            str(runtime_summary_path.resolve())
            if runtime_summary_path is not None and runtime_summary_path.exists()
            else None
        ),
        "runtime_summary": _load_runtime_summary(runtime_summary_path),
        "dataset_stats": {},
    }


def _split_indices(sample_count: int, *, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 1:
        return np.arange(sample_count), np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    val_count = int(round(sample_count * max(0.0, min(0.5, val_split))))
    val_count = min(max(val_count, 1), sample_count - 1)
    return indices[val_count:], indices[:val_count]


def _sample_id(sample: OrchestrationSample, index: int) -> str:
    return str(sample.metadata.get("sample_id") or f"orch_sample_{index}")


def _source_domain_coverage(samples: Sequence[OrchestrationSample]) -> dict[str, Any]:
    source_counts = Counter(str(sample.source_type or "unknown") for sample in samples)
    runtime_domain_counts = Counter(
        str(sample.metadata.get("source_domain") or "unknown")
        for sample in samples
    )
    objective_counts = Counter(
        str(
            sample.metadata.get("objective_preset")
            or (
                sample.econ_semantic_summary.objective_preset
                if sample.econ_semantic_summary is not None
                else "balanced"
            )
        )
        for sample in samples
    )
    return {
        "dataset_kind": "orchestration_training_dataset",
        "source_type_counts": dict(sorted(source_counts.items())),
        "runtime_source_domain_counts": dict(sorted(runtime_domain_counts.items())),
        "objective_preset_counts": dict(sorted(objective_counts.items())),
    }


def _build_dataset_summary(
    *,
    samples: Sequence[OrchestrationSample],
    dataset_info: Mapping[str, Any],
    vocab_size: int,
    instruction_seq_len: int,
    hidden: int,
    ctx_dim: int,
) -> dict[str, Any]:
    source_counts = Counter(str(sample.source_type or "unknown") for sample in samples)
    tool_counts = Counter(
        str(sample.target_tool_sequence[0].name)
        for sample in samples
        if sample.target_tool_sequence
    )
    runtime_like_count = sum(1 for sample in samples if sample.source_type == "semantic_runtime_corpus")
    benchmark_gate_ready = runtime_like_count >= ORCHESTRATION_BENCHMARK_MIN_RUNTIME_SAMPLES
    return {
        "schema_version": "orchestration_transformer_dataset_summary_v1",
        "dataset_source": dataset_info.get("dataset_source"),
        "dataset_path": dataset_info.get("dataset_path"),
        "runtime_summary_path": dataset_info.get("runtime_summary_path"),
        "dataset_digest": sha256_json(
            {
                "dataset_path": dataset_info.get("dataset_path"),
                "sample_ids": [_sample_id(sample, idx) for idx, sample in enumerate(samples)],
                "dataset_source": dataset_info.get("dataset_source"),
            }
        ),
        "num_samples": len(samples),
        "runtime_like_count": runtime_like_count,
        "source_type_counts": dict(sorted(source_counts.items())),
        "first_tool_counts": dict(sorted(tool_counts.items())),
        "model_contract": {
            "hidden": int(hidden),
            "ctx_dim": int(ctx_dim),
            "vocab_size": int(vocab_size),
            "instruction_seq_len": int(instruction_seq_len),
            "tool_prediction_contract": "first_tool_only_v1",
        },
        "source_domain_coverage": _source_domain_coverage(samples),
        "runtime_summary": dict(dataset_info.get("runtime_summary", {}) or {}),
        "dataset_stats": dict(dataset_info.get("dataset_stats", {}) or {}),
        "benchmark_gate": {
            "name": "orchestration_min_runtime_samples",
            "ready": benchmark_gate_ready,
            "required_runtime_samples": ORCHESTRATION_BENCHMARK_MIN_RUNTIME_SAMPLES,
            "observed_runtime_samples": runtime_like_count,
            "synthetic_fallback": dataset_info.get("dataset_source")
            in {"heuristic_synthetic_fallback", "synthetic_mixed_fallback"},
        },
    }


def _build_execution_preconditions(
    *,
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_type_counts = dict(dataset_summary.get("source_type_counts", {}) or {})
    satisfied = {
        "artifact::dataset_present": int(bool(dataset_summary.get("dataset_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("num_samples", 0)) > 0),
        "dataset::runtime_backed_examples_present": int(int(dataset_summary.get("runtime_like_count", 0)) > 0),
        "dataset::heuristic_only_fallback": int(bool(source_type_counts.get("heuristic", 0))),
        "benchmark::orchestration_min_runtime_samples": int(
            bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready"))
        ),
    }
    return {
        "schema_version": "orchestration_transformer_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [key for key, value in sorted(satisfied.items()) if not value],
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def _build_model_config(*, hidden: int, ctx_dim: int, vocab_size: int, instruction_seq_len: int) -> dict[str, Any]:
    return {
        "schema_version": "orchestration_transformer_model_config_v1",
        "hidden": int(hidden),
        "ctx_dim": int(ctx_dim),
        "vocab_size": int(vocab_size),
        "instruction_seq_len": int(instruction_seq_len),
        "tool_prediction_contract": "first_tool_only_v1",
    }


def _build_trajectory_audits(samples: Sequence[OrchestrationSample]) -> list[Any]:
    audits: list[Any] = []
    for index, sample in enumerate(samples):
        first_tool = sample.target_tool_sequence[0].name if sample.target_tool_sequence else "NONE"
        reward = 1.0 if sample.source_type == "semantic_runtime_corpus" else 0.5
        audits.append(
            create_trajectory_audit(
                episode_id=_sample_id(sample, index),
                num_steps=len(sample.target_tool_sequence) or 1,
                rewards=[reward] * max(len(sample.target_tool_sequence), 1),
                reward_components={
                    "runtime_backed": [1.0 if sample.source_type == "semantic_runtime_corpus" else 0.0],
                    "tool_count": [float(len(sample.target_tool_sequence))],
                },
                events=[f"first_tool:{first_tool}", f"source:{sample.source_type}"],
            )
        )
    return audits


def _train_epoch(
    model: OrchestrationTransformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_ctx, batch_instr, batch_tools in dataloader:
        optimizer.zero_grad()
        tool_logits, _arg_vec = model(batch_instr, batch_ctx)
        target_first_tool = batch_tools[:, 0]
        loss = criterion(tool_logits, target_first_tool)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pred = torch.argmax(tool_logits, dim=-1)
        correct += int((pred == target_first_tool).sum().item())
        total += int(batch_ctx.shape[0])
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
    }


def _evaluate(
    model: OrchestrationTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_ctx, batch_instr, batch_tools in dataloader:
            tool_logits, _arg_vec = model(batch_instr, batch_ctx)
            target_first_tool = batch_tools[:, 0]
            loss = criterion(tool_logits, target_first_tool)
            total_loss += float(loss.item())
            pred = torch.argmax(tool_logits, dim=-1)
            correct += int((pred == target_first_tool).sum().item())
            total += int(batch_ctx.shape[0])
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
    }


def _subset_metrics(
    model: OrchestrationTransformer,
    samples: Sequence[OrchestrationSample],
    criterion: nn.Module,
    *,
    batch_size: int,
    vocab_size: int,
    instruction_seq_len: int,
) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "loss": 0.0, "accuracy": 0.0}
    X, instr, Y, _ = dataset_to_model_tensors(
        list(samples),
        vocab_size=vocab_size,
        instruction_seq_len=instruction_seq_len,
    )
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(instr).long(),
        torch.from_numpy(Y).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = _evaluate(model, loader, criterion)
    return {
        "count": len(samples),
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
    }


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.save_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    samples, dataset_info = _resolve_samples(args, output_root)
    if not samples:
        raise ValueError("Orchestration training dataset is empty")

    X, instr, Y, tool_names = dataset_to_model_tensors(
        samples,
        vocab_size=args.vocab_size,
        instruction_seq_len=args.instruction_seq_len,
    )
    actual_ctx_dim = int(X.shape[1])
    if args.ctx_dim and args.ctx_dim != actual_ctx_dim:
        raise ValueError(f"Explicit ctx_dim={args.ctx_dim} does not match dataset ctx_dim={actual_ctx_dim}")

    dataset_summary = _build_dataset_summary(
        samples=samples,
        dataset_info=dataset_info,
        vocab_size=args.vocab_size,
        instruction_seq_len=args.instruction_seq_len,
        hidden=args.hidden,
        ctx_dim=actual_ctx_dim,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary=dataset_summary)
    model_config = _build_model_config(
        hidden=args.hidden,
        ctx_dim=actual_ctx_dim,
        vocab_size=args.vocab_size,
        instruction_seq_len=args.instruction_seq_len,
    )
    config_digest = sha256_json(
        {
            "dataset_digest": dataset_summary.get("dataset_digest"),
            "hidden": args.hidden,
            "ctx_dim": actual_ctx_dim,
            "vocab_size": args.vocab_size,
            "instruction_seq_len": args.instruction_seq_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "run_name": args.run_name,
        }
    )

    train_indices, val_indices = _split_indices(len(samples), val_split=args.val_split, seed=args.seed)
    train_dataset = TensorDataset(
        torch.from_numpy(X[train_indices]).float(),
        torch.from_numpy(instr[train_indices]).long(),
        torch.from_numpy(Y[train_indices]).long(),
    )
    eval_indices = val_indices if len(val_indices) > 0 else train_indices
    eval_dataset = TensorDataset(
        torch.from_numpy(X[eval_indices]).float(),
        torch.from_numpy(instr[eval_indices]).long(),
        torch.from_numpy(Y[eval_indices]).long(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = OrchestrationTransformer(
        vocab_size=args.vocab_size,
        hidden=args.hidden,
        ctx_dim=actual_ctx_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    latest_checkpoint_path = output_root / "final_model.pt"
    best_checkpoint_path = output_root / "best_model.pt"
    history_path = output_root / "training_history.json"
    dataset_path = output_root / "orchestration_training_dataset.json"
    dataset_summary_path = output_root / "orchestration_dataset_summary.json"
    model_config_path = output_root / "orchestration_model_config.json"
    preconditions_path = output_root / "orchestration_training_preconditions.json"
    subset_metrics_path = output_root / "orchestration_subset_metrics.json"
    training_summary_path = output_root / "orchestration_training_summary.json"
    training_job_result_path = output_root / "training_job_result.json"

    history: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_epoch = 0
    optimizer_steps = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, criterion)
        val_metrics = _evaluate(model, eval_loader, criterion)
        epoch_summary = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(epoch_summary)
        optimizer_steps += max(len(train_loader), 1)
        print(json.dumps({"event": "epoch_complete", "data": epoch_summary}, sort_keys=True))
        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_epoch = int(epoch)
            torch.save(model.state_dict(), best_checkpoint_path)

    torch.save(model.state_dict(), latest_checkpoint_path)
    if not best_checkpoint_path.exists():
        torch.save(model.state_dict(), best_checkpoint_path)

    save_dataset(samples, str(dataset_path))
    subset_metrics = {
        source_type: _subset_metrics(
            model,
            [sample for sample in samples if sample.source_type == source_type],
            criterion,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            instruction_seq_len=args.instruction_seq_len,
        )
        for source_type in sorted({sample.source_type for sample in samples})
    }
    training_summary = {
        "schema_version": "orchestration_transformer_training_summary_v1",
        "status": "completed",
        "run_name": args.run_name,
        "seed": int(args.seed),
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "config_digest": config_digest,
        "num_samples": len(samples),
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "subset_metrics": subset_metrics,
        "artifacts": {
            "dataset": str(dataset_path),
            "dataset_summary": str(dataset_summary_path),
            "model_config": str(model_config_path),
            "preconditions": str(preconditions_path),
            "history": str(history_path),
            "subset_metrics": str(subset_metrics_path),
            "checkpoint_latest": str(latest_checkpoint_path),
            "checkpoint_best": str(best_checkpoint_path),
            "runtime_dataset_ref": dataset_info.get("dataset_path"),
            "runtime_summary_ref": dataset_info.get("runtime_summary_path"),
        },
    }

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(preconditions_path, execution_preconditions)
    _write_json(history_path, {"history": history})
    _write_json(subset_metrics_path, subset_metrics)
    _write_json(training_summary_path, training_summary)

    result = {
        "checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "preconditions": str(preconditions_path),
        "history": str(history_path),
        "subset_metrics": str(subset_metrics_path),
        "training_summary": str(training_summary_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "orchestration_transformer",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        datapack_ids = [_sample_id(sample, idx) for idx, sample in enumerate(samples)]
        runner.set_eligible_datapacks(datapack_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for idx, sample in enumerate(samples):
            runner.record_sample(
                str(sample.context.task_type or "orchestration"),
                datapack_id=_sample_id(sample, idx),
                slice_id=f"{sample.source_type}:{_sample_id(sample, idx)}",
            )
        for audit in _build_trajectory_audits(samples):
            runner.add_trajectory_audit(audit)
        runner.update_step(optimizer_steps)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "runtime_like_count": int(dataset_summary.get("runtime_like_count", 0)),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="orchestration_transformer",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "orchestration_transformer"},
            promotion_policy_snapshot={},
            source_domain_coverage=dataset_summary.get("source_domain_coverage", {}),
            receipt_label_coverage={"runtime_like_samples": int(dataset_summary.get("runtime_like_count", 0))},
            metadata={
                "trajectory_audit_kind": "orchestration_sample_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "tool_prediction_contract": "first_tool_only_v1",
                "dataset_source": dataset_info.get("dataset_source"),
            },
        )
        runner.register_artifact("orchestration_training_dataset", dataset_path)
        runner.register_artifact("orchestration_dataset_summary", dataset_summary_path)
        runner.register_artifact("orchestration_model_config", model_config_path)
        runner.register_artifact("orchestration_training_preconditions", preconditions_path)
        runner.register_artifact("orchestration_training_history", history_path)
        runner.register_artifact("orchestration_subset_metrics", subset_metrics_path)
        runner.register_artifact("orchestration_training_summary", training_summary_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="orchestration_transformer_latest",
                model_family="orchestration_transformer",
                model_version="orchestration_transformer_v1",
                path=latest_checkpoint_path,
                step=optimizer_steps,
                epoch=args.epochs,
                metadata={
                    "config_digest": config_digest,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                },
            )
        )
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="orchestration_transformer_best",
                model_family="orchestration_transformer",
                model_version="orchestration_transformer_v1",
                path=best_checkpoint_path,
                step=optimizer_steps,
                epoch=best_epoch,
                is_best=True,
                metadata={
                    "best_val_accuracy": best_val_acc,
                    "dataset_digest": dataset_summary.get("dataset_digest"),
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
            "training_kind": "orchestration_transformer",
            "runtime_export_dir": args.runtime_export_dir,
            "dataset_json": args.dataset_json,
            "runtime_summary_json": args.runtime_summary_json,
            "num_samples": args.num_samples,
            "use_mixed_dataset": args.use_mixed_dataset,
            "econ_semantic_ratio": args.econ_semantic_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden": args.hidden,
            "ctx_dim": args.ctx_dim,
            "vocab_size": args.vocab_size,
            "instruction_seq_len": args.instruction_seq_len,
            "seed": args.seed,
            "run_name": args.run_name,
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
            num_episodes=max(1, args.num_samples),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="orchestration_transformer",
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
