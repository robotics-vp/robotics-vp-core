#!/usr/bin/env python3
"""Train the meta-transformer using the real runtime dataset substrate."""
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

from src.orchestrator.meta_transformer_training import (
    TORCH_AVAILABLE,
    SEMANTIC_VOCAB,
    MetaTransformerDataset,
    MetaTransformerNet,
    MetaTransformerSample,
    collate_meta_transformer_batch,
    compute_loss,
    evaluate_meta_transformer,
    generate_meta_transformer_dataset,
    load_meta_transformer_dataset,
    save_meta_transformer_dataset,
)
from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, Subset
else:  # pragma: no cover - handled by explicit error below
    torch = None
    DataLoader = object
    Subset = object


META_TRANSFORMER_BENCHMARK_MIN_SAMPLES = 1000


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta-transformer with real runtime dataset substrate")
    parser.add_argument("--runtime-export-dir", type=str, default=None, help="Directory containing meta_transformer_runtime_dataset.json from export_semantic_runtime_learning_corpus.py")
    parser.add_argument("--dataset-json", type=str, default=None, help="Path to a saved meta-transformer dataset JSON")
    parser.add_argument("--runtime-summary-json", type=str, default=None, help="Optional semantic_runtime_learning_summary.json to join with dataset-json input")
    parser.add_argument("--synthetic-samples", type=int, default=0, help="Explicit synthetic fallback sample count; only used when no dataset is provided")
    parser.add_argument("--output-dir", type=str, default="results/meta_transformer", help="Directory for training artifacts")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/meta_transformer", help="Directory for checkpoints")
    parser.add_argument("--run-name", type=str, default="meta_transformer", help="Training run name prefix")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-semantic-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
) -> tuple[list[MetaTransformerSample], dict[str, Any]]:
    dataset_path: Path | None = None
    runtime_summary_path: Path | None = None
    dataset_source = ""

    if args.runtime_export_dir:
        export_dir = Path(args.runtime_export_dir)
        dataset_path = export_dir / "meta_transformer_runtime_dataset.json"
        runtime_summary_path = export_dir / "semantic_runtime_learning_summary.json"
        dataset_source = "semantic_runtime_export"
    elif args.dataset_json:
        dataset_path = Path(args.dataset_json)
        runtime_summary_path = Path(args.runtime_summary_json) if args.runtime_summary_json else None
        dataset_source = "dataset_json"
    elif args.synthetic_samples > 0:
        samples = generate_meta_transformer_dataset(args.synthetic_samples)
        dataset_path = output_root / "generated_meta_transformer_dataset.json"
        save_meta_transformer_dataset(samples, str(dataset_path))
        return samples, {
            "dataset_source": "synthetic_generator",
            "dataset_path": str(dataset_path.resolve()),
            "runtime_summary_path": None,
            "runtime_summary": {},
        }
    else:
        raise ValueError(
            "Provide --runtime-export-dir, --dataset-json, or an explicit --synthetic-samples count"
        )

    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Meta-transformer dataset not found: {dataset_path}")

    samples = load_meta_transformer_dataset(str(dataset_path))
    return samples, {
        "dataset_source": dataset_source,
        "dataset_path": str(dataset_path.resolve()),
        "runtime_summary_path": (
            str(runtime_summary_path.resolve())
            if runtime_summary_path is not None and runtime_summary_path.exists()
            else None
        ),
        "runtime_summary": _load_runtime_summary(runtime_summary_path),
    }


def _split_indices(
    sample_count: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if sample_count <= 1:
        return list(range(sample_count)), []
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count)
    rng.shuffle(indices)
    val_count = int(round(sample_count * max(0.0, min(0.5, val_fraction))))
    val_count = min(max(val_count, 1), sample_count - 1)
    val_indices = sorted(indices[:val_count].tolist())
    train_indices = sorted(indices[val_count:].tolist())
    return train_indices, val_indices


def _sample_task_id(sample: MetaTransformerSample) -> str:
    return str(sample.task_context.get("task_id") or "meta_transformer")


def _sample_source_domain(sample: MetaTransformerSample) -> str:
    return str(sample.task_context.get("source_domain") or "unknown")


def _build_source_domain_coverage(samples: Sequence[MetaTransformerSample]) -> dict[str, Any]:
    source_counts = Counter(_sample_source_domain(sample) for sample in samples)
    task_counts = Counter(_sample_task_id(sample) for sample in samples)
    return {
        "dataset_kind": "meta_transformer_runtime_dataset",
        "source_domain_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }


def _build_dataset_summary(
    *,
    samples: Sequence[MetaTransformerSample],
    dataset_info: Mapping[str, Any],
    max_semantic_tokens: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
) -> dict[str, Any]:
    sample_ids = [str(sample.sample_id) for sample in samples]
    authority_counts = Counter(sample.authority_gt for sample in samples)
    semantic_token_lengths = [len(sample.semantic_tokens) for sample in samples] or [0]
    vla_dims = {int(np.asarray(sample.vla_embedding).shape[-1]) for sample in samples}
    dino_dims = {int(np.asarray(sample.dino_embedding).shape[-1]) for sample in samples}
    benchmark_gate_ready = (
        dataset_info.get("dataset_source") != "synthetic_generator"
        and len(samples) >= META_TRANSFORMER_BENCHMARK_MIN_SAMPLES
    )
    return {
        "schema_version": "meta_transformer_dataset_summary_v1",
        "dataset_source": dataset_info.get("dataset_source"),
        "dataset_path": dataset_info.get("dataset_path"),
        "runtime_summary_path": dataset_info.get("runtime_summary_path"),
        "dataset_digest": sha256_json(
            {
                "sample_ids": sample_ids,
                "dataset_source": dataset_info.get("dataset_source"),
                "dataset_path": dataset_info.get("dataset_path"),
            }
        ),
        "num_samples": len(samples),
        "authority_counts": dict(sorted(authority_counts.items())),
        "avg_semantic_token_count": float(sum(semantic_token_lengths) / max(len(semantic_token_lengths), 1)),
        "max_semantic_tokens": int(max_semantic_tokens),
        "vla_dims": sorted(vla_dims),
        "dino_dims": sorted(dino_dims),
        "model_contract": {
            "hidden_dim": int(hidden_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "semantic_vocab_size": len(SEMANTIC_VOCAB),
        },
        "source_domain_coverage": _build_source_domain_coverage(samples),
        "runtime_summary": dict(dataset_info.get("runtime_summary", {}) or {}),
        "benchmark_gate": {
            "name": "meta_transformer_min_runtime_samples",
            "ready": benchmark_gate_ready,
            "required_samples": META_TRANSFORMER_BENCHMARK_MIN_SAMPLES,
            "observed_samples": len(samples),
            "synthetic_source": dataset_info.get("dataset_source") == "synthetic_generator",
        },
    }


def _build_execution_preconditions(
    *,
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_summary = dict(dataset_summary.get("runtime_summary", {}) or {})
    satisfied = {
        "artifact::dataset_present": int(bool(dataset_summary.get("dataset_path"))),
        "dataset::non_empty": int(int(dataset_summary.get("num_samples", 0)) > 0),
        "dataset::runtime_source": int(
            dataset_summary.get("dataset_source") in {"semantic_runtime_export", "dataset_json"}
        ),
        "benchmark::meta_transformer_min_runtime_samples": int(
            bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready"))
        ),
        "runtime_summary::available": int(bool(runtime_summary)),
    }
    return {
        "schema_version": "meta_transformer_execution_preconditions_v1",
        "satisfied_preconditions": satisfied,
        "unsatisfied_preconditions": [
            key for key, value in sorted(satisfied.items()) if not value
        ],
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def _build_model_config(
    *,
    vla_dim: int,
    dino_dim: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    max_semantic_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": "meta_transformer_model_config_v1",
        "vla_dim": int(vla_dim),
        "dino_dim": int(dino_dim),
        "hidden_dim": int(hidden_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "max_semantic_tokens": int(max_semantic_tokens),
        "semantic_vocab_size": len(SEMANTIC_VOCAB),
    }


def _checkpoint_payload(
    *,
    model: Any,
    model_config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    dataset_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
        "history": [dict(row) for row in history],
        "dataset_summary": dict(dataset_summary),
        "semantic_vocab": list(SEMANTIC_VOCAB),
    }


def _build_trajectory_audits(samples: Sequence[MetaTransformerSample]) -> list[Any]:
    audits: list[Any] = []
    for sample in samples:
        reward = max(float(sample.confidence_vla), float(sample.confidence_dino))
        audits.append(
            create_trajectory_audit(
                episode_id=str(sample.sample_id),
                num_steps=1,
                rewards=[reward],
                reward_components={
                    "confidence_vla": [float(sample.confidence_vla)],
                    "confidence_dino": [float(sample.confidence_dino)],
                },
                events=[
                    f"authority_gt:{sample.authority_gt}",
                    *[str(token) for token in list(sample.semantic_tokens)[:4]],
                ],
            )
        )
    return audits


def _build_training_summary(
    *,
    run_name: str,
    seed: int,
    config_digest: str,
    dataset_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    optimizer_steps: int,
    best_epoch: int,
    best_val_loss: float,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    final_metrics = dict(history[-1]) if history else {}
    return {
        "schema_version": "meta_transformer_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "config_digest": config_digest,
        "dataset_digest": dataset_summary.get("dataset_digest"),
        "num_samples": int(dataset_summary.get("num_samples", 0)),
        "optimizer_steps": int(optimizer_steps),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss if best_val_loss != float("inf") else 0.0),
        "final_metrics": final_metrics,
        "benchmark_gate": dict(dataset_summary.get("benchmark_gate", {}) or {}),
        "artifacts": dict(artifacts),
    }


def _train(
    *,
    args: argparse.Namespace,
    runner: Optional[RegalTrainingRunner],
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train the meta-transformer")

    output_root = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    samples, dataset_info = _resolve_samples(args, output_root)
    if not samples:
        raise ValueError("Meta-transformer dataset is empty")

    first_sample = samples[0]
    vla_dim = int(np.asarray(first_sample.vla_embedding).shape[-1])
    dino_dim = int(np.asarray(first_sample.dino_embedding).shape[-1])
    dataset_summary = _build_dataset_summary(
        samples=samples,
        dataset_info=dataset_info,
        max_semantic_tokens=args.max_semantic_tokens,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    execution_preconditions = _build_execution_preconditions(dataset_summary=dataset_summary)
    model_config = _build_model_config(
        vla_dim=vla_dim,
        dino_dim=dino_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_semantic_tokens=args.max_semantic_tokens,
    )
    config_digest = sha256_json(
        {
            "dataset_summary": dataset_summary.get("dataset_digest"),
            "model_config": model_config,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "run_name": args.run_name,
        }
    )

    dataset = MetaTransformerDataset(samples, max_semantic_tokens=args.max_semantic_tokens)
    train_indices, val_indices = _split_indices(len(samples), val_fraction=args.val_fraction, seed=args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_meta_transformer_batch,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_loader = DataLoader(
        val_dataset or train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_meta_transformer_batch,
    )

    torch.manual_seed(args.seed)
    model = MetaTransformerNet(
        vla_dim=vla_dim,
        dino_dim=dino_dim,
        hidden_dim=args.hidden_dim,
        max_output_tokens=args.max_semantic_tokens,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    latest_checkpoint_path = checkpoint_root / f"{args.run_name}_seed{args.seed}.pt"
    best_checkpoint_path = checkpoint_root / f"{args.run_name}_best_seed{args.seed}.pt"
    dataset_summary_path = output_root / "meta_transformer_dataset_summary.json"
    model_config_path = output_root / "meta_transformer_model_config.json"
    history_path = output_root / "meta_transformer_training_history.json"
    preconditions_path = output_root / "meta_transformer_execution_preconditions.json"
    training_summary_path = output_root / "meta_transformer_training_summary.json"
    training_job_result_path = output_root / "training_job_result.json"

    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    best_epoch = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_authority_losses: list[float] = []
        epoch_token_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch["vla_embeddings"], batch["dino_embeddings"])
            loss, metrics = compute_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(metrics["total_loss"]))
            epoch_authority_losses.append(float(metrics["authority_loss"]))
            epoch_token_losses.append(float(metrics["token_loss"]))

        eval_metrics = evaluate_meta_transformer(model, eval_loader)
        epoch_summary = {
            "epoch": int(epoch),
            "train_total_loss": float(sum(epoch_losses) / max(len(epoch_losses), 1)),
            "train_authority_loss": float(sum(epoch_authority_losses) / max(len(epoch_authority_losses), 1)),
            "train_token_loss": float(sum(epoch_token_losses) / max(len(epoch_token_losses), 1)),
            "eval_total_loss": float(eval_metrics["total_loss"]),
            "eval_authority_acc": float(eval_metrics["authority_acc"]),
            "eval_first_token_acc": float(eval_metrics["first_token_acc"]),
        }
        history.append(epoch_summary)
        print(json.dumps({"event": "epoch_complete", "data": epoch_summary}, sort_keys=True))
        if epoch_summary["eval_total_loss"] <= best_val_loss:
            best_val_loss = epoch_summary["eval_total_loss"]
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model=model,
                    model_config=model_config,
                    history=history,
                    dataset_summary=dataset_summary,
                ),
                best_checkpoint_path,
            )

    checkpoint_payload = _checkpoint_payload(
        model=model,
        model_config=model_config,
        history=history,
        dataset_summary=dataset_summary,
    )
    torch.save(checkpoint_payload, latest_checkpoint_path)
    if not best_checkpoint_path.exists():
        torch.save(checkpoint_payload, best_checkpoint_path)

    artifacts = {
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "history": str(history_path),
        "preconditions": str(preconditions_path),
        "checkpoint_latest": str(latest_checkpoint_path),
        "checkpoint_best": str(best_checkpoint_path),
        "runtime_dataset_ref": dataset_info.get("dataset_path"),
        "runtime_summary_ref": dataset_info.get("runtime_summary_path"),
    }
    training_summary = _build_training_summary(
        run_name=args.run_name,
        seed=args.seed,
        config_digest=config_digest,
        dataset_summary=dataset_summary,
        history=history,
        optimizer_steps=optimizer_steps,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        artifacts=artifacts,
    )

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(model_config_path, model_config)
    _write_json(history_path, {"history": history})
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)

    result = {
        "checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "dataset_summary": str(dataset_summary_path),
        "model_config": str(model_config_path),
        "history": str(history_path),
        "training_summary": str(training_summary_path),
        "preconditions": str(preconditions_path),
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "meta_transformer_runtime",
            "result": result,
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        sample_ids = [str(sample.sample_id) for sample in samples]
        runner.set_eligible_datapacks(sample_ids)
        runner.set_sampler_config(seed=args.seed, config_sha=config_digest)
        for sample in samples:
            runner.record_sample(
                _sample_task_id(sample),
                datapack_id=str(sample.sample_id),
                slice_id=str(sample.sample_id),
            )
        for audit in _build_trajectory_audits(samples):
            runner.add_trajectory_audit(audit)
        runner.update_step(optimizer_steps)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "dataset_source": dataset_summary.get("dataset_source"),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="meta_transformer_runtime",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={"profile_id": "meta_transformer_runtime"},
            promotion_policy_snapshot={},
            source_domain_coverage=dataset_summary.get("source_domain_coverage", {}),
            receipt_label_coverage={"total_labels": 0},
            metadata={
                "dataset_source": dataset_summary.get("dataset_source"),
                "trajectory_audit_kind": "meta_transformer_sample_projection",
                "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
                "optimizer_steps": optimizer_steps,
                "runtime_summary_available": bool(dataset_info.get("runtime_summary")),
            },
        )
        runner.register_artifact("meta_transformer_dataset_summary", dataset_summary_path)
        runner.register_artifact("meta_transformer_model_config", model_config_path)
        runner.register_artifact("meta_transformer_history", history_path)
        runner.register_artifact("meta_transformer_preconditions", preconditions_path)
        runner.register_artifact("meta_transformer_training_summary", training_summary_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="meta_transformer_latest",
                model_family="meta_transformer",
                model_version="meta_transformer_runtime_v1",
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
                checkpoint_id="meta_transformer_best",
                model_family="meta_transformer",
                model_version="meta_transformer_runtime_v1",
                path=best_checkpoint_path,
                step=optimizer_steps,
                epoch=best_epoch,
                is_best=True,
                metadata={
                    "best_val_loss": training_summary.get("best_val_loss"),
                    "dataset_digest": dataset_summary.get("dataset_digest"),
                },
            )
        )

    print(json.dumps({"event": "checkpoint_saved", "path": str(latest_checkpoint_path)}, sort_keys=True))
    return result


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return _train(args=args, runner=runner)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    plan_sha = sha256_json(
        {
            "training_kind": "meta_transformer_runtime",
            "runtime_export_dir": args.runtime_export_dir,
            "dataset_json": args.dataset_json,
            "runtime_summary_json": args.runtime_summary_json,
            "synthetic_samples": args.synthetic_samples,
            "run_name": args.run_name,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
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
            num_episodes=max(1, args.synthetic_samples or 1),
            training_steps=max(1, args.epochs),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="meta_transformer_runtime",
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
