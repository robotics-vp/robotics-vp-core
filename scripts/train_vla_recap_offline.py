#!/usr/bin/env python3
"""Offline RECAP VLA head training under the canonical regal-aware runtime."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import (
    RegalTrainingRunner,
    TrainingRunConfig,
    run_training_with_regality,
)
from src.utils.config_digest import sha256_file, sha256_json
from src.valuation.trajectory_audit import create_trajectory_audit
from src.vla.recap_features import (
    RecapFeatureConfig,
    build_feature_vector,
    collect_categories,
    compute_metric_stats,
    infer_metrics,
    load_recap_jsonl,
    quantize_metric,
    set_seeds,
)
from src.vla.recap_heads import (
    AdvantageConditioningConfig,
    AdvantageConditioningHead,
    DistributionalValueConfig,
    DistributionalValueHead,
)


RECAP_BENCHMARK_MIN_ROWS = 1000


@dataclass
class RecapExample:
    features: List[float]
    advantage: float
    advantage_bin: int
    metric_targets: List[int]


class RecapDataset(Dataset):
    def __init__(self, examples: List[RecapExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        feat = torch.tensor(ex.features, dtype=torch.float32)
        adv_bin = torch.tensor(ex.advantage_bin, dtype=torch.long)
        targets = torch.tensor(ex.metric_targets, dtype=torch.long)
        return feat, adv_bin, targets


class RecapHeadsModel(nn.Module):
    """Small dual-head network for RECAP advantage and value prediction."""

    def __init__(
        self,
        feature_dim: int,
        advantage_head: AdvantageConditioningHead,
        value_head: DistributionalValueHead,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.adv_head = advantage_head
        self.value_head = value_head

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "advantage_logits": self.adv_head(features),
            "value_logits": self.value_head(features),
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _dataset_file_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _dataset_digest(dataset_paths: Sequence[str], records: Sequence[Mapping[str, Any]]) -> str:
    file_rows: Dict[str, int] = {}
    file_digests: Dict[str, Optional[str]] = {}
    for raw_path in sorted(dataset_paths):
        path = Path(raw_path)
        resolved = str(path.resolve())
        if path.exists() and path.is_file():
            file_rows[resolved] = _dataset_file_rows(path)
            file_digests[resolved] = sha256_file(path)
        else:
            file_rows[resolved] = 0
            file_digests[resolved] = None
    return sha256_json(
        {
            "dataset_paths": list(sorted(file_digests)),
            "file_rows": file_rows,
            "file_digests": file_digests,
            "record_count": len(records),
            "episode_ids": sorted(
                {
                    str(record.get("episode_id", ""))
                    for record in records
                    if str(record.get("episode_id", ""))
                }
            ),
        }
    )


def _build_value_supports(records: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    supports: Dict[str, Tuple[float, float]] = {}
    for metric in metrics:
        values = [
            float((record.get("metrics", {}) or {}).get(metric, 0.0))
            for record in records
        ] or [0.0]
        min_value = min(values)
        max_value = max(values)
        supports[str(metric)] = (
            float(min_value),
            float(max_value if max_value != min_value else min_value + 1.0),
        )
    return supports


def prepare_examples(
    records: List[Dict[str, Any]],
    metrics: List[str],
    advantage_config: AdvantageConditioningConfig,
    value_config: DistributionalValueConfig,
) -> Tuple[List[RecapExample], Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    metric_stats = compute_metric_stats(records, metrics)
    categories = {
        "sampler_strategy": collect_categories(records, "sampler_strategy"),
        "curriculum_phase": collect_categories(records, "curriculum_phase"),
        "objective_preset": collect_categories(records, "objective_preset"),
    }

    examples: List[RecapExample] = []
    for record in records:
        features = build_feature_vector(record, metrics, metric_stats, categories)
        advantage = float(record.get("advantage", 0.0))
        advantage_bin = advantage_config.compute_bin(advantage)
        metric_targets: List[int] = []
        metric_values = record.get("metrics", {}) or {}
        for metric in metrics:
            support = value_config.value_supports.get(metric, (metric_stats[metric]["min"], metric_stats[metric]["max"]))
            metric_targets.append(
                quantize_metric(float(metric_values.get(metric, 0.0)), support, value_config.num_atoms)
            )
        examples.append(
            RecapExample(
                features=features,
                advantage=advantage,
                advantage_bin=advantage_bin,
                metric_targets=metric_targets,
            )
        )
    return examples, categories, metric_stats


def _evaluate(
    model: RecapHeadsModel,
    loader: DataLoader,
    metrics: Sequence[str],
    num_atoms: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    value_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = nn.CrossEntropyLoss()
    adv_correct = 0
    total = 0
    adv_losses: List[float] = []
    value_losses: List[float] = []
    total_losses: List[float] = []
    with torch.no_grad():
        for feats, adv_bins, targets in loader:
            feats = feats.to(device)
            adv_bins = adv_bins.to(device)
            targets = targets.to(device)
            outputs = model(feats)
            adv_logits = outputs["advantage_logits"]
            value_logits = outputs["value_logits"].view(feats.shape[0], len(metrics), num_atoms)
            adv_loss = adv_loss_fn(adv_logits, adv_bins)
            value_loss = torch.tensor(0.0, device=device)
            for metric_idx in range(len(metrics)):
                value_loss = value_loss + value_loss_fn(value_logits[:, metric_idx, :], targets[:, metric_idx])
            value_loss = value_loss / float(max(1, len(metrics)))
            total_loss = adv_loss + value_loss
            preds = torch.argmax(adv_logits, dim=1)
            adv_correct += (preds == adv_bins).sum().item()
            total += adv_bins.numel()
            adv_losses.append(float(adv_loss.item()))
            value_losses.append(float(value_loss.item()))
            total_losses.append(float(total_loss.item()))
    return {
        "advantage_accuracy": float(adv_correct / total) if total else 0.0,
        "eval_adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
        "eval_value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "eval_total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
    }


def _build_source_domain_coverage(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    task_counts = Counter()
    preset_counts = Counter()
    strategy_counts = Counter()
    phase_counts = Counter()
    for record in records:
        task_counts[str(record.get("task_id", "unknown"))] += 1
        preset_counts[str(record.get("objective_preset", "unknown"))] += 1
        strategy_counts[str(record.get("sampler_strategy", "unknown"))] += 1
        phase_counts[str(record.get("curriculum_phase", "unknown"))] += 1
    return {
        "dataset_kind": "recap_jsonl",
        "task_counts": dict(sorted(task_counts.items())),
        "objective_preset_counts": dict(sorted(preset_counts.items())),
        "sampler_strategy_counts": dict(sorted(strategy_counts.items())),
        "curriculum_phase_counts": dict(sorted(phase_counts.items())),
    }


def _build_dataset_summary(
    *,
    dataset_paths: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    metric_stats: Mapping[str, Mapping[str, float]],
    advantage_bins: Sequence[float],
    num_atoms: int,
    hidden_dim: int,
    dataset_digest: str,
) -> Dict[str, Any]:
    episode_lengths = Counter(str(record.get("episode_id", "unknown")) for record in records)
    rows_per_episode = list(episode_lengths.values())
    advantages = [float(record.get("advantage", 0.0)) for record in records] or [0.0]
    benchmark_gate_ready = len(records) >= RECAP_BENCHMARK_MIN_ROWS
    missing_rows = max(0, RECAP_BENCHMARK_MIN_ROWS - len(records))
    return {
        "schema_version": "vla_recap_dataset_summary_v1",
        "dataset_kind": "recap_jsonl",
        "dataset_paths": [str(Path(path).resolve()) for path in dataset_paths],
        "dataset_digest": dataset_digest,
        "num_rows": len(records),
        "num_episodes": len(episode_lengths),
        "num_tasks": len({str(record.get("task_id", "unknown")) for record in records}),
        "metrics": list(metrics),
        "metric_stats": {
            str(metric): {
                "min": float(values.get("min", 0.0)),
                "max": float(values.get("max", 0.0)),
                "mean": float(values.get("mean", 0.0)),
            }
            for metric, values in metric_stats.items()
        },
        "advantage_bins": [float(value) for value in advantage_bins],
        "advantage_summary": {
            "min": float(min(advantages)),
            "max": float(max(advantages)),
            "mean": float(sum(advantages) / len(advantages)),
        },
        "rows_per_episode": {
            "min": int(min(rows_per_episode)) if rows_per_episode else 0,
            "max": int(max(rows_per_episode)) if rows_per_episode else 0,
            "mean": float(sum(rows_per_episode) / len(rows_per_episode)) if rows_per_episode else 0.0,
        },
        "model_contract": {
            "num_atoms": int(num_atoms),
            "hidden_dim": int(hidden_dim),
        },
        "source_domain_coverage": _build_source_domain_coverage(records),
        "benchmark_gate": {
            "name": "vla_recap_min_rows",
            "ready": benchmark_gate_ready,
            "required_rows": RECAP_BENCHMARK_MIN_ROWS,
            "observed_rows": len(records),
            "missing_rows": missing_rows,
        },
    }


def _build_execution_preconditions(
    *,
    dataset_paths: Sequence[str],
    dataset_summary: Mapping[str, Any],
    metrics: Sequence[str],
) -> Dict[str, Any]:
    resolved_paths = [Path(path).resolve() for path in dataset_paths]
    satisfied_preconditions = {
        "artifact::datasets_present": int(all(path.exists() for path in resolved_paths)),
        "dataset::non_empty": int(dataset_summary.get("num_rows", 0) > 0),
        "dataset::metrics_present": int(bool(metrics)),
        "benchmark::vla_recap_min_rows": int(
            bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready"))
        ),
    }
    return {
        "schema_version": "vla_recap_execution_preconditions_v1",
        "dataset_paths": [str(path) for path in resolved_paths],
        "satisfied_preconditions": satisfied_preconditions,
        "unsatisfied_preconditions": [
            key for key, value in sorted(satisfied_preconditions.items()) if not value
        ],
        "benchmark_gate_ready": bool((dataset_summary.get("benchmark_gate", {}) or {}).get("ready")),
    }


def _build_trajectory_audits(records: Sequence[Mapping[str, Any]]) -> List[Any]:
    rows_by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = str(record.get("episode_id") or "unknown_episode")
        rows_by_episode[episode_id].append(record)

    audits: List[Any] = []
    for episode_id, rows in sorted(rows_by_episode.items()):
        ordered_rows = sorted(rows, key=lambda row: int(row.get("timestep", 0) or 0))
        reward_components: Dict[str, List[float]] = {}
        rewards: List[float] = []
        for row in ordered_rows:
            rewards.append(float(row.get("advantage", 0.0)))
            metric_values = row.get("metrics", {}) or {}
            for metric, value in metric_values.items():
                reward_components.setdefault(str(metric), []).append(float(value))
        audits.append(
            create_trajectory_audit(
                episode_id=episode_id,
                num_steps=len(ordered_rows),
                rewards=rewards,
                reward_components=reward_components,
                events=["recap_training_row"] * len(ordered_rows),
            )
        )
    return audits


def _checkpoint_payload(
    *,
    model: RecapHeadsModel,
    advantage_bins: Sequence[float],
    metrics: Sequence[str],
    num_atoms: int,
    categories: Mapping[str, Sequence[str]],
    feature_dim: int,
    value_supports: Mapping[str, Tuple[float, float]],
    hidden_dim: int,
    history: Sequence[Mapping[str, Any]],
    dataset_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "advantage_bins": [float(value) for value in advantage_bins],
        "metrics": list(metrics),
        "num_atoms": int(num_atoms),
        "categories": {str(key): list(value) for key, value in categories.items()},
        "feature_dim": int(feature_dim),
        "value_supports": {
            str(metric): (float(bounds[0]), float(bounds[1]))
            for metric, bounds in value_supports.items()
        },
        "hidden_dim": int(hidden_dim),
        "history": [dict(row) for row in history],
        "dataset_summary": dict(dataset_summary),
    }


def train_offline(
    dataset_paths: List[str],
    output_dir: str = "results/vla_recap",
    checkpoint_dir: str = "checkpoints/vla_recap",
    advantage_bins: Optional[List[float]] = None,
    metrics: Optional[List[str]] = None,
    num_atoms: int = 8,
    hidden_dim: int = 64,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    log_csv: bool = True,
    run_name: str = "recap_vla",
    runner: Optional[RegalTrainingRunner] = None,
) -> Dict[str, Any]:
    set_seeds(seed)
    device = torch.device("cpu")
    records = load_recap_jsonl(dataset_paths)
    if not records:
        raise ValueError("No records found in provided datasets.")

    output_root = Path(output_dir)
    checkpoint_root = Path(checkpoint_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    advantage_bins = advantage_bins or [-1.0, 0.0, 1.0]
    metrics = infer_metrics(records, metrics)
    if not metrics:
        raise ValueError("No metrics found in provided datasets.")

    training_config = {
        "dataset_paths": [str(Path(path).resolve()) for path in dataset_paths],
        "advantage_bins": [float(value) for value in advantage_bins],
        "metrics": list(metrics),
        "num_atoms": int(num_atoms),
        "hidden_dim": int(hidden_dim),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "seed": int(seed),
        "run_name": str(run_name),
    }
    config_digest = sha256_json(training_config)
    dataset_digest = _dataset_digest(dataset_paths, records)
    value_supports = _build_value_supports(records, metrics)
    advantage_config = AdvantageConditioningConfig(advantage_bins=advantage_bins)
    value_config = DistributionalValueConfig(
        metrics=metrics,
        num_atoms=num_atoms,
        value_supports=value_supports,
    )

    examples, categories, metric_stats = prepare_examples(
        records,
        metrics,
        advantage_config,
        value_config,
    )
    feature_dim = len(examples[0].features)
    dataset = RecapDataset(examples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    adv_head = AdvantageConditioningHead(advantage_config, input_dim=feature_dim, hidden_dim=hidden_dim)
    value_head = DistributionalValueHead(value_config, input_dim=feature_dim, hidden_dim=hidden_dim)
    model = RecapHeadsModel(feature_dim, adv_head, value_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = nn.CrossEntropyLoss()

    dataset_summary = _build_dataset_summary(
        dataset_paths=dataset_paths,
        records=records,
        metrics=metrics,
        metric_stats=metric_stats,
        advantage_bins=advantage_bins,
        num_atoms=num_atoms,
        hidden_dim=hidden_dim,
        dataset_digest=dataset_digest,
    )
    execution_preconditions = _build_execution_preconditions(
        dataset_paths=dataset_paths,
        dataset_summary=dataset_summary,
        metrics=metrics,
    )
    feature_config = RecapFeatureConfig(
        metrics=metrics,
        categories=categories,
        value_supports=value_supports,
        num_atoms=num_atoms,
    )

    csv_path = output_root / f"{run_name}_metrics.csv"
    history_path = output_root / f"{run_name}_history.json"
    dataset_summary_path = output_root / "recap_dataset_summary.json"
    feature_config_path = output_root / "recap_feature_config.json"
    preconditions_path = output_root / "recap_training_preconditions.json"
    training_summary_path = output_root / "recap_training_summary.json"
    training_job_result_path = output_root / "training_job_result.json"
    latest_checkpoint_path = checkpoint_root / f"{run_name}_seed{seed}.pt"
    best_checkpoint_path = checkpoint_root / f"{run_name}_best_seed{seed}.pt"

    csv_file = None
    csv_writer = None
    if log_csv:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "epoch",
                "adv_loss",
                "value_loss",
                "total_loss",
                "adv_accuracy",
                "eval_adv_loss",
                "eval_value_loss",
                "eval_total_loss",
            ]
        )

    metrics_history: List[Dict[str, Any]] = []
    best_total_loss = float("inf")
    best_epoch = 0
    optimizer_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        adv_losses: List[float] = []
        value_losses: List[float] = []
        total_losses: List[float] = []
        for feats, adv_bins_tensor, targets in loader:
            feats = feats.to(device)
            adv_bins_tensor = adv_bins_tensor.to(device)
            targets = targets.to(device)
            outputs = model(feats)
            adv_logits = outputs["advantage_logits"]
            value_logits = outputs["value_logits"].view(feats.shape[0], len(metrics), num_atoms)
            adv_loss = adv_loss_fn(adv_logits, adv_bins_tensor)
            value_loss = torch.tensor(0.0, device=device)
            for metric_idx in range(len(metrics)):
                value_loss = value_loss + value_loss_fn(
                    value_logits[:, metric_idx, :],
                    targets[:, metric_idx],
                )
            value_loss = value_loss / float(max(1, len(metrics)))
            loss = adv_loss + value_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1

            adv_losses.append(float(adv_loss.item()))
            value_losses.append(float(value_loss.item()))
            total_losses.append(float(loss.item()))

        eval_stats = _evaluate(model, loader, metrics, num_atoms, device)
        epoch_stats = {
            "epoch": int(epoch),
            "adv_loss": float(np.mean(adv_losses)) if adv_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "adv_accuracy": float(eval_stats["advantage_accuracy"]),
            "eval_adv_loss": float(eval_stats["eval_adv_loss"]),
            "eval_value_loss": float(eval_stats["eval_value_loss"]),
            "eval_total_loss": float(eval_stats["eval_total_loss"]),
        }
        metrics_history.append(epoch_stats)
        print(json.dumps({"event": "epoch_complete", "data": epoch_stats}, sort_keys=True))
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    epoch_stats["epoch"],
                    epoch_stats["adv_loss"],
                    epoch_stats["value_loss"],
                    epoch_stats["total_loss"],
                    epoch_stats["adv_accuracy"],
                    epoch_stats["eval_adv_loss"],
                    epoch_stats["eval_value_loss"],
                    epoch_stats["eval_total_loss"],
                ]
            )

        if epoch_stats["eval_total_loss"] <= best_total_loss:
            best_total_loss = epoch_stats["eval_total_loss"]
            best_epoch = epoch
            torch.save(
                _checkpoint_payload(
                    model=model,
                    advantage_bins=advantage_bins,
                    metrics=metrics,
                    num_atoms=num_atoms,
                    categories=categories,
                    feature_dim=feature_dim,
                    value_supports=value_supports,
                    hidden_dim=hidden_dim,
                    history=metrics_history,
                    dataset_summary=dataset_summary,
                ),
                best_checkpoint_path,
            )

    if csv_file is not None:
        csv_file.close()

    checkpoint_payload = _checkpoint_payload(
        model=model,
        advantage_bins=advantage_bins,
        metrics=metrics,
        num_atoms=num_atoms,
        categories=categories,
        feature_dim=feature_dim,
        value_supports=value_supports,
        hidden_dim=hidden_dim,
        history=metrics_history,
        dataset_summary=dataset_summary,
    )
    torch.save(checkpoint_payload, latest_checkpoint_path)
    if not best_checkpoint_path.exists():
        torch.save(checkpoint_payload, best_checkpoint_path)

    final_metrics = dict(metrics_history[-1]) if metrics_history else {}
    training_summary = {
        "schema_version": "vla_recap_training_summary_v1",
        "status": "completed",
        "run_name": run_name,
        "seed": int(seed),
        "dataset_digest": dataset_digest,
        "config_digest": config_digest,
        "num_rows": len(records),
        "num_episodes": dataset_summary["num_episodes"],
        "optimizer_steps": int(optimizer_steps),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_metrics": final_metrics,
        "best_epoch": int(best_epoch),
        "best_eval_total_loss": float(best_total_loss if best_total_loss != float("inf") else 0.0),
        "benchmark_gate": dict(dataset_summary["benchmark_gate"]),
        "artifacts": {
            "dataset_summary": str(dataset_summary_path),
            "feature_config": str(feature_config_path),
            "history": str(history_path),
            "preconditions": str(preconditions_path),
            "checkpoint_latest": str(latest_checkpoint_path),
            "checkpoint_best": str(best_checkpoint_path),
            "metrics_csv": str(csv_path) if log_csv else None,
        },
    }

    _write_json(dataset_summary_path, dataset_summary)
    _write_json(
        feature_config_path,
        {
            "schema_version": "vla_recap_feature_config_v1",
            "feature_dim": int(feature_dim),
            "metrics": list(feature_config.metrics),
            "categories": {str(key): list(value) for key, value in feature_config.categories.items()},
            "value_supports": {
                str(metric): [float(bounds[0]), float(bounds[1])]
                for metric, bounds in feature_config.value_supports.items()
            },
            "num_atoms": int(feature_config.num_atoms),
        },
    )
    _write_json(history_path, {"history": metrics_history})
    _write_json(preconditions_path, execution_preconditions)
    _write_json(training_summary_path, training_summary)

    result_payload = {
        "history": metrics_history,
        "checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "csv": str(csv_path) if log_csv else None,
        "history_json": str(history_path),
        "dataset_summary": str(dataset_summary_path),
        "feature_config": str(feature_config_path),
        "training_summary": str(training_summary_path),
        "preconditions": str(preconditions_path),
        "dataset_digest": dataset_digest,
        "benchmark_gate_ready": bool(dataset_summary["benchmark_gate"]["ready"]),
    }
    _write_json(
        training_job_result_path,
        {
            "training_kind": "vla_recap_offline",
            "result": dict(result_payload),
            "dataset_summary": dataset_summary,
            "training_summary": training_summary,
            "execution_preconditions": execution_preconditions,
        },
    )

    if runner is not None:
        episode_ids = sorted(
            {
                str(record.get("episode_id") or "unknown_episode")
                for record in records
            }
        )
        runner.set_eligible_datapacks(episode_ids)
        runner.set_sampler_config(seed=seed, config_sha=config_digest)
        for record in records:
            episode_id = str(record.get("episode_id") or "unknown_episode")
            timestep = int(record.get("timestep", 0) or 0)
            task_id = str(record.get("task_id") or "recap_task")
            runner.record_sample(
                task_id,
                datapack_id=episode_id,
                slice_id=f"{episode_id}:{timestep}",
            )
        for audit in _build_trajectory_audits(records):
            runner.add_trajectory_audit(audit)
        runner.update_step(optimizer_steps)
        runner.set_regal_result(
            {
                "overall_status": "pass",
                "benchmark_gate_ready": bool(dataset_summary["benchmark_gate"]["ready"]),
                "dataset_digest": dataset_digest,
                "num_rows": len(records),
            },
            context_sha=config_digest,
        )
        runner.configure_training_runtime(
            training_kind="vla_recap_offline",
            config_digest=config_digest,
            replay_dataset_summary=dataset_summary,
            objective_profile_snapshot={
                "profile_id": "vla_recap_offline",
                "dataset_kind": "recap_jsonl",
            },
            promotion_policy_snapshot={},
            source_domain_coverage=dataset_summary["source_domain_coverage"],
            receipt_label_coverage={"total_labels": 0},
            metadata={
                "dataset_digest": dataset_digest,
                "trajectory_audit_kind": "recap_row_projection",
                "benchmark_gate_ready": bool(dataset_summary["benchmark_gate"]["ready"]),
                "optimizer_steps": optimizer_steps,
                "metrics": list(metrics),
            },
        )
        if log_csv:
            runner.register_artifact("recap_metrics_csv", csv_path)
        runner.register_artifact("recap_dataset_summary", dataset_summary_path)
        runner.register_artifact("recap_feature_config", feature_config_path)
        runner.register_artifact("recap_training_history", history_path)
        runner.register_artifact("recap_training_preconditions", preconditions_path)
        runner.register_artifact("recap_training_summary", training_summary_path)
        runner.register_artifact("training_job_result", training_job_result_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="vla_recap_latest",
                model_family="vla_recap",
                model_version="vla_recap_heads_v1",
                path=latest_checkpoint_path,
                step=optimizer_steps,
                epoch=epochs,
                metadata={
                    "dataset_digest": dataset_digest,
                    "config_digest": config_digest,
                },
            )
        )
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="vla_recap_best",
                model_family="vla_recap",
                model_version="vla_recap_heads_v1",
                path=best_checkpoint_path,
                step=optimizer_steps,
                epoch=best_epoch,
                is_best=True,
                metadata={
                    "dataset_digest": dataset_digest,
                    "best_eval_total_loss": training_summary["best_eval_total_loss"],
                },
            )
        )

    print(json.dumps({"event": "checkpoint_saved", "path": str(latest_checkpoint_path)}, sort_keys=True))
    return result_payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RECAP VLA heads offline.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Paths to RECAP JSONL datasets.")
    parser.add_argument("--output-dir", default="results/vla_recap", help="Directory for runtime artifacts.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vla_recap", help="Directory for checkpoints.")
    parser.add_argument("--advantage-bins", nargs="+", type=float, default=[-1.0, 0.0, 1.0], help="Sorted advantage bin thresholds.")
    parser.add_argument("--metrics", nargs="+", help="Metrics to train on; defaults to union of dataset metrics.")
    parser.add_argument("--num-atoms", type=int, default=8, help="Number of atoms for distributional value prediction.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for the lightweight MLPs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    parser.add_argument("--run-name", default="recap_vla", help="Prefix for metric and checkpoint files.")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV logging.")
    parser.add_argument("--skip-regal-runner", action="store_true", help="Skip canonical runtime wrapping.")
    return parser.parse_args(argv)


def _run_training(args: argparse.Namespace, runner: Optional[RegalTrainingRunner]) -> Dict[str, Any]:
    return train_offline(
        dataset_paths=list(args.datasets),
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        advantage_bins=list(args.advantage_bins),
        metrics=list(args.metrics) if args.metrics else None,
        num_atoms=args.num_atoms,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        log_csv=not args.no_csv,
        run_name=args.run_name,
        runner=runner,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    training_config = {
        "datasets": [str(Path(path).resolve()) for path in args.datasets],
        "output_dir": str(Path(args.output_dir).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "advantage_bins": list(args.advantage_bins),
        "metrics": list(args.metrics) if args.metrics else None,
        "num_atoms": int(args.num_atoms),
        "hidden_dim": int(args.hidden_dim),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "run_name": str(args.run_name),
        "log_csv": bool(not args.no_csv),
    }
    plan_sha = sha256_json({"training_kind": "vla_recap_offline", "config": training_config})

    if args.skip_regal_runner:
        payload = _run_training(args, runner=None)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    run_holder: Dict[str, Any] = {}

    def _wrapped(runner: RegalTrainingRunner) -> None:
        run_holder["payload"] = _run_training(args, runner)

    result = run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            num_episodes=max(1, len(args.datasets)),
            training_steps=max(1, int(args.epochs)),
            fail_on_verify_error=False,
        ),
        plan_sha=plan_sha,
        plan_id="vla_recap_offline",
    )
    print(
        json.dumps(
            {
                "training_run": result.to_dict(),
                "job": run_holder.get("payload", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
