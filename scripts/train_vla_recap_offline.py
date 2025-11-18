#!/usr/bin/env python3
"""
Offline RECAP VLA head training (advantage + distributional value).

Consumes RECAP JSONL datasets and trains lightweight heads on CPU.
Deterministic given the same inputs/seed and JSON-safe outputs.
"""
import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.vla.recap_heads import (
    AdvantageConditioningConfig,
    AdvantageConditioningHead,
    DistributionalValueConfig,
    DistributionalValueHead,
)


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
    """
    Small dual-head network:
    - Advantage head: predicts advantage bin.
    - Distributional value head: per-metric atom logits.
    """

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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_recap_jsonl(paths: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for path in sorted(paths):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    # Deterministic ordering by (episode_id, timestep, sampler_strategy)
    records.sort(key=lambda r: (r.get("episode_id", ""), r.get("timestep", 0), r.get("sampler_strategy") or ""))
    return records


def _infer_metrics(records: List[Dict], override: List[str] = None) -> List[str]:
    if override:
        return list(override)
    metrics = set()
    for rec in records:
        metrics.update(rec.get("metrics", {}).keys())
    return sorted(metrics)


def _collect_categories(records: List[Dict], field: str) -> List[str]:
    vals = sorted({rec.get(field) for rec in records if rec.get(field) is not None})
    return [str(v) for v in vals]


def _build_feature_vector(
    rec: Dict,
    metrics: List[str],
    metric_stats: Dict[str, Dict[str, float]],
    categories: Dict[str, List[str]],
) -> List[float]:
    features: List[float] = []
    metric_values = rec.get("metrics", {}) or {}
    for m in metrics:
        stats = metric_stats[m]
        val = float(metric_values.get(m, 0.0))
        rng = stats["max"] - stats["min"]
        norm = 0.0 if rng == 0 else (val - stats["mean"]) / max(rng, 1e-6)
        features.append(norm)

    for field, cats in categories.items():
        one_hot = [0.0] * len(cats)
        if cats:
            val = rec.get(field)
            if val is not None and str(val) in cats:
                idx = cats.index(str(val))
                one_hot[idx] = 1.0
        features.extend(one_hot)
    return features


def _quantize_metric(value: float, support: Tuple[float, float], num_atoms: int) -> int:
    lo, hi = support
    if hi == lo:
        return 0
    pos = (value - lo) / max(hi - lo, 1e-6)
    pos = min(max(pos, 0.0), 1.0)
    return int(round(pos * (num_atoms - 1)))


def prepare_examples(
    records: List[Dict],
    metrics: List[str],
    advantage_config: AdvantageConditioningConfig,
    value_config: DistributionalValueConfig,
) -> Tuple[List[RecapExample], Dict[str, List[float]]]:
    # Metric stats for normalization and supports
    metric_stats: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [float((rec.get("metrics", {}) or {}).get(m, 0.0)) for rec in records]
        vals = vals or [0.0]
        metric_stats[m] = {
            "min": min(vals),
            "max": max(vals),
            "mean": float(sum(vals) / len(vals)),
        }

    categories = {
        "sampler_strategy": _collect_categories(records, "sampler_strategy"),
        "curriculum_phase": _collect_categories(records, "curriculum_phase"),
        "objective_preset": _collect_categories(records, "objective_preset"),
    }

    examples: List[RecapExample] = []
    for rec in records:
        features = _build_feature_vector(rec, metrics, metric_stats, categories)
        adv = float(rec.get("advantage", 0.0))
        adv_bin = advantage_config.compute_bin(adv)
        metric_targets: List[int] = []
        metric_vals = rec.get("metrics", {}) or {}
        for m in metrics:
            support = value_config.value_supports.get(m, (metric_stats[m]["min"], metric_stats[m]["max"]))
            metric_targets.append(_quantize_metric(float(metric_vals.get(m, 0.0)), support, value_config.num_atoms))
        examples.append(RecapExample(features=features, advantage=adv, advantage_bin=adv_bin, metric_targets=metric_targets))
    return examples, categories


def _evaluate(
    model: RecapHeadsModel,
    loader: DataLoader,
    num_metrics: int,
    num_atoms: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for feats, adv_bins, targets in loader:
            feats = feats.to(device)
            adv_bins = adv_bins.to(device)
            targets = targets.to(device)
            outputs = model(feats)
            preds = torch.argmax(outputs["advantage_logits"], dim=1)
            adv_correct += (preds == adv_bins).sum().item()
            total += adv_bins.numel()
    acc = (adv_correct / total) if total else 0.0
    return {"advantage_accuracy": acc}


def train_offline(
    dataset_paths: List[str],
    output_dir: str = "results/vla_recap",
    checkpoint_dir: str = "checkpoints/vla_recap",
    advantage_bins: List[float] = None,
    metrics: List[str] = None,
    num_atoms: int = 8,
    hidden_dim: int = 64,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    log_csv: bool = True,
    run_name: str = "recap_vla",
) -> Dict[str, Any]:
    set_seeds(seed)
    device = torch.device("cpu")
    records = _load_recap_jsonl(dataset_paths)
    if not records:
        raise ValueError("No records found in provided datasets.")

    advantage_bins = advantage_bins or [-1.0, 0.0, 1.0]
    metrics = _infer_metrics(records, metrics)
    adv_config = AdvantageConditioningConfig(advantage_bins=advantage_bins)

    # Supports for each metric derived from dataset values
    supports = {}
    for m in metrics:
        vals = [float((rec.get("metrics", {}) or {}).get(m, 0.0)) for rec in records] or [0.0]
        supports[m] = (min(vals), max(vals) if max(vals) != min(vals) else min(vals) + 1.0)
    value_config = DistributionalValueConfig(metrics=metrics, num_atoms=num_atoms, value_supports=supports)

    examples, categories = prepare_examples(records, metrics, adv_config, value_config)
    feature_dim = len(examples[0].features)

    dataset = RecapDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))

    adv_head = AdvantageConditioningHead(adv_config, input_dim=feature_dim, hidden_dim=hidden_dim)
    val_head = DistributionalValueHead(value_config, input_dim=feature_dim, hidden_dim=hidden_dim)
    model = RecapHeadsModel(feature_dim, adv_head, val_head).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    value_loss_fn = nn.CrossEntropyLoss()
    adv_loss_fn = nn.CrossEntropyLoss()

    metrics_history: List[Dict[str, float]] = []
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_path = Path(output_dir) / f"{run_name}_metrics.csv"
    csv_writer = None
    if log_csv:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "adv_loss", "value_loss", "total_loss", "adv_accuracy"])
    else:
        csv_file = None

    for epoch in range(1, epochs + 1):
        model.train()
        adv_losses = []
        value_losses = []
        total_losses = []
        for feats, adv_bins, targets in loader:
            feats = feats.to(device)
            adv_bins = adv_bins.to(device)
            targets = targets.to(device)
            outputs = model(feats)
            adv_logits = outputs["advantage_logits"]
            value_logits = outputs["value_logits"].view(feats.shape[0], len(metrics), num_atoms)
            adv_loss = adv_loss_fn(adv_logits, adv_bins)
            v_loss = 0.0
            for mi in range(len(metrics)):
                v_loss = v_loss + value_loss_fn(value_logits[:, mi, :], targets[:, mi])
            v_loss = v_loss / float(len(metrics))
            loss = adv_loss + v_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            adv_losses.append(adv_loss.item())
            value_losses.append(v_loss.item())
            total_losses.append(loss.item())

        eval_stats = _evaluate(model, loader, num_metrics=len(metrics), num_atoms=num_atoms, device=device)
        epoch_stats = {
            "epoch": epoch,
            "adv_loss": float(np.mean(adv_losses)),
            "value_loss": float(np.mean(value_losses)),
            "total_loss": float(np.mean(total_losses)),
            "adv_accuracy": eval_stats["advantage_accuracy"],
        }
        metrics_history.append(epoch_stats)
        print(json.dumps({"event": "epoch_complete", "data": epoch_stats}, sort_keys=True))
        if csv_writer:
            csv_writer.writerow([epoch, epoch_stats["adv_loss"], epoch_stats["value_loss"], epoch_stats["total_loss"], epoch_stats["adv_accuracy"]])

    if csv_writer:
        csv_file.close()

    ckpt_path = Path(checkpoint_dir) / f"{run_name}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "advantage_bins": advantage_bins,
            "metrics": metrics,
            "num_atoms": num_atoms,
            "categories": categories,
            "feature_dim": feature_dim,
            "history": metrics_history,
        },
        ckpt_path,
    )
    print(json.dumps({"event": "checkpoint_saved", "path": str(ckpt_path)}, sort_keys=True))
    return {
        "history": metrics_history,
        "checkpoint": str(ckpt_path),
        "csv": str(csv_path) if log_csv else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RECAP VLA heads offline.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Paths to RECAP JSONL datasets.")
    parser.add_argument("--output-dir", default="results/vla_recap", help="Directory for metric logs.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vla_recap", help="Directory for checkpoints.")
    parser.add_argument("--advantage-bins", nargs="+", type=float, default=[-1.0, 0.0, 1.0], help="Sorted advantage bin thresholds.")
    parser.add_argument("--metrics", nargs="+", help="Metrics to train on; defaults to union of dataset metrics.")
    parser.add_argument("--num-atoms", type=int, default=8, help="Number of atoms for distributional value prediction.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden size for the lightweight MLPs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    parser.add_argument("--run-name", default="recap_vla", help="Prefix for metric/ckpt files.")
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV logging.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_offline(
        dataset_paths=args.datasets,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        advantage_bins=args.advantage_bins,
        metrics=args.metrics,
        num_atoms=args.num_atoms,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        log_csv=not args.no_csv,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
