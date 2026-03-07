#!/usr/bin/env python3
"""Train learned shadow pricing/data-value/regal-support models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.data_value_models import train_data_value_model
from src.learning.pricing_models import train_pricing_delta_model
from src.learning.regal_support_models import train_regal_support_model
from src.replay.dataset import load_replay_dataset
from src.utils.config_digest import sha256_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned shadow pricing/value/regal-support models")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    dataset = load_replay_dataset(args.dataset_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    training_cfg = dict(config.get("training", {}) or {})
    device_name = str(training_cfg.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    seed = int(training_cfg.get("seed", 42))
    epochs = int(training_cfg.get("epochs", 8))
    lr = float(training_cfg.get("lr", 1e-3))
    hidden_dim = int(training_cfg.get("hidden_dim", 64))

    pricing_model, pricing_metrics = train_pricing_delta_model(
        dataset.episodes,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )
    data_model, data_metrics = train_data_value_model(
        dataset.episodes,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )
    regal_model, regal_metrics = train_regal_support_model(
        dataset.episodes,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )

    pricing_path = output_root / "pricing_delta.pt"
    data_path = output_root / "data_value.pt"
    regal_path = output_root / "regal_support.pt"
    _save_checkpoint(pricing_path, pricing_model, pricing_metrics, config, dataset.manifest.dataset_digest, hidden_dim)
    _save_checkpoint(data_path, data_model, data_metrics, config, dataset.manifest.dataset_digest, hidden_dim)
    _save_checkpoint(regal_path, regal_model, regal_metrics, config, dataset.manifest.dataset_digest, hidden_dim)

    summary = {
        "dataset_digest": dataset.manifest.dataset_digest,
        "config_digest": sha256_json(config),
        "device": device_name,
        "checkpoints": {
            "pricing_delta": str(pricing_path),
            "data_value": str(data_path),
            "regal_support": str(regal_path),
        },
        "metrics": {
            "pricing_delta": pricing_metrics,
            "data_value": data_metrics,
            "regal_support": regal_metrics,
        },
    }
    summary_path = output_root / "shadow_model_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metrics: dict,
    config: dict,
    dataset_digest: str,
    hidden_dim: int,
) -> None:
    first_weight = next(iter(model.state_dict().values()))
    input_dim = int(first_weight.shape[-1])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "training_config": config,
            "dataset_digest": dataset_digest,
            "config_digest": sha256_json(config),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "model_version": metrics.get("model_version"),
        },
        path,
    )


if __name__ == "__main__":
    main()
