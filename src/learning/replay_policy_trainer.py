"""CPU-safe replay policy training and evaluation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.determinism.determinism_context import get_context_summary, set_determinism
from src.learning.replay_policy_model import ReplayPolicyConfig, ReplayPolicyModel
from src.replay.dataset import ReplayDatasetBundle, load_replay_dataset
from src.replay.schema import ReplayStepRecord
from src.utils.config_digest import sha256_json
from src.utils.training_env import configure_training_env, get_device


@dataclass(frozen=True)
class ReplayPolicyTrainResult:
    """Training result metadata for the replay BC policy."""

    checkpoint_path: str
    best_checkpoint_path: str
    metrics_path: str
    summary_path: str
    config_digest: str
    dataset_digest: str
    best_val_mse: float
    train_steps: int
    epochs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "metrics_path": self.metrics_path,
            "summary_path": self.summary_path,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "best_val_mse": float(self.best_val_mse),
            "train_steps": int(self.train_steps),
            "epochs": int(self.epochs),
        }


class ReplayPolicyDataset(Dataset):
    """Torch dataset over canonical replay steps."""

    def __init__(
        self,
        records: Sequence[ReplayStepRecord],
        *,
        obs_dim: int,
        action_dim: int,
        condition_dim: int,
    ) -> None:
        self.records = list(records)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "obs_vector": _pad_vector(record.obs_vector, self.obs_dim),
            "action_vector": _pad_vector(record.action_vector, self.action_dim),
            "condition_vector": _pad_vector(record.condition_vector_values, self.condition_dim),
            "skill_mode": record.skill_mode,
            "reward": float(record.reward),
            "episode_id": record.episode_id,
            "step_idx": int(record.step_idx),
        }


def load_training_config(path: str | Path) -> Dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Training config at {path} must be a mapping")
    return dict(payload)


def train_replay_policy(
    *,
    dataset_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    resume_checkpoint: Optional[str | Path] = None,
) -> ReplayPolicyTrainResult:
    dataset = load_replay_dataset(dataset_dir)
    config = load_training_config(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    training_cfg = dict(config.get("training", {}) or {})
    seed = int(training_cfg.get("seed", 42))
    set_determinism(seed=seed)
    configure_training_env(config)
    device = _resolve_device(config)

    model_cfg = ReplayPolicyConfig(
        obs_dim=max(dataset.manifest.obs_dim, int(config.get("model", {}).get("obs_dim_override", 0) or 0)),
        action_dim=max(dataset.manifest.action_dim, int(config.get("model", {}).get("action_dim_override", 0) or 0)),
        condition_dim=max(dataset.manifest.condition_dim, int(config.get("model", {}).get("condition_dim_override", 0) or 0)),
        skill_modes=list(config.get("model", {}).get("skill_modes", dataset.manifest.skill_modes) or dataset.manifest.skill_modes),
        hidden_dim=int(config.get("model", {}).get("hidden_dim", 128)),
        head_hidden_dim=int(config.get("model", {}).get("head_hidden_dim", 64)),
        vision_dim=int(config.get("model", {}).get("vision_dim", 16)),
        use_condition_film=bool(config.get("model", {}).get("use_condition_film", True)),
        use_condition_vector_for_policy=bool(config.get("model", {}).get("use_condition_vector_for_policy", True)),
        condition_fusion_mode=str(config.get("model", {}).get("condition_fusion_mode", "film")),
        default_skill_mode=str(config.get("model", {}).get("default_skill_mode", "efficiency_throughput")),
        enable_value_head=bool(config.get("model", {}).get("enable_value_head", True)),
        metadata={
            "dataset_digest": dataset.manifest.dataset_digest,
            "config_path": str(config_path),
        },
    )
    model = ReplayPolicyModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    start_epoch = 0
    best_val_mse = float("inf")
    train_steps = 0

    if resume_checkpoint:
        start_epoch, best_val_mse, train_steps = _load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_checkpoint,
            device=device,
        )

    train_records, val_records = split_step_records(
        dataset.steps,
        val_fraction=float(training_cfg.get("val_fraction", 0.25)),
    )
    train_loader = DataLoader(
        ReplayPolicyDataset(train_records, obs_dim=model_cfg.obs_dim, action_dim=model_cfg.action_dim, condition_dim=model_cfg.condition_dim),
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        ReplayPolicyDataset(val_records, obs_dim=model_cfg.obs_dim, action_dim=model_cfg.action_dim, condition_dim=model_cfg.condition_dim),
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=False,
        collate_fn=_collate_batch,
    )

    epochs = int(training_cfg.get("epochs", 8))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    metrics_path = output_root / "train_metrics.jsonl"
    checkpoint_path = output_root / "replay_policy_latest.pt"
    best_checkpoint_path = output_root / "replay_policy_best.pt"

    for epoch in range(start_epoch, epochs):
        train_metrics, train_steps = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            train_steps=train_steps,
        )
        val_metrics = evaluate_policy_model(model=model, loader=val_loader, device=device)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "train_steps": train_steps,
        }
        _append_jsonl(metrics_path, epoch_metrics)
        _save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_mse=min(best_val_mse, val_metrics["mse"]),
            train_steps=train_steps,
            model_config=model_cfg.to_dict(),
            training_config=config,
        )
        if val_metrics["mse"] <= best_val_mse:
            best_val_mse = val_metrics["mse"]
            _save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_val_mse=best_val_mse,
                train_steps=train_steps,
                model_config=model_cfg.to_dict(),
                training_config=config,
            )

    summary = {
        "model_version": model_cfg.model_version,
        "config_digest": model_cfg.config_digest,
        "dataset_digest": dataset.manifest.dataset_digest,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "best_val_mse": best_val_mse,
        "epochs": epochs,
        "train_steps": train_steps,
        "device": str(device),
        "determinism": get_context_summary(),
    }
    summary_path = output_root / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return ReplayPolicyTrainResult(
        checkpoint_path=str(checkpoint_path),
        best_checkpoint_path=str(best_checkpoint_path),
        metrics_path=str(metrics_path),
        summary_path=str(summary_path),
        config_digest=model_cfg.config_digest,
        dataset_digest=dataset.manifest.dataset_digest,
        best_val_mse=float(best_val_mse),
        train_steps=train_steps,
        epochs=epochs,
    )


def evaluate_replay_policy(
    *,
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "val",
) -> Dict[str, Any]:
    dataset = load_replay_dataset(dataset_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ReplayPolicyConfig.from_mapping(checkpoint["model_config"])
    model = ReplayPolicyModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_records, val_records = split_step_records(dataset.steps, val_fraction=0.25)
    records = train_records if split == "train" else val_records
    loader = DataLoader(
        ReplayPolicyDataset(records, obs_dim=model_config.obs_dim, action_dim=model_config.action_dim, condition_dim=model_config.condition_dim),
        batch_size=16,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    metrics = evaluate_policy_model(model=model, loader=loader, device=torch.device("cpu"))
    predictions = _collect_predictions(model=model, loader=loader, device=torch.device("cpu"))
    (output_root / "policy_eval.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _append_jsonl(output_root / "policy_predictions.jsonl", predictions, append=False)
    return {
        "metrics_path": str(output_root / "policy_eval.json"),
        "predictions_path": str(output_root / "policy_predictions.jsonl"),
        "metrics": metrics,
    }


def load_policy_checkpoint(checkpoint_path: str | Path, *, device: Optional[torch.device] = None) -> tuple[ReplayPolicyModel, ReplayPolicyConfig, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    model_config = ReplayPolicyConfig.from_mapping(checkpoint["model_config"])
    model = ReplayPolicyModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if device is not None:
        model = model.to(device)
    return model, model_config, dict(checkpoint)


def split_step_records(records: Sequence[ReplayStepRecord], *, val_fraction: float = 0.25) -> Tuple[List[ReplayStepRecord], List[ReplayStepRecord]]:
    threshold = int(float(val_fraction) * 100)
    train_records: List[ReplayStepRecord] = []
    val_records: List[ReplayStepRecord] = []
    for record in records:
        bucket = int(sha256_json({"episode_id": record.episode_id})[:4], 16) % 100
        if bucket < threshold:
            val_records.append(record)
        else:
            train_records.append(record)
    if not val_records and train_records:
        val_records.append(train_records.pop())
    if not train_records and val_records:
        train_records.append(val_records[0])
    return train_records, val_records


def evaluate_policy_model(
    *,
    model: ReplayPolicyModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            metrics = _batch_loss(model=model, batch=batch, device=device)
            batch_size = int(batch["obs_vector"].shape[0])
            totals["loss"] += float(metrics["loss"].item()) * batch_size
            totals["mse"] += float(metrics["mse"].item()) * batch_size
            totals["mae"] += float(metrics["mae"].item()) * batch_size
            totals["count"] += batch_size
    count = max(totals["count"], 1)
    return {
        "loss": totals["loss"] / count,
        "mse": totals["mse"] / count,
        "mae": totals["mae"] / count,
        "count": totals["count"],
    }


def _collect_predictions(
    *,
    model: ReplayPolicyModel,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            obs_vector = batch["obs_vector"].to(device)
            action_vector = batch["action_vector"].to(device)
            condition_vector = batch["condition_vector"].to(device)
            outputs = model(obs_vector, condition_vector, skill_modes=batch["skill_modes"])
            predicted = outputs["action_mean"]
            confidence = outputs["confidence"]
            errors = torch.mean(torch.abs(predicted - action_vector), dim=-1)
            for index, episode_id in enumerate(batch["episode_ids"]):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "step_idx": int(batch["step_idx"][index].item()),
                        "skill_mode": batch["skill_modes"][index],
                        "target_action_vector": action_vector[index].detach().cpu().tolist(),
                        "predicted_action_vector": predicted[index].detach().cpu().tolist(),
                        "confidence": float(confidence[index].item()),
                        "mae": float(errors[index].item()),
                    }
                )
    return rows


def _run_epoch(
    *,
    model: ReplayPolicyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    train_steps: int,
) -> tuple[Dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "count": 0}
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        metrics = _batch_loss(model=model, batch=batch, device=device)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(batch["obs_vector"].shape[0])
        totals["loss"] += float(metrics["loss"].item()) * batch_size
        totals["mse"] += float(metrics["mse"].item()) * batch_size
        totals["count"] += batch_size
        train_steps += batch_size
    count = max(totals["count"], 1)
    return {"loss": totals["loss"] / count, "mse": totals["mse"] / count}, train_steps


def _batch_loss(
    *,
    model: ReplayPolicyModel,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obs_vector = batch["obs_vector"].to(device)
    action_vector = batch["action_vector"].to(device)
    condition_vector = batch["condition_vector"].to(device)
    reward = batch["reward"].to(device)
    outputs = model(obs_vector, condition_vector, skill_modes=batch["skill_modes"])
    action_mean = outputs["action_mean"]
    action_log_std = outputs["action_log_std"]
    confidence = outputs["confidence"]
    action_std = torch.exp(action_log_std)
    normalized_error = (action_vector - action_mean) / torch.clamp(action_std, min=1e-4)
    gaussian_nll = 0.5 * normalized_error.pow(2) + action_log_std
    bc_loss = gaussian_nll.mean()
    mse = F.mse_loss(action_mean, action_vector)
    mae = F.l1_loss(action_mean, action_vector)
    confidence_target = torch.exp(-torch.mean(torch.abs(action_vector - action_mean).detach(), dim=-1))
    confidence_loss = F.mse_loss(confidence, confidence_target)
    value_loss = torch.zeros((), device=device)
    if "value" in outputs:
        value_loss = F.mse_loss(outputs["value"], reward)
    loss = bc_loss + 0.25 * mse + 0.10 * confidence_loss + 0.10 * value_loss
    return {"loss": loss, "mse": mse, "mae": mae}


def _resolve_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return get_device(dict(config))


def _collate_batch(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "obs_vector": torch.stack([row["obs_vector"] for row in rows]).float(),
        "action_vector": torch.stack([row["action_vector"] for row in rows]).float(),
        "condition_vector": torch.stack([row["condition_vector"] for row in rows]).float(),
        "reward": torch.as_tensor([row["reward"] for row in rows], dtype=torch.float32),
        "skill_modes": [str(row["skill_mode"]) for row in rows],
        "episode_ids": [str(row["episode_id"]) for row in rows],
        "step_idx": torch.as_tensor([row["step_idx"] for row in rows], dtype=torch.long),
    }


def _pad_vector(values: Sequence[float], target_dim: int) -> torch.Tensor:
    payload = [float(value) for value in values[:target_dim]]
    if len(payload) < target_dim:
        payload.extend([0.0] * (target_dim - len(payload)))
    return torch.as_tensor(payload, dtype=torch.float32)


def _append_jsonl(path: Path, rows: Mapping[str, Any] | Sequence[Mapping[str, Any]], *, append: bool = True) -> None:
    mode = "a" if append else "w"
    row_list = [dict(rows)] if isinstance(rows, Mapping) else [dict(row) for row in rows]
    with path.open(mode, encoding="utf-8") as handle:
        for row in row_list:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _save_checkpoint(
    *,
    path: Path,
    model: ReplayPolicyModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    train_steps: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_mse": float(best_val_mse),
            "train_steps": int(train_steps),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": dict(model_config),
            "training_config": dict(training_config),
        },
        path,
    )


def _load_checkpoint(
    *,
    model: ReplayPolicyModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_val_mse", float("inf"))), int(checkpoint.get("train_steps", 0))
