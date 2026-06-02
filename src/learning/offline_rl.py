"""Experimental offline-to-online bridge with a TD3+BC-style shadow path."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from src.determinism.determinism_context import get_context_summary, set_determinism
from src.learning.replay_policy_model import ReplayPolicyConfig, ReplayPolicyModel
from src.learning.replay_policy_trainer import _append_jsonl, _resolve_device, split_step_records
from src.replay.dataset import load_replay_dataset
from src.replay.schema import ReplayStepRecord
from src.rl.contract_aware_critic import CriticBundleConfig, ContractAwareCriticBundle
from src.rl.contract_aware_losses import ContractAwareLossWeights, contract_aware_losses
from src.utils.config_digest import sha256_json
from src.utils.training_env import configure_training_env


@dataclass(frozen=True)
class OfflineRLTrainResult:
    """Checkpoint and metric outputs for offline RL shadow training."""

    actor_checkpoint_path: str
    critic_checkpoint_path: str
    summary_path: str
    metrics_path: str
    config_digest: str
    dataset_digest: str
    algorithm: str
    epochs: int
    train_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_checkpoint_path": self.actor_checkpoint_path,
            "critic_checkpoint_path": self.critic_checkpoint_path,
            "summary_path": self.summary_path,
            "metrics_path": self.metrics_path,
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "algorithm": self.algorithm,
            "epochs": int(self.epochs),
            "train_steps": int(self.train_steps),
        }


class OfflineReplayTransitionDataset(Dataset):
    """Deterministic replay transition view for staged offline RL."""

    def __init__(
        self,
        records: Sequence[ReplayStepRecord],
        *,
        obs_dim: int,
        action_dim: int,
        condition_dim: int,
        objective_axes: Sequence[str],
        econ_axes: Sequence[str],
    ) -> None:
        sorted_records = sorted(records, key=lambda row: (row.run_id, row.episode_id, row.step_idx))
        self.rows: list[Dict[str, Any]] = []
        next_lookup: dict[tuple[str, str, int], ReplayStepRecord] = {
            (row.run_id, row.episode_id, row.step_idx): row for row in sorted_records
        }
        for row in sorted_records:
            next_row = next_lookup.get((row.run_id, row.episode_id, row.step_idx + 1))
            done = bool(row.done or next_row is None or next_row.episode_id != row.episode_id)
            self.rows.append(
                {
                    "obs_vector": _pad_vector(row.obs_vector, obs_dim),
                    "action_vector": _pad_vector(row.action_vector, action_dim),
                    "condition_vector": _pad_vector(row.condition_vector_values, condition_dim),
                    "next_obs_vector": _pad_vector(next_row.obs_vector if next_row is not None else [], obs_dim),
                    "next_condition_vector": _pad_vector(next_row.condition_vector_values if next_row is not None else [], condition_dim),
                    "reward": float(row.reward),
                    "done": float(done),
                    "skill_mode": row.skill_mode,
                    "objective_target": _summary_vector(row.objective_tensor_summary, objective_axes),
                    "econ_target": _summary_vector(row.econ_tensor_summary, econ_axes),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.rows[index])


def load_offline_rl_config(path: str | Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Offline RL config at {path} must be a mapping")
    return dict(payload)


def train_offline_rl(
    *,
    dataset_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    episode_ids: Optional[Sequence[str]] = None,
) -> OfflineRLTrainResult:
    dataset = load_replay_dataset(dataset_dir)
    config = load_offline_rl_config(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected_episode_ids = {str(value) for value in (episode_ids or []) if str(value)}
    filtered_steps = (
        [row for row in dataset.steps if row.episode_id in selected_episode_ids]
        if selected_episode_ids
        else list(dataset.steps)
    )

    training_cfg = dict(config.get("training", {}) or {})
    algorithm = str(config.get("algorithm", "td3_bc_shadow"))
    seed = int(training_cfg.get("seed", 42))
    set_determinism(seed=seed)
    configure_training_env(config)
    device = _resolve_device(config)

    actor_config = ReplayPolicyConfig(
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
        enable_value_head=False,
        metadata={"algorithm": algorithm, "dataset_digest": dataset.manifest.dataset_digest},
    )
    actor = ReplayPolicyModel(actor_config).to(device)
    actor_target = ReplayPolicyModel(actor_config).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic_config = CriticBundleConfig(
        obs_dim=actor_config.obs_dim,
        action_dim=actor_config.action_dim,
        condition_dim=actor_config.condition_dim,
        skill_modes=actor_config.skill_modes,
        hidden_dim=actor_config.hidden_dim,
        head_hidden_dim=actor_config.head_hidden_dim,
        vision_dim=actor_config.vision_dim,
        use_condition_film=actor_config.use_condition_film,
        use_condition_vector_for_policy=actor_config.use_condition_vector_for_policy,
        condition_fusion_mode=actor_config.condition_fusion_mode,
        default_skill_mode=actor_config.default_skill_mode,
        metadata={"algorithm": algorithm},
    )
    critic = ContractAwareCriticBundle(critic_config).to(device)
    critic_target = ContractAwareCriticBundle(critic_config).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=float(training_cfg.get("actor_lr", 1e-3)))
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=float(training_cfg.get("critic_lr", 1e-3)))

    train_records, _ = split_step_records(filtered_steps, val_fraction=float(training_cfg.get("val_fraction", 0.25)))
    transition_dataset = OfflineReplayTransitionDataset(
        train_records,
        obs_dim=actor_config.obs_dim,
        action_dim=actor_config.action_dim,
        condition_dim=actor_config.condition_dim,
        objective_axes=critic_config.objective_axes,
        econ_axes=critic_config.econ_axes,
    )
    loader = DataLoader(
        transition_dataset,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate_transitions,
    )

    gamma = float(training_cfg.get("gamma", 0.98))
    tau = float(training_cfg.get("tau", 0.01))
    bc_weight = float(training_cfg.get("bc_weight", 2.5))
    policy_delay = max(1, int(training_cfg.get("policy_delay", 2)))
    epochs = int(training_cfg.get("epochs", 6))
    metrics_path = output_root / "offline_rl_metrics.jsonl"
    actor_checkpoint = output_root / "offline_rl_actor.pt"
    critic_checkpoint = output_root / "offline_rl_critic.pt"
    train_steps = 0

    for epoch in range(epochs):
        for batch in loader:
            train_steps += 1
            obs = torch.as_tensor(batch["obs_vector"], dtype=torch.float32, device=device)
            actions = torch.as_tensor(batch["action_vector"], dtype=torch.float32, device=device)
            condition = torch.as_tensor(batch["condition_vector"], dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(batch["next_obs_vector"], dtype=torch.float32, device=device)
            next_condition = torch.as_tensor(batch["next_condition_vector"], dtype=torch.float32, device=device)
            rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=device)
            dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=device)
            objective_target = torch.as_tensor(batch["objective_target"], dtype=torch.float32, device=device)
            econ_target = torch.as_tensor(batch["econ_target"], dtype=torch.float32, device=device)

            with torch.no_grad():
                next_policy = actor_target(next_obs, next_condition, skill_modes=batch["skill_mode"])
                next_actions = next_policy["action_mean"]
                target_output = critic_target(next_obs, next_actions, next_condition)
                scalar_target = rewards + (1.0 - dones) * gamma * target_output.compiled_scalar

            critic_optimizer.zero_grad(set_to_none=True)
            critic_output = critic(obs, actions, condition)
            critic_losses = contract_aware_losses(
                outputs=critic_output,
                scalar_targets=scalar_target,
                objective_targets=objective_target,
                econ_targets=econ_target,
                weights=ContractAwareLossWeights(),
            )
            critic_losses["total_loss"].backward()
            critic_optimizer.step()

            actor_metrics: Dict[str, float] = {}
            if train_steps % policy_delay == 0:
                actor_optimizer.zero_grad(set_to_none=True)
                actor_output = actor(obs, condition, skill_modes=batch["skill_mode"])
                predicted_actions = actor_output["action_mean"]
                critic_for_actor = critic(obs, predicted_actions, condition)
                bc_loss = F.mse_loss(predicted_actions, actions)
                q_scale = critic_for_actor.compiled_scalar.detach().abs().mean().clamp(min=1.0)
                actor_loss = -critic_for_actor.compiled_scalar.mean() / q_scale + bc_weight * bc_loss
                actor_loss.backward()
                actor_optimizer.step()
                _soft_update(actor_target, actor, tau=tau)
                _soft_update(critic_target, critic, tau=tau)
                actor_metrics = {
                    "actor_loss": float(actor_loss.item()),
                    "bc_loss": float(bc_loss.item()),
                    "policy_q_mean": float(critic_for_actor.compiled_scalar.mean().item()),
                }

            metrics = {
                "epoch": epoch + 1,
                "train_step": train_steps,
                "critic_total_loss": float(critic_losses["total_loss"].item()),
                "critic_scalar_loss": float(critic_losses["scalar_loss"].item()),
                "critic_objective_loss": float(critic_losses.get("objective_loss", torch.tensor(0.0)).item()),
                "critic_econ_loss": float(critic_losses.get("econ_loss", torch.tensor(0.0)).item()),
            }
            metrics.update(actor_metrics)
            _append_jsonl(metrics_path, metrics)

    _save_offline_checkpoint(actor_checkpoint, actor, actor_config.to_dict(), config, train_steps)
    _save_offline_checkpoint(critic_checkpoint, critic, critic_config.to_dict(), config, train_steps)
    summary = {
        "algorithm": algorithm,
        "config_digest": sha256_json(config),
        "dataset_digest": dataset.manifest.dataset_digest,
        "epochs": epochs,
        "train_steps": train_steps,
        "actor_checkpoint": str(actor_checkpoint),
        "critic_checkpoint": str(critic_checkpoint),
        "selected_episode_count": len(selected_episode_ids) if selected_episode_ids else len({row.episode_id for row in filtered_steps}),
        "device": str(device),
        "determinism": get_context_summary(),
        "sac_backbone_preserved": True,
        "legacy_default_untouched": True,
    }
    summary_path = output_root / "offline_rl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return OfflineRLTrainResult(
        actor_checkpoint_path=str(actor_checkpoint),
        critic_checkpoint_path=str(critic_checkpoint),
        summary_path=str(summary_path),
        metrics_path=str(metrics_path),
        config_digest=sha256_json(config),
        dataset_digest=dataset.manifest.dataset_digest,
        algorithm=algorithm,
        epochs=epochs,
        train_steps=train_steps,
    )


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _summary_vector(summary: Mapping[str, Any], axes: Sequence[str]) -> list[float]:
    axis_map = dict(summary.get("axes", {}) or {})
    return [float(axis_map.get(axis, 0.0)) for axis in axes]


def _pad_vector(values: Sequence[float], target_dim: int) -> list[float]:
    row = [float(value) for value in values]
    if len(row) >= target_dim:
        return row[:target_dim]
    return row + [0.0] * (target_dim - len(row))


def _save_offline_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    train_steps: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_config),
            "training_config": dict(training_config),
            "train_steps": int(train_steps),
        },
        path,
    )


def _collate_transitions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "obs_vector": torch.as_tensor([row["obs_vector"] for row in rows], dtype=torch.float32),
        "action_vector": torch.as_tensor([row["action_vector"] for row in rows], dtype=torch.float32),
        "condition_vector": torch.as_tensor([row["condition_vector"] for row in rows], dtype=torch.float32),
        "next_obs_vector": torch.as_tensor([row["next_obs_vector"] for row in rows], dtype=torch.float32),
        "next_condition_vector": torch.as_tensor([row["next_condition_vector"] for row in rows], dtype=torch.float32),
        "reward": torch.as_tensor([row["reward"] for row in rows], dtype=torch.float32),
        "done": torch.as_tensor([row["done"] for row in rows], dtype=torch.float32),
        "skill_mode": [str(row["skill_mode"]) for row in rows],
        "objective_target": torch.as_tensor([row["objective_target"] for row in rows], dtype=torch.float32),
        "econ_target": torch.as_tensor([row["econ_target"] for row in rows], dtype=torch.float32),
    }
