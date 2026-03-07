"""Optional contract-aware critic sidecar for the online SAC backbone."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from src.rl.contract_aware_critic import CriticBundleConfig, ContractAwareCriticBundle
from src.rl.contract_aware_losses import ContractAwareLossWeights, contract_aware_losses
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class SACContractAwareAdapterConfig:
    """Config surface for optional SAC contract-aware critic training."""

    enabled: bool = False
    latent_dim: int = 128
    action_dim: int = 2
    condition_dim: int = 0
    skill_modes: list[str] = field(default_factory=lambda: ["efficiency_throughput"])
    hidden_dim: int = 128
    head_hidden_dim: int = 64
    device: str = "cpu"
    lr: float = 1e-3
    log_interval: int = 50
    artifact_dir: Optional[str] = None
    consistency_loss_weight: float = 0.1
    objective_loss_weight: float = 0.35
    econ_loss_weight: float = 0.35
    model_version: str = "sac_contract_aware_adapter_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "latent_dim": int(self.latent_dim),
            "action_dim": int(self.action_dim),
            "condition_dim": int(self.condition_dim),
            "skill_modes": list(self.skill_modes),
            "hidden_dim": int(self.hidden_dim),
            "head_hidden_dim": int(self.head_hidden_dim),
            "device": self.device,
            "lr": float(self.lr),
            "log_interval": int(self.log_interval),
            "artifact_dir": self.artifact_dir,
            "consistency_loss_weight": float(self.consistency_loss_weight),
            "objective_loss_weight": float(self.objective_loss_weight),
            "econ_loss_weight": float(self.econ_loss_weight),
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SACContractAwareAdapterConfig":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            latent_dim=int(payload.get("latent_dim", 128)),
            action_dim=int(payload.get("action_dim", 2)),
            condition_dim=int(payload.get("condition_dim", 0)),
            skill_modes=[str(value) for value in payload.get("skill_modes", ["efficiency_throughput"]) or ["efficiency_throughput"]],
            hidden_dim=int(payload.get("hidden_dim", 128)),
            head_hidden_dim=int(payload.get("head_hidden_dim", 64)),
            device=str(payload.get("device", "cpu")),
            lr=float(payload.get("lr", 1e-3)),
            log_interval=int(payload.get("log_interval", 50)),
            artifact_dir=payload.get("artifact_dir"),
            consistency_loss_weight=float(payload.get("consistency_loss_weight", 0.1)),
            objective_loss_weight=float(payload.get("objective_loss_weight", 0.35)),
            econ_loss_weight=float(payload.get("econ_loss_weight", 0.35)),
            model_version=str(payload.get("model_version", "sac_contract_aware_adapter_v1")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


class SACContractAwareAdapter:
    """Optional sidecar that trains a multi-head critic alongside SAC."""

    def __init__(self, config: SACContractAwareAdapterConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, SACContractAwareAdapterConfig)
            else SACContractAwareAdapterConfig.from_mapping(config)
        )
        self.enabled = bool(self.config.enabled)
        self.device = torch.device(self.config.device)
        self.training_steps = 0
        self.last_metrics: Dict[str, Any] = {}
        self._artifact_path: Optional[Path] = None

        if not self.enabled:
            self.bundle = None
            self.optimizer = None
            return

        bundle_config = CriticBundleConfig(
            obs_dim=self.config.latent_dim,
            action_dim=self.config.action_dim,
            condition_dim=max(1, self.config.condition_dim),
            skill_modes=list(self.config.skill_modes),
            hidden_dim=self.config.hidden_dim,
            head_hidden_dim=self.config.head_hidden_dim,
            metadata={"adapter": self.config.model_version, **dict(self.config.metadata)},
        )
        self.bundle = ContractAwareCriticBundle(bundle_config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.bundle.parameters(), lr=self.config.lr)
        if self.config.artifact_dir:
            artifact_root = Path(self.config.artifact_dir)
            artifact_root.mkdir(parents=True, exist_ok=True)
            self._artifact_path = artifact_root / "sac_contract_aware_metrics.jsonl"

    def update_from_batch(
        self,
        *,
        latent_batch: np.ndarray | torch.Tensor,
        action_batch: np.ndarray | torch.Tensor,
        reward_batch: np.ndarray | torch.Tensor,
        done_batch: Optional[np.ndarray | torch.Tensor] = None,
        condition_batch: Optional[np.ndarray | torch.Tensor] = None,
        skill_modes: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled or self.bundle is None or self.optimizer is None:
            return {"enabled": False}

        self.training_steps += 1
        latents = _as_tensor(latent_batch, device=self.device)
        actions = _as_tensor(action_batch, device=self.device)
        rewards = _as_tensor(reward_batch, device=self.device).reshape(-1)
        condition = _condition_tensor(
            condition_batch,
            batch_size=latents.shape[0],
            condition_dim=self.bundle.config.condition_dim,
            device=self.device,
        )
        outputs = self.bundle(
            latents,
            actions,
            condition,
        )
        scalar_targets = rewards
        objective_targets = _objective_targets(rewards, outputs.objective_axes, device=self.device)
        econ_targets = _econ_targets(rewards, outputs.econ_axes, done_batch, device=self.device)

        weights = ContractAwareLossWeights(
            scalar=1.0,
            objective=self.config.objective_loss_weight,
            econ=self.config.econ_loss_weight,
            consistency=self.config.consistency_loss_weight,
            confidence=0.05,
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses = contract_aware_losses(
            outputs=outputs,
            scalar_targets=scalar_targets,
            objective_targets=objective_targets,
            econ_targets=econ_targets,
            weights=weights,
        )
        losses["total_loss"].backward()
        self.optimizer.step()

        metrics = {
            "enabled": True,
            "training_step": int(self.training_steps),
            "total_loss": float(losses["total_loss"].item()),
            "scalar_loss": float(losses["scalar_loss"].item()),
            "objective_loss": float(losses.get("objective_loss", torch.tensor(0.0)).item()),
            "econ_loss": float(losses.get("econ_loss", torch.tensor(0.0)).item()),
            "consistency_loss": float(losses["consistency_loss"].item()),
            "compiled_scalar_mean": float(outputs.compiled_scalar.detach().mean().item()),
            "objective_prediction_mean": float(outputs.objective_vector.detach().mean().item()),
            "econ_prediction_mean": float(outputs.econ_vector.detach().mean().item()),
            "config_digest": self.config.config_digest,
            "model_version": self.config.model_version,
            "skill_mode": (skill_modes[0] if skill_modes else self.config.skill_modes[0]),
        }
        self.last_metrics = metrics
        self._append_metrics(metrics)
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        if not self.enabled or self.bundle is None:
            return {"enabled": False, "config": self.config.to_dict()}
        return {
            "enabled": True,
            "config": self.config.to_dict(),
            "bundle_state_dict": self.bundle.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "training_steps": self.training_steps,
            "last_metrics": dict(self.last_metrics),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.bundle is None:
            return
        if payload.get("bundle_state_dict"):
            self.bundle.load_state_dict(payload["bundle_state_dict"])
        if self.optimizer is not None and payload.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.training_steps = int(payload.get("training_steps", 0))
        self.last_metrics = dict(payload.get("last_metrics", {}) or {})

    def _append_metrics(self, metrics: Mapping[str, Any]) -> None:
        if self._artifact_path is None:
            return
        if self.training_steps % max(1, self.config.log_interval) != 0:
            return
        with self._artifact_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(metrics), sort_keys=True) + "\n")


def _as_tensor(value: np.ndarray | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _condition_tensor(
    condition_batch: Optional[np.ndarray | torch.Tensor],
    *,
    batch_size: int,
    condition_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if condition_batch is None:
        return torch.zeros(batch_size, condition_dim, dtype=torch.float32, device=device)
    tensor = _as_tensor(condition_batch, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] == condition_dim:
        return tensor
    if tensor.shape[-1] > condition_dim:
        return tensor[:, :condition_dim]
    pad = torch.zeros(batch_size, condition_dim - tensor.shape[-1], dtype=torch.float32, device=device)
    return torch.cat([tensor, pad], dim=-1)


def _objective_targets(
    rewards: torch.Tensor,
    objective_axes: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    targets = torch.zeros(rewards.shape[0], len(objective_axes), dtype=torch.float32, device=device)
    for index, axis in enumerate(objective_axes):
        if axis == "throughput":
            targets[:, index] = torch.clamp(rewards, min=0.0)
        elif axis == "error":
            targets[:, index] = torch.clamp(-rewards, min=0.0)
        elif axis == "safety":
            targets[:, index] = torch.sigmoid(rewards)
        elif axis == "energy":
            targets[:, index] = torch.abs(rewards) * 0.1
        elif axis == "uncertainty":
            targets[:, index] = torch.zeros_like(rewards)
        else:
            targets[:, index] = torch.tanh(rewards)
    return targets


def _econ_targets(
    rewards: torch.Tensor,
    econ_axes: Sequence[str],
    done_batch: Optional[np.ndarray | torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    done_tensor = (
        _as_tensor(done_batch, device=device).reshape(-1)
        if done_batch is not None
        else torch.zeros_like(rewards)
    )
    targets = torch.zeros(rewards.shape[0], len(econ_axes), dtype=torch.float32, device=device)
    for index, axis in enumerate(econ_axes):
        if axis == "expected_value_delta":
            targets[:, index] = rewards
        elif axis == "expected_pricing_confidence":
            targets[:, index] = torch.sigmoid(rewards)
        elif axis == "expected_adaptation_benefit":
            targets[:, index] = torch.clamp(rewards, min=0.0)
        elif axis == "expected_compute_cost":
            targets[:, index] = torch.full_like(rewards, 0.05)
        elif axis == "expected_risk_cost":
            targets[:, index] = torch.clamp(-rewards, min=0.0) + 0.1 * done_tensor
        elif axis == "data_value_proxy":
            targets[:, index] = torch.sigmoid(rewards) * 0.5
        else:
            targets[:, index] = torch.tanh(rewards)
    return targets


__all__ = [
    "SACContractAwareAdapterConfig",
    "SACContractAwareAdapter",
]
