"""Replay-trainable shadow policy aligned with trunk/condition-vector routing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from src.rl.trunk_net import TrunkNet
from src.utils.config_digest import sha256_json


def _pad_or_trim_batch(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    current = tensor.shape[-1]
    if current == target_dim:
        return tensor
    if current > target_dim:
        return tensor[..., :target_dim]
    pad = torch.zeros(*tensor.shape[:-1], target_dim - current, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


@dataclass(frozen=True)
class ReplayPolicyConfig:
    """Serializable replay policy configuration."""

    obs_dim: int
    action_dim: int
    condition_dim: int
    skill_modes: List[str]
    hidden_dim: int = 128
    head_hidden_dim: int = 64
    vision_dim: int = 16
    use_condition_film: bool = True
    use_condition_vector_for_policy: bool = True
    condition_fusion_mode: str = "film"
    default_skill_mode: str = "efficiency_throughput"
    enable_value_head: bool = True
    model_version: str = "shadow_replay_policy_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "condition_dim": int(self.condition_dim),
            "skill_modes": list(self.skill_modes),
            "hidden_dim": int(self.hidden_dim),
            "head_hidden_dim": int(self.head_hidden_dim),
            "vision_dim": int(self.vision_dim),
            "use_condition_film": bool(self.use_condition_film),
            "use_condition_vector_for_policy": bool(self.use_condition_vector_for_policy),
            "condition_fusion_mode": self.condition_fusion_mode,
            "default_skill_mode": self.default_skill_mode,
            "enable_value_head": bool(self.enable_value_head),
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReplayPolicyConfig":
        return cls(
            obs_dim=int(payload.get("obs_dim", 0)),
            action_dim=int(payload.get("action_dim", 0)),
            condition_dim=int(payload.get("condition_dim", 0)),
            skill_modes=[str(value) for value in payload.get("skill_modes", []) or []],
            hidden_dim=int(payload.get("hidden_dim", 128)),
            head_hidden_dim=int(payload.get("head_hidden_dim", 64)),
            vision_dim=int(payload.get("vision_dim", 16)),
            use_condition_film=bool(payload.get("use_condition_film", True)),
            use_condition_vector_for_policy=bool(payload.get("use_condition_vector_for_policy", True)),
            condition_fusion_mode=str(payload.get("condition_fusion_mode", "film")),
            default_skill_mode=str(payload.get("default_skill_mode", "efficiency_throughput")),
            enable_value_head=bool(payload.get("enable_value_head", True)),
            model_version=str(payload.get("model_version", "shadow_replay_policy_v1")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


class ReplayTrunkBridge(nn.Module):
    """Batch-safe bridge around the repo's TrunkNet modules."""

    def __init__(self, config: ReplayPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = TrunkNet(
            vision_dim=config.vision_dim,
            state_dim=config.obs_dim,
            condition_dim=config.condition_dim,
            hidden_dim=config.hidden_dim,
            use_condition_film=config.use_condition_film,
            use_condition_vector=True,
            use_condition_vector_for_policy=config.use_condition_vector_for_policy,
            condition_fusion_mode=config.condition_fusion_mode,
        )

    def forward(
        self,
        obs_vector: torch.Tensor,
        condition_vector: torch.Tensor,
        vision_vector: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if obs_vector.ndim == 1:
            obs_vector = obs_vector.unsqueeze(0)
        if condition_vector.ndim == 1:
            condition_vector = condition_vector.unsqueeze(0)
        if vision_vector is None:
            vision_vector = torch.zeros(
                obs_vector.shape[0],
                self.config.vision_dim,
                device=obs_vector.device,
                dtype=obs_vector.dtype,
            )
        if vision_vector.ndim == 1:
            vision_vector = vision_vector.unsqueeze(0)

        obs_vector = _pad_or_trim_batch(obs_vector, self.config.obs_dim)
        condition_vector = _pad_or_trim_batch(condition_vector, self.config.condition_dim)
        vision_vector = _pad_or_trim_batch(vision_vector, self.config.vision_dim)

        vision_embed = self.trunk.vision_proj(vision_vector)
        state_embed = self.trunk.state_proj(obs_vector)
        condition_embed = self.trunk.condition_proj(condition_vector)

        if self.trunk.use_condition_film:
            gating = torch.sigmoid(condition_embed)
            vision_embed = vision_embed * gating
            state_embed = state_embed * gating

        fused = torch.cat([vision_embed, state_embed, condition_embed], dim=-1)
        trunk_features = self.trunk.fusion(fused)

        if self.config.use_condition_vector_for_policy:
            conditioned = self.trunk._condition_policy_features(trunk_features, condition_vector)
            return trunk_features, conditioned
        return trunk_features, None


class GaussianActionHead(nn.Module):
    """Skill head that predicts action mean, scale, and confidence."""

    def __init__(self, in_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {
            "action_mean": self.mean(hidden),
            "action_log_std": torch.clamp(self.log_std(hidden), min=-5.0, max=2.0),
            "confidence": torch.sigmoid(self.confidence(hidden)).squeeze(-1),
        }


class ReplayValueHead(nn.Module):
    """Optional value head for staged offline RL support."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ReplayPolicyModel(nn.Module):
    """Shared-trunk, skill-routed shadow replay policy."""

    def __init__(self, config: ReplayPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.skill_modes = list(config.skill_modes or [config.default_skill_mode])
        self.default_skill_mode = config.default_skill_mode if config.default_skill_mode in self.skill_modes else self.skill_modes[0]
        self.trunk = ReplayTrunkBridge(config)
        in_dim = config.hidden_dim * 2 if config.use_condition_vector_for_policy and config.condition_fusion_mode == "concat" else config.hidden_dim
        self.heads = nn.ModuleDict(
            {
                skill_mode: GaussianActionHead(
                    in_dim=in_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.head_hidden_dim,
                )
                for skill_mode in self.skill_modes
            }
        )
        self.value_head = ReplayValueHead(in_dim=in_dim, hidden_dim=config.head_hidden_dim) if config.enable_value_head else None

    def forward(
        self,
        obs_vector: torch.Tensor,
        condition_vector: torch.Tensor,
        *,
        skill_modes: Optional[Sequence[str]] = None,
        vision_vector: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, int]]:
        base_features, conditioned_features = self.trunk(obs_vector, condition_vector, vision_vector=vision_vector)
        policy_features = conditioned_features if conditioned_features is not None else base_features
        batch_size = policy_features.shape[0]
        resolved_modes = list(skill_modes or [self.default_skill_mode] * batch_size)
        if len(resolved_modes) != batch_size:
            raise ValueError("skill_modes length must match batch size")

        action_mean = torch.zeros(batch_size, self.config.action_dim, device=policy_features.device, dtype=policy_features.dtype)
        action_log_std = torch.zeros_like(action_mean)
        confidence = torch.zeros(batch_size, device=policy_features.device, dtype=policy_features.dtype)
        head_usage: Dict[str, int] = {}

        for skill_mode in sorted(set(resolved_modes)):
            head_key = skill_mode if skill_mode in self.heads else self.default_skill_mode
            head_usage[head_key] = head_usage.get(head_key, 0) + sum(1 for value in resolved_modes if value == skill_mode)
            indices = [index for index, value in enumerate(resolved_modes) if value == skill_mode]
            if not indices:
                continue
            index_tensor = torch.as_tensor(indices, device=policy_features.device, dtype=torch.long)
            head_out = self.heads[head_key](policy_features.index_select(0, index_tensor))
            action_mean.index_copy_(0, index_tensor, head_out["action_mean"])
            action_log_std.index_copy_(0, index_tensor, head_out["action_log_std"])
            confidence.index_copy_(0, index_tensor, head_out["confidence"])

        outputs: Dict[str, torch.Tensor | Dict[str, int]] = {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "confidence": confidence,
            "head_usage": head_usage,
            "trunk_features": base_features,
        }
        if self.value_head is not None:
            outputs["value"] = self.value_head(policy_features)
        return outputs


def build_replay_policy_model(config: Mapping[str, Any]) -> ReplayPolicyModel:
    """Instantiate a replay policy model from config."""

    return ReplayPolicyModel(ReplayPolicyConfig.from_mapping(config))
