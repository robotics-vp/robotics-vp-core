"""Additive contract-aware critic stack for structured objective/econ prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from src.learning.replay_policy_model import ReplayPolicyConfig, ReplayTrunkBridge
from src.objectives.compiler import ObjectiveCompiler
from src.objectives.profile import ObjectiveProfile
from src.objectives.tensor import objective_tensor_from_axes
from src.utils.config_digest import sha256_json


DEFAULT_OBJECTIVE_AXES = (
    "throughput",
    "error",
    "safety",
    "energy",
    "constraint_risk",
    "uncertainty",
)
DEFAULT_ECON_AXES = (
    "expected_value_delta",
    "expected_pricing_confidence",
    "expected_adaptation_benefit",
    "expected_compute_cost",
    "expected_risk_cost",
    "data_value_proxy",
)


def _pad_or_trim_batch(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    current = tensor.shape[-1]
    if current == target_dim:
        return tensor
    if current > target_dim:
        return tensor[..., :target_dim]
    pad = torch.zeros(
        *tensor.shape[:-1],
        target_dim - current,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, pad], dim=-1)


def _objective_profile(
    profile: Mapping[str, Any] | ObjectiveProfile | None,
) -> ObjectiveProfile:
    if isinstance(profile, ObjectiveProfile):
        return profile
    if isinstance(profile, Mapping):
        return ObjectiveProfile.from_dict(profile)
    return ObjectiveProfile.weighted_sum(
        {
            "throughput": 1.0,
            "error": 1.0,
            "safety": 1.0,
            "energy": 1.0,
        },
        profile_id="contract_aware_default",
        maximize={"error": False, "energy": False},
    )


@dataclass(frozen=True)
class CriticBundleConfig:
    """Serializable config for contract-aware critics."""

    obs_dim: int
    action_dim: int
    condition_dim: int
    skill_modes: list[str]
    objective_axes: list[str] = field(
        default_factory=lambda: list(DEFAULT_OBJECTIVE_AXES)
    )
    econ_axes: list[str] = field(default_factory=lambda: list(DEFAULT_ECON_AXES))
    compile_axes: list[str] = field(
        default_factory=lambda: ["throughput", "error", "safety", "energy"]
    )
    hidden_dim: int = 128
    head_hidden_dim: int = 64
    vision_dim: int = 16
    use_condition_film: bool = True
    use_condition_vector_for_policy: bool = True
    condition_fusion_mode: str = "film"
    default_skill_mode: str = "efficiency_throughput"
    model_version: str = "contract_aware_critic_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "condition_dim": int(self.condition_dim),
            "skill_modes": list(self.skill_modes),
            "objective_axes": list(self.objective_axes),
            "econ_axes": list(self.econ_axes),
            "compile_axes": list(self.compile_axes),
            "hidden_dim": int(self.hidden_dim),
            "head_hidden_dim": int(self.head_hidden_dim),
            "vision_dim": int(self.vision_dim),
            "use_condition_film": bool(self.use_condition_film),
            "use_condition_vector_for_policy": bool(
                self.use_condition_vector_for_policy
            ),
            "condition_fusion_mode": self.condition_fusion_mode,
            "default_skill_mode": self.default_skill_mode,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CriticBundleConfig":
        return cls(
            obs_dim=int(payload.get("obs_dim", 0)),
            action_dim=int(payload.get("action_dim", 0)),
            condition_dim=int(payload.get("condition_dim", 0)),
            skill_modes=[str(value) for value in payload.get("skill_modes", []) or []],
            objective_axes=[
                str(value)
                for value in payload.get("objective_axes", DEFAULT_OBJECTIVE_AXES)
                or DEFAULT_OBJECTIVE_AXES
            ],
            econ_axes=[
                str(value)
                for value in payload.get("econ_axes", DEFAULT_ECON_AXES)
                or DEFAULT_ECON_AXES
            ],
            compile_axes=[
                str(value)
                for value in payload.get(
                    "compile_axes", ["throughput", "error", "safety", "energy"]
                )
                or ["throughput", "error", "safety", "energy"]
            ],
            hidden_dim=int(payload.get("hidden_dim", 128)),
            head_hidden_dim=int(payload.get("head_hidden_dim", 64)),
            vision_dim=int(payload.get("vision_dim", 16)),
            use_condition_film=bool(payload.get("use_condition_film", True)),
            use_condition_vector_for_policy=bool(
                payload.get("use_condition_vector_for_policy", True)
            ),
            condition_fusion_mode=str(payload.get("condition_fusion_mode", "film")),
            default_skill_mode=str(
                payload.get("default_skill_mode", "efficiency_throughput")
            ),
            model_version=str(payload.get("model_version", "contract_aware_critic_v1")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass
class CriticOutput:
    """Typed critic predictions used by offline RL and budget gates."""

    objective_axes: list[str]
    objective_vector: torch.Tensor
    objective_confidence: torch.Tensor
    econ_axes: list[str]
    econ_vector: torch.Tensor
    econ_confidence: torch.Tensor
    compiled_scalar: torch.Tensor
    compiled_scalar_baseline: torch.Tensor
    scalar_confidence: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def detach(self) -> "CriticOutput":
        return CriticOutput(
            objective_axes=list(self.objective_axes),
            objective_vector=self.objective_vector.detach(),
            objective_confidence=self.objective_confidence.detach(),
            econ_axes=list(self.econ_axes),
            econ_vector=self.econ_vector.detach(),
            econ_confidence=self.econ_confidence.detach(),
            compiled_scalar=self.compiled_scalar.detach(),
            compiled_scalar_baseline=self.compiled_scalar_baseline.detach(),
            scalar_confidence=self.scalar_confidence.detach(),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_axes": list(self.objective_axes),
            "objective_vector": self.objective_vector.detach().cpu().tolist(),
            "objective_confidence": self.objective_confidence.detach().cpu().tolist(),
            "econ_axes": list(self.econ_axes),
            "econ_vector": self.econ_vector.detach().cpu().tolist(),
            "econ_confidence": self.econ_confidence.detach().cpu().tolist(),
            "compiled_scalar": self.compiled_scalar.detach().cpu().tolist(),
            "compiled_scalar_baseline": self.compiled_scalar_baseline.detach()
            .cpu()
            .tolist(),
            "scalar_confidence": self.scalar_confidence.detach().cpu().tolist(),
            "metadata": dict(self.metadata),
        }


class _VectorHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, out_dim)
        self.confidence = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.value(hidden), torch.sigmoid(self.confidence(hidden)).squeeze(-1)


class ObjectiveVectorCriticHead(_VectorHead):
    """Predict decomposed objective quantities without premature scalarization."""


class EconCriticHead(_VectorHead):
    """Predict econ-facing quantities for adaptation and pricing decisions."""


class ScalarCompiledCriticHead(_VectorHead):
    """Residual scalar head downstream of explicit contract compilation."""


class ContractAwareCriticBundle(nn.Module):
    """Shared-trunk critic that preserves objective/econ structure until compile."""

    def __init__(self, config: CriticBundleConfig) -> None:
        super().__init__()
        self.config = config
        trunk_config = ReplayPolicyConfig(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            condition_dim=config.condition_dim,
            skill_modes=list(config.skill_modes or [config.default_skill_mode]),
            hidden_dim=config.hidden_dim,
            head_hidden_dim=config.head_hidden_dim,
            vision_dim=config.vision_dim,
            use_condition_film=config.use_condition_film,
            use_condition_vector_for_policy=config.use_condition_vector_for_policy,
            condition_fusion_mode=config.condition_fusion_mode,
            default_skill_mode=config.default_skill_mode,
            enable_value_head=False,
            metadata={"critic_bundle": True},
        )
        self.trunk = ReplayTrunkBridge(trunk_config)
        feature_dim = (
            config.hidden_dim * 2
            if config.use_condition_vector_for_policy
            and config.condition_fusion_mode == "concat"
            else config.hidden_dim
        )
        critic_dim = feature_dim + config.action_dim
        self.objective_head = ObjectiveVectorCriticHead(
            critic_dim, len(config.objective_axes), config.head_hidden_dim
        )
        self.econ_head = EconCriticHead(
            critic_dim, len(config.econ_axes), config.head_hidden_dim
        )
        self.scalar_head = ScalarCompiledCriticHead(
            critic_dim + len(config.objective_axes) + len(config.econ_axes) + 1,
            1,
            config.head_hidden_dim,
        )

    def forward(
        self,
        obs_vector: torch.Tensor,
        action_vector: torch.Tensor,
        condition_vector: torch.Tensor,
        *,
        objective_profile: Mapping[str, Any] | ObjectiveProfile | None = None,
    ) -> CriticOutput:
        base_features, conditioned_features = self.trunk(obs_vector, condition_vector)
        policy_features = (
            conditioned_features if conditioned_features is not None else base_features
        )
        action_batch = _pad_or_trim_batch(action_vector, self.config.action_dim)
        critic_features = torch.cat([policy_features, action_batch], dim=-1)
        objective_vector, objective_confidence = self.objective_head(critic_features)
        econ_vector, econ_confidence = self.econ_head(critic_features)
        compiled_scalar_baseline = self.compile_objective_vector(
            objective_vector=objective_vector,
            objective_profile=objective_profile,
        )
        scalar_input = torch.cat(
            [
                critic_features,
                objective_vector,
                econ_vector,
                compiled_scalar_baseline.unsqueeze(-1),
            ],
            dim=-1,
        )
        scalar_residual, scalar_confidence = self.scalar_head(scalar_input)
        compiled_scalar = compiled_scalar_baseline + scalar_residual.squeeze(-1)
        return CriticOutput(
            objective_axes=list(self.config.objective_axes),
            objective_vector=objective_vector,
            objective_confidence=objective_confidence,
            econ_axes=list(self.config.econ_axes),
            econ_vector=econ_vector,
            econ_confidence=econ_confidence,
            compiled_scalar=compiled_scalar,
            compiled_scalar_baseline=compiled_scalar_baseline,
            scalar_confidence=scalar_confidence,
            metadata={
                "model_version": self.config.model_version,
                "config_digest": self.config.config_digest,
            },
        )

    def compile_objective_vector(
        self,
        *,
        objective_vector: torch.Tensor,
        objective_profile: Mapping[str, Any] | ObjectiveProfile | None = None,
    ) -> torch.Tensor:
        profile = _objective_profile(objective_profile)
        compiler = ObjectiveCompiler(profile)
        rows = objective_vector.detach().cpu().tolist()
        compiled: list[float] = []
        for row in rows:
            axis_values = {
                axis: float(row[index])
                for index, axis in enumerate(self.config.objective_axes)
                if axis in set(self.config.compile_axes)
            }
            tensor = objective_tensor_from_axes(axis_values)
            compiled.append(float(compiler.scalarize(tensor)))
        return torch.as_tensor(
            compiled, dtype=objective_vector.dtype, device=objective_vector.device
        )
