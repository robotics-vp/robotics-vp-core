"""OSS-inspired neural architecture scaffolds for Phase 3 Embodiment / Actuation.

This module defines CPU-runnable model skeletons and manifests for future GPU
training. The architectures borrow public design patterns from V-JEPA-style
latent prediction, ACT-style action chunking, Diffusion Policy-style denoising,
TD-MPC-style latent control, and topology/contrastive morphology consistency.
They are not provider imports, benchmark claims, or promoted policies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import mapping, safe_int, stable_id, strings
from .neural_seams import encode_state_features

NEURAL_ARCHITECTURE_MANIFEST_VERSION = "phase3_embodiment_neural_architecture_manifest_v1"


@dataclass(frozen=True)
class EmbodimentNeuralArchitectureSpec:
    architecture_id: str
    family: str
    purpose: str
    input_shapes: dict[str, Any] = field(default_factory=dict)
    output_shapes: dict[str, Any] = field(default_factory=dict)
    oss_inspirations: list[str] = field(default_factory=list)
    training_objectives: list[str] = field(default_factory=list)
    required_training_rows: list[str] = field(default_factory=list)
    promotion_requirements: list[str] = field(default_factory=list)
    blocker_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "embodiment_neural_architecture_spec_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "family": self.family,
            "purpose": self.purpose,
            "input_shapes": mapping(self.input_shapes),
            "output_shapes": mapping(self.output_shapes),
            "oss_inspirations": strings(self.oss_inspirations),
            "training_objectives": strings(self.training_objectives),
            "required_training_rows": strings(self.required_training_rows),
            "promotion_requirements": strings(self.promotion_requirements),
            "blocker_reasons": strings(self.blocker_reasons),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class EmbodimentNeuralArchitectureManifest:
    manifest_id: str
    architecture_specs: list[EmbodimentNeuralArchitectureSpec]
    smoke_results: dict[str, Any] = field(default_factory=dict)
    promotion_eligible: bool = False
    blocker_reasons: list[str] = field(default_factory=list)
    source_refs: dict[str, Any] = field(default_factory=dict)
    version: str = NEURAL_ARCHITECTURE_MANIFEST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "architecture_count": len(self.architecture_specs),
            "architecture_specs": [spec.to_dict() for spec in self.architecture_specs],
            "smoke_results": mapping(self.smoke_results),
            "promotion_eligible": bool(self.promotion_eligible),
            "blocker_reasons": strings(self.blocker_reasons),
            "source_refs": mapping(self.source_refs),
            "version": self.version,
        }


class TemporalJEPALatentPredictor(nn.Module):
    """Latent action-conditioned predictor for local dynamics scaffolding."""

    architecture_family = "temporal_jepa_action_conditioned_predictor"

    def __init__(
        self,
        feature_dim: int = 32,
        action_dim: int = 12,
        latent_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.input = nn.Linear(self.feature_dim + self.action_dim, self.latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=max(1, heads),
            dim_feedforward=self.latent_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.energy_head = nn.Sequential(nn.LayerNorm(self.latent_dim), nn.Linear(self.latent_dim, 1))

    def forward(self, state_features: torch.Tensor, action_context: torch.Tensor) -> dict[str, torch.Tensor]:
        state_seq = _ensure_sequence(state_features.float(), feature_dim=self.feature_dim)
        action_seq = _ensure_sequence(action_context.float(), feature_dim=self.action_dim)
        state_seq, action_seq = _align_time(state_seq, action_seq)
        tokens = self.input(torch.cat([state_seq, action_seq], dim=-1))
        encoded = self.context_encoder(tokens)
        context = encoded.mean(dim=1)
        next_latent = self.predictor(context)
        return {
            "context_latent": context,
            "predicted_next_latent": next_latent,
            "latent_energy": F.softplus(self.energy_head(next_latent)).squeeze(-1),
        }

    def describe(self) -> dict[str, Any]:
        return _module_description(
            self,
            family=self.architecture_family,
            inputs={"state_features": ["B", "T", self.feature_dim], "action_context": ["B", "T", self.action_dim]},
            outputs={"predicted_next_latent": ["B", self.latent_dim], "latent_energy": ["B"]},
        )


class ActionChunkingTransformerHead(nn.Module):
    """ACT-style chunk proposal head with observation context and query tokens."""

    architecture_family = "act_style_chunked_transformer_head"

    def __init__(
        self,
        feature_dim: int = 32,
        action_dim: int = 12,
        chunk_len: int = 8,
        hidden_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.hidden_dim = int(hidden_dim)
        self.context_proj = nn.Linear(self.feature_dim, self.hidden_dim)
        self.chunk_queries = nn.Parameter(torch.zeros(1, self.chunk_len, self.hidden_dim))
        nn.init.normal_(self.chunk_queries, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(1, heads),
            dim_feedforward=self.hidden_dim * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, depth))
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.quality_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))

    def forward(self, context_features: torch.Tensor) -> dict[str, torch.Tensor]:
        context = _ensure_sequence(context_features.float(), feature_dim=self.feature_dim)
        context_tokens = self.context_proj(context)
        queries = self.chunk_queries.expand(context.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([context_tokens, queries], dim=1))
        chunk_tokens = encoded[:, -self.chunk_len :, :]
        return {
            "action_chunk": torch.tanh(self.action_head(chunk_tokens)),
            "chunk_quality": torch.sigmoid(self.quality_head(chunk_tokens.mean(dim=1))).squeeze(-1),
        }

    def describe(self) -> dict[str, Any]:
        return _module_description(
            self,
            family=self.architecture_family,
            inputs={"context_features": ["B", "T", self.feature_dim]},
            outputs={"action_chunk": ["B", self.chunk_len, self.action_dim], "chunk_quality": ["B"]},
        )


class DiffusionActionDenoiserHead(nn.Module):
    """Diffusion Policy-style conditional denoiser for action chunks."""

    architecture_family = "diffusion_policy_action_denoiser"

    def __init__(
        self,
        feature_dim: int = 32,
        action_dim: int = 12,
        chunk_len: int = 8,
        hidden_dim: int = 96,
        timestep_dim: int = 16,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.timestep_dim = int(timestep_dim)
        input_dim = self.feature_dim + self.action_dim * self.chunk_len + self.timestep_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_dim * self.chunk_len),
        )
        self.confidence = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        noisy_action_chunk: torch.Tensor,
        condition_features: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        chunk = _ensure_chunk(noisy_action_chunk.float(), self.chunk_len, self.action_dim)
        cond = _ensure_sequence(condition_features.float(), feature_dim=self.feature_dim).mean(dim=1)
        if timesteps is None:
            timesteps = torch.zeros((chunk.shape[0],), dtype=chunk.dtype, device=chunk.device)
        timestep_embedding = _sinusoidal_timestep_embedding(timesteps.float(), self.timestep_dim)
        flat_chunk = chunk.reshape(chunk.shape[0], self.chunk_len * self.action_dim)
        model_input = torch.cat([flat_chunk, cond, timestep_embedding], dim=-1)
        residual = self.net(model_input).reshape(chunk.shape[0], self.chunk_len, self.action_dim)
        return {
            "denoised_action_chunk": torch.tanh(chunk - residual),
            "predicted_noise": residual,
            "denoise_confidence": torch.sigmoid(self.confidence(model_input)).squeeze(-1),
        }

    def describe(self) -> dict[str, Any]:
        return _module_description(
            self,
            family=self.architecture_family,
            inputs={
                "noisy_action_chunk": ["B", self.chunk_len, self.action_dim],
                "condition_features": ["B", "T", self.feature_dim],
                "timesteps": ["B"],
            },
            outputs={
                "denoised_action_chunk": ["B", self.chunk_len, self.action_dim],
                "predicted_noise": ["B", self.chunk_len, self.action_dim],
            },
        )


class EmbodimentTopologyContrastiveHead(nn.Module):
    """Topology-aware contrastive projector for morphology/action consistency."""

    architecture_family = "embodiment_topology_contrastive_head"

    def __init__(
        self,
        feature_dim: int = 32,
        group_dim: int = 4,
        embedding_dim: int = 48,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.group_dim = int(group_dim)
        self.embedding_dim = int(embedding_dim)
        self.temperature = float(temperature)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim + self.group_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.adjacency_head = nn.Linear(self.embedding_dim, self.group_dim * self.group_dim)

    def forward(
        self,
        state_features: torch.Tensor,
        morphology_groups: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        features = _ensure_sequence(state_features.float(), feature_dim=self.feature_dim).mean(dim=1)
        if morphology_groups is None:
            morphology_groups = torch.zeros(
                (features.shape[0], self.group_dim),
                dtype=features.dtype,
                device=features.device,
            )
        groups = _fit_last_dim(morphology_groups.float(), self.group_dim)
        embedding = F.normalize(self.projector(torch.cat([features, groups], dim=-1)), dim=-1)
        logits = self.adjacency_head(embedding).reshape(features.shape[0], self.group_dim, self.group_dim)
        sim = embedding @ embedding.transpose(0, 1) / max(self.temperature, 1e-6)
        return {
            "embedding": embedding,
            "adjacency_logits": logits,
            "batch_contrastive_logits": sim,
        }

    def describe(self) -> dict[str, Any]:
        return _module_description(
            self,
            family=self.architecture_family,
            inputs={"state_features": ["B", "T", self.feature_dim], "morphology_groups": ["B", self.group_dim]},
            outputs={
                "embedding": ["B", self.embedding_dim],
                "adjacency_logits": ["B", self.group_dim, self.group_dim],
            },
        )


def build_embodiment_neural_architecture_specs(state: Any) -> list[EmbodimentNeuralArchitectureSpec]:
    action_dim = _action_dim(state)
    joint_groups = _joint_group_count(state)
    blockers = _base_blockers(state)
    source_refs = {"state_id": str(getattr(state, "state_id", ""))}
    common_metadata = {
        "authority_level": "none",
        "source_refs": source_refs,
        "notes": "architecture_contract_only_no_training_claim",
    }
    return [
        EmbodimentNeuralArchitectureSpec(
            architecture_id=stable_id("embodiment_neural_arch", {"family": "jepa", **source_refs}),
            family=TemporalJEPALatentPredictor.architecture_family,
            purpose="action_conditioned_latent_local_dynamics_prediction",
            input_shapes={"state_features": ["B", "T", 32], "action_context": ["B", "T", action_dim]},
            output_shapes={"predicted_next_latent": ["B", 64], "latent_energy": ["B"]},
            oss_inspirations=["V-JEPA/V-JEPA2 masked latent prediction", "TD-MPC2 decoder-free latent control"],
            training_objectives=["masked_next_latent_prediction", "action_conditioned_energy_ranking"],
            required_training_rows=["local_contact_dynamics", "drift_calibration"],
            promotion_requirements=_promotion_requirements(),
            blocker_reasons=blockers,
            metadata=common_metadata,
        ),
        EmbodimentNeuralArchitectureSpec(
            architecture_id=stable_id("embodiment_neural_arch", {"family": "act", **source_refs}),
            family=ActionChunkingTransformerHead.architecture_family,
            purpose="chunked_action_proposal_from_embodiment_context",
            input_shapes={"context_features": ["B", "T", 32]},
            output_shapes={"action_chunk": ["B", 8, action_dim], "chunk_quality": ["B"]},
            oss_inspirations=["Action Chunking Transformer", "LeRobot/ACT action chunking discipline"],
            training_objectives=["chunk_imitation_loss", "temporal_ensembling_smoothness"],
            required_training_rows=["action_proposal", "inverse_retargeting"],
            promotion_requirements=_promotion_requirements(),
            blocker_reasons=blockers,
            metadata=common_metadata,
        ),
        EmbodimentNeuralArchitectureSpec(
            architecture_id=stable_id("embodiment_neural_arch", {"family": "diffusion", **source_refs}),
            family=DiffusionActionDenoiserHead.architecture_family,
            purpose="multimodal_action_chunk_denoising_under_embodiment_constraints",
            input_shapes={"noisy_action_chunk": ["B", 8, action_dim], "condition_features": ["B", "T", 32]},
            output_shapes={"denoised_action_chunk": ["B", 8, action_dim]},
            oss_inspirations=["Diffusion Policy conditional action denoising"],
            training_objectives=["noise_prediction_loss", "constraint_conditioned_action_score_matching"],
            required_training_rows=["action_proposal", "drift_calibration"],
            promotion_requirements=_promotion_requirements(),
            blocker_reasons=blockers,
            metadata=common_metadata,
        ),
        EmbodimentNeuralArchitectureSpec(
            architecture_id=stable_id("embodiment_neural_arch", {"family": "topology_contrastive", **source_refs}),
            family=EmbodimentTopologyContrastiveHead.architecture_family,
            purpose="morphology_topology_and_action_feasibility_consistency",
            input_shapes={"state_features": ["B", "T", 32], "morphology_groups": ["B", joint_groups]},
            output_shapes={"embedding": ["B", 48], "adjacency_logits": ["B", joint_groups, joint_groups]},
            oss_inspirations=["topology-aware contrastive representation learning", "morphology/action-space invariance"],
            training_objectives=["positive_same_embodiment_contrast", "negative_mismatched_morphology_contrast"],
            required_training_rows=["inverse_retargeting", "local_contact_dynamics"],
            promotion_requirements=_promotion_requirements(),
            blocker_reasons=blockers,
            metadata=common_metadata,
        ),
    ]


def smoke_forward_neural_architectures(state: Any) -> dict[str, dict[str, Any]]:
    """Run deterministic CPU forward passes for all architecture scaffolds."""

    torch.manual_seed(17)
    action_dim = _action_dim(state)
    group_dim = _joint_group_count(state)
    features = encode_state_features(state, 32).reshape(1, 1, 32).repeat(2, 3, 1)
    action_context = torch.zeros((2, 3, action_dim), dtype=torch.float32)
    noisy_chunk = torch.zeros((2, 8, action_dim), dtype=torch.float32)
    morphology_groups = _morphology_group_tensor(state, group_dim).repeat(2, 1)
    modules = {
        TemporalJEPALatentPredictor.architecture_family: (
            TemporalJEPALatentPredictor(feature_dim=32, action_dim=action_dim),
            (features, action_context),
        ),
        ActionChunkingTransformerHead.architecture_family: (
            ActionChunkingTransformerHead(feature_dim=32, action_dim=action_dim, chunk_len=8),
            (features,),
        ),
        DiffusionActionDenoiserHead.architecture_family: (
            DiffusionActionDenoiserHead(feature_dim=32, action_dim=action_dim, chunk_len=8),
            (noisy_chunk, features, torch.tensor([0.0, 3.0])),
        ),
        EmbodimentTopologyContrastiveHead.architecture_family: (
            EmbodimentTopologyContrastiveHead(feature_dim=32, group_dim=group_dim),
            (features, morphology_groups),
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for family, (module, args) in modules.items():
            outputs = module(*args)
            results[family] = {
                "describe": module.describe(),
                "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
                "finite": all(bool(torch.isfinite(value).all().item()) for value in outputs.values()),
                "param_count": _param_count(module),
                "authority_level": "none",
            }
    return results


def build_embodiment_neural_architecture_manifest(
    state: Any,
    *,
    source_refs: Mapping[str, Any] | None = None,
    include_smoke: bool = True,
) -> EmbodimentNeuralArchitectureManifest:
    specs = build_embodiment_neural_architecture_specs(state)
    smoke = smoke_forward_neural_architectures(state) if include_smoke else {}
    blockers = sorted({reason for spec in specs for reason in spec.blocker_reasons})
    refs = {"state_id": str(getattr(state, "state_id", "")), **mapping(source_refs)}
    return EmbodimentNeuralArchitectureManifest(
        manifest_id=stable_id(
            "embodiment_neural_architecture_manifest",
            {"state_id": refs.get("state_id", ""), "families": [spec.family for spec in specs]},
        ),
        architecture_specs=specs,
        smoke_results=smoke,
        promotion_eligible=False,
        blocker_reasons=blockers,
        source_refs=refs,
    )


def _ensure_sequence(value: torch.Tensor, *, feature_dim: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value.reshape(1, 1, -1)
    elif value.ndim == 2:
        value = value.unsqueeze(1)
    if value.shape[-1] != feature_dim:
        value = _fit_last_dim(value, feature_dim)
    return value


def _ensure_chunk(value: torch.Tensor, chunk_len: int, action_dim: int) -> torch.Tensor:
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim == 1:
        value = value.reshape(1, 1, -1)
    if value.shape[1] != chunk_len:
        if value.shape[1] > chunk_len:
            value = value[:, :chunk_len, :]
        else:
            pad = torch.zeros(
                (value.shape[0], chunk_len - value.shape[1], value.shape[-1]),
                dtype=value.dtype,
                device=value.device,
            )
            value = torch.cat([value, pad], dim=1)
    if value.shape[-1] != action_dim:
        value = _fit_last_dim(value, action_dim)
    return value


def _fit_last_dim(value: torch.Tensor, size: int) -> torch.Tensor:
    if value.shape[-1] == size:
        return value
    if value.shape[-1] > size:
        return value[..., :size]
    pad_shape = (*value.shape[:-1], size - value.shape[-1])
    pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
    return torch.cat([value, pad], dim=-1)


def _align_time(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    steps = max(left.shape[1], right.shape[1])
    if left.shape[1] != steps:
        left = left.repeat(1, steps, 1) if left.shape[1] == 1 else left[:, :steps, :]
    if right.shape[1] != steps:
        right = right.repeat(1, steps, 1) if right.shape[1] == 1 else right[:, :steps, :]
    return left, right


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = max(1, dim // 2)
    frequencies = torch.exp(
        torch.arange(half, dtype=timesteps.dtype, device=timesteps.device)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    args = timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return _fit_last_dim(embedding, dim)


def _module_description(module: nn.Module, *, family: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": family,
        "input_shapes": mapping(inputs),
        "output_shapes": mapping(outputs),
        "param_count": _param_count(module),
        "promotion_required": True,
        "default_posture": "architecture_scaffold_only",
        "authority_level": "none",
    }


def _param_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def _action_dim(state: Any) -> int:
    dim = safe_int(getattr(getattr(state, "action_space", None), "dimension", 0), 0)
    return max(1, dim or 12)


def _joint_group_count(state: Any) -> int:
    metadata = mapping(getattr(getattr(state, "capability", None), "metadata", {}))
    group_counts = mapping(metadata.get("group_counts"))
    if group_counts:
        return max(1, len(group_counts))
    joint_names = strings(getattr(getattr(state, "joint_state", None), "joint_names", []))
    if joint_names:
        groups = set()
        for name in joint_names:
            if "hip" in name or "knee" in name or "ankle" in name:
                groups.add("legs")
            elif "waist" in name:
                groups.add("waist")
            elif "shoulder" in name or "elbow" in name or "wrist" in name:
                groups.add("arms")
            elif "hand" in name or "thumb" in name or "index" in name or "middle" in name:
                groups.add("hands")
        return max(1, len(groups) or 4)
    return 4


def _morphology_group_tensor(state: Any, group_dim: int) -> torch.Tensor:
    metadata = mapping(getattr(getattr(state, "capability", None), "metadata", {}))
    group_counts = mapping(metadata.get("group_counts"))
    values = [float(group_counts.get(key, 0.0)) for key in sorted(group_counts)] if group_counts else []
    if not values:
        values = [0.0 for _ in range(group_dim)]
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
    if tensor.sum() > 0:
        tensor = tensor / tensor.sum().clamp_min(1.0)
    return _fit_last_dim(tensor, group_dim)


def _base_blockers(state: Any) -> list[str]:
    blockers = [
        "no_gpu_training_run",
        "no_provider_runtime_eval",
        "no_benchmark_promotion_evidence",
    ]
    safety = getattr(state, "safety_envelope", None)
    calibration = getattr(state, "calibration_targets", None)
    blockers.extend(strings(getattr(safety, "missing_evidence", [])))
    blockers.extend(strings(getattr(calibration, "missing_evidence", [])))
    return sorted(set(blockers))


def _promotion_requirements() -> list[str]:
    return [
        "gpu_training_run_manifest",
        "heldout_benchmark_metrics",
        "provider_runtime_eval_receipts",
        "latency_watchdog_evidence",
        "demotion_path_test",
    ]


__all__ = [
    "ActionChunkingTransformerHead",
    "DiffusionActionDenoiserHead",
    "EmbodimentNeuralArchitectureManifest",
    "EmbodimentNeuralArchitectureSpec",
    "EmbodimentTopologyContrastiveHead",
    "NEURAL_ARCHITECTURE_MANIFEST_VERSION",
    "TemporalJEPALatentPredictor",
    "build_embodiment_neural_architecture_manifest",
    "build_embodiment_neural_architecture_specs",
    "smoke_forward_neural_architectures",
]
