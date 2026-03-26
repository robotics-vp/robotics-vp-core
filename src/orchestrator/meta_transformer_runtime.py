"""Runtime helper loading for the trained meta-transformer package."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.orchestrator.meta_transformer_planning import (
    META_BACKEND_LABELS,
    META_DATA_MIX_LABELS,
    META_ENERGY_PROFILE_LABELS,
    META_OBJECTIVE_PRESET_LABELS,
    META_PLANNING_CONTEXT_DIM,
    decode_backend_label,
    decode_expected_delta_vector,
    decode_named_distribution,
    decode_objective_preset,
)
from src.orchestrator.meta_transformer_training import (
    TORCH_AVAILABLE,
    MetaTransformerNet,
    decode_semantic_tokens,
)

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover - handled by explicit errors in the loader
    torch = None


@dataclass(frozen=True)
class MetaTransformerRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    best_checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MetaTransformerRuntimeInference:
    authority: str
    authority_confidence: float
    alternate_authority_confidence: float
    policy_state: np.ndarray
    diffusion_conditioning: np.ndarray
    ontology_tokens: list[str]
    objective_preset: str
    objective_confidence: float
    objective_alternate_confidence: float
    chosen_backend: str
    backend_confidence: float
    backend_alternate_confidence: float
    energy_profile_weights: Dict[str, float]
    data_mix_weights: Dict[str, float]
    expected_deltas: Dict[str, float]
    planning_trace: Dict[str, Any]
    planning_heads_available: bool
    benchmark_gate_ready: bool
    promotion_stage: str


def _mapping(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def load_meta_transformer_runtime_package(path: str | Path) -> MetaTransformerRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    return MetaTransformerRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(payload.get("checkpoint_path") or ""),
        best_checkpoint_path=str(payload.get("best_checkpoint_path") or ""),
        model_config=_mapping(payload.get("model_config")),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


class LoadedMetaTransformerRuntime:
    """CPU-loaded inference helper for a trained meta-transformer package."""

    def __init__(self, package: MetaTransformerRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load a trained meta-transformer package")
        self.package = package
        checkpoint_path = Path(package.best_checkpoint_path or package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Meta-transformer checkpoint not found: {checkpoint_path}")
        checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model_config = dict(package.model_config or checkpoint_payload.get("model_config", {}) or {})
        self.vla_dim = int(model_config.get("vla_dim", 128))
        self.dino_dim = int(model_config.get("dino_dim", 256))
        self.objective_labels = list(model_config.get("objective_labels") or META_OBJECTIVE_PRESET_LABELS)
        self.backend_labels = list(model_config.get("backend_labels") or META_BACKEND_LABELS)
        self.energy_profile_labels = list(
            model_config.get("energy_profile_labels") or META_ENERGY_PROFILE_LABELS
        )
        self.data_mix_labels = list(model_config.get("data_mix_labels") or META_DATA_MIX_LABELS)
        self.planning_context_dim = int(
            model_config.get("planning_context_dim", META_PLANNING_CONTEXT_DIM)
        )
        self.model = MetaTransformerNet(
            vla_dim=self.vla_dim,
            dino_dim=self.dino_dim,
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            max_output_tokens=int(model_config.get("max_semantic_tokens", 16)),
            num_heads=int(model_config.get("num_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            planning_context_dim=self.planning_context_dim,
            objective_label_count=len(self.objective_labels),
            backend_label_count=len(self.backend_labels),
            energy_profile_dim=len(self.energy_profile_labels),
            data_mix_dim=len(self.data_mix_labels),
        )
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        load_result = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = set(load_result.missing_keys)
        planning_prefixes = (
            "planning_context_proj",
            "planning_fusion",
            "objective_head",
            "backend_head",
            "energy_head",
            "data_mix_head",
            "expected_delta_head",
        )
        self.planning_heads_available = not any(
            key.startswith(planning_prefixes) for key in missing_keys
        )
        self.model.eval()

    def infer(
        self,
        *,
        dino_features: np.ndarray,
        vla_features: np.ndarray,
        planning_context: Optional[np.ndarray] = None,
    ) -> MetaTransformerRuntimeInference:
        vla_vector = np.asarray(vla_features, dtype=np.float32).reshape(-1)
        dino_vector = np.asarray(dino_features, dtype=np.float32).reshape(-1)
        if vla_vector.size < self.vla_dim:
            vla_vector = np.pad(vla_vector, (0, self.vla_dim - vla_vector.size))
        if dino_vector.size < self.dino_dim:
            dino_vector = np.pad(dino_vector, (0, self.dino_dim - dino_vector.size))
        vla_tensor = torch.from_numpy(vla_vector[: self.vla_dim]).float().unsqueeze(0)
        dino_tensor = torch.from_numpy(dino_vector[: self.dino_dim]).float().unsqueeze(0)
        context_vector = np.asarray(
            planning_context if planning_context is not None else np.zeros(self.planning_context_dim, dtype=np.float32),
            dtype=np.float32,
        ).reshape(-1)
        if context_vector.size < self.planning_context_dim:
            context_vector = np.pad(context_vector, (0, self.planning_context_dim - context_vector.size))
        context_vector = context_vector[: self.planning_context_dim]
        planning_context_tensor = torch.from_numpy(context_vector).float().unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(vla_tensor, dino_tensor, planning_context=planning_context_tensor)
        authority_probs = torch.softmax(outputs["authority_logits"][0], dim=-1).cpu().numpy()
        authority_index = int(np.argmax(authority_probs))
        authority = "dino" if authority_index == 0 else "vla"
        alternate_authority_confidence = float(authority_probs[1 - authority_index])
        token_ids = torch.argmax(outputs["token_logits"][0], dim=-1).cpu().numpy()
        benchmark_gate_ready = bool(self.package.benchmark_gate.get("ready", False))
        objective_probs = torch.softmax(outputs["objective_logits"][0], dim=-1).cpu().numpy()
        objective_index = int(np.argmax(objective_probs))
        objective_sorted = np.sort(objective_probs)[::-1]
        objective_preset = (
            self.objective_labels[objective_index]
            if objective_index < len(self.objective_labels)
            else decode_objective_preset(objective_index)
        )
        backend_probs = torch.softmax(outputs["backend_logits"][0], dim=-1).cpu().numpy()
        backend_index = int(np.argmax(backend_probs))
        backend_sorted = np.sort(backend_probs)[::-1]
        chosen_backend = (
            self.backend_labels[backend_index]
            if backend_index < len(self.backend_labels)
            else decode_backend_label(backend_index)
        )
        energy_logits = outputs["energy_logits"][0].cpu().numpy()
        data_mix_logits = outputs["data_mix_logits"][0].cpu().numpy()
        expected_delta_vector = outputs["expected_delta_vector"][0].cpu().numpy()
        return MetaTransformerRuntimeInference(
            authority=authority,
            authority_confidence=float(authority_probs[authority_index]),
            alternate_authority_confidence=alternate_authority_confidence,
            policy_state=outputs["policy_state"][0].cpu().numpy().astype(np.float32),
            diffusion_conditioning=outputs["diffusion_cond"][0].cpu().numpy().astype(np.float32),
            ontology_tokens=decode_semantic_tokens(token_ids),
            objective_preset=objective_preset,
            objective_confidence=float(objective_probs[objective_index]),
            objective_alternate_confidence=float(objective_sorted[1] if objective_sorted.size > 1 else 0.0),
            chosen_backend=chosen_backend,
            backend_confidence=float(backend_probs[backend_index]),
            backend_alternate_confidence=float(backend_sorted[1] if backend_sorted.size > 1 else 0.0),
            energy_profile_weights=decode_named_distribution(
                energy_logits,
                self.energy_profile_labels,
            ),
            data_mix_weights=decode_named_distribution(
                data_mix_logits,
                self.data_mix_labels,
            ),
            expected_deltas=decode_expected_delta_vector(expected_delta_vector),
            planning_trace={
                "planning_heads_available": bool(self.planning_heads_available),
                "planning_context_used": planning_context is not None,
                "planning_context_norm": float(np.linalg.norm(context_vector)),
                "objective_distribution": {
                    self.objective_labels[idx]: float(prob)
                    for idx, prob in enumerate(objective_probs.tolist())
                    if idx < len(self.objective_labels)
                },
                "backend_distribution": {
                    self.backend_labels[idx]: float(prob)
                    for idx, prob in enumerate(backend_probs.tolist())
                    if idx < len(self.backend_labels)
                },
            },
            planning_heads_available=bool(self.planning_heads_available),
            benchmark_gate_ready=benchmark_gate_ready,
            promotion_stage=(
                "promoted"
                if benchmark_gate_ready
                else str(self.package.promotion_stage or "shadow_candidate")
            ),
        )


__all__ = [
    "LoadedMetaTransformerRuntime",
    "MetaTransformerRuntimeInference",
    "MetaTransformerRuntimePackage",
    "load_meta_transformer_runtime_package",
]
