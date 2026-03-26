"""Runtime helper loading for the trained meta-transformer package."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

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
        self.model = MetaTransformerNet(
            vla_dim=int(model_config.get("vla_dim", 128)),
            dino_dim=int(model_config.get("dino_dim", 256)),
            hidden_dim=int(model_config.get("hidden_dim", 128)),
            max_output_tokens=int(model_config.get("max_semantic_tokens", 16)),
            num_heads=int(model_config.get("num_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
        )
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def infer(
        self,
        *,
        dino_features: np.ndarray,
        vla_features: np.ndarray,
    ) -> MetaTransformerRuntimeInference:
        vla_tensor = torch.from_numpy(np.asarray(vla_features, dtype=np.float32)).float().unsqueeze(0)
        dino_tensor = torch.from_numpy(np.asarray(dino_features, dtype=np.float32)).float().unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(vla_tensor, dino_tensor)
        authority_probs = torch.softmax(outputs["authority_logits"][0], dim=-1).cpu().numpy()
        authority_index = int(np.argmax(authority_probs))
        authority = "dino" if authority_index == 0 else "vla"
        alternate_authority_confidence = float(authority_probs[1 - authority_index])
        token_ids = torch.argmax(outputs["token_logits"][0], dim=-1).cpu().numpy()
        benchmark_gate_ready = bool(self.package.benchmark_gate.get("ready", False))
        return MetaTransformerRuntimeInference(
            authority=authority,
            authority_confidence=float(authority_probs[authority_index]),
            alternate_authority_confidence=alternate_authority_confidence,
            policy_state=outputs["policy_state"][0].cpu().numpy().astype(np.float32),
            diffusion_conditioning=outputs["diffusion_cond"][0].cpu().numpy().astype(np.float32),
            ontology_tokens=decode_semantic_tokens(token_ids),
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
