from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.orchestrator.pipeline_stage_policy import (
    PIPELINE_CONFIG_FLAG_KEYS,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGE_POLICY_FEATURE_NAMES,
)
from src.orchestrator.pipeline_stage_policy_training import (
    PipelineStagePolicyNet,
    TORCH_AVAILABLE,
)

torch: Any

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover
    torch = None


def _clamp01(value: Any) -> float:
    try:
        candidate = float(value)
    except Exception:
        candidate = 0.0
    return max(0.0, min(1.0, candidate))


def _normalize_distribution(
    values: np.ndarray, labels: tuple[str, ...]
) -> Dict[str, float]:
    clipped = np.maximum(values.astype(np.float64), 0.0)
    total = float(np.sum(clipped))
    if total <= 0.0:
        fallback = {label: 0.0 for label in labels}
        if labels:
            fallback[labels[0]] = 1.0
        return fallback
    return {label: float(clipped[idx] / total) for idx, label in enumerate(labels)}


@dataclass(frozen=True)
class PipelineStagePolicyRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_pipeline_stage_policy_runtime_package(
    path: str | Path,
) -> PipelineStagePolicyRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return PipelineStagePolicyRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        model_config=dict(payload.get("model_config", {}) or {}),
        benchmark_gate=dict(payload.get("benchmark_gate", {}) or {}),
        execution_preconditions=dict(payload.get("execution_preconditions", {}) or {}),
        inference_contract=dict(payload.get("inference_contract", {}) or {}),
        promotion_stage=str(
            payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"
        ),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


class LoadedPipelineStagePolicyHelper:
    def __init__(self, package: PipelineStagePolicyRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to load the pipeline stage policy helper"
            )
        checkpoint_path = Path(package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"pipeline stage policy checkpoint not found: {checkpoint_path}"
            )
        payload = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        input_dim = int(
            payload.get("input_dim", len(PIPELINE_STAGE_POLICY_FEATURE_NAMES))
        )
        hidden_dim = int(payload.get("hidden_dim", 32))
        self.package = package
        self.model = PipelineStagePolicyNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.benchmark_gate = dict(package.benchmark_gate or {})
        self.promotion_stage = str(package.promotion_stage or "shadow_candidate")
        self.inference_contract = dict(package.inference_contract or {})

    def apply_to_policy(
        self,
        *,
        feature_map: Mapping[str, Any],
        heuristic_policy: Mapping[str, Any],
        helper_mode: str = "auto",
    ) -> Dict[str, Any]:
        vector = np.asarray(
            [
                float(feature_map.get(name, 0.0))
                for name in PIPELINE_STAGE_POLICY_FEATURE_NAMES
            ],
            dtype=np.float32,
        )
        tensor = torch.from_numpy(vector).float().unsqueeze(0)
        with torch.no_grad():
            stage_logits, config_logits, activation_logits = self.model(tensor)
        stage_probs = torch.softmax(stage_logits[0], dim=-1).cpu().numpy()
        config_probs = torch.sigmoid(config_logits[0]).cpu().numpy()
        activation_score = float(torch.sigmoid(activation_logits[0]).item())

        prior_stage_distribution = {
            label: float(
                dict(heuristic_policy.get("stage_distribution", {}) or {}).get(
                    label, 0.0
                )
            )
            for label in PIPELINE_STAGE_LABELS
        }
        prior_config_flags = {
            key: _clamp01(
                dict(heuristic_policy.get("config_flag_scores", {}) or {}).get(key, 0.0)
            )
            for key in PIPELINE_CONFIG_FLAG_KEYS
        }
        prior_activation = _clamp01(heuristic_policy.get("activation_label", 0.0))

        benchmark_gate_ready = bool(self.benchmark_gate.get("ready", False))
        blend = dict(self.inference_contract.get("helper_blend_policy", {}) or {})
        helper_weight = float(
            blend.get(
                "promoted_helper_weight"
                if benchmark_gate_ready
                else "shadow_candidate_helper_weight",
                0.35 if benchmark_gate_ready else 0.12,
            )
        )
        max_stage_delta = float(
            blend.get(
                "promoted_max_stage_delta"
                if benchmark_gate_ready
                else "shadow_candidate_max_stage_delta",
                0.4 if benchmark_gate_ready else 0.18,
            )
        )
        max_config_delta = float(
            blend.get(
                "promoted_max_config_delta"
                if benchmark_gate_ready
                else "shadow_candidate_max_config_delta",
                0.35 if benchmark_gate_ready else 0.18,
            )
        )

        learned_stage_distribution = _normalize_distribution(
            stage_probs, PIPELINE_STAGE_LABELS
        )
        blended_stage = {}
        for label in PIPELINE_STAGE_LABELS:
            raw = ((1.0 - helper_weight) * prior_stage_distribution.get(label, 0.0)) + (
                helper_weight * learned_stage_distribution.get(label, 0.0)
            )
            low = max(0.0, prior_stage_distribution.get(label, 0.0) - max_stage_delta)
            high = min(1.0, prior_stage_distribution.get(label, 0.0) + max_stage_delta)
            blended_stage[label] = min(max(raw, low), high)
        final_stage_distribution = _normalize_distribution(
            np.asarray(
                [blended_stage[label] for label in PIPELINE_STAGE_LABELS],
                dtype=np.float64,
            ),
            PIPELINE_STAGE_LABELS,
        )

        learned_config_flags = {
            key: _clamp01(config_probs[idx])
            for idx, key in enumerate(PIPELINE_CONFIG_FLAG_KEYS)
        }
        final_config_flags = {}
        for key in PIPELINE_CONFIG_FLAG_KEYS:
            raw = ((1.0 - helper_weight) * prior_config_flags.get(key, 0.0)) + (
                helper_weight * learned_config_flags.get(key, 0.0)
            )
            low = max(0.0, prior_config_flags.get(key, 0.0) - max_config_delta)
            high = min(1.0, prior_config_flags.get(key, 0.0) + max_config_delta)
            final_config_flags[key] = min(max(raw, low), high)

        final_activation_label = prior_activation
        if (
            benchmark_gate_ready
            and prior_activation > 0.5
            and helper_mode == "required"
        ):
            final_activation_label = 1.0 if activation_score >= 0.25 else 0.0

        return {
            "stage_distribution": final_stage_distribution,
            "config_flag_scores": final_config_flags,
            "activation_label": final_activation_label,
            "policy_source": "heuristic_plus_learned_helper",
            "promotion_stage": "promoted"
            if benchmark_gate_ready
            else self.promotion_stage,
            "helper_trace": {
                "package_id": self.package.package_id,
                "helper_weight": helper_weight,
                "benchmark_gate_ready": benchmark_gate_ready,
                "prior_stage_distribution": prior_stage_distribution,
                "learned_stage_distribution": learned_stage_distribution,
                "final_stage_distribution": final_stage_distribution,
                "prior_config_flags": prior_config_flags,
                "learned_config_flags": learned_config_flags,
                "final_config_flags": final_config_flags,
                "prior_activation_label": prior_activation,
                "learned_activation_label": activation_score,
                "final_activation_label": final_activation_label,
            },
        }


def resolve_pipeline_stage_policy_helper(
    *,
    helper_mode: str = "disabled",
    package: Optional[PipelineStagePolicyRuntimePackage] = None,
    package_path: Optional[str | Path] = None,
) -> Optional[LoadedPipelineStagePolicyHelper]:
    mode = str(helper_mode or "disabled")
    if mode == "disabled":
        return None
    if package is None and package_path is None:
        if mode == "required":
            raise ValueError("pipeline stage policy helper requires a package path")
        return None
    if package is None:
        assert package_path is not None
        package = load_pipeline_stage_policy_runtime_package(package_path)
    helper = LoadedPipelineStagePolicyHelper(package)
    if mode == "required" and not bool(helper.benchmark_gate.get("ready", False)):
        raise ValueError(
            "pipeline stage policy helper requires a benchmark-gated package"
        )
    return helper


__all__ = [
    "LoadedPipelineStagePolicyHelper",
    "PipelineStagePolicyRuntimePackage",
    "load_pipeline_stage_policy_runtime_package",
    "resolve_pipeline_stage_policy_helper",
]
