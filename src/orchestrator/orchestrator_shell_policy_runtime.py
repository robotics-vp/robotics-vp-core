from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.orchestrator.orchestrator_shell_policy import (
    SHELL_POLICY_PRESET_LABELS,
    SHELL_POLICY_STRATEGY_KEYS,
    build_shell_policy_feature_vector,
    heuristic_preset_distribution,
    normalize_strategy_overrides,
)
from src.orchestrator.orchestrator_shell_policy_training import (
    OrchestratorShellPolicyNet,
    TORCH_AVAILABLE,
)
from src.semantic.models import SemanticSnapshot

if TORCH_AVAILABLE:  # pragma: no branch
    import torch
else:  # pragma: no cover
    torch = None


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_vector(values: np.ndarray, labels: tuple[str, ...]) -> Dict[str, float]:
    clipped = np.maximum(values.astype(np.float64), 0.0)
    total = float(np.sum(clipped))
    if total <= 0.0:
        fallback = {label: 0.0 for label in labels}
        fallback[labels[0]] = 1.0
        return fallback
    return {
        label: float(clipped[idx] / total)
        for idx, label in enumerate(labels)
    }


def _focus_presets_from_distribution(distribution: Mapping[str, Any]) -> list[str]:
    ranked = sorted(
        ((str(label), float(value)) for label, value in distribution.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return ["balanced"]
    selected = [ranked[0][0]]
    for label, score in ranked[1:]:
        if score >= 0.28 and len(selected) < 2:
            selected.append(label)
    return selected


@dataclass(frozen=True)
class OrchestratorShellPolicyRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_orchestrator_shell_policy_runtime_package(path: str | Path) -> OrchestratorShellPolicyRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return OrchestratorShellPolicyRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        model_config=_mapping(payload.get("model_config")),
        benchmark_gate=_mapping(payload.get("benchmark_gate")),
        execution_preconditions=_mapping(payload.get("execution_preconditions")),
        inference_contract=_mapping(payload.get("inference_contract")),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=_mapping(payload.get("metadata")),
    )


class LoadedOrchestratorShellPolicyHelper:
    def __init__(self, package: OrchestratorShellPolicyRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load the orchestrator shell policy helper")
        checkpoint_path = Path(package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"orchestrator shell policy checkpoint not found: {checkpoint_path}")
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        input_dim = int(payload.get("input_dim", 0))
        hidden_dim = int(payload.get("hidden_dim", 32))
        self.package = package
        self.model = OrchestratorShellPolicyNet(input_dim=input_dim, hidden_dim=hidden_dim)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.model_sha = str(package.package_id)
        self.benchmark_gate = dict(package.benchmark_gate or {})
        self.promotion_stage = str(package.promotion_stage or "shadow_candidate")
        self.inference_contract = dict(package.inference_contract or {})

    def apply_to_advisory(
        self,
        *,
        snapshot: SemanticSnapshot,
        heuristic_advisory: Mapping[str, Any],
        trust_matrix: Optional[Mapping[str, Any]] = None,
        helper_mode: str = "auto",
    ) -> Dict[str, Any]:
        feature_vector = build_shell_policy_feature_vector(snapshot, trust_matrix=trust_matrix)
        tensor = torch.from_numpy(feature_vector).float().unsqueeze(0)
        with torch.no_grad():
            preset_logits, strategy_logits, safety_logits, activation_logits = self.model(tensor)
        preset_probs = torch.softmax(preset_logits[0], dim=-1).cpu().numpy()
        strategy_probs = torch.softmax(strategy_logits[0], dim=-1).cpu().numpy()
        safety_score = float(torch.sigmoid(safety_logits[0]).item())
        activation_score = float(torch.sigmoid(activation_logits[0]).item())

        benchmark_gate_ready = bool(self.benchmark_gate.get("ready", False))
        helper_blend = dict(self.inference_contract.get("helper_blend_policy", {}) or {})
        helper_weight = float(
            helper_blend.get(
                "promoted_helper_weight" if benchmark_gate_ready else "shadow_candidate_helper_weight",
                0.4 if benchmark_gate_ready else 0.15,
            )
        )
        max_safety_delta = float(
            helper_blend.get(
                "promoted_max_safety_delta" if benchmark_gate_ready else "shadow_candidate_max_safety_delta",
                0.25 if benchmark_gate_ready else 0.1,
            )
        )

        prior_preset_distribution = heuristic_preset_distribution(
            heuristic_advisory.get("focus_objective_presets", []) or []
        )
        prior_strategy_distribution = normalize_strategy_overrides(
            heuristic_advisory.get("sampler_strategy_overrides", {}) or {}
        )
        prior_safety = _clamp01(float(heuristic_advisory.get("safety_emphasis", 0.0) or 0.0))
        learned_preset_distribution = _normalize_vector(preset_probs, SHELL_POLICY_PRESET_LABELS)
        learned_strategy_distribution = _normalize_vector(strategy_probs, SHELL_POLICY_STRATEGY_KEYS)

        blended_preset_distribution = {
            label: ((1.0 - helper_weight) * prior_preset_distribution.get(label, 0.0))
            + (helper_weight * learned_preset_distribution.get(label, 0.0))
            for label in SHELL_POLICY_PRESET_LABELS
        }
        blended_strategy_distribution = {
            label: ((1.0 - helper_weight) * prior_strategy_distribution.get(label, 0.0))
            + (helper_weight * learned_strategy_distribution.get(label, 0.0))
            for label in SHELL_POLICY_STRATEGY_KEYS
        }
        blended_preset_distribution = _normalize_vector(
            np.asarray([blended_preset_distribution[label] for label in SHELL_POLICY_PRESET_LABELS], dtype=np.float64),
            SHELL_POLICY_PRESET_LABELS,
        )
        blended_strategy_distribution = _normalize_vector(
            np.asarray([blended_strategy_distribution[label] for label in SHELL_POLICY_STRATEGY_KEYS], dtype=np.float64),
            SHELL_POLICY_STRATEGY_KEYS,
        )
        raw_safety = ((1.0 - helper_weight) * prior_safety) + (helper_weight * safety_score)
        blended_safety = _clamp01(
            min(max(raw_safety, prior_safety - max_safety_delta), prior_safety + max_safety_delta)
        )

        execution_mode = str(heuristic_advisory.get("execution_mode", "advisory") or "advisory")
        activation_plan = dict(heuristic_advisory.get("activation_plan", {}) or {})
        activation_work_order = heuristic_advisory.get("activation_work_order")
        if benchmark_gate_ready and execution_mode != "advisory":
            if activation_score < 0.3 and helper_mode == "required":
                execution_mode = "advisory"
                activation_plan = {}
                activation_work_order = None
            elif activation_plan:
                activation_plan["helper_activation_preference"] = activation_score
                activation_plan["helper_recommended_mode"] = execution_mode if activation_score >= 0.5 else "advisory"

        return {
            "focus_objective_presets": _focus_presets_from_distribution(blended_preset_distribution),
            "sampler_strategy_overrides": blended_strategy_distribution,
            "safety_emphasis": blended_safety,
            "execution_mode": execution_mode,
            "activation_plan": activation_plan,
            "activation_work_order": activation_work_order,
            "policy_source": "heuristic_plus_learned_helper",
            "promotion_stage": "promoted" if benchmark_gate_ready else self.promotion_stage,
            "helper_trace": {
                "package_id": self.package.package_id,
                "helper_weight": helper_weight,
                "benchmark_gate_ready": benchmark_gate_ready,
                "prior_preset_distribution": prior_preset_distribution,
                "prior_strategy_distribution": prior_strategy_distribution,
                "prior_safety_emphasis": prior_safety,
                "learned_preset_distribution": learned_preset_distribution,
                "learned_strategy_distribution": learned_strategy_distribution,
                "learned_safety_emphasis": safety_score,
                "learned_activation_preference": activation_score,
                "final_preset_distribution": blended_preset_distribution,
                "final_strategy_distribution": blended_strategy_distribution,
                "final_safety_emphasis": blended_safety,
            },
        }


def resolve_orchestrator_shell_policy_helper(
    *,
    helper_mode: str = "disabled",
    package: Optional[OrchestratorShellPolicyRuntimePackage] = None,
    package_path: Optional[str | Path] = None,
) -> Optional[LoadedOrchestratorShellPolicyHelper]:
    mode = str(helper_mode or "disabled")
    if mode == "disabled":
        return None
    if package is None and package_path is None:
        if mode == "required":
            raise ValueError("orchestrator shell helper requires a package path")
        return None
    if package is None:
        package = load_orchestrator_shell_policy_runtime_package(package_path)
    helper = LoadedOrchestratorShellPolicyHelper(package)
    if mode == "required" and not bool(helper.benchmark_gate.get("ready", False)):
        raise ValueError("orchestrator shell helper requires a benchmark-gated package")
    return helper


__all__ = [
    "LoadedOrchestratorShellPolicyHelper",
    "OrchestratorShellPolicyRuntimePackage",
    "load_orchestrator_shell_policy_runtime_package",
    "resolve_orchestrator_shell_policy_helper",
]
