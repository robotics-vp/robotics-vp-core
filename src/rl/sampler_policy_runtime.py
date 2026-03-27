from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.rl.sampler_policy import (
    SAMPLER_EPISODE_FEATURE_NAMES,
    SAMPLER_PLAN_PARAMETER_NAMES,
    SAMPLER_POLICY_STRATEGIES,
    SAMPLER_POOL_FEATURE_NAMES,
    build_sampler_episode_feature_map,
    build_sampler_pool_feature_map,
)
from src.rl.sampler_policy_training import (
    SamplerPolicyEpisodeNet,
    SamplerPolicyPoolNet,
    TORCH_AVAILABLE,
)

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


@dataclass(frozen=True)
class SamplerPolicyRuntimePackage:
    package_id: str
    package_path: str
    checkpoint_path: str
    model_config: Dict[str, Any]
    benchmark_gate: Dict[str, Any]
    execution_preconditions: Dict[str, Any]
    inference_contract: Dict[str, Any]
    promotion_stage: str
    metadata: Dict[str, Any]


def load_sampler_policy_runtime_package(path: str | Path) -> SamplerPolicyRuntimePackage:
    package_path = Path(path)
    payload = json.loads(package_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(payload.get("checkpoint_path") or ""))
    if checkpoint_path and not checkpoint_path.is_absolute():
        checkpoint_path = (package_path.parent / checkpoint_path).resolve()
    return SamplerPolicyRuntimePackage(
        package_id=str(payload.get("package_id") or package_path.stem),
        package_path=str(package_path.resolve()),
        checkpoint_path=str(checkpoint_path),
        model_config=dict(payload.get("model_config", {}) or {}),
        benchmark_gate=dict(payload.get("benchmark_gate", {}) or {}),
        execution_preconditions=dict(payload.get("execution_preconditions", {}) or {}),
        inference_contract=dict(payload.get("inference_contract", {}) or {}),
        promotion_stage=str(payload.get("promotion_stage", "shadow_candidate") or "shadow_candidate"),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


class LoadedSamplerPolicyHelper:
    def __init__(self, package: SamplerPolicyRuntimePackage) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load the sampler policy helper")
        checkpoint_path = Path(package.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"sampler policy checkpoint not found: {checkpoint_path}")
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        hidden_dim = int(payload.get("hidden_dim", 32))
        pool_input_dim = int(payload.get("pool_input_dim", len(SAMPLER_POOL_FEATURE_NAMES)))
        episode_input_dim = int(
            payload.get("episode_input_dim", len(SAMPLER_EPISODE_FEATURE_NAMES) + len(SAMPLER_POLICY_STRATEGIES))
        )
        self.package = package
        self.pool_model = SamplerPolicyPoolNet(input_dim=pool_input_dim, hidden_dim=hidden_dim)
        self.episode_model = SamplerPolicyEpisodeNet(input_dim=episode_input_dim, hidden_dim=hidden_dim)
        self.pool_model.load_state_dict(payload["pool_state_dict"])
        self.episode_model.load_state_dict(payload["episode_state_dict"])
        self.pool_model.eval()
        self.episode_model.eval()
        self.benchmark_gate = dict(package.benchmark_gate or {})
        self.inference_contract = dict(package.inference_contract or {})
        self.promotion_stage = str(package.promotion_stage or "shadow_candidate")

    def score_pool(
        self,
        episodes: Sequence[Mapping[str, Any]],
        *,
        heuristic_strategy_distribution: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        feature_map = build_sampler_pool_feature_map(
            episodes,
            heuristic_strategy_distribution=heuristic_strategy_distribution,
        )
        vector = np.asarray([float(feature_map.get(name, 0.0)) for name in SAMPLER_POOL_FEATURE_NAMES], dtype=np.float32)
        tensor = torch.from_numpy(vector).float().unsqueeze(0)
        with torch.no_grad():
            strategy_logits, plan_logits = self.pool_model(tensor)
            strategy_probs = torch.softmax(strategy_logits[0], dim=-1).cpu().numpy()
            plan_probs = torch.sigmoid(plan_logits[0]).cpu().numpy()
        return {
            "strategy_distribution": {
                strategy: _clamp01(score)
                for strategy, score in zip(SAMPLER_POLICY_STRATEGIES, strategy_probs.tolist())
            },
            "sampling_plan": {
                name: _clamp01(score)
                for name, score in zip(SAMPLER_PLAN_PARAMETER_NAMES, plan_probs.tolist())
            },
            "pool_feature_map": feature_map,
        }

    def score_episode(self, episode: Mapping[str, Any], strategy: str) -> Dict[str, Any]:
        feature_map = build_sampler_episode_feature_map(episode)
        vector = np.asarray(
            [
                *[float(feature_map.get(name, 0.0)) for name in SAMPLER_EPISODE_FEATURE_NAMES],
                *[1.0 if strategy == candidate else 0.0 for candidate in SAMPLER_POLICY_STRATEGIES],
            ],
            dtype=np.float32,
        )
        tensor = torch.from_numpy(vector).float().unsqueeze(0)
        with torch.no_grad():
            weight_score = float(torch.sigmoid(self.episode_model(tensor)[0]).item())
        return {
            "weight_score": _clamp01(weight_score),
            "feature_map": feature_map,
        }


def resolve_sampler_policy_helper(
    *,
    helper_mode: str = "disabled",
    package: Optional[SamplerPolicyRuntimePackage] = None,
    package_path: Optional[str | Path] = None,
) -> Optional[LoadedSamplerPolicyHelper]:
    mode = str(helper_mode or "disabled")
    if mode == "disabled":
        return None
    if package is None and package_path is None:
        if mode == "required":
            raise ValueError("sampler policy helper requires a package path")
        return None
    if package is None:
        package = load_sampler_policy_runtime_package(package_path)
    helper = LoadedSamplerPolicyHelper(package)
    if mode == "required" and not bool(helper.benchmark_gate.get("ready", False)):
        raise ValueError("sampler policy helper requires a benchmark-gated package")
    return helper


__all__ = [
    "LoadedSamplerPolicyHelper",
    "SamplerPolicyRuntimePackage",
    "load_sampler_policy_runtime_package",
    "resolve_sampler_policy_helper",
]
