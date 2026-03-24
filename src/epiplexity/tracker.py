"""Epiplexity tracker with deterministic caching."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional
import hashlib
import json
import os

import torch

from src.epiplexity.estimators import EpiplexityEstimator, PrequentialAUCLossEstimator


@dataclass(frozen=True)
class ComputeBudget:
    max_steps: int
    batch_size: int = 16

    def budget_id(self) -> str:
        return f"steps_{int(self.max_steps)}_bs_{int(self.batch_size)}"


@dataclass(frozen=True)
class EpiplexityRunKey:
    repr_id: str
    repr_version_hash: str
    tokenizer_version: str
    transform_chain_hash: str
    dataset_slice_id: str
    probe_model_id: str
    compute_budget_id: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class EpiplexityResult:
    key: EpiplexityRunKey
    S_T_proxy: float
    H_T_proxy: float
    epi_per_flop: float
    delta_epi_vs_baseline: float
    loss_curve: List[float]
    flops_estimate: float = 0.0
    compute_normalizer: str = "flops_estimate"
    estimator_id: str = ""
    estimator_config_sha: str = ""
    score_mode: str = "absolute"
    baseline_repr_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "S_T_proxy": self.S_T_proxy,
            "H_T_proxy": self.H_T_proxy,
            "epi_per_flop": self.epi_per_flop,
            "delta_epi_vs_baseline": self.delta_epi_vs_baseline,
            "loss_curve": self.loss_curve,
            "flops_estimate": self.flops_estimate,
            "compute_normalizer": self.compute_normalizer,
            "estimator_id": self.estimator_id,
            "estimator_config_sha": self.estimator_config_sha,
            "score_mode": self.score_mode,
            "baseline_repr_id": self.baseline_repr_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpiplexityResult":
        return cls(
            key=EpiplexityRunKey(**data["key"]),
            S_T_proxy=float(data.get("S_T_proxy", 0.0)),
            H_T_proxy=float(data.get("H_T_proxy", 0.0)),
            epi_per_flop=float(data.get("epi_per_flop", 0.0)),
            delta_epi_vs_baseline=float(data.get("delta_epi_vs_baseline", 0.0)),
            loss_curve=list(data.get("loss_curve", [])),
            flops_estimate=float(data.get("flops_estimate", 0.0) or 0.0),
            compute_normalizer=str(data.get("compute_normalizer", "flops_estimate") or "flops_estimate"),
            estimator_id=str(data.get("estimator_id", "") or ""),
            estimator_config_sha=str(data.get("estimator_config_sha", "") or ""),
            score_mode=str(data.get("score_mode", "absolute") or "absolute"),
            baseline_repr_id=(
                str(data.get("baseline_repr_id"))
                if data.get("baseline_repr_id") is not None
                else None
            ),
        )


class EpiplexityTracker:
    def __init__(
        self,
        cache_dir: str = "artifacts/epiplexity_cache",
        estimator: Optional[EpiplexityEstimator] = None,
        cache_enabled: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.estimator = estimator or PrequentialAUCLossEstimator()
        self.cache_enabled = cache_enabled

    def evaluate_tokens(
        self,
        tokens: torch.Tensor,
        key: EpiplexityRunKey,
        budget: ComputeBudget,
        baseline_result: Optional[EpiplexityResult] = None,
    ) -> EpiplexityResult:
        cached = self._load_cache(key) if self.cache_enabled else None
        if cached is None:
            cached = self._compute_absolute(tokens=tokens, key=key, budget=budget)
            if self.cache_enabled:
                self._save_cache(cached)
        if baseline_result is None:
            return cached
        return self._with_baseline_delta(cached, baseline_result)

    def _compute_absolute(
        self,
        tokens: torch.Tensor,
        key: EpiplexityRunKey,
        budget: ComputeBudget,
    ) -> EpiplexityResult:
        tokens = _ensure_tokens_tensor(tokens)
        s_t, h_t, losses = self.estimator.fit_and_score(
            tokens=tokens,
            steps=budget.max_steps,
            batch_size=budget.batch_size,
            seed=key.seed,
        )
        flops_estimate = float(self.estimator.estimate_flops(tokens, budget.max_steps, budget.batch_size))
        if flops_estimate > 0.0:
            epi_per_flop = float(s_t) / flops_estimate
        else:
            epi_per_flop = float(s_t) / max(1.0, float(budget.max_steps))
            flops_estimate = float(max(1, budget.max_steps))

        return EpiplexityResult(
            key=key,
            S_T_proxy=float(s_t),
            H_T_proxy=float(h_t),
            epi_per_flop=float(epi_per_flop),
            delta_epi_vs_baseline=0.0,
            loss_curve=losses,
            flops_estimate=float(flops_estimate),
            compute_normalizer="flops_estimate",
            estimator_id=self.estimator.estimator_id(),
            estimator_config_sha=self.estimator.config_sha(),
            score_mode="absolute",
            baseline_repr_id=None,
        )

    def _with_baseline_delta(
        self,
        absolute_result: EpiplexityResult,
        baseline_result: EpiplexityResult,
    ) -> EpiplexityResult:
        return replace(
            absolute_result,
            delta_epi_vs_baseline=float(absolute_result.epi_per_flop - float(baseline_result.epi_per_flop)),
            score_mode="relative",
            baseline_repr_id=baseline_result.key.repr_id,
        )

    def _cache_path(self, key: EpiplexityRunKey) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = json.dumps(
            {
                "run_key": key.to_dict(),
                "estimator_id": self.estimator.estimator_id(),
                "estimator_config_sha": self.estimator.config_sha(),
            },
            sort_keys=True,
        )
        cache_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{cache_hash}.json")

    def _load_cache(self, key: EpiplexityRunKey) -> Optional[EpiplexityResult]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = EpiplexityResult.from_dict(data)
            if result.score_mode != "absolute":
                result.delta_epi_vs_baseline = 0.0
                result.score_mode = "absolute"
                result.baseline_repr_id = None
            return result
        except Exception:
            return None

    def _save_cache(self, result: EpiplexityResult) -> None:
        path = self._cache_path(result.key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, sort_keys=True)


def _ensure_tokens_tensor(tokens: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        out = tokens
    else:
        stacked = []
        for t in tokens:
            if t.dim() == 2:
                stacked.append(t)
            elif t.dim() == 3:
                stacked.append(t.squeeze(0))
        out = torch.stack(stacked, dim=0)
    if out.dim() != 3:
        raise ValueError("tokens must be [N, T, D]")
    return out.to(dtype=torch.float32)


__all__ = [
    "ComputeBudget",
    "EpiplexityRunKey",
    "EpiplexityResult",
    "EpiplexityTracker",
]
