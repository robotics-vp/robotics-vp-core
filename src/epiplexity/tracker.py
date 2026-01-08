"""Epiplexity tracker with deterministic caching."""
from __future__ import annotations

from dataclasses import dataclass, asdict
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "S_T_proxy": self.S_T_proxy,
            "H_T_proxy": self.H_T_proxy,
            "epi_per_flop": self.epi_per_flop,
            "delta_epi_vs_baseline": self.delta_epi_vs_baseline,
            "loss_curve": self.loss_curve,
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
        if cached is not None:
            return cached

        tokens = _ensure_tokens_tensor(tokens)
        s_t, h_t, losses = self.estimator.fit_and_score(
            tokens=tokens,
            steps=budget.max_steps,
            batch_size=budget.batch_size,
            seed=key.seed,
        )
        epi_per_flop = float(s_t) / max(1.0, float(budget.max_steps))
        delta = 0.0
        if baseline_result is not None:
            delta = epi_per_flop - float(baseline_result.epi_per_flop)

        result = EpiplexityResult(
            key=key,
            S_T_proxy=float(s_t),
            H_T_proxy=float(h_t),
            epi_per_flop=float(epi_per_flop),
            delta_epi_vs_baseline=float(delta),
            loss_curve=losses,
        )
        if self.cache_enabled:
            self._save_cache(result)
        return result

    def _cache_path(self, key: EpiplexityRunKey) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key.to_hash()}.json")

    def _load_cache(self, key: EpiplexityRunKey) -> Optional[EpiplexityResult]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EpiplexityResult.from_dict(data)
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
