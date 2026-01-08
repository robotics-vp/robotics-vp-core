"""Harness for representation/transform epiplexity comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import json
import os
import statistics

import torch

from src.epiplexity.tracker import EpiplexityTracker, EpiplexityRunKey, EpiplexityResult, ComputeBudget
from src.epiplexity.metadata import attach_epiplexity_result, attach_epiplexity_summary

RepresentationFn = Callable[[Union[Sequence[Any], Any]], torch.Tensor]


@dataclass
class EpiplexityLeaderboard:
    dataset_slice_id: str
    baseline_repr: str
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
    summaries: Dict[str, Dict[str, Dict[str, Any]]]


class TokenizerAblationHarness:
    def __init__(
        self,
        tracker: Optional[EpiplexityTracker] = None,
        representation_fns: Optional[Dict[str, RepresentationFn]] = None,
        output_dir: str = "artifacts/epiplexity_leaderboards",
    ) -> None:
        self.tracker = tracker or EpiplexityTracker()
        self.representation_fns = representation_fns or {}
        self.output_dir = output_dir

    def evaluate(
        self,
        episodes: Sequence[Any] | Any,
        repr_ids: Sequence[str],
        budgets: Sequence[ComputeBudget],
        seeds: Sequence[int],
        baseline_repr: str,
        dataset_slice_id: str,
        repr_version_hashes: Optional[Dict[str, str]] = None,
        tokenizer_versions: Optional[Dict[str, str]] = None,
        transform_chain_hashes: Optional[Dict[str, str]] = None,
        probe_model_id: str = "probe_mlp",
        datapacks: Optional[Sequence[Any]] = None,
        store_full_runs: bool = False,
    ) -> EpiplexityLeaderboard:
        repr_version_hashes = repr_version_hashes or {}
        tokenizer_versions = tokenizer_versions or {}
        transform_chain_hashes = transform_chain_hashes or {}

        results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}

        baseline_tokens = self._tokens_for_repr(baseline_repr, episodes)
        baseline_cache: Dict[str, Dict[int, EpiplexityResult]] = {}

        for budget in budgets:
            budget_id = budget.budget_id()
            baseline_cache[budget_id] = {}
            for seed in seeds:
                base_key = EpiplexityRunKey(
                    repr_id=baseline_repr,
                    repr_version_hash=repr_version_hashes.get(baseline_repr, "v1"),
                    tokenizer_version=tokenizer_versions.get(baseline_repr, "v1"),
                    transform_chain_hash=transform_chain_hashes.get(baseline_repr, "v1"),
                    dataset_slice_id=dataset_slice_id,
                    probe_model_id=probe_model_id,
                    compute_budget_id=budget_id,
                    seed=int(seed),
                )
                baseline_cache[budget_id][seed] = self.tracker.evaluate_tokens(
                    tokens=baseline_tokens,
                    key=base_key,
                    budget=budget,
                )

        for repr_id in repr_ids:
            tokens = self._tokens_for_repr(repr_id, episodes)
            results.setdefault(repr_id, {})
            summaries.setdefault(repr_id, {})

            for budget in budgets:
                budget_id = budget.budget_id()
                results[repr_id].setdefault(budget_id, {})
                metrics_per_seed = []
                for seed in seeds:
                    key = EpiplexityRunKey(
                        repr_id=repr_id,
                        repr_version_hash=repr_version_hashes.get(repr_id, "v1"),
                        tokenizer_version=tokenizer_versions.get(repr_id, "v1"),
                        transform_chain_hash=transform_chain_hashes.get(repr_id, "v1"),
                        dataset_slice_id=dataset_slice_id,
                        probe_model_id=probe_model_id,
                        compute_budget_id=budget_id,
                        seed=int(seed),
                    )
                    baseline_result = baseline_cache[budget_id][seed]
                    result = self.tracker.evaluate_tokens(tokens, key, budget, baseline_result=baseline_result)
                    metrics_per_seed.append(result)
                    results[repr_id][budget_id][str(seed)] = {
                        "S_T_proxy": result.S_T_proxy,
                        "H_T_proxy": result.H_T_proxy,
                        "epi_per_flop": result.epi_per_flop,
                        "delta_epi_vs_baseline": result.delta_epi_vs_baseline,
                    }
                    if datapacks and store_full_runs:
                        for dp in datapacks:
                            attach_epiplexity_result(dp, result)

                summaries[repr_id][budget_id] = _summarize_results(metrics_per_seed)
                if datapacks:
                    for dp in datapacks:
                        attach_epiplexity_summary(dp, repr_id, budget_id, summaries[repr_id][budget_id])

        leaderboard = EpiplexityLeaderboard(
            dataset_slice_id=dataset_slice_id,
            baseline_repr=baseline_repr,
            results=results,
            summaries=summaries,
        )
        self._write_leaderboard(leaderboard)
        return leaderboard

    def _tokens_for_repr(self, repr_id: str, episodes: Sequence[Any] | Any) -> torch.Tensor:
        if repr_id not in self.representation_fns:
            raise ValueError(f"Representation '{repr_id}' not registered")
        tokens = self.representation_fns[repr_id](episodes)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.as_tensor(tokens, dtype=torch.float32)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        return tokens

    def _write_leaderboard(self, leaderboard: EpiplexityLeaderboard) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{leaderboard.dataset_slice_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_slice_id": leaderboard.dataset_slice_id,
                    "baseline_repr": leaderboard.baseline_repr,
                    "results": leaderboard.results,
                    "summaries": leaderboard.summaries,
                },
                f,
                indent=2,
                sort_keys=True,
            )


def _summarize_results(results: List[EpiplexityResult]) -> Dict[str, Any]:
    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _std(vals: List[float]) -> float:
        return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    s_vals = [r.S_T_proxy for r in results]
    h_vals = [r.H_T_proxy for r in results]
    epi_vals = [r.epi_per_flop for r in results]
    delta_vals = [r.delta_epi_vs_baseline for r in results]

    confidence = 1.0 / (1.0 + _std(delta_vals)) if delta_vals else 0.0

    return {
        "mean": {
            "S_T_proxy": _mean(s_vals),
            "H_T_proxy": _mean(h_vals),
            "epi_per_flop": _mean(epi_vals),
            "delta_epi_vs_baseline": _mean(delta_vals),
        },
        "std": {
            "S_T_proxy": _std(s_vals),
            "H_T_proxy": _std(h_vals),
            "epi_per_flop": _std(epi_vals),
            "delta_epi_vs_baseline": _std(delta_vals),
        },
        "confidence": confidence,
    }


__all__ = ["TokenizerAblationHarness", "EpiplexityLeaderboard"]
