"""Pareto frontier tracking for objective tensors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from src.objectives.tensor import ObjectiveTensor


@dataclass
class FrontierPoint:
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    compute_cost: float = 1.0


class ParetoFrontierTracker:
    """Track non-dominated objective outcomes and marginal gains."""

    def __init__(self, maximize: Optional[Mapping[str, bool]] = None):
        self.maximize = dict(maximize or {})
        self._frontiers: Dict[Tuple[str, str, str], List[FrontierPoint]] = {}

    def _key(self, task_id: str, env_id: str, profile_id: str) -> Tuple[str, str, str]:
        return (str(task_id), str(env_id), str(profile_id))

    def _direction(self, axis: str) -> float:
        return 1.0 if self.maximize.get(axis, True) else -1.0

    def _directed(self, vec: np.ndarray, axes: Iterable[str]) -> np.ndarray:
        signs = np.asarray([self._direction(axis) for axis in axes], dtype=np.float32)
        return vec * signs

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a >= b) and np.any(a > b))

    def marginal_gain(
        self,
        candidate: ObjectiveTensor,
        *,
        task_id: str,
        env_id: str,
        profile_id: str,
        compute_cost: float = 1.0,
    ) -> float:
        vec = candidate.mean_vector(normalize=True)
        axes = list(candidate.schema.axes)
        directed = self._directed(vec, axes)
        key = self._key(task_id, env_id, profile_id)
        frontier = self._frontiers.get(key, [])

        if not frontier:
            return 1.0 / max(compute_cost, 1e-6)

        directed_frontier = [self._directed(fp.vector, axes) for fp in frontier]

        if any(self._dominates(existing, directed) for existing in directed_frontier):
            return 0.0

        dominated_count = sum(1 for existing in directed_frontier if self._dominates(directed, existing))
        novelty = 0.0
        if directed_frontier:
            distances = [float(np.linalg.norm(directed - existing)) for existing in directed_frontier]
            novelty = float(np.mean(distances))
        raw_gain = 1.0 + dominated_count + 0.1 * novelty
        return raw_gain / max(compute_cost, 1e-6)

    def add(
        self,
        candidate: ObjectiveTensor,
        *,
        task_id: str,
        env_id: str,
        profile_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
        compute_cost: float = 1.0,
    ) -> float:
        gain = self.marginal_gain(
            candidate,
            task_id=task_id,
            env_id=env_id,
            profile_id=profile_id,
            compute_cost=compute_cost,
        )
        if gain <= 0.0:
            return 0.0

        vec = candidate.mean_vector(normalize=True).astype(np.float32)
        axes = list(candidate.schema.axes)
        directed_new = self._directed(vec, axes)

        key = self._key(task_id, env_id, profile_id)
        frontier = self._frontiers.setdefault(key, [])
        keep: List[FrontierPoint] = []
        for point in frontier:
            directed_existing = self._directed(point.vector, axes)
            if self._dominates(directed_new, directed_existing):
                continue
            keep.append(point)
        keep.append(
            FrontierPoint(
                vector=vec,
                metadata=dict(metadata or {}),
                compute_cost=float(compute_cost),
            )
        )
        self._frontiers[key] = keep
        return gain

    def frontier(self, *, task_id: str, env_id: str, profile_id: str) -> List[FrontierPoint]:
        return list(self._frontiers.get(self._key(task_id, env_id, profile_id), []))
