from __future__ import annotations

# Repo map (recon):
# - Tasks/envs: src/envs/*, src/env/isaac_adapter.py, src/physics/backends/*
# - Econ metrics: src/economics/reward_engine.py (EconVector aggregation), src/analytics/econ_reports.py (reporting)
# - Datapacks: src/ontology/store.py + src/ontology/models.py, src/valuation/datapack_schema.py, src/valuation/datapacks.py
# - Objectives: src/config/objective_profile.py (ObjectiveVector), src/valuation/reward_builder.py (combine_reward)

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from src.objectives.economic_objective import EconomicObjectiveSpec


@dataclass
class MotorTrainingResult:
    policy_id: str
    raw_metrics: Mapping[str, float]
    econ_metrics: Mapping[str, float]


@dataclass
class MotorEvalResult:
    policy_id: str
    raw_metrics: Mapping[str, float]
    econ_metrics: Mapping[str, float]


class MotorBackend(Protocol):
    """Abstract interface for underlying motor stacks (Holosoma, others later)."""

    def train_policy(
        self,
        task_id: str,
        objective: EconomicObjectiveSpec,
        datapack_ids: Sequence[str],
        num_envs: int,
        max_steps: int,
        seed: int | None = None,
    ) -> MotorTrainingResult:
        ...

    def evaluate_policy(
        self,
        policy_id: str,
        task_id: str,
        objective: EconomicObjectiveSpec,
        num_episodes: int,
        seed: int | None = None,
    ) -> MotorEvalResult:
        ...

    def deploy_policy_handle(self, policy_id: str) -> Any:
        """Return a backend-specific handle for hooking into real robot / sim loops."""
        ...
