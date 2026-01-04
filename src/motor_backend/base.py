from __future__ import annotations

# Repo map (recon):
# - Tasks/envs: src/envs/*, src/env/isaac_adapter.py, src/physics/backends/*
# - Econ metrics: src/economics/reward_engine.py (EconVector aggregation), src/analytics/econ_reports.py (reporting)
# - Datapacks: src/ontology/store.py + src/ontology/models.py, src/valuation/datapack_schema.py, src/valuation/datapacks.py
# - Objectives: src/config/objective_profile.py (ObjectiveVector), src/valuation/reward_builder.py (combine_reward)

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING
from pathlib import Path

from src.motor_backend.datapacks import DatapackConfig
from src.objectives.economic_objective import EconomicObjectiveSpec

if TYPE_CHECKING:
    from src.motor_backend.rollout_capture import RolloutBundle


@dataclass
class MotorTrainingResult:
    policy_id: str
    raw_metrics: Mapping[str, float]
    econ_metrics: Mapping[str, float]
    rollout_bundle: "RolloutBundle | None" = None


@dataclass
class MotorEvalResult:
    policy_id: str
    raw_metrics: Mapping[str, float]
    econ_metrics: Mapping[str, float]
    rollout_bundle: "RolloutBundle | None" = None


class MotorBackend(Protocol):
    """Abstract interface for underlying motor stacks (Holosoma, others later)."""

    def train_policy(
        self,
        task_id: str,
        objective: EconomicObjectiveSpec,
        datapack_ids: Sequence[str],
        num_envs: int,
        max_steps: int,
        datapack_configs: Sequence[DatapackConfig] | None = None,
        scenario_id: str | None = None,
        rollout_base_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> MotorTrainingResult:
        ...

    def evaluate_policy(
        self,
        policy_id: str,
        task_id: str,
        objective: EconomicObjectiveSpec,
        num_episodes: int,
        scenario_id: str | None = None,
        rollout_base_dir: str | Path | None = None,
        seed: int | None = None,
    ) -> MotorEvalResult:
        ...

    def deploy_policy_handle(self, policy_id: str) -> Any:
        """Return a backend-specific handle for hooking into real robot / sim loops."""
        ...
