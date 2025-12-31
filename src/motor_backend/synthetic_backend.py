from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.datapacks import DatapackConfig
from src.motor_backend.rollout_capture import (
    EpisodeMetadata,
    finalize_rollout_bundle,
    record_episode_rollout,
    start_rollout_capture,
)
from src.objectives.economic_objective import EconomicObjectiveSpec


@dataclass(frozen=True)
class SyntheticPolicyHandle:
    policy_id: str

    def act(self, obs: Any) -> dict[str, float]:
        return {"action": 0.0}

    def step(self, obs: Any) -> dict[str, float]:
        return self.act(obs)


class SyntheticBackend:
    """Deterministic motor backend for smoke tests and non-sim integration."""

    def __init__(self, econ_meter: EconomicMeter) -> None:
        self._econ_meter = econ_meter

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
        policy_id = f"synthetic_{uuid.uuid4().hex[:8]}"
        raw_metrics = _simulate_metrics(
            objective=objective,
            num_envs=num_envs,
            max_steps=max_steps,
            seed=seed,
            phase="train",
        )
        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))
        econ_metrics.setdefault("anti_reward_hacking_suspicious", 0.0)
        return MotorTrainingResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=None,
        )

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
        raw_metrics = _simulate_metrics(
            objective=objective,
            num_envs=1,
            max_steps=max(num_episodes, 1),
            seed=seed,
            phase="eval",
        )
        raw_metrics["num_episodes"] = float(num_episodes)
        econ_metrics = dict(self._econ_meter.summarize(raw_metrics))
        econ_metrics.setdefault("anti_reward_hacking_suspicious", 0.0)

        rollout_bundle = None
        if scenario_id and rollout_base_dir and num_episodes > 0:
            base_dir = Path(rollout_base_dir)
            start_rollout_capture(scenario_id, base_dir)
            episodes_to_record = max(1, min(num_episodes, 3))
            for idx in range(episodes_to_record):
                episode_meta = EpisodeMetadata(
                    episode_id=f"{scenario_id}_synthetic_{idx:03d}",
                    task_id=task_id,
                    robot_family=None,
                    seed=seed,
                    env_params={"objective": _objective_to_dict(objective)},
                )
                record_episode_rollout(
                    scenario_id=scenario_id,
                    episode_idx=idx,
                    metadata=episode_meta,
                    trajectory_data={"policy_id": policy_id, "metrics": dict(raw_metrics)},
                    rgb_frames=None,
                    depth_frames=None,
                    metrics=raw_metrics,
                    base_dir=base_dir,
                )
            rollout_bundle = finalize_rollout_bundle(scenario_id, base_dir)

        return MotorEvalResult(
            policy_id=policy_id,
            raw_metrics=raw_metrics,
            econ_metrics=econ_metrics,
            rollout_bundle=rollout_bundle,
        )

    def deploy_policy_handle(self, policy_id: str) -> Any:
        return SyntheticPolicyHandle(policy_id=policy_id)


def _simulate_metrics(
    *,
    objective: EconomicObjectiveSpec,
    num_envs: int,
    max_steps: int,
    seed: int | None,
    phase: str,
) -> dict[str, float]:
    rng = random.Random(seed if seed is not None else 0)
    mpl_base = 40.0 + rng.uniform(-4.0, 4.0) + (objective.mpl_weight * 5.0)
    error_rate = max(0.0, 0.2 - objective.error_weight * 0.02 + rng.uniform(-0.03, 0.03))
    success_rate = max(0.0, min(1.0, 1.0 - error_rate))
    mean_episode_length_s = max(5.0, 60.0 - objective.mpl_weight * 2.0 + rng.uniform(-5.0, 5.0))
    energy_kwh = max(0.05, 0.4 + objective.energy_weight * 0.1 + rng.uniform(-0.05, 0.05))
    reward = mpl_base * 0.05 - (error_rate * 5.0) - (energy_kwh * 2.0)
    if phase == "eval":
        mpl_base *= 1.05
        reward *= 1.1
        mean_episode_length_s *= 0.95

    return {
        "mpl_units_per_hour": float(max(mpl_base, 0.0)),
        "success_rate": float(success_rate),
        "error_rate": float(error_rate),
        "energy_kwh": float(energy_kwh),
        "mean_episode_length_s": float(mean_episode_length_s),
        "mean_reward": float(reward),
        "train_steps": float(max_steps),
        "num_envs": float(num_envs),
    }


def _objective_to_dict(objective: EconomicObjectiveSpec) -> dict[str, float]:
    payload = {
        "mpl_weight": objective.mpl_weight,
        "energy_weight": objective.energy_weight,
        "error_weight": objective.error_weight,
        "novelty_weight": objective.novelty_weight,
        "risk_weight": objective.risk_weight,
    }
    if objective.extra_weights:
        payload.update({str(k): float(v) for k, v in objective.extra_weights.items()})
    return payload
