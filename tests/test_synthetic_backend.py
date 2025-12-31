"""Tests for the synthetic motor backend."""

from pathlib import Path

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.synthetic_backend import SyntheticBackend
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.ontology.models import Robot, Task


def test_synthetic_backend_train_and_eval(tmp_path: Path) -> None:
    task = Task(
        task_id="task_synth",
        name="SyntheticTask",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=20.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id="robot_synth", name="robot_synth", energy_cost_per_wh=0.12)
    meter = EconomicMeter(task=task, robot=robot)
    backend = SyntheticBackend(econ_meter=meter)

    objective = EconomicObjectiveSpec(mpl_weight=1.0, energy_weight=0.3, error_weight=0.2)
    train = backend.train_policy(
        task_id=task.task_id,
        objective=objective,
        datapack_ids=[],
        num_envs=4,
        max_steps=100,
        seed=123,
    )
    assert train.policy_id
    assert train.econ_metrics["mpl_units_per_hour"] > 0.0

    eval_result = backend.evaluate_policy(
        policy_id=train.policy_id,
        task_id=task.task_id,
        objective=objective,
        num_episodes=2,
        scenario_id="scenario_synth",
        rollout_base_dir=tmp_path,
        seed=123,
    )
    assert eval_result.econ_metrics["mpl_units_per_hour"] > 0.0
    assert eval_result.rollout_bundle is not None
    assert eval_result.rollout_bundle.episodes
