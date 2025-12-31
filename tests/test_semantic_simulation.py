"""Tests for semantic simulation runner."""
from pathlib import Path

from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.rollout_capture import RolloutCaptureConfig
from src.ontology.models import Datapack, Task
from src.ontology.store import OntologyStore
from src.orchestrator import semantic_simulation


def test_run_semantic_simulation_with_stub_backend(monkeypatch, tmp_path: Path):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_a",
            name="Task A",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.12,
        )
    )
    store.append_datapacks(
        [
            Datapack(
                datapack_id="dp1",
                source_type="holosoma",
                task_id="task_a",
                modality="motion",
                storage_uri="data/mocap/test.npz",
                metadata={
                    "tags": ["humanoid"],
                    "robot_families": ["G1"],
                    "objective_hint": "baseline",
                },
            )
        ]
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    policy_path = run_dir / "model.onnx"
    policy_path.write_text("stub")

    class DummyBackend:
        def train_policy(self, **kwargs):
            return MotorTrainingResult(
                policy_id=str(policy_path),
                raw_metrics={"mean_reward": 1.0},
                econ_metrics={"mpl_units_per_hour": 50.0, "anti_reward_hacking_suspicious": 0.0},
            )

        def evaluate_policy(self, **kwargs):
            return MotorEvalResult(
                policy_id=str(policy_path),
                raw_metrics={"mean_reward": 1.0},
                econ_metrics={"mpl_units_per_hour": 55.0},
            )

    monkeypatch.setattr(semantic_simulation, "make_motor_backend", lambda *_args, **_kwargs: DummyBackend())

    result = semantic_simulation.run_semantic_simulation(
        store=store,
        tags=["humanoid"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_a",
        eval_episodes=1,
        rollout_capture_config=RolloutCaptureConfig(output_dir=tmp_path / "rollouts"),
        datapack_output_dir=tmp_path / "datapacks",
    )

    assert result.scenario.task_id == "task_a"
    assert store.list_scenarios()
    assert any(dp.datapack_id == "dp1_vla" for dp in store.list_datapacks())
    assert (tmp_path / "datapacks" / "dp1_vla.yaml").exists()
