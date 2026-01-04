"""End-to-end test for semantic simulation loop."""
from pathlib import Path

from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.ontology.models import Datapack, Task
from src.ontology.store import OntologyStore
from src.orchestrator import semantic_simulation
from src.orchestrator.schedule import BudgetConfig, reset_budget_state, set_budget_config


def test_semantic_simulation_e2e(monkeypatch, tmp_path: Path):
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=2, daily_step_budget=20_000_000, daily_run_budget=10))

    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_logging",
            name="Logging Task",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.12,
        )
    )
    store.append_datapacks(
        [
            Datapack(
                datapack_id="dp_logging",
                source_type="holosoma",
                task_id="task_logging",
                modality="motion",
                storage_uri="data/mocap/logging_clip.npz",
                metadata={
                    "tags": ["warehouse", "logging"],
                    "robot_families": ["G1"],
                    "objective_hint": "baseline",
                },
            )
        ]
    )

    trajectory_path = tmp_path / "traj.npz"
    trajectory_path.write_text("stub")

    class DummyBackend:
        def train_policy(self, **kwargs):
            return MotorTrainingResult(
                policy_id=str(tmp_path / "model.onnx"),
                raw_metrics={"train_steps": 100},
                econ_metrics={
                    "mpl_units_per_hour": 55.0,
                    "anti_reward_hacking_suspicious": 1.0,
                },
            )

        def evaluate_policy(self, **kwargs):
            rollout_bundle = RolloutBundle(
                scenario_id="scenario_stub",
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id="ep1",
                            task_id="task_logging",
                            robot_family="G1",
                            seed=42,
                            env_params={},
                        ),
                        trajectory_path=trajectory_path,
                    )
                ],
            )
            return MotorEvalResult(
                policy_id=str(tmp_path / "model.onnx"),
                raw_metrics={"mean_reward": 1.0},
                econ_metrics={"mpl_units_per_hour": 60.0},
                rollout_bundle=rollout_bundle,
            )

    monkeypatch.setattr(semantic_simulation, "make_motor_backend", lambda *_args, **_kwargs: DummyBackend())

    result = semantic_simulation.run_semantic_simulation(
        store=store,
        intent="reduce logging defects",
        tags=["warehouse", "logging"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_logging",
        eval_episodes=1,
        rollout_base_dir=tmp_path / "rollouts",
        datapack_output_dir=tmp_path / "datapacks",
        run_log_path=tmp_path / "runs.jsonl",
    )

    assert result.status == "completed"
    assert result.scenario is not None
    assert result.simulation is not None
    assert store.list_scenarios()

    scenario_record = store.list_scenarios()[0]
    assert scenario_record.get("task_id") == "task_logging"
    assert "train_metrics_anti_reward_hacking_suspicious" in scenario_record
    scenario_tags = scenario_record.get("datapack_tags", [])
    assert "auto_labeled" in scenario_tags
    assert "warehouse" in scenario_tags
    assert "logging" in scenario_tags

    new_datapack_ids = {dp.datapack_id for dp in store.list_datapacks()}
    assert "dp_logging_vla" in new_datapack_ids
    labeled_dp = next(dp for dp in store.list_datapacks() if dp.datapack_id == "dp_logging_vla")
    assert "auto_labeled" in labeled_dp.metadata.get("tags", [])
    reset_budget_state()
    set_budget_config(BudgetConfig())
