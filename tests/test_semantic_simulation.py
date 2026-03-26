"""Tests for semantic simulation runner."""
import json
from pathlib import Path

from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.ontology.models import Datapack, Task
from src.ontology.store import OntologyStore
from src.orchestrator import semantic_simulation
from src.orchestrator.schedule import BudgetConfig, reset_budget_state, set_budget_config


def test_run_semantic_simulation_with_stub_backend(monkeypatch, tmp_path: Path):
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=2, daily_step_budget=20_000_000, daily_run_budget=10))
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
    trajectory_path = run_dir / "trajectory.npz"
    trajectory_path.write_text("stub")

    class DummyBackend:
        def train_policy(self, **kwargs):
            return MotorTrainingResult(
                policy_id=str(policy_path),
                raw_metrics={"mean_reward": 1.0},
                econ_metrics={"mpl_units_per_hour": 50.0, "anti_reward_hacking_suspicious": 0.0},
            )

        def evaluate_policy(self, **kwargs):
            rollout_bundle = RolloutBundle(
                scenario_id="scenario_stub",
                episodes=[
                    EpisodeRollout(
                        metadata=EpisodeMetadata(
                            episode_id="ep1",
                            task_id="task_a",
                            robot_family="G1",
                            seed=None,
                            env_params={},
                        ),
                        trajectory_path=trajectory_path,
                    )
                ],
            )
            return MotorEvalResult(
                policy_id=str(policy_path),
                raw_metrics={"mean_reward": 1.0},
                econ_metrics={"mpl_units_per_hour": 55.0},
                rollout_bundle=rollout_bundle,
            )

    monkeypatch.setattr(semantic_simulation, "make_motor_backend", lambda *_args, **_kwargs: DummyBackend())

    result = semantic_simulation.run_semantic_simulation(
        store=store,
        tags=["humanoid"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_a",
        eval_episodes=1,
        rollout_base_dir=tmp_path / "rollouts",
        datapack_output_dir=tmp_path / "datapacks",
        run_log_path=tmp_path / "logs.jsonl",
        selection_scorer_package={
            "package_id": "selection_helper_smoke",
            "feature_weights": {"quality_score": 1.0},
            "max_adjustment": 0.25,
        },
    )

    assert result.status == "completed"
    assert result.scenario.task_id == "task_a"
    assert result.simulation is not None
    assert result.simulation.selection_summary is not None
    assert result.simulation.selection_summary["selected_ids"][0] == "dp1"
    assert result.simulation.selection_summary["selection_policy"] == "heuristic_plus_learned_helper"
    assert result.simulation.selection_summary["scorer_package_id"] == "selection_helper_smoke"
    assert result.simulation.selection_summary["selection_helper_status"]["status"] == "available"
    assert result.simulation.selection_summary["selection_helper_status"]["promotion_stage"] == "shadow_candidate"
    assert result.simulation.selection_summary["selection_helper_status"]["benchmark_gate_ready"] is False
    assert result.simulation.selection_summary["selection_context"]["candidate_pool_size_norm"] > 0.0
    selection_sidecar_path = trajectory_path.parent / "ep1_selection_summary_v1.json"
    sidecar_payload = json.loads(selection_sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_payload["selection_summary"]["selected_ids"] == ["dp1"]
    assert store.list_scenarios()
    assert any(dp.datapack_id == "dp1_vla" for dp in store.list_datapacks())
    labeled_dp = next(dp for dp in store.list_datapacks() if dp.datapack_id == "dp1_vla")
    assert "execution_preconditions" in labeled_dp.metadata
    assert "future_training_artifacts" in labeled_dp.metadata
    assert labeled_dp.metadata["semantic_fusion"]["status"] in {"ready", "blocked", "mixed"}
    assert (tmp_path / "datapacks" / "dp1_vla.yaml").exists()
    reset_budget_state()
    set_budget_config(BudgetConfig())


def test_run_semantic_simulation_required_selection_helper_must_exist(monkeypatch, tmp_path: Path):
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=2, daily_step_budget=20_000_000, daily_run_budget=10))
    monkeypatch.chdir(tmp_path)
    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_required",
            name="Task Required",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.12,
        )
    )
    store.append_datapacks(
        [
            Datapack(
                datapack_id="dp_required",
                source_type="holosoma",
                task_id="task_required",
                modality="motion",
                storage_uri="data/mocap/test_required.npz",
                metadata={
                    "tags": ["humanoid"],
                    "robot_families": ["G1"],
                    "objective_hint": "baseline",
                },
            )
        ]
    )

    result = semantic_simulation.run_semantic_simulation(
        store=store,
        tags=["humanoid"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_required",
        selection_scorer_mode="required",
    )

    assert result.status == "failed"
    assert result.reason is not None
    assert "selection_scorer_mode='required'" in result.reason

    reset_budget_state()
    set_budget_config(BudgetConfig())


def test_run_semantic_simulation_required_selection_helper_must_be_benchmark_ready(monkeypatch, tmp_path: Path):
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=2, daily_step_budget=20_000_000, daily_run_budget=10))
    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_ready",
            name="Task Ready",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.12,
        )
    )
    store.append_datapacks(
        [
            Datapack(
                datapack_id="dp_ready",
                source_type="holosoma",
                task_id="task_ready",
                modality="motion",
                storage_uri="data/mocap/test_ready.npz",
                metadata={
                    "tags": ["humanoid"],
                    "robot_families": ["G1"],
                    "objective_hint": "baseline",
                },
            )
        ]
    )

    result = semantic_simulation.run_semantic_simulation(
        store=store,
        tags=["humanoid"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_ready",
        selection_scorer_package={
            "package_id": "selection_helper_shadow",
            "feature_weights": {"quality_score": 1.0},
            "max_adjustment": 0.25,
        },
        selection_scorer_mode="required",
    )

    assert result.status == "failed"
    assert result.reason is not None
    assert "benchmark-gated ready" in result.reason

    reset_budget_state()
    set_budget_config(BudgetConfig())
