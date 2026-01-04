"""Tests for scenario metadata builder."""
from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.scenarios.metadata import build_scenario_metadata


def test_build_scenario_metadata_dedupes_tags():
    objective = EconomicObjectiveSpec(mpl_weight=1.0, error_weight=2.0, extra_weights={"foo": 3.0})
    datapacks = [
        DatapackConfig(
            id="dp_a",
            description="A",
            motion_clips=[MotionClipSpec(path="data/a.npz")],
            tags=["humanoid", "warehouse"],
            task_tags=["reach"],
            robot_families=["G1"],
        ),
        DatapackConfig(
            id="dp_b",
            description="B",
            motion_clips=[MotionClipSpec(path="data/b.npz")],
            tags=["humanoid", "logging"],
            task_tags=["locomotion"],
            robot_families=["G1", "T1"],
        ),
    ]

    scenario = build_scenario_metadata(
        run_id="run123",
        task_id="logging_task",
        motor_backend="holosoma",
        objective_name="baseline",
        objective=objective,
        datapacks=datapacks,
        notes="test run",
    )

    assert scenario.scenario_id == "holosoma:logging_task:baseline:run123"
    assert scenario.datapack_ids == ["dp_a", "dp_b"]
    assert scenario.datapack_tags == ["humanoid", "logging", "warehouse"]
    assert scenario.task_tags == ["locomotion", "reach"]
    assert scenario.robot_families == ["G1", "T1"]
    assert scenario.objective_weights["mpl_weight"] == 1.0
    assert scenario.objective_weights["error_weight"] == 2.0
    assert scenario.objective_weights["foo"] == 3.0
    assert scenario.notes == "test run"
