"""Tests for semantic policy helpers."""
from src.motor_backend.datapacks import DatapackConfig
from src.orchestrator.semantic_policy import (
    MissingScenarioSpec,
    apply_arh_penalty,
    detect_semantic_gaps,
    select_datapacks_for_intent,
)
from src.scenarios.metadata import ScenarioMetadata


def test_apply_arh_penalty_adjusts_mpl():
    metrics = {"mpl_units_per_hour": 100.0, "anti_reward_hacking_suspicious": 1.0}
    adjusted = apply_arh_penalty(metrics, penalty_factor=0.2)
    assert adjusted["mpl_units_per_hour_adjusted"] == 80.0
    assert adjusted["anti_reward_hacking_penalty"] == 0.2


def test_detect_semantic_gaps():
    scenario = ScenarioMetadata(
        scenario_id="holosoma:task:obj:run",
        task_id="task",
        motor_backend="holosoma",
        objective_name="obj",
        objective_weights={"mpl_weight": 1.0},
        datapack_ids=["dp1"],
        datapack_tags=["humanoid"],
        task_tags=[],
        robot_families=["G1"],
    )
    missing = detect_semantic_gaps(["humanoid", "warehouse"], "G1", [scenario])
    assert missing == [MissingScenarioSpec(tags=["warehouse"], robot_family="G1")]


def test_select_datapacks_prefers_non_arh():
    candidates = [
        DatapackConfig(id="dp1", description="", tags=["humanoid", "warehouse"]),
        DatapackConfig(id="dp2", description="", tags=["humanoid", "warehouse"]),
    ]
    scenarios = [
        {
            "datapack_ids": ["dp1"],
            "datapack_tags": ["humanoid", "warehouse"],
            "robot_families": ["G1"],
            "train_metrics_anti_reward_hacking_suspicious": 1.0,
        }
    ]

    selected = select_datapacks_for_intent(
        tags=["humanoid", "warehouse"],
        robot_family="G1",
        objective_hint=None,
        candidates=candidates,
        scenarios=scenarios,
    )
    assert selected
    assert selected[0].id == "dp2"
