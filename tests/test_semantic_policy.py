"""Tests for semantic policy helpers."""
from src.orchestrator.semantic_policy import apply_arh_penalty, detect_semantic_gaps
from src.ontology.store import OntologyStore
from src.scenarios.metadata import ScenarioMetadata


def test_apply_arh_penalty_adjusts_mpl():
    metrics = {"mpl_units_per_hour": 100.0, "anti_reward_hacking_suspicious": 1.0}
    adjusted = apply_arh_penalty(metrics, penalty_factor=0.2)
    assert adjusted["mpl_units_per_hour_adjusted"] == 80.0
    assert adjusted["anti_reward_hacking_penalty"] == 0.2


def test_detect_semantic_gaps(tmp_path):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    scenario = ScenarioMetadata(
        scenario_id="holosoma:task:obj:run",
        task_id="task",
        motor_backend="holosoma",
        objective_name="obj",
        objective_weights={"mpl_weight": 1.0},
        datapack_ids=["dp1"],
        datapack_tags=["humanoid"],
        task_tags=[],
        robot_families=[],
    )
    store.record_scenario(scenario, train_metrics={}, eval_metrics={})

    missing = detect_semantic_gaps(store, ["humanoid", "warehouse"])
    assert missing == ["warehouse"]
