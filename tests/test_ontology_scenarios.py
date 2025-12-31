"""Tests for scenario persistence in ontology store."""
from src.ontology.store import OntologyStore
from src.scenarios.metadata import ScenarioMetadata


def test_record_scenario_persists_flattened_metrics(tmp_path):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    scenario = ScenarioMetadata(
        scenario_id="holosoma:task:obj:run1",
        task_id="task",
        motor_backend="holosoma",
        objective_name="obj",
        objective_weights={"mpl_weight": 1.0},
        datapack_ids=["dp_a"],
        datapack_tags=["humanoid"],
        task_tags=["reach"],
        robot_families=["G1"],
        notes="test",
    )
    train_metrics = {"mpl_units_per_hour": 50.0, "anti_reward_hacking_reason": "none"}
    eval_metrics = {"mpl_units_per_hour": 60.0}

    store.record_scenario(scenario, train_metrics=train_metrics, eval_metrics=eval_metrics)

    records = store.list_scenarios()
    assert len(records) == 1
    record = records[0]
    assert record["scenario_id"] == scenario.scenario_id
    assert record["train_metrics_mpl_units_per_hour"] == 50.0
    assert record["eval_metrics_mpl_units_per_hour"] == 60.0
    assert record["train_metrics_anti_reward_hacking_reason"] == "none"
