"""Tests for ontology query helpers."""
from src.ontology.models import Datapack
from src.ontology.query import find_datapacks, find_scenarios
from src.ontology.store import OntologyStore
from src.scenarios.metadata import ScenarioMetadata


def test_find_datapacks_filters_by_tags(tmp_path):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    dp = Datapack(
        datapack_id="dp1",
        source_type="holosoma",
        task_id="task_a",
        modality="motion",
        storage_uri="data/mocap/a.npz",
        metadata={"tags": ["humanoid", "warehouse"], "robot_families": ["G1"]},
    )
    store.append_datapacks([dp])

    results = find_datapacks(store, tags=["humanoid"], robot_family="G1")
    assert len(results) == 1
    assert results[0].datapack_id == "dp1"

    results = find_datapacks(store, tags=["missing"])
    assert results == []


def test_find_scenarios_filters_by_tags(tmp_path):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    scenario = ScenarioMetadata(
        scenario_id="holosoma:task_a:obj:run1",
        task_id="task_a",
        motor_backend="holosoma",
        objective_name="obj",
        objective_weights={"mpl_weight": 1.0},
        datapack_ids=["dp1"],
        datapack_tags=["humanoid"],
        task_tags=["reach"],
        robot_families=["G1"],
    )
    store.record_scenario(scenario, train_metrics={"mpl_units_per_hour": 10.0}, eval_metrics={})

    results = find_scenarios(store, datapack_tags=["humanoid"], robot_families=["G1"])
    assert len(results) == 1
    assert results[0]["scenario_id"] == scenario.scenario_id

    results = find_scenarios(store, datapack_tags=["missing"])
    assert results == []
