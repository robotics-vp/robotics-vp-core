"""Tests for schedule budget manager."""
import pytest

from src.orchestrator.schedule import (
    BudgetConfig,
    BudgetExceeded,
    acquire_run_budget,
    release_run_budget,
    reset_budget_state,
    set_budget_config,
)
from src.ontology.store import OntologyStore
from src.orchestrator import semantic_simulation


def test_budget_acquire_release():
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=2, daily_step_budget=100, daily_run_budget=2))

    acquire_run_budget(50)
    acquire_run_budget(40)

    with pytest.raises(BudgetExceeded):
        acquire_run_budget(10)

    release_run_budget(50)
    release_run_budget(40)
    reset_budget_state()
    set_budget_config(BudgetConfig())


def test_run_semantic_simulation_defers_on_budget(tmp_path):
    reset_budget_state()
    set_budget_config(BudgetConfig(max_concurrent_runs=1, daily_step_budget=10, daily_run_budget=1))

    store = OntologyStore(root_dir=tmp_path / "ontology")
    result = semantic_simulation.run_semantic_simulation(
        store=store,
        intent="test",
        tags=["warehouse"],
        robot_family="G1",
        objective_hint="baseline",
        task_id="task_a",
        num_envs=10,
        max_steps=10,
        run_log_path=tmp_path / "runs.jsonl",
    )

    assert result.status == "deferred"
    reset_budget_state()
    set_budget_config(BudgetConfig())
