"""Tests for ARH-adjusted econ reporting."""

from src.analytics.econ_reports import compute_task_econ_summary
from src.ontology.models import EconVector, Episode, Task
from src.ontology.store import OntologyStore


def _build_store(tmp_path, components):
    store = OntologyStore(root_dir=tmp_path / "ontology")
    task = Task(
        task_id="task_arh",
        name="ARH Task",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=20.0,
        default_energy_cost_per_wh=0.1,
    )
    store.upsert_task(task)
    store.upsert_episode(Episode(episode_id="ep1", task_id=task.task_id, robot_id="robot_1", status="success"))
    store.upsert_econ_vector(
        EconVector(
            episode_id="ep1",
            mpl_units_per_hour=100.0,
            wage_parity=2.0,
            energy_cost=1.0,
            damage_cost=0.0,
            novelty_delta=0.0,
            reward_scalar_sum=1.0,
            components=components,
        )
    )
    return store, task.task_id


def test_arh_default_penalty_applies(monkeypatch, tmp_path):
    monkeypatch.delenv("ARH_PENALTY_FACTOR", raising=False)
    monkeypatch.delenv("ARH_EXCLUSION_THRESH", raising=False)
    store, task_id = _build_store(tmp_path, {"anti_reward_hacking_suspicious": 1.0})

    summary = compute_task_econ_summary(store, task_id)
    assert summary["mpl_raw"]["mean"] == 100.0
    assert summary["mpl"]["mean"] == 50.0
    assert summary["arh"]["suspicious_count"] == 1


def test_arh_penalty_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ARH_PENALTY_FACTOR", "0.2")
    monkeypatch.delenv("ARH_EXCLUSION_THRESH", raising=False)
    store, task_id = _build_store(tmp_path, {"anti_reward_hacking_suspicious": 1.0})

    summary = compute_task_econ_summary(store, task_id)
    assert summary["mpl"]["mean"] == 80.0


def test_arh_hard_exclusion(monkeypatch, tmp_path):
    monkeypatch.setenv("ARH_EXCLUSION_THRESH", "0.3")
    monkeypatch.delenv("ARH_PENALTY_FACTOR", raising=False)
    store, task_id = _build_store(tmp_path, {"anti_reward_hacking_score": 0.5})

    summary = compute_task_econ_summary(store, task_id)
    assert summary["mpl"]["mean"] == 0.0
    assert summary["arh"]["excluded_count"] == 1
