#!/usr/bin/env python3
"""
Smoke test for econ report helpers.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.analytics.econ_reports import (
    compute_task_econ_summary,
    compute_datapack_mix_summary,
    compute_pricing_snapshot,
)
from src.ontology.models import Task, Robot, Datapack, Episode, EconVector
from src.ontology.store import OntologyStore


def main():
    root = Path("data/ontology/test_econ_reports")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))

    task = Task(
        task_id="task_econ",
        name="EconTask",
        environment_id="env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    store.upsert_task(task)
    store.upsert_robot(Robot(robot_id="robot_econ", name="EconBot"))

    now = datetime.utcnow()
    store.append_datapacks([
        Datapack(datapack_id="dp_human", source_type="human_video", task_id="task_econ", modality="video", storage_uri="/tmp/dp1", novelty_score=0.2, quality_score=0.9, created_at=now),
        Datapack(datapack_id="dp_phys", source_type="physics", task_id="task_econ", modality="state", storage_uri="/tmp/dp2", novelty_score=0.6, quality_score=0.8, created_at=now),
    ])
    episodes = [
        Episode(episode_id="ep1", task_id="task_econ", robot_id="robot_econ", started_at=now, status="success"),
        Episode(episode_id="ep2", task_id="task_econ", robot_id="robot_econ", started_at=now, status="failure"),
    ]
    for ep in episodes:
        store.upsert_episode(ep)
    evs = [
        EconVector(episode_id="ep1", mpl_units_per_hour=90, wage_parity=1.2, energy_cost=0.5, damage_cost=0.1, novelty_delta=0.2, reward_scalar_sum=10.0),
        EconVector(episode_id="ep2", mpl_units_per_hour=70, wage_parity=1.0, energy_cost=0.7, damage_cost=0.2, novelty_delta=0.1, reward_scalar_sum=5.0),
    ]
    for ev in evs:
        store.upsert_econ_vector(ev)

    task_summary = compute_task_econ_summary(store, "task_econ")
    dp_summary = compute_datapack_mix_summary(store, "task_econ")
    pricing = compute_pricing_snapshot(store, "task_econ")

    assert task_summary["counts"]["episodes"] == 2
    assert "mpl" in task_summary and task_summary["mpl"]["mean"] > 0
    assert "sources" in dp_summary and len(dp_summary["sources"]) == 2
    assert pricing["human_unit_cost"] > 0

    print("[smoke_test_econ_reports] All tests passed.")


if __name__ == "__main__":
    main()
