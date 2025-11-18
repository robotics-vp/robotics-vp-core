#!/usr/bin/env python3
"""
Smoke test for ontology models + JSONL store (Phase A).
"""
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.models import Task, Robot, Datapack, Episode, EpisodeEvent, EconVector
from src.ontology.store import OntologyStore


def main():
    root = Path("data/ontology/test_smoke")
    if root.exists():
        shutil.rmtree(root)

    store = OntologyStore(root_dir=str(root))

    # Create models
    task = Task(
        task_id="task_drawer",
        name="Drawer Vase",
        description="Open drawer and move vase",
        environment_id="drawer_vase_env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(
        robot_id="robot_v1",
        name="DrawerBot",
        hardware_profile={"arms": 1, "gripper": "parallel"},
        energy_cost_per_wh=0.10,
    )
    now = datetime.utcnow()
    datapacks = [
        Datapack(
            datapack_id="dp1",
            source_type="human_video",
            task_id="task_drawer",
            modality="video",
            storage_uri="/tmp/dp1",
            novelty_score=0.2,
            quality_score=0.8,
            created_at=now,
        ),
        Datapack(
            datapack_id="dp2",
            source_type="synthetic_video",
            task_id="task_drawer",
            modality="video",
            storage_uri="/tmp/dp2",
            novelty_score=0.7,
            quality_score=0.9,
            created_at=now + timedelta(seconds=1),
        ),
    ]
    episode = Episode(
        episode_id="ep1",
        task_id="task_drawer",
        robot_id="robot_v1",
        datapack_id="dp1",
        started_at=now,
        ended_at=now + timedelta(minutes=5),
        status="success",
    )
    events = [
        EpisodeEvent(
            episode_id="ep1",
            timestep=0,
            event_type="step",
            timestamp=now,
            reward_scalar=0.1,
        ),
        EpisodeEvent(
            episode_id="ep1",
            timestep=1,
            event_type="collision",
            timestamp=now + timedelta(seconds=1),
            reward_scalar=-0.5,
            reward_components={"collision_penalty": -0.5},
        ),
        EpisodeEvent(
            episode_id="ep1",
            timestep=2,
            event_type="success",
            timestamp=now + timedelta(seconds=2),
            reward_scalar=1.0,
        ),
    ]
    econ = EconVector(
        episode_id="ep1",
        mpl_units_per_hour=95.0,
        wage_parity=0.8,
        energy_cost=0.12,
        damage_cost=0.01,
        novelty_delta=0.3,
        reward_scalar_sum=1.5,
        components={"mpl": 1.2, "safety": 0.3},
    )

    # Upserts/append
    store.upsert_task(task)
    store.upsert_robot(robot)
    store.append_datapacks(datapacks)
    store.upsert_episode(episode)
    store.append_events(events)
    store.upsert_econ_vector(econ)

    # Assertions
    assert store.get_task("task_drawer") is not None
    assert store.get_robot("robot_v1") is not None
    assert len(store.list_datapacks(task_id="task_drawer")) == 2
    ep = store.get_episode("ep1")
    assert ep is not None and ep.status == "success"
    econ_loaded = store.get_econ_vector("ep1")
    assert econ_loaded is not None and econ_loaded.reward_scalar_sum == econ.reward_scalar_sum
    evts = store.get_events("ep1")
    assert len(evts) == 3
    assert [e.event_type for e in evts] == ["step", "collision", "success"]

    print("[smoke_test_ontology_store] All tests passed.")


if __name__ == "__main__":
    main()
