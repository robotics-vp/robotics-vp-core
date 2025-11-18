#!/usr/bin/env python3
"""
Smoke test for EpisodeLogger + RewardEngine + OntologyStore wiring.
"""
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.logging.episode_logger import EpisodeLogger
from src.economics.reward_engine import RewardEngine
from src.ontology.models import Task, Robot
from src.ontology.store import OntologyStore


def main():
    root = Path("data/ontology/test_logging_smoke")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))

    task = Task(
        task_id="task_logging",
        name="LoggingTask",
        environment_id="env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id="robot_logging", name="LoggerBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    logger = EpisodeLogger(store=store, task=task, robot=robot)
    reward_engine = RewardEngine(task=task, robot=robot, config={"wage_parity_stub": 0.9})

    episode = logger.start_episode()
    # Fake steps
    for t in range(3):
        scalar, components = reward_engine.step_reward(
            raw_env_reward=1.0 + t,
            info={"mpl_component": 10 + t, "energy_penalty": 0.1 * t, "collision_penalty": 0.0},
        )
        logger.log_step(
            timestep=t,
            reward_scalar=scalar,
            reward_components=components,
            state_summary={"t": t},
        )

    econ = reward_engine.compute_econ_vector(episode, logger._events)
    logger.mark_outcome(status="success")
    logger.finalize(econ_vector=econ)

    # Assertions
    ep_loaded = store.get_episode(episode.episode_id)
    assert ep_loaded is not None and ep_loaded.status == "success"
    events = store.get_events(episode.episode_id)
    assert len(events) == 3
    econ_loaded = store.get_econ_vector(episode.episode_id)
    assert econ_loaded is not None
    assert econ_loaded.reward_scalar_sum > 0
    assert econ_loaded.mpl_units_per_hour > 0
    # Deterministic ID check
    episode2 = logger.start_episode()
    logger.finalize()
    assert episode.episode_id == episode2.episode_id, "Episode ID should be deterministic for same task/robot/datapack"

    print("[smoke_test_episode_logging_and_econ_vector] All tests passed.")


if __name__ == "__main__":
    main()
