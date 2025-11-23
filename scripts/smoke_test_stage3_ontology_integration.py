#!/usr/bin/env python3
"""
Integrated smoke: Stage2 enrichment -> Datapack, Stage3 descriptor -> Episode + EconVector via logging.
"""
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.datapack_adapters import datapack_from_stage2_enrichment
from src.ontology.episode_adapters import episode_from_descriptor
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot
from src.logging.episode_logger import EpisodeLogger
from src.economics.reward_engine import RewardEngine


def main():
    root = Path("data/ontology/test_stage3_integration")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))
    task = Task(
        task_id="task_int",
        name="IntegrationTask",
        environment_id="env",
        human_mpl_units_per_hour=60.0,
        human_wage_per_hour=18.0,
        default_energy_cost_per_wh=0.12,
    )
    robot = Robot(robot_id="robot_int", name="IntBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    # Stage 2 enrichment -> datapack
    enrichment = {
        "episode_id": "ep_stage2",
        "enrichment": {
            "novelty_tags": [{"novelty_score": 0.6, "expected_mpl_gain": 1.0}],
            "coherence_score": 0.8,
        },
    }
    dp = datapack_from_stage2_enrichment(enrichment, task_id=task.task_id)
    store.append_datapacks([dp])

    # Stage 3 descriptor -> episode projection
    descriptor = {
        "episode_id": "ep_stage3",
        "pack_id": dp.datapack_id,
        "env_name": "drawer_vase",
        "backend": "pybullet",
        "engine_type": "pybullet",
        "objective_preset": "balanced",
        "objective_vector": [1, 1, 1, 1, 0],
        "tier": 2,
        "trust_score": 0.9,
        "sampling_metadata": {
            "strategy": "frontier_prioritized",
            "phase": "frontier",
            "skill_mode": "frontier_exploration",
            "condition_metadata": {"skill_mode": "frontier_exploration", "curriculum_phase": "frontier", "tag_fingerprint": "abc", "tag_count": 1},
        },
        "semantic_tags": ["fragile"],
    }
    episode = episode_from_descriptor(descriptor, task_id=task.task_id, robot_id=robot.robot_id)
    store.upsert_episode(episode)

    # Log events and econ vector
    logger = EpisodeLogger(store=store, task=task, robot=robot)
    reward_engine = RewardEngine(task=task, robot=robot, config={"wage_parity_stub": 1.1})
    logger._current_episode = episode
    logger._events = []
    for t in range(2):
        scalar, comps = reward_engine.step_reward(1.0 + t, {"mpl_component": 80 + t, "energy_penalty": 0.2})
        logger.log_step(timestep=t, reward_scalar=scalar, reward_components=comps, state_summary={"t": t})
    econ = reward_engine.compute_econ_vector(episode, logger._events)
    logger.mark_outcome(status="success", metadata={"note": "integration_smoke"})
    logger.finalize(econ_vector=econ)

    ep_loaded = store.get_episode(episode.episode_id)
    econ_loaded = store.get_econ_vector(episode.episode_id)
    assert ep_loaded is not None
    assert econ_loaded is not None
    assert econ_loaded.episode_id == ep_loaded.episode_id
    assert ep_loaded.datapack_id == dp.datapack_id
    assert ep_loaded.metadata.get("curriculum_phase") == "frontier"
    assert ep_loaded.metadata.get("sampler_strategy") == "frontier_prioritized"
    assert ep_loaded.metadata.get("skill_mode") == "frontier_exploration"
    assert ep_loaded.metadata.get("condition_vector_summary", {}).get("skill_mode") == "frontier_exploration"
    print("[smoke_test_stage3_ontology_integration] All tests passed.")


if __name__ == "__main__":
    main()
