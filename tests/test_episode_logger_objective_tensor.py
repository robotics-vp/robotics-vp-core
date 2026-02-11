from datetime import datetime

from src.logging.episode_logger import EpisodeLogger
from src.objectives.tensor import objective_tensor_from_axes
from src.ontology.models import EconVector, Robot, Task
from src.ontology.store import OntologyStore


def test_episode_logger_persists_objective_tensor(tmp_path):
    store = OntologyStore(root_dir=str(tmp_path / "ontology"))
    task = Task(task_id="t1", name="Task")
    robot = Robot(robot_id="r1", name="Robot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    logger = EpisodeLogger(store=store, task=task, robot=robot)
    episode = logger.start_episode()
    logger.log_step(
        timestep=0,
        reward_scalar=1.0,
        reward_components={"mpl_component": 0.8},
        state_summary={"ok": True},
    )

    objective_tensor = objective_tensor_from_axes(
        {"throughput": 0.8, "error": 0.1, "safety": 0.9, "energy": 0.2},
        context={"episode_id": episode.episode_id},
    )
    econ_vector = EconVector(
        episode_id=episode.episode_id,
        mpl_units_per_hour=1.0,
        wage_parity=1.0,
        energy_cost=0.1,
        damage_cost=0.0,
        novelty_delta=0.0,
        reward_scalar_sum=1.0,
        metadata={"created": datetime.utcnow().isoformat()},
    )

    logger.mark_outcome("success")
    logger.finalize(econ_vector=econ_vector, objective_tensor=objective_tensor)

    stored = store.get_objective_tensor(episode.episode_id)
    assert stored is not None
    assert stored["objective_tensor"]["version"] == "objective_tensor_v1"
    ep_loaded = store.get_episode(episode.episode_id)
    assert ep_loaded is not None
    assert ep_loaded.metadata.get("objective_tensor_present") is True
