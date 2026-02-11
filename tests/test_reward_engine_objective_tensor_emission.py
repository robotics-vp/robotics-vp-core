from src.economics.reward_engine import RewardEngine
from src.objectives.profile import ObjectiveProfile
from src.ontology.models import Robot, Task


def test_reward_engine_emits_objective_tensor_when_profile_set():
    task = Task(task_id="task", name="Task")
    robot = Robot(robot_id="robot", name="Robot")
    profile = ObjectiveProfile(
        scalarizer="weighted_sum",
        weights={"throughput": 1.0, "error": 1.0, "safety": 1.0, "energy": 1.0},
        maximize={"throughput": True, "error": False, "safety": True, "energy": False},
    )
    engine = RewardEngine(task, robot, config={}, objective_profile=profile)
    scalar, components = engine.step_reward(
        0.5,
        {
            "mpl_component": 0.8,
            "delta_errors": 0.2,
            "energy_penalty": 0.1,
            "safety_bonus": 0.1,
            "map_first_quality_score": 0.9,
        },
    )
    assert "objective_tensor_v1" in components
    assert "objective_constraint_flags" in components
    assert isinstance(scalar, float)
