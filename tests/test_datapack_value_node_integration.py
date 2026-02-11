from src.regal.base import RegalDecision
from src.regal.data_value import RegalDataValueNode
from src.objectives.tensor import objective_tensor_from_axes


def test_datapack_value_node_allows_positive_frontier_gain():
    node = RegalDataValueNode()
    tensor = objective_tensor_from_axes(
        {"throughput": 0.8, "error": 0.2, "safety": 0.9, "energy": 0.2}
    )
    report = node.evaluate(
        {
            "objective_tensor": tensor,
            "task_id": "task",
            "env_id": "env",
            "profile_id": "profile",
            "compute_cost": 1.0,
            "plausibility_score": 0.9,
            "reward_safety_score": 0.9,
            "gen2sim_validity_score": 0.9,
        }
    )
    assert report.decision == RegalDecision.ALLOW
    assert "frontier_gain_positive" in report.reason_codes
