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


class _DummyHelper:
    def predict_context(self, *, context):
        return {
            "predicted_validity_score": 0.95,
            "predicted_value_support_score": 0.85,
            "promotion_stage": "shadow_candidate",
        }


def test_datapack_value_node_uses_gen2sim_assessment_for_generated_sources():
    node = RegalDataValueNode(gen2sim_validity_helper=_DummyHelper())
    tensor = objective_tensor_from_axes(
        {"throughput": 0.85, "error": 0.15, "safety": 0.9, "energy": 0.25}
    )
    report = node.evaluate(
        {
            "objective_tensor": tensor,
            "task_id": "task",
            "env_id": "env",
            "profile_id": "profile",
            "compute_cost": 1.0,
            "source": "synthetic_branch",
            "trust_score": 0.93,
            "std_ratio": 1.0,
            "branch_value": 0.8,
            "gap_labels": {
                "coverage_gap_contribution": 0.5,
                "economic_priority": 0.4,
            },
            "metadata": {
                "scene_tracks_backend": "real",
                "vision_backbone_selected": "real",
                "semantic_grounding_mode": "non_heuristic",
                "semantic_memory_grounded": True,
            },
        }
    )

    assert report.decision == RegalDecision.ALLOW
    assert report.details["reliability"] != report.details["base_reliability"]
    assert (
        report.details["gen2sim_validity_assessment"]["metadata"]["helper_status"]["status"]
        == "loaded_direct"
    )
