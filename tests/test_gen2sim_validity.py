from src.evidence.gen2sim_validity import (
    build_gen2sim_feature_vector,
    resolve_gen2sim_validity_assessment,
)
from src.objectives.tensor import objective_tensor_from_axes


class _DummyGen2SimHelper:
    def predict_context(self, *, context):
        return {
            "predicted_validity_score": 0.95,
            "predicted_value_support_score": 0.85,
            "promotion_stage": "shadow_candidate",
        }


def test_gen2sim_not_applicable_for_non_generated_sources() -> None:
    assessment = resolve_gen2sim_validity_assessment(
        {
            "source": "real_seed",
            "plausibility_score": 0.8,
            "reward_safety_score": 0.9,
        },
        helper_status={"status": "package_missing", "promotion_stage": "heuristic_fallback"},
    )

    assert "gen2sim_not_applicable" in assessment.reason_codes
    assert assessment.metadata["helper_status"]["status"] == "package_missing"


def test_gen2sim_helper_adjustment_is_bounded() -> None:
    context = {
        "source": "synthetic_branch",
        "trust_score": 0.92,
        "std_ratio": 1.0,
        "branch_value": 0.8,
        "gap_labels": {"coverage_gap_contribution": 0.6, "economic_priority": 0.4},
        "metadata": {
            "scene_tracks_backend": "real",
            "vision_backbone_selected": "real",
            "semantic_grounding_mode": "non_heuristic",
            "semantic_memory_grounded": True,
        },
    }
    baseline = resolve_gen2sim_validity_assessment(context)
    adjusted = resolve_gen2sim_validity_assessment(
        context,
        helper=_DummyGen2SimHelper(),
        helper_status={
            "status": "loaded_direct",
            "promotion_stage": "shadow_candidate",
            "benchmark_gate_ready": False,
        },
    )

    expected_delta = 0.12 * (0.95 - baseline.validity_score)
    assert adjusted.validity_score > baseline.validity_score
    assert abs((adjusted.validity_score - baseline.validity_score) - expected_delta) < 1e-6
    assert "learned_helper_adjustment_applied" in adjusted.reason_codes


def test_gen2sim_feature_vector_preserves_objective_conditioning() -> None:
    tensor = objective_tensor_from_axes(
        {"throughput": 0.9, "error": 0.1, "safety": 0.8, "energy": 0.2}
    )
    feature_vector = build_gen2sim_feature_vector(
        {
            "source": "synthetic_branch",
            "objective_tensor": tensor,
            "metadata": {"scene_tracks_backend": "real"},
        }
    )

    assert len(feature_vector) == 16
    assert any(value > 0.0 for value in feature_vector[-4:])
