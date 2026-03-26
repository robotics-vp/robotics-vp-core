from src.orchestrator.queue_selection import build_live_queue_selection
from src.regality.promotion_policy import load_regal_promotion_policy
from src.rl.econ_regal_sampling import recommend_sampling


def test_econ_regal_sampling_and_live_queue_shim():
    policy = load_regal_promotion_policy("configs/regality/promotion_default.yaml")
    recommendation = recommend_sampling(
        objective_profile_coverage_gap=0.4,
        constraint_violation_count=2,
        uncertainty=0.5,
        datapack_value=1.2,
        signal_yield_score=0.2,
        regal_support_score=0.8,
        deploy_recommendation="require_review",
        pricing_recommendation="publish_discounted",
        datapack_recommendation="review",
        promotion_policy=policy,
        replay_policy_error=0.3,
        provenance_quality=0.4,
    )
    assert recommendation.priority_label == "high"
    assert "pricing_review" in recommendation.queue_tags
    assert "pricing_truth_review" in recommendation.queue_tags
    assert "signal_yield_candidate" in recommendation.queue_tags

    queue = build_live_queue_selection(
        {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "sampling_priority_score": recommendation.priority_score,
                    "replay_queue_tags": recommendation.queue_tags,
                    "replay_action": recommendation.replay_action,
                    "deploy_recommendation": "require_review",
                    "pricing_recommendation": "publish_discounted",
                    "datapack_recommendation": "review",
                }
            ]
        }
    )
    assert queue["summary"]["num_entries"] == 1
    assert queue["entries"][0]["episode_id"] == "ep_001"


def test_recommend_sampling_accepts_semantic_runtime_support():
    policy = load_regal_promotion_policy("configs/regality/promotion_default.yaml")
    recommendation = recommend_sampling(
        objective_profile_coverage_gap=0.1,
        constraint_violation_count=0,
        uncertainty=0.1,
        datapack_value=0.4,
        signal_yield_score=0.0,
        regal_support_score=0.1,
        deploy_recommendation="allow_shadow",
        pricing_recommendation="publish",
        datapack_recommendation="keep",
        promotion_policy=policy,
        replay_policy_error=0.05,
        provenance_quality=0.8,
        semantic_runtime_route_score=0.9,
        semantic_runtime_authority_confidence=0.8,
        semantic_runtime_counterfactual_value=0.7,
        semantic_runtime_predicted_regret=0.4,
        semantic_runtime_authority_switch_recommended=True,
    )

    assert recommendation.learned_route_score == 0.9
    assert recommendation.learned_authority_confidence == 0.8
    assert recommendation.learned_counterfactual_value == 0.7
    assert recommendation.learned_predicted_regret == 0.4
    assert "runtime_score_candidate" in recommendation.queue_tags
    assert "runtime_counterfactual_candidate" in recommendation.queue_tags
    assert "runtime_regret_review" in recommendation.queue_tags
    assert "authority_switch_review" in recommendation.queue_tags
