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
