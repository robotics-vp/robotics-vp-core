from src.economics.inferential_training_gate import InferentialTrainingCandidate, InferentialTrainingGate
from src.regality.promotion_policy import load_regal_promotion_policy


def _candidate(**overrides):
    payload = {
        "run_id": "run_a",
        "episode_id": "ep_001",
        "objective_profile_id": "balanced_contract",
        "source_domain": "synthetic",
        "expected_value_gain": 1.5,
        "compute_cost": 0.2,
        "risk_cost": 0.1,
        "uncertainty": 0.2,
        "ood_score": 0.1,
        "data_quality": 0.8,
        "provenance_quality": 0.8,
        "pricing_summary": {"confidence": 0.8},
        "regal_statuses": {
            "objective_integrity_regal": "pass",
            "reward_safety_regal": "pass",
            "pricing_truth_regal": "pass",
        },
        "regal_scores": {"overall": 0.8},
        "replay_policy_uncertainty": 0.2,
        "learned_data_value": 0.5,
        "expected_adaptation_benefit": 0.3,
    }
    payload.update(overrides)
    return InferentialTrainingCandidate(**payload)


def test_inferential_training_gate_adapt_collect_review():
    gate = InferentialTrainingGate(promotion_policy=load_regal_promotion_policy("configs/regality/promotion_default.yaml"))
    adapt = gate.evaluate(_candidate())
    assert adapt.decision == "adapt_now"
    assert adapt.recommended_training_mode == "offline_td3_bc_shadow"

    collect = gate.evaluate(_candidate(uncertainty=0.8, ood_score=0.7))
    assert collect.decision == "collect_more_data"

    review = gate.evaluate(_candidate(regal_statuses={"objective_integrity_regal": "fail", "reward_safety_regal": "pass", "pricing_truth_regal": "pass"}))
    assert review.decision == "require_review"
