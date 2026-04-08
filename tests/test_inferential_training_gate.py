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
    assert "inferential_reward" in adapt.artifact_summary
    assert adapt.artifact_summary["inferential_reward"]["expected_gain"] >= adapt.expected_gain - 1e-9
    assert adapt.receipt_kind == "inferential_training_decision_v1"
    assert adapt.authority_class == "work_order"
    assert adapt.decision_scope == "training_admission_and_data_collection"
    assert adapt.reward_math_mutation is False

    collect = gate.evaluate(_candidate(uncertainty=0.8, ood_score=0.7))
    assert collect.decision == "collect_more_data"

    review = gate.evaluate(_candidate(regal_statuses={"objective_integrity_regal": "fail", "reward_safety_regal": "pass", "pricing_truth_regal": "pass"}))
    assert review.decision == "require_review"


def test_inferential_training_gate_promotes_signal_yield_support():
    gate = InferentialTrainingGate(
        promotion_policy=load_regal_promotion_policy("configs/regality/promotion_default.yaml"),
        min_net_benefit=0.05,
    )
    candidate = _candidate(
        expected_value_gain=0.0,
        expected_adaptation_benefit=0.0,
        learned_data_value=0.0,
        compute_cost=0.08,
        risk_cost=0.0,
        uncertainty=0.0,
        ood_score=0.0,
        data_quality=1.0,
        provenance_quality=1.0,
        frontier_gain=0.2,
        epiplexity_delta=0.5,
        epiplexity_confidence=0.8,
        transfer_score=0.4,
    )
    decision = gate.evaluate(candidate)

    assert decision.decision == "adapt_now"
    inferential = decision.artifact_summary["inferential_reward"]
    assert inferential["signal_yield"]["epiplexity_term"] > 0.0
    assert inferential["signal_yield"]["score"] > 0.0
    assert decision.to_dict()["authority_class"] == "work_order"
