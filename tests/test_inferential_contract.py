from src.economics.inferential_contract import (
    build_inferential_admission_contract,
    build_inferential_execution_work_order,
    build_inferential_learnability_contract,
)
from src.economics.inferential_training_gate import (
    InferentialTrainingCandidate,
    InferentialTrainingDecision,
)
from src.evidence.preconditions import build_execution_preconditions


def test_inferential_learnability_contract_classification():
    missing = build_inferential_learnability_contract(
        subject_id="ep_missing",
        subject_kind="replay_episode",
        datapack_id="dp_missing",
    )
    assert missing.learnability_class == "missing"

    portable = build_inferential_learnability_contract(
        subject_id="ep_portable",
        subject_kind="replay_episode",
        datapack_id="dp_portable",
        epiplexity_delta=0.3,
        epiplexity_confidence=0.8,
        overlay_joined=True,
    )
    assert portable.learnability_class == "portable_receipt_backed"

    benchmark = build_inferential_learnability_contract(
        subject_id="ep_benchmark",
        subject_kind="replay_episode",
        datapack_id="dp_benchmark",
        frontier_gain=0.2,
        epiplexity_delta=0.4,
        epiplexity_confidence=0.9,
        overlay_joined=True,
        benchmark_eligible=True,
        semantic_grounding_non_heuristic=True,
        promotion_trace_complete=True,
    )
    assert benchmark.learnability_class == "benchmark_receipt_backed"
    assert benchmark.receipt_backed is True


def test_inferential_execution_work_order_carries_contract():
    learnability_contract = build_inferential_learnability_contract(
        subject_id="ep_001",
        subject_kind="replay_episode",
        datapack_id="dp_001",
        frontier_gain=0.1,
        epiplexity_delta=0.25,
        epiplexity_confidence=0.75,
        overlay_joined=True,
    )
    decision = InferentialTrainingDecision(
        decision="adapt_now",
        expected_gain=1.2,
        expected_cost=0.2,
        expected_risk=0.1,
        net_benefit=0.9,
        allowed_budget=0.9,
        recommended_training_mode="offline_td3_bc_shadow",
        reasons=["adaptation_budget_admitted"],
        artifact_summary={
            "inferential_reward": {"expected_gain": 1.2},
            "inferential_learnability_contract": learnability_contract.to_dict(),
        },
    )
    readiness = build_execution_preconditions(
        subject_id="ep_001",
        subject_kind="replay_episode",
        signal_values={"benchmark_eligible": True},
        soft_boolean_signals={"benchmark_eligible": True},
    )

    work_order = build_inferential_execution_work_order(
        decision=decision,
        readiness=readiness,
        run_id="run_001",
        episode_id="ep_001",
        objective_profile_id="balanced_contract",
        source_domain="synthetic",
        datapack_id="dp_001",
    )

    payload = work_order.to_dict()
    assert payload["order_type"] == "adaptation_training"
    assert payload["metadata"]["contract_kind"] == "inferential_execution_work_order_v1"
    assert payload["metadata"]["inferential_learnability_contract"]["learnability_class"] == "portable_receipt_backed"
    assert payload["artifact_refs"]["datapack_id"] == "dp_001"


def test_inferential_admission_contract_summarizes_decisions():
    learnability_contract = build_inferential_learnability_contract(
        subject_id="ep_001",
        subject_kind="replay_episode",
        datapack_id="dp_001",
        frontier_gain=0.2,
        epiplexity_delta=0.35,
        epiplexity_confidence=0.8,
        overlay_joined=True,
    )
    candidate = InferentialTrainingCandidate(
        run_id="run_001",
        episode_id="ep_001",
        objective_profile_id="balanced_contract",
        source_domain="synthetic",
        expected_value_gain=1.0,
        compute_cost=0.2,
        risk_cost=0.1,
        uncertainty=0.2,
        ood_score=0.1,
        data_quality=0.9,
        provenance_quality=0.8,
        pricing_summary={"confidence": 0.85},
        regal_statuses={"objective_integrity_regal": "pass"},
        regal_scores={"overall": 0.9},
        replay_policy_uncertainty=0.2,
        learned_data_value=0.4,
        expected_adaptation_benefit=0.3,
        metadata={
            "datapack_id": "dp_001",
            "inferential_learnability_contract": learnability_contract.to_dict(),
        },
    )
    decision = InferentialTrainingDecision(
        decision="adapt_now",
        expected_gain=1.1,
        expected_cost=0.3,
        expected_risk=0.1,
        net_benefit=0.8,
        allowed_budget=0.8,
        recommended_training_mode="offline_td3_bc_shadow",
        reasons=["adaptation_budget_admitted"],
        artifact_summary={
            "inferential_reward": {"expected_gain": 1.1},
            "inferential_learnability_contract": learnability_contract.to_dict(),
        },
    )
    contract = build_inferential_admission_contract(
        candidates=[candidate],
        decisions=[decision],
        work_orders=[],
    )

    assert contract["contract_kind"] == "inferential_admission_contract_v1"
    assert contract["authority_class"] == "work_order"
    assert contract["summary"]["decision_count"] == 1
    assert contract["summary"]["decision_counts"]["adapt_now"] == 1
    assert contract["summary"]["learnability_summary"]["receipt_backed_count"] == 1
    assert contract["episode_decisions"][0]["learnability_class"] == "portable_receipt_backed"
