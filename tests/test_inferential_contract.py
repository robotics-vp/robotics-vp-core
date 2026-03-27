from src.economics.inferential_contract import (
    build_inferential_execution_work_order,
    build_inferential_learnability_contract,
)
from src.economics.inferential_training_gate import InferentialTrainingDecision
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
