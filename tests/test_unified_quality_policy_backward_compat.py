from src.objectives.profile import ObjectiveProfile
from src.policies.unified_quality import UnifiedQualityPolicy


def test_unified_quality_default_matches_multiplicative_formula():
    policy = UnifiedQualityPolicy()
    weights = policy.compute(
        mhn_plausibility=0.8,
        mhn_difficulty=0.2,
        scene_ir_convergence=0.9,
        scene_ir_visibility=0.9,
        process_reward_conf=0.7,
        process_reward_conf_p10=0.7,
        process_reward_delta=0.2,
        process_reward_disagreement=0.1,
        map_first_quality=0.9,
    )
    expected = (
        weights.w_mhn
        * weights.w_scene_ir
        * weights.w_process_reward
        * weights.w_map_first
    )
    assert abs(weights.w_combined - expected) < 1e-6


def test_unified_quality_sampler_profile_scalarizes_at_boundary():
    policy = UnifiedQualityPolicy()
    profile = ObjectiveProfile(
        scalarizer="weighted_sum",
        weights={"throughput": 1.0, "error": 1.0, "safety": 1.0, "energy": 1.0},
        maximize={"throughput": True, "error": False, "safety": True, "energy": False},
    )
    weights = policy.compute(
        mhn_plausibility=0.8,
        scene_ir_convergence=0.8,
        scene_ir_visibility=0.8,
        process_reward_conf=0.8,
        process_reward_conf_p10=0.8,
        process_reward_delta=0.2,
        process_reward_disagreement=0.1,
        map_first_quality=0.8,
        objective_profile=profile,
    )
    assert weights.objective_tensor_slice is not None
    assert weights.w_combined >= 0.0


def test_unified_quality_execution_preconditions_can_block_eligibility():
    policy = UnifiedQualityPolicy()
    weights = policy.compute(
        mhn_plausibility=0.8,
        scene_ir_convergence=0.8,
        scene_ir_visibility=0.8,
        process_reward_conf=0.8,
        process_reward_conf_p10=0.8,
        process_reward_delta=0.2,
        process_reward_disagreement=0.1,
        map_first_quality=0.8,
        execution_preconditions={
            "ready": False,
            "blocking_preconditions": ["artifact::runtime_packet_ref"],
            "readiness_score": 0.5,
        },
    )
    assert weights.is_eligible is False
    assert weights.eligibility_reason == "execution_preconditions=artifact::runtime_packet_ref"
