from __future__ import annotations

import pytest

from src.world_model.sim_synth_physics.agenda import SimulationAgenda
from src.world_model.sim_synth_physics.backend_adapters import BackendAdapterDescriptor
from src.world_model.sim_synth_physics.calibration import (
    build_physics_adaptation_receipt,
    build_physics_calibration_receipt,
)
from src.world_model.sim_synth_physics.inferential import (
    agenda_score_with_inferential_prior,
    benchmark_provenance_quality,
    build_branch_plan_inferential_contract,
    build_simulation_job_inferential_contract,
)
from src.world_model.sim_synth_physics.physics_contracts import PhysicsExecutionContract
from src.world_model.sim_synth_physics.randomization import compile_physics_adaptation_policy
from src.world_model.sim_synth_physics.state import (
    PhysicsContextState,
    SimSynthPhysicsWorldState,
)


def test_inferential_job_contract_reacts_to_priority_and_provenance() -> None:
    low_priority = build_simulation_job_inferential_contract(
        job_id="job_low",
        coverage_gap_score=0.7,
        economic_priority=0.2,
        trust_priority=0.9,
        readiness=0.5,
        ranking_policy="receipt_gated_with_inferential_contracts",
        wm_validation_pressure=0.3,
        benchmark_signals={},
    )
    high_priority = build_simulation_job_inferential_contract(
        job_id="job_high",
        coverage_gap_score=0.7,
        economic_priority=0.8,
        trust_priority=0.2,
        readiness=0.5,
        ranking_policy="receipt_gated_with_inferential_contracts",
        wm_validation_pressure=0.3,
        benchmark_signals={
            "benchmark_eligible": True,
            "semantic_grounding_non_heuristic": True,
            "scene_tracks_backend_real": True,
            "vision_backbone_real": True,
        },
    )

    assert high_priority.frontier_gain == pytest.approx(0.7)
    assert high_priority.transfer_score > low_priority.transfer_score
    assert high_priority.epiplexity_delta > low_priority.epiplexity_delta
    assert high_priority.epiplexity_confidence > low_priority.epiplexity_confidence
    assert high_priority.provenance_quality == pytest.approx(
        benchmark_provenance_quality(
            {
                "benchmark_eligible": True,
                "semantic_grounding_non_heuristic": True,
                "scene_tracks_backend_real": True,
                "vision_backbone_real": True,
            }
        )
    )
    assert agenda_score_with_inferential_prior(
        base_ranking_score=0.4,
        contract=high_priority,
    ) > 0.4


def test_branch_inferential_contract_confidence_reacts_to_backend_provenance() -> None:
    upstream_contract = build_simulation_job_inferential_contract(
        job_id="job_upstream",
        coverage_gap_score=0.6,
        economic_priority=0.7,
        trust_priority=0.4,
        readiness=0.6,
        ranking_policy="receipt_gated_with_inferential_contracts",
        wm_validation_pressure=0.25,
        benchmark_signals={"benchmark_eligible": True},
    )

    low_provenance = build_branch_plan_inferential_contract(
        plan_id="plan_low",
        job_id="job_upstream",
        expected_yield_score=0.5,
        job_contract=upstream_contract,
        benchmark_signals={},
    )
    high_provenance = build_branch_plan_inferential_contract(
        plan_id="plan_high",
        job_id="job_upstream",
        expected_yield_score=0.5,
        job_contract=upstream_contract,
        benchmark_signals={
            "benchmark_eligible": True,
            "semantic_grounding_non_heuristic": True,
            "scene_tracks_backend_real": True,
            "vision_backbone_real": True,
        },
    )

    assert high_provenance.epiplexity_confidence > low_provenance.epiplexity_confidence
    assert high_provenance.trust_score > low_provenance.trust_score
    assert high_provenance.metadata["upstream_job_learnability_class"] == upstream_contract.learnability_class


def test_humanoid_adaptation_policy_tracks_randomization_axes() -> None:
    physics_context = PhysicsContextState(
        context_id="physics_ctx_1",
        backend="isaac",
        fidelity_tier="high_fidelity",
        timestep_ms=4.0,
        domain_randomization_regime="benchmark_focus",
        calibration_profile="receipt_backed",
        selection_policy="receipt_gated_with_inferential_contracts",
    )
    adapter = BackendAdapterDescriptor(
        backend="isaac",
        adapter_name="backend_isaac_unitree_target_v1",
        adapter_status="shadow_ready",
        supports_execution=False,
        simulator_family="isaac",
        target_hardware_class="unitree_g1_r1_class",
        metadata={"target_runtime_stack": ["isaacsim", "unitree_sdk2"]},
    )

    policy = compile_physics_adaptation_policy(
        physics_context,
        adapter=adapter,
        benchmark_signals={"benchmark_eligible": True},
        embodiment_context={
            "active_embodiments": ["unitree_g1"],
            "control_constraints": {
                "latency_budget_ms": 8.0,
                "contact_risk_score": 0.85,
            },
        },
    )

    assert policy.target_hardware_class == "unitree_g1_r1_class"
    assert policy.domain_randomization_profile == "humanoid_contact_latency_and_sensor_calibration"
    assert policy.system_identification_profile == "whole_body_latency_contact_and_actuator_id"
    assert "battery_voltage_sag" in policy.randomization_axes
    assert "foot_contact_threshold" in policy.randomization_axes
    assert "actuator_delay_profile" in policy.calibration_targets
    assert "whole_body_joint_map" in policy.calibration_targets


def test_calibration_and_adaptation_receipts_react_to_route_status_and_runtime_evidence() -> None:
    physics_context = PhysicsContextState(
        context_id="physics_ctx_2",
        backend="isaac",
        fidelity_tier="high_fidelity",
        timestep_ms=4.0,
        domain_randomization_regime="calibration_focus",
        calibration_profile="receipt_backed",
        selection_policy="receipt_gated_with_inferential_contracts",
        metadata={
            "benchmark_signals": {"benchmark_eligible": True},
            "backend_helper_status": {"promotion_stage": "promoted"},
        },
    )
    adapter = BackendAdapterDescriptor(
        backend="isaac",
        adapter_name="backend_isaac_unitree_target_v1",
        adapter_status="shadow_ready",
        supports_execution=False,
        simulator_family="isaac",
        target_hardware_class="unitree_g1_r1_class",
        metadata={"target_runtime_stack": ["isaacsim", "unitree_sdk2"]},
    )
    adaptation_policy = compile_physics_adaptation_policy(
        physics_context,
        adapter=adapter,
        benchmark_signals={"benchmark_eligible": True},
        embodiment_context={
            "active_embodiments": ["unitree_g1"],
            "control_constraints": {
                "latency_budget_ms": 6.0,
                "contact_risk_score": 0.75,
            },
        },
    )
    world_state = SimSynthPhysicsWorldState(
        state_id="world_state_1",
        simulation_agenda=SimulationAgenda(
            agenda_id="agenda_1",
            coverage_window_ref="coverage_window_1",
            jobs=[],
            ranking_policy="receipt_gated_with_inferential_contracts",
        ),
        physics_context=physics_context,
        physics_adaptation_policy=adaptation_policy,
    )
    ready_contract = PhysicsExecutionContract(
        contract_id="contract_ready",
        requested_backend="isaac",
        resolved_backend="isaac",
        fidelity_tier="high_fidelity",
        domain_randomization_regime="calibration_focus",
        calibration_profile="receipt_backed",
        backend_selection_policy="receipt_gated_with_inferential_contracts",
        adapter_name="backend_isaac_unitree_target_v1",
        simulator_family="isaac",
        target_hardware_class="unitree_g1_r1_class",
        adaptation_policy_id=adaptation_policy.policy_id,
        route_status="ready",
    )
    fallback_contract = PhysicsExecutionContract(
        contract_id="contract_fallback",
        requested_backend="isaac",
        resolved_backend="pybullet",
        fidelity_tier="branch_balanced",
        domain_randomization_regime="calibration_focus",
        calibration_profile="receipt_backed",
        backend_selection_policy="receipt_gated_with_inferential_contracts",
        adapter_name="backend_pybullet_v2",
        simulator_family="pybullet",
        target_hardware_class="unitree_g1_r1_class",
        adaptation_policy_id=adaptation_policy.policy_id,
        route_status="fallback",
        fallback_reason="isaac runtime unavailable",
    )

    adaptation_ready = build_physics_adaptation_receipt(
        world_state,
        ready_contract,
        runtime_evidence={
            "runtime_concrete_completed": True,
            "shadow_execution_status": "shadow_with_data_harvest",
            "materialized_render_provider_count": 2,
        },
    )
    adaptation_fallback = build_physics_adaptation_receipt(
        world_state,
        fallback_contract,
        runtime_evidence={},
    )
    calibration_ready = build_physics_calibration_receipt(
        world_state,
        ready_contract,
        adaptation_receipt=adaptation_ready,
        runtime_evidence={
            "runtime_concrete_completed": True,
            "shadow_execution_status": "shadow_with_data_harvest",
            "materialized_render_provider_count": 2,
            "render_artifact_count": 4,
            "runtime_output_artifact_count": 3,
        },
    )
    calibration_fallback = build_physics_calibration_receipt(
        world_state,
        fallback_contract,
        adaptation_receipt=adaptation_fallback,
        runtime_evidence={"shadow_missing_asset_count": 2},
    )

    assert adaptation_ready.readiness_score > adaptation_fallback.readiness_score
    assert calibration_ready.quality_score > calibration_fallback.quality_score
    assert adaptation_ready.domain_randomization_profile == adaptation_policy.domain_randomization_profile
    assert adaptation_ready.metadata["runtime_evidence"]["runtime_concrete_completed"] is True
    assert calibration_ready.metadata["benchmark_gate_ready"] is True
    assert calibration_fallback.metadata["resolved_backend"] == "pybullet"
