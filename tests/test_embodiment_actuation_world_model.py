"""Tests for Phase 3 Embodiment / Actuation WM shadow contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.embodiment.core import EmbodimentInputs, compute_embodiment
from src.embodiment.registry import CapabilityProfile, EmbodimentRegistryEntry
from src.runtime.action_adapter_v2 import ActionAdapterV2
from src.runtime.observation_adapter_v2 import ObservationAdapterV2
from src.world_model.embodiment_actuation import (
    EMBODIMENT_ACTUATION_WORLD_STATE_VERSION,
    ActionSpaceValidationReceipt,
    ActionProposalReceipt,
    CalibrationTargetReceipt,
    CapabilityProfileReceipt,
    ContactAffordanceReceipt,
    EmbodimentCompilationReceipt,
    EmbodimentCostReceipt,
    EmbodimentDriftReceipt,
    InverseRetargetReceipt,
    LocalDynamicsReceipt,
    ObservationInterfaceReceipt,
    SafetyEnvelopeReceipt,
    SimEmbodimentTransferReceipt,
    build_economic_embodiment_receipt_bundle,
    build_perception_embodiment_feedback,
    build_runtime_adapter_validation_context,
    build_sim_embodiment_transfer_context,
    compile_embodiment_actuation_with_receipts,
    compile_embodiment_actuation_world_state,
    holosoma_contract,
    resolve_embodiment_seam,
    unitree_g1_contract,
)
from src.world_model.sim_synth_physics.adapters.embodiment_inputs import (
    build_embodiment_input_context,
)


def _scene_tracks_payload() -> dict[str, np.ndarray]:
    poses_r = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], 4, axis=0)
    poses_r = np.repeat(poses_r, 2, axis=1)
    poses_t = np.zeros((4, 2, 3), dtype=np.float32)
    poses_t[:, 1, 0] = 0.04
    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(["hand", "drawer"], dtype="U32"),
        "scene_tracks_v1/entity_types": np.array([0, 0], dtype=np.int32),
        "scene_tracks_v1/class_ids": np.array([0, 1], dtype=np.int32),
        "scene_tracks_v1/class_names": np.array(["hand", "drawer"], dtype="U32"),
        "scene_tracks_v1/poses_R": poses_r,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": np.full((4, 2), 0.05, dtype=np.float32),
        "scene_tracks_v1/visibility": np.ones((4, 2), dtype=np.float32),
        "scene_tracks_v1/occlusion": np.zeros((4, 2), dtype=np.float32),
        "scene_tracks_v1/ir_loss": np.zeros((4, 2), dtype=np.float32),
        "scene_tracks_v1/converged": np.ones((4, 2), dtype=bool),
    }


def _embodiment_result():
    return compute_embodiment(
        EmbodimentInputs(
            scene_tracks=_scene_tracks_payload(),
            action_stream=[{"dx": 0.1}, {"dx": 0.0}],
            task_constraints={"forbidden_contacts": []},
            failure_events={},
        )
    )


def _registry_entry(with_safety_refs: bool = False) -> EmbodimentRegistryEntry:
    safety = {}
    if with_safety_refs:
        safety = {
            "watchdog_ref": "watchdog://unitree/g1/local-smoke",
            "latency_profile_ref": "latency://unitree/g1/local-smoke",
            "margin_fraction": 0.72,
        }
    profile = CapabilityProfile(
        profile_id="g1_profile",
        robot_family="unitree_g1",
        sensor_modalities=["rgb", "depth", "proprio"],
        action_spaces=["g1_29dof_joint_position"],
        workspace_bounds={"x": {"min": -1.0, "max": 1.0}},
        skill_capabilities={"reach": 0.8, "grasp": 0.6},
        timing={"control_hz": 50.0},
        safety_envelopes=safety,
    )
    return EmbodimentRegistryEntry(
        embodiment_id="unitree_g1_shadow",
        robot_id="g1_local",
        robot_family="unitree_g1",
        capability_profile=profile,
        observation_schema_id="obs_g1_v1",
        action_schema_id="act_g1_v1",
        translator_refs={"retarget": "retarget://unitree/g1/shadow"},
    )


def _action_adapter() -> ActionAdapterV2:
    return ActionAdapterV2(
        schema_id="act_g1_v1",
        channel_order=["joint_0", "joint_1", "joint_2"],
        control_hz=50.0,
        latency_ms=12.0,
        translator_ref="retarget://unitree/g1/shadow",
        embodiment_id="unitree_g1_shadow",
        bounds={"joint_0": {"min": -1.0, "max": 1.0}},
    )


def _observation_adapter() -> ObservationAdapterV2:
    return ObservationAdapterV2(
        schema_id="obs_g1_v1",
        proprio_fields=["q0", "q1", "dq0"],
        sensor_refs=["camera://head/rgb", "proprio://g1"],
        sample_hz=50.0,
        latency_ms=8.0,
        translator_ref="obs://unitree/g1/shadow",
        embodiment_id="unitree_g1_shadow",
    )


def _perception_shadow_surface() -> SimpleNamespace:
    return SimpleNamespace(
        surface_id="embodiment_shadow_surface_test",
        actionable_object_count=2,
        obstructed_object_count=0,
        scene_contact_feasibility=0.74,
        scene_affordance_coverage=0.68,
        scene_obstruction_severity=0.12,
        body_object_engagement_summary={"g1_default_body": 0.61},
        evidence_quality_for_embodiment={
            "fusion_confidence": 0.72,
            "fusion_disagreement": 0.18,
        },
    )


def _compile(with_safety_refs: bool = False):
    return compile_embodiment_actuation_with_receipts(
        episode_id="ep_phase3",
        frame_index=3,
        embodiment_registry_entry=_registry_entry(with_safety_refs=with_safety_refs),
        advisory_embodiment_result=_embodiment_result(),
        action_adapter=_action_adapter(),
        observation_adapter=_observation_adapter(),
        perception_shadow_surface=_perception_shadow_surface(),
        provider_contracts=[
            unitree_g1_contract(policy_ref="/tmp/g1_policy.pt"),
            holosoma_contract(policy_ref="/tmp/g1_policy.onnx"),
        ],
        joint_state={
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "positions": [0.1, 0.0, -0.1],
            "velocities": [0.0, 0.0, 0.0],
            "timestamp_s": 1.2,
        },
        source_refs={"embodiment_profile_ref": "artifact://embodiment/profile"},
    )


def test_phase3_compiler_emits_canonical_state_and_receipts() -> None:
    result = _compile()
    state = result.state

    assert state.version == EMBODIMENT_ACTUATION_WORLD_STATE_VERSION
    assert state.authority_level == "none"
    assert state.compilation_mode == "shadow_advisory"
    assert state.capability.robot_family == "unitree_g1"
    assert state.action_space.dimension == 3
    assert state.observation_interface.schema_id == "obs_g1_v1"
    assert state.contact_state.contact_event_count > 0
    assert state.contact_affordance_graph.actionable_object_count == 2
    assert state.action_proposal_bundle.authority_level == "none"
    assert state.safety_envelope.status == "external_blocked"
    assert "safety_watchdog_profile" in state.safety_envelope.missing_evidence
    assert state.receipt_manifest["receipt_count"] == len(result.receipts) - 1


def test_phase3_receipt_family_is_complete_and_json_safe() -> None:
    result = _compile()
    receipt_types = {type(receipt) for receipt in result.receipts}

    expected = {
        EmbodimentCompilationReceipt,
        CapabilityProfileReceipt,
        ActionSpaceValidationReceipt,
        ObservationInterfaceReceipt,
        ContactAffordanceReceipt,
        LocalDynamicsReceipt,
        InverseRetargetReceipt,
        ActionProposalReceipt,
        SafetyEnvelopeReceipt,
        EmbodimentDriftReceipt,
        CalibrationTargetReceipt,
        EmbodimentCostReceipt,
        SimEmbodimentTransferReceipt,
    }
    assert expected.issubset(receipt_types)
    for receipt in result.receipts:
        payload = receipt.to_dict()
        assert payload["authority_level"] == "none"
        assert payload["receipt_id"]


def test_phase3_compiler_is_permissive_but_explicit_when_inputs_are_missing() -> None:
    state = compile_embodiment_actuation_world_state(episode_id="missing_inputs")

    assert state.authority_level == "none"
    assert state.capability.truth_class == "unavailable"
    assert "capability_profile" in state.capability.missing_fields
    assert state.action_space.validation_status == "unavailable"
    assert state.observation_interface.validation_status == "unavailable"
    assert state.safety_envelope.status == "external_blocked"
    assert state.calibration_targets.missing_evidence


def test_phase3_with_safety_refs_allows_shadow_runtime_validation() -> None:
    result = _compile(with_safety_refs=True)
    state = result.state
    validation = build_runtime_adapter_validation_context(state)

    assert state.safety_envelope.status == "available"
    assert validation.runtime_validation_status == "shadow_validated"
    assert validation.authority_level == "none"


def test_phase3_downstream_consumers_are_shadow_only_and_sim_consumable() -> None:
    result = _compile()
    state = result.state

    sim_context = build_sim_embodiment_transfer_context(state)
    normalized = build_embodiment_input_context(sim_context.to_dict())
    perception_feedback = build_perception_embodiment_feedback(state)
    runtime_validation = build_runtime_adapter_validation_context(state)
    economic_bundle = build_economic_embodiment_receipt_bundle(state, result.receipts)

    assert normalized["active_embodiment_count"] == 1
    assert normalized["action_feasibility_score"] > 0.0
    assert normalized["retargeting_readiness_score"] > 0.0
    assert normalized["embodiment_authority_level"] == "none"
    assert perception_feedback.authority_level == "none"
    assert runtime_validation.runtime_validation_status == "shadow_blocked"
    assert economic_bundle.allocative_authority == "none"
    assert len(economic_bundle.receipt_refs) == len(result.receipts)


def test_provider_contracts_keep_external_evidence_honest() -> None:
    unitree = unitree_g1_contract(policy_ref="/tmp/g1_policy.pt")
    holosoma = holosoma_contract(policy_ref="/tmp/g1_policy.onnx")

    assert unitree.resolved_status() == "external_blocked"
    assert "actuator_latency_profile" in unitree.missing_components
    assert "safety_watchdog_profile" in unitree.missing_components
    assert holosoma.truth_class == "local_deploy_smoke"
    assert holosoma.resolved_status() == "external_blocked"
    assert "native_holosoma_runtime_execution" in holosoma.missing_components


def test_embodiment_seam_promotion_requires_provider_and_benchmark() -> None:
    blocked = resolve_embodiment_seam(
        "action_proposal", posture="auto", benchmark_signals={"score": 0.9}
    )
    promoted = resolve_embodiment_seam(
        "action_proposal",
        posture="auto",
        benchmark_signals={"score": 0.9, "benchmark_ready": True},
        provider_available=True,
    )

    assert blocked.can_execute is False
    assert "provider_unavailable" in blocked.blocked_reasons
    assert promoted.can_execute is True
    assert promoted.promotion_stage == "promoted"
