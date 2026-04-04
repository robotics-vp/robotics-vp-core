"""Tests for the embodiment-facing shadow consumer (Phase 2).

Validates that:
- Perception state is consumable by embodiment-relevant logic
- Per-object action relevance is typed and correct
- Scene-level summaries are computed
- Provider truth and deployment posture feed through
- Receipt emission is complete
- Shadow/advisory posture is maintained (no control authority)
- Reduced-quality but honest behavior when inputs are degraded
"""

from __future__ import annotations

import numpy as np

from src.evidence.belief_state import BeliefState
from src.world_model.perception_grounding import (
    EmbodimentShadowConsumptionReceipt,
    EmbodimentShadowSurface,
    ObjectActionRelevance,
    PerceptionCompilationResult,
    compile_perception_grounding_with_receipts,
    compile_perception_grounding_world_state,
    consume_perception_for_embodiment,
)
from src.world_model.perception_grounding.receipts import (
    DeploymentResourceReceipt,
    GroundingCalibrationReceipt,
    InferenceHeadroomReceipt,
    PerceptionContributionReceipt,
    ProviderAvailabilityReceipt,
    TemporalGroundingReceipt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scene_tracks_payload() -> dict[str, np.ndarray]:
    poses_r = np.stack(
        [np.stack([np.eye(3), np.eye(3)]), np.stack([np.eye(3), np.eye(3)])]
    ).astype(np.float32)
    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(
            ["drawer_track", "vase_track"], dtype="U32"
        ),
        "scene_tracks_v1/entity_types": np.array([0, 0], dtype=np.int32),
        "scene_tracks_v1/class_ids": np.array([0, 1], dtype=np.int32),
        "scene_tracks_v1/class_names": np.array(["drawer", "vase"], dtype="U32"),
        "scene_tracks_v1/poses_R": poses_r,
        "scene_tracks_v1/poses_t": np.array(
            [
                [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
                [[0.01, 0.0, 0.0], [0.31, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "scene_tracks_v1/scales": np.ones((2, 2), dtype=np.float32),
        "scene_tracks_v1/visibility": np.array(
            [[1.0, 0.8], [1.0, 0.9]], dtype=np.float32
        ),
        "scene_tracks_v1/occlusion": np.array(
            [[0.0, 0.2], [0.0, 0.1]], dtype=np.float32
        ),
        "scene_tracks_v1/ir_loss": np.zeros((2, 2), dtype=np.float32),
        "scene_tracks_v1/converged": np.ones((2, 2), dtype=bool),
        "scene_tracks_v1/summary_json": np.array(
            [
                '{"quality_score":0.92,"topology":{"temporal_stability":0.83,"grounded_track_object_count":2,"track_count":2}}'
            ],
            dtype="U256",
        ),
    }


def _belief_state() -> BeliefState:
    return BeliefState(
        belief_id="belief_test",
        episode_id="ep_test",
        timestamp="2026-04-04T10:00:00Z",
        semantic_tags=["drawer", "fragile"],
        state_vector={
            "semantic_quality": 0.78,
            "evidence_coverage": 0.81,
            "evidence_disagreement_mean": 0.18,
            "teacher_alignment": 0.74,
        },
        uncertainty={"semantic": 0.18, "coverage_gap": 0.19},
        evidence_refs=[],
        artifact_refs={},
        provenance={},
        metadata={},
    )


def _compiled_state():
    return compile_perception_grounding_world_state(
        episode_id="ep_embod_test",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )


# ---------------------------------------------------------------------------
# Embodiment shadow consumer tests
# ---------------------------------------------------------------------------


def test_consume_perception_for_embodiment_returns_typed_surface() -> None:
    state = _compiled_state()
    surface, receipt = consume_perception_for_embodiment(state)

    assert isinstance(surface, EmbodimentShadowSurface)
    assert isinstance(receipt, EmbodimentShadowConsumptionReceipt)
    assert surface.version == "embodiment_shadow_surface_v1"
    assert receipt.version == "embodiment_shadow_consumption_receipt_v1"


def test_consumer_produces_per_object_action_relevance() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    assert len(surface.object_action_relevances) >= 2
    for oar in surface.object_action_relevances:
        assert isinstance(oar, ObjectActionRelevance)
        assert oar.track_id != ""
        assert 0.0 <= oar.reachability_score <= 1.0
        assert 0.0 <= oar.obstruction_score <= 1.0
        assert 0.0 <= oar.affordance_feasibility <= 1.0
        assert 0.0 <= oar.contact_precondition_met <= 1.0
        assert 0.0 <= oar.misalignment_risk <= 1.0
        assert 0.0 <= oar.perception_confidence <= 1.0


def test_consumer_scene_level_summaries() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    assert 0.0 <= surface.scene_contact_feasibility <= 1.0
    assert 0.0 <= surface.scene_affordance_coverage <= 1.0
    assert 0.0 <= surface.scene_obstruction_severity <= 1.0
    assert surface.actionable_object_count >= 0
    assert surface.obstructed_object_count >= 0


def test_consumer_body_object_engagement() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    assert "g1_default_body" in surface.body_object_engagement_summary
    score = surface.body_object_engagement_summary["g1_default_body"]
    assert 0.0 <= score <= 1.0


def test_consumer_resource_readiness() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    rr = surface.resource_readiness
    assert "deployment_posture" in rr
    assert "compute_available" in rr
    assert "companion_available" in rr


def test_consumer_provider_truth() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    pt = surface.provider_truth_for_embodiment
    assert "providers_available" in pt
    assert "truth_classes" in pt
    assert "stub_only_providers" in pt
    # vision_backbone_stub should be flagged
    assert "vision_backbone_stub" in pt["stub_only_providers"]
    assert pt["all_providers_real"] is False


def test_consumer_evidence_quality() -> None:
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    eq = surface.evidence_quality_for_embodiment
    assert "fusion_confidence" in eq
    assert "fusion_disagreement" in eq
    assert "embodiment_trust_score" in eq
    assert eq["fusion_confidence"] > 0.0
    assert 0.0 <= eq["embodiment_trust_score"] <= 1.0


def test_consumer_shadow_posture() -> None:
    """Shadow consumer MUST maintain advisory posture, no sovereignty."""
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)

    assert surface.consumption_mode == "shadow_advisory"
    assert surface.authority_level == "none"


def test_consumer_receipt_is_complete() -> None:
    state = _compiled_state()
    _, receipt = consume_perception_for_embodiment(state)

    assert receipt.source_state_id == state.state_id
    assert receipt.source_episode_id == state.episode_id
    assert receipt.object_count_consumed >= 2
    assert receipt.evidence_fusion_confidence > 0.0
    assert receipt.deployment_posture in ("unavailable", "available", "degraded")
    assert isinstance(receipt.reduced_quality, bool)


def test_consumer_serialization_roundtrip() -> None:
    state = _compiled_state()
    surface, receipt = consume_perception_for_embodiment(state)

    surface_dict = surface.to_dict()
    receipt_dict = receipt.to_dict()

    assert surface_dict["surface_id"] == surface.surface_id
    assert surface_dict["consumption_mode"] == "shadow_advisory"
    assert surface_dict["authority_level"] == "none"
    assert len(surface_dict["object_action_relevances"]) >= 2

    assert receipt_dict["receipt_id"] == receipt.receipt_id
    assert receipt_dict["object_count_consumed"] == receipt.object_count_consumed


def test_consumer_reduced_quality_with_empty_state() -> None:
    """Consumer should emit reduced-quality but honest output for empty state."""
    from src.world_model.perception_grounding.state import (
        PerceptionGroundingWorldState,
    )

    empty_state = PerceptionGroundingWorldState(
        state_id="empty_test",
        frame_index=0,
        episode_id="ep_empty",
        maturity_stage="schema_only",
    )
    surface, receipt = consume_perception_for_embodiment(empty_state)

    assert surface.consumption_mode == "shadow_advisory"
    assert len(surface.object_action_relevances) == 0
    assert receipt.reduced_quality is True
    assert receipt.reduced_quality_reason != ""
    assert receipt.object_count_consumed == 0


def test_consumer_does_not_assert_embodiment_wm_sovereignty() -> None:
    """Ensure the consumer output has no planner/controller fields."""
    state = _compiled_state()
    surface, _ = consume_perception_for_embodiment(state)
    d = surface.to_dict()

    # These fields should NOT exist in perception-side shadow output
    for forbidden_key in [
        "action_proposal",
        "control_policy",
        "inverse_dynamics",
        "skill_plan",
        "motor_command",
        "joint_trajectory",
    ]:
        assert forbidden_key not in d, f"Found forbidden key: {forbidden_key}"


# ---------------------------------------------------------------------------
# Full receipt family tests
# ---------------------------------------------------------------------------


def test_compile_with_receipts_emits_full_receipt_family() -> None:
    """compile_with_receipts now returns all 8 receipt types."""
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_full_receipts",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    assert isinstance(result, PerceptionCompilationResult)
    assert result.state.maturity_stage == "shadow_runtime"

    # Collect receipt types
    receipt_types = {type(r).__name__ for r in result.receipts}

    # Must have all these receipt families
    assert "EvidenceFusionReceipt" in receipt_types
    assert "ProviderAvailabilityReceipt" in receipt_types
    assert "GroundingCalibrationReceipt" in receipt_types
    assert "InferenceHeadroomReceipt" in receipt_types
    assert "DeploymentResourceReceipt" in receipt_types
    assert "TemporalGroundingReceipt" in receipt_types
    assert "PerceptionContributionReceipt" in receipt_types


def test_provider_availability_receipts_cover_all_providers() -> None:
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_avail",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    avail_receipts = [
        r for r in result.receipts if isinstance(r, ProviderAvailabilityReceipt)
    ]
    provider_ids = {r.provider_id for r in avail_receipts}

    assert "scene_tracks" in provider_ids
    assert "vision_backbone_stub" in provider_ids

    # stub provider should have stub_only install status
    stub_r = next(r for r in avail_receipts if r.provider_id == "vision_backbone_stub")
    assert stub_r.provider_truth_class == "stub_smoke_only"
    assert stub_r.install_status == "stub_only"


def test_grounding_calibration_receipt_has_quality_metrics() -> None:
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_cal",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    cal_receipts = [
        r for r in result.receipts if isinstance(r, GroundingCalibrationReceipt)
    ]
    assert len(cal_receipts) == 1
    cal = cal_receipts[0]

    assert 0.0 <= cal.grounding_accuracy <= 1.0
    assert 0.0 <= cal.spatial_accuracy <= 1.0
    assert 0.0 <= cal.temporal_consistency <= 1.0
    assert 0.0 <= cal.provider_agreement <= 1.0
    assert 0.0 <= cal.cross_provider_disagreement <= 1.0


def test_deployment_resource_receipt_identifies_bottlenecks() -> None:
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_deploy",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    dr_receipts = [
        r for r in result.receipts if isinstance(r, DeploymentResourceReceipt)
    ]
    assert len(dr_receipts) == 1
    dr = dr_receipts[0]

    # Default deployment surface has no compute, so bottlenecks expected
    assert len(dr.bottleneck_ids) > 0
    assert "deployment_posture_unavailable" in dr.bottleneck_ids


def test_temporal_grounding_receipt_has_persistence_metrics() -> None:
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_temporal",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    tg_receipts = [
        r for r in result.receipts if isinstance(r, TemporalGroundingReceipt)
    ]
    assert len(tg_receipts) == 1
    tg = tg_receipts[0]

    assert tg.tracks_maintained >= 2
    assert 0.0 <= tg.temporal_coherence_score <= 1.0


def test_perception_contribution_receipt_for_economic_wm() -> None:
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_econ",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    pc_receipts = [
        r for r in result.receipts if isinstance(r, PerceptionContributionReceipt)
    ]
    assert len(pc_receipts) == 1
    pc = pc_receipts[0]

    assert pc.episode_id == "ep_econ"
    assert pc.object_count >= 2
    assert pc.provider_count >= 1
    assert 0.0 <= pc.grounding_quality <= 1.0
    assert 0.0 <= pc.semantic_yield <= 1.0
    assert 0.0 <= pc.calibration_confidence <= 1.0
    assert 0.0 <= pc.temporal_stability <= 1.0


def test_metadata_contains_all_receipt_dicts() -> None:
    """State metadata carries serialized dicts for all receipt families."""
    state = _compiled_state()
    md = state.metadata

    assert "evidence_fusion_receipt" in md
    assert "provider_availability_receipts" in md
    assert "grounding_calibration_receipt" in md
    assert "inference_headroom_receipts" in md
    assert "deployment_resource_receipt" in md
    assert "temporal_grounding_receipt" in md
    assert "perception_contribution_receipt" in md

    # provider_availability_receipts should cover all known providers
    avail = md["provider_availability_receipts"]
    assert isinstance(avail, list)
    assert len(avail) >= 2  # at least scene_tracks + vision_backbone_stub


# ---------------------------------------------------------------------------
# Integration: embodiment consumer + full receipt family
# ---------------------------------------------------------------------------


def test_full_perception_to_embodiment_pipeline() -> None:
    """End-to-end: compile → receipt family → embodiment shadow consumer."""
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_pipeline",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    # Full receipt family present
    receipt_types = {type(r).__name__ for r in result.receipts}
    assert len(receipt_types) >= 6

    # Embodiment shadow consumer works on the compiled state
    surface, receipt = consume_perception_for_embodiment(result.state)

    assert surface.consumption_mode == "shadow_advisory"
    assert len(surface.object_action_relevances) >= 2
    assert receipt.object_count_consumed >= 2
    assert receipt.evidence_fusion_confidence > 0.0

    # Serialization works
    surface_dict = surface.to_dict()
    receipt_dict = receipt.to_dict()
    assert "object_action_relevances" in surface_dict
    assert "receipt_id" in receipt_dict
