"""Tests for the Perception / Grounding world model schema, receipts, and promotion."""

from __future__ import annotations

from src.vla.semantic_vla import (
    SEMANTIC_VLA_STATUS,
    SEMANTIC_VLA_SUCCESSOR,
    SemanticVLA,
)
from src.world_model.perception_grounding import (
    AnnotationSemanticBridgeState,
    BatteryState,
    ComputeEnvelopeState,
    DatasetSurfaceState,
    DeploymentResourceReceipt,
    DeploymentResourceSurface,
    DepthProviderContract,
    EmbodimentSemanticBridgeState,
    EconomicSemanticBridgeState,
    EvidenceFusionReceipt,
    EvidenceRoutingState,
    GroundingCalibrationReceipt,
    InferenceCapacityState,
    InferenceHeadroomReceipt,
    ObjectTrackState,
    PerceptionContributionReceipt,
    PerceptionGroundingWorldState,
    PerceptionProviderContract,
    PerceptionProviderRegistry,
    ProviderAvailabilityReceipt,
    ProviderSurfaceState,
    ProviderInvocationReceipt,
    SAMProviderContract,
    SemanticBridgeReceipt,
    SemanticBridgeRegistry,
    SceneEdge,
    SceneGraphState,
    SimSynthSemanticBridgeState,
    TaskMeasurementSurface,
    ThermalState,
    TemporalGroundingReceipt,
    TemporalGroundingState,
    VisionBackboneProviderContract,
    VJEPAProviderContract,
    resolve_evidence_fusion_helper,
    resolve_graph_transformer_helper,
    resolve_semantic_bridge_helper,
    resolve_temporal_grounding_helper,
)


# -------------------------------------------------------------------------
# State serialization round-trip tests
# -------------------------------------------------------------------------


class TestObjectTrackState:
    def test_roundtrip(self):
        track = ObjectTrackState(
            track_id="track_001",
            object_label="cup",
            object_category="container",
            confidence=0.95,
            epistemic_uncertainty=0.05,
            pose_3d=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.3, 0.1, 1.0],
            feature_token=[0.1] * 128,
            provider_sources=["dinov2", "sam_3_1"],
            visibility=0.9,
            occlusion_score=0.1,
            temporal_persistence_frames=30,
            affordance_hints=["graspable"],
            risk_hints=["fragile"],
        )
        d = track.to_dict()
        assert d["track_id"] == "track_001"
        assert d["object_label"] == "cup"
        assert d["confidence"] == 0.95
        assert len(d["feature_token"]) == 128
        assert d["provider_sources"] == ["dinov2", "sam_3_1"]
        assert d["affordance_hints"] == ["graspable"]
        assert d["version"] == "object_track_state_v1"

    def test_confidence_clipped(self):
        track = ObjectTrackState(
            track_id="t",
            object_label="x",
            object_category="y",
            confidence=1.5,
            epistemic_uncertainty=-0.1,
        )
        d = track.to_dict()
        assert d["confidence"] == 1.0
        assert d["epistemic_uncertainty"] == 0.0


class TestSceneEdge:
    def test_roundtrip(self):
        edge = SceneEdge(
            edge_id="edge_001",
            source_track_id="track_001",
            target_track_id="track_002",
            edge_type="contact",
            confidence=0.8,
            spatial_distance=0.05,
        )
        d = edge.to_dict()
        assert d["edge_type"] == "contact"
        assert d["confidence"] == 0.8
        assert d["version"] == "scene_edge_v1"


class TestSceneGraphState:
    def test_empty_graph(self):
        graph = SceneGraphState(graph_id="g1")
        d = graph.to_dict()
        assert d["object_count"] == 0
        assert d["edge_count"] == 0
        assert d["object_tracks"] == []
        assert d["edges"] == []

    def test_graph_with_objects_and_edges(self):
        track1 = ObjectTrackState(
            track_id="t1", object_label="cup", object_category="container",
            confidence=0.9, epistemic_uncertainty=0.1,
        )
        track2 = ObjectTrackState(
            track_id="t2", object_label="drawer", object_category="furniture",
            confidence=0.85, epistemic_uncertainty=0.15,
        )
        edge = SceneEdge(
            edge_id="e1", source_track_id="t1", target_track_id="t2",
            edge_type="containment", confidence=0.7,
        )
        graph = SceneGraphState(
            graph_id="g1",
            object_tracks=[track1, track2],
            edges=[edge],
            object_count=2,
            edge_count=1,
            edge_type_counts={"containment": 1},
        )
        d = graph.to_dict()
        assert len(d["object_tracks"]) == 2
        assert len(d["edges"]) == 1
        assert d["edge_type_counts"]["containment"] == 1


class TestTemporalGroundingState:
    def test_roundtrip(self):
        state = TemporalGroundingState(
            grounding_id="tg1",
            frame_index=42,
            total_tracks=10,
            visible_tracks=8,
            occluded_tracks=1,
            lost_tracks=1,
            recovered_tracks=0,
            helper_posture="auto",
            helper_promotion_stage="heuristic_fallback",
        )
        d = state.to_dict()
        assert d["total_tracks"] == 10
        assert d["helper_posture"] == "auto"
        assert d["version"] == "temporal_grounding_state_v1"


class TestEvidenceRoutingState:
    def test_roundtrip(self):
        state = EvidenceRoutingState(
            routing_id="er1",
            provider_contributions={"dinov2": 0.4, "sam_3_1": 0.6},
            fusion_method="heuristic_weighted",
            fusion_confidence=0.85,
            provider_availability={"dinov2": "available", "sam_3_1": "unavailable"},
        )
        d = state.to_dict()
        assert d["provider_contributions"]["dinov2"] == 0.4
        assert d["provider_availability"]["sam_3_1"] == "unavailable"


class TestProviderDatasetMeasurementState:
    def test_provider_surface_roundtrip(self):
        state = ProviderSurfaceState(
            surface_id="ps_1",
            provider_ids=["sam_3_1", "dinov2"],
            provider_kinds={"sam_3_1": "concept_segmentation", "dinov2": "vision_backbone"},
            provider_availability={"sam_3_1": "available", "dinov2": "available"},
            provider_truth_class={"sam_3_1": "real", "dinov2": "real"},
            sensor_modalities={"sam_3_1": ["rgb"], "dinov2": ["rgb"]},
            vectorized_runtime_supported=True,
            provider_batch_capacity=8,
            provider_latency_budget_ms=60.0,
        )
        d = state.to_dict()
        assert d["provider_ids"] == ["sam_3_1", "dinov2"]
        assert d["vectorized_runtime_supported"] is True
        assert d["version"] == "provider_surface_state_v1"

    def test_dataset_surface_roundtrip(self):
        state = DatasetSurfaceState(
            surface_id="ds_1",
            dataset_ids=["ego4d_like"],
            world_inventory_ids=["kitchen_world_a"],
            split_name="train",
            sensor_inventory=["rgb_front", "depth_front"],
            scene_hierarchy_levels=["object", "surface", "room"],
            available_sequences=42,
            calibration_assets_present=True,
            annotation_sources=["teacher_runtime"],
        )
        d = state.to_dict()
        assert d["dataset_ids"] == ["ego4d_like"]
        assert d["available_sequences"] == 42
        assert d["calibration_assets_present"] is True

    def test_task_measurement_surface_roundtrip(self):
        state = TaskMeasurementSurface(
            surface_id="tm_1",
            task_id="perception_shadow_eval",
            measurement_names=["grounding_quality", "track_persistence"],
            measurement_values={"grounding_quality": 0.8, "track_persistence": 0.75},
            measurement_status={"grounding_quality": "good", "track_persistence": "degraded"},
            vector_env_count=4,
            measurement_window_frames=32,
        )
        d = state.to_dict()
        assert d["task_id"] == "perception_shadow_eval"
        assert d["measurement_values"]["track_persistence"] == 0.75
        assert d["version"] == "task_measurement_surface_v1"


class TestDeploymentResourceSurface:
    def test_roundtrip(self):
        state = DeploymentResourceSurface(
            surface_id="dr_1",
            compute_envelope=ComputeEnvelopeState(
                profile_id="ce_1",
                on_device_available=True,
                companion_available=True,
                placement_class="hybrid",
                latency_budget_ms=45.0,
                qos_class="bounded",
            ),
            inference_capacity=InferenceCapacityState(
                profile_id="ic_1",
                provider_capacity_by_id={"sam_3_1": 0.5, "dinov2": 0.8},
                headroom_fraction=0.4,
                batch_headroom=2,
                max_parallel_providers=2,
            ),
            battery_state=BatteryState(
                battery_id="bat_1",
                charge_fraction=0.65,
                reserve_fraction=0.2,
                projected_runtime_minutes=18.0,
                charging_state="discharging",
            ),
            thermal_state=ThermalState(
                thermal_id="therm_1",
                thermal_headroom_fraction=0.55,
                throttled=False,
                max_temperature_c=62.0,
                thermal_zone="compute",
            ),
            bandwidth_mbps=120.0,
            companion_compute_available=True,
            deployment_posture="preflight_ready",
        )
        d = state.to_dict()
        assert d["compute_envelope"]["placement_class"] == "hybrid"
        assert d["inference_capacity"]["headroom_fraction"] == 0.4
        assert d["battery_state"]["charge_fraction"] == 0.65
        assert d["deployment_posture"] == "preflight_ready"


class TestSemanticBridgeState:
    def test_registry_roundtrip(self):
        sim_bridge = SimSynthSemanticBridgeState(
            bridge_id="sb_1",
            source_graph_id="g1",
            branch_relevance_scores=[0.8, 0.6],
            object_preservation_scores=[0.9, 0.7],
            helper_posture="auto",
            helper_promotion_stage="promoted",
        )
        embodiment_bridge = EmbodimentSemanticBridgeState(
            bridge_id="emb_1",
            source_graph_id="g1",
            per_object_affordance_scores={"t1": 0.9},
            per_object_affordance_classes={"t1": ["graspable", "liftable"]},
            body_object_pairwise_scores={"left_hand": {"t1": 0.8}},
            resource_conditioned=True,
            embodiment_dof=29,
        )
        annotation_bridge = AnnotationSemanticBridgeState(
            bridge_id="ab_1",
            source_graph_id="g1",
            object_class_labels={"t1": "cup"},
            teacher_alignment_score=0.8,
        )
        economic_bridge = EconomicSemanticBridgeState(
            bridge_id="eb_1",
            source_graph_id="g1",
            economic_summary_token=[0.1] * 8,
            semantic_density=0.75,
        )
        registry = SemanticBridgeRegistry(
            registry_id="reg_1",
            source_graph_id="g1",
            sim_synth_bridge=sim_bridge,
            embodiment_bridge=embodiment_bridge,
            annotation_bridge=annotation_bridge,
            economic_bridge=economic_bridge,
        )
        d = registry.to_dict()
        assert d["sim_synth_bridge"]["helper_promotion_stage"] == "promoted"
        assert d["embodiment_bridge"]["resource_conditioned"] is True
        assert d["embodiment_bridge"]["body_object_pairwise_scores"]["left_hand"]["t1"] == 0.8
        assert d["annotation_bridge"]["object_class_labels"]["t1"] == "cup"
        assert d["economic_bridge"]["semantic_density"] == 0.75
        assert d["semantic_vla_successor_status"] == "distributed_bridge_family"


class TestPerceptionGroundingWorldState:
    def test_minimal(self):
        state = PerceptionGroundingWorldState(
            state_id="pg_001",
            frame_index=0,
            episode_id="ep_001",
        )
        d = state.to_dict()
        assert d["state_id"] == "pg_001"
        assert d["scene_graph"] is None
        assert d["maturity_stage"] == "schema_only"
        assert d["version"] == "perception_grounding_world_state_v1"

    def test_full_state(self):
        track = ObjectTrackState(
            track_id="t1", object_label="cup", object_category="container",
            confidence=0.9, epistemic_uncertainty=0.1,
        )
        scene = SceneGraphState(
            graph_id="g1", object_tracks=[track], object_count=1,
        )
        temporal = TemporalGroundingState(
            grounding_id="tg1", frame_index=5, total_tracks=1,
            visible_tracks=1, occluded_tracks=0, lost_tracks=0,
            recovered_tracks=0,
        )
        routing = EvidenceRoutingState(
            routing_id="er1",
            provider_contributions={"dinov2": 1.0},
            fusion_confidence=0.9,
        )
        provider_surface = ProviderSurfaceState(
            surface_id="ps1",
            provider_ids=["dinov2"],
            provider_availability={"dinov2": "available"},
        )
        dataset_surface = DatasetSurfaceState(
            surface_id="ds1",
            dataset_ids=["ego_like"],
            world_inventory_ids=["kitchen_a"],
        )
        task_measurements = TaskMeasurementSurface(
            surface_id="tm1",
            task_id="shadow_eval",
            measurement_names=["grounding_quality"],
            measurement_values={"grounding_quality": 0.9},
        )
        deployment_resource_surface = DeploymentResourceSurface(
            surface_id="dr1",
            deployment_posture="preflight_ready",
            compute_envelope=ComputeEnvelopeState(profile_id="ce1", on_device_available=True),
        )
        semantic_bridge_registry = SemanticBridgeRegistry(
            registry_id="sbr1",
            source_graph_id="g1",
            sim_synth_bridge=SimSynthSemanticBridgeState(
                bridge_id="sb1",
                source_graph_id="g1",
            ),
        )
        state = PerceptionGroundingWorldState(
            state_id="pg_002",
            frame_index=5,
            episode_id="ep_002",
            scene_graph=scene,
            temporal_grounding=temporal,
            evidence_routing=routing,
            provider_surface=provider_surface,
            dataset_surface=dataset_surface,
            task_measurements=task_measurements,
            deployment_resource_surface=deployment_resource_surface,
            semantic_bridge_registry=semantic_bridge_registry,
            maturity_stage="shadow_runtime",
        )
        d = state.to_dict()
        assert d["scene_graph"] is not None
        assert len(d["scene_graph"]["object_tracks"]) == 1
        assert d["temporal_grounding"]["total_tracks"] == 1
        assert d["evidence_routing"]["fusion_confidence"] == 0.9
        assert d["provider_surface"]["provider_ids"] == ["dinov2"]
        assert d["dataset_surface"]["dataset_ids"] == ["ego_like"]
        assert d["task_measurements"]["measurement_values"]["grounding_quality"] == 0.9
        assert d["deployment_resource_surface"]["deployment_posture"] == "preflight_ready"
        assert d["semantic_bridge_registry"]["source_graph_id"] == "g1"
        assert d["maturity_stage"] == "shadow_runtime"


# -------------------------------------------------------------------------
# Receipt serialization tests
# -------------------------------------------------------------------------


class TestProviderInvocationReceipt:
    def test_roundtrip(self):
        receipt = ProviderInvocationReceipt(
            receipt_id="pir_001",
            provider_id="sam_3_1",
            provider_kind="concept_segmentation",
            invocation_status="executed",
            output_quality_score=0.85,
            latency_ms=45.0,
            output_token_count=12,
        )
        d = receipt.to_dict()
        assert d["provider_id"] == "sam_3_1"
        assert d["invocation_status"] == "executed"
        assert d["version"] == "provider_invocation_receipt_v1"

    def test_fallback_receipt(self):
        receipt = ProviderInvocationReceipt(
            receipt_id="pir_002",
            provider_id="sam_3_1",
            provider_kind="concept_segmentation",
            invocation_status="skipped",
            fallback_used=True,
            fallback_reason="sam_weights_or_gpu_unavailable",
        )
        d = receipt.to_dict()
        assert d["fallback_used"] is True
        assert d["fallback_reason"] == "sam_weights_or_gpu_unavailable"


class TestProviderAvailabilityReceipt:
    def test_roundtrip(self):
        receipt = ProviderAvailabilityReceipt(
            receipt_id="par_001",
            provider_surface_id="ps_1",
            provider_id="dinov2",
            availability_status="available",
            install_status="install_ready",
            provider_truth_class="real",
            sensor_modalities=["rgb"],
        )
        d = receipt.to_dict()
        assert d["provider_id"] == "dinov2"
        assert d["install_status"] == "install_ready"
        assert d["version"] == "provider_availability_receipt_v1"


class TestGroundingCalibrationReceipt:
    def test_roundtrip(self):
        receipt = GroundingCalibrationReceipt(
            receipt_id="gcr_001",
            calibration_method="cross_provider_agreement",
            grounding_accuracy=0.8,
            provider_agreement=0.75,
        )
        d = receipt.to_dict()
        assert d["grounding_accuracy"] == 0.8
        assert d["version"] == "grounding_calibration_receipt_v1"


class TestEvidenceFusionReceipt:
    def test_roundtrip(self):
        receipt = EvidenceFusionReceipt(
            receipt_id="efr_001",
            fusion_method="heuristic_weighted",
            provider_ids=["dinov2", "sam_3_1"],
            provider_weights={"dinov2": 0.4, "sam_3_1": 0.6},
            fusion_confidence=0.85,
            output_object_count=5,
        )
        d = receipt.to_dict()
        assert d["provider_ids"] == ["dinov2", "sam_3_1"]
        assert d["version"] == "evidence_fusion_receipt_v1"


class TestInferenceHeadroomReceipt:
    def test_roundtrip(self):
        receipt = InferenceHeadroomReceipt(
            receipt_id="ihr_001",
            deployment_surface_id="dr_1",
            provider_id="sam_3_1",
            headroom_fraction=0.45,
            estimated_latency_ms=38.0,
            on_device_available=True,
            companion_available=True,
            bandwidth_mbps=100.0,
        )
        d = receipt.to_dict()
        assert d["headroom_fraction"] == 0.45
        assert d["on_device_available"] is True
        assert d["version"] == "inference_headroom_receipt_v1"


class TestDeploymentResourceReceipt:
    def test_roundtrip(self):
        receipt = DeploymentResourceReceipt(
            receipt_id="drr_001",
            deployment_surface_id="dr_1",
            deployment_posture="preflight_ready",
            compute_ready=True,
            battery_ready=True,
            thermal_ready=False,
            bottleneck_ids=["thermal_headroom"],
        )
        d = receipt.to_dict()
        assert d["deployment_posture"] == "preflight_ready"
        assert d["thermal_ready"] is False
        assert d["bottleneck_ids"] == ["thermal_headroom"]


class TestSemanticBridgeReceipt:
    def test_roundtrip(self):
        receipt = SemanticBridgeReceipt(
            receipt_id="sbr_001",
            bridge_kind="sim_synth",
            source_graph_id="g1",
            output_quality_score=0.7,
            downstream_usefulness_score=0.8,
            helper_posture="auto",
            helper_promotion_stage="promoted",
        )
        d = receipt.to_dict()
        assert d["bridge_kind"] == "sim_synth"
        assert d["downstream_usefulness_score"] == 0.8
        assert d["version"] == "semantic_bridge_receipt_v1"


class TestTemporalGroundingReceipt:
    def test_roundtrip(self):
        receipt = TemporalGroundingReceipt(
            receipt_id="tgr_001",
            frame_index=42,
            tracks_maintained=8,
            tracks_lost=1,
            temporal_coherence_score=0.9,
        )
        d = receipt.to_dict()
        assert d["tracks_maintained"] == 8
        assert d["version"] == "temporal_grounding_receipt_v1"


class TestPerceptionContributionReceipt:
    def test_roundtrip(self):
        receipt = PerceptionContributionReceipt(
            receipt_id="pcr_001",
            episode_id="ep_001",
            grounding_quality=0.85,
            semantic_yield=0.7,
            calibration_confidence=0.9,
            action_relevance_prior=0.6,
            provider_count=3,
            object_count=12,
        )
        d = receipt.to_dict()
        assert d["episode_id"] == "ep_001"
        assert d["provider_count"] == 3
        assert d["version"] == "perception_contribution_receipt_v1"


# -------------------------------------------------------------------------
# Provider contract tests
# -------------------------------------------------------------------------


class TestSAMProviderContract:
    def test_defaults_unavailable(self):
        contract = SAMProviderContract()
        d = contract.to_dict()
        assert d["availability"] == "unavailable"
        assert d["provider_truth_class"] == "unavailable"
        assert d["fallback_posture"] == "scene_tracks_only"
        assert d["version"] == "sam_provider_contract_v1"

    def test_available(self):
        contract = SAMProviderContract(
            availability="available",
            provider_truth_class="real",
            weights_available=True,
            weights_path="/models/sam2.1_hiera_large.pt",
            image_predictor_available=True,
        )
        d = contract.to_dict()
        assert d["availability"] == "available"
        assert d["weights_available"] is True


class TestVisionBackboneProviderContract:
    def test_defaults_unavailable(self):
        contract = VisionBackboneProviderContract()
        d = contract.to_dict()
        assert d["availability"] == "unavailable"
        assert d["fallback_posture"] == "deterministic_stub"
        assert d["backbone_dim"] == 1024
        assert d["projection_output_dim"] == 128


class TestVJEPAProviderContract:
    def test_defaults_unavailable(self):
        contract = VJEPAProviderContract()
        d = contract.to_dict()
        assert d["availability"] == "unavailable"
        assert d["upstream_repo"] == "facebookresearch/vjepa2"
        assert d["fallback_posture"] == "planning_only"


class TestDepthProviderContract:
    def test_defaults_unavailable(self):
        contract = DepthProviderContract()
        d = contract.to_dict()
        assert d["availability"] == "unavailable"
        assert d["fallback_posture"] == "scene_tracks_geometry_only"
        assert d["camera_intrinsics_required"] is True


class TestPerceptionProviderRegistry:
    def test_default_registry(self):
        registry = PerceptionProviderRegistry(registry_id="reg_001")
        d = registry.to_dict()
        assert d["sam_contract"]["availability"] == "unavailable"
        assert d["vision_backbone_contract"]["availability"] == "unavailable"
        assert d["vjepa_contract"]["availability"] == "unavailable"
        assert d["depth_contract"]["availability"] == "unavailable"
        assert d["version"] == "perception_provider_registry_v1"

    def test_base_contract_roundtrip(self):
        contract = PerceptionProviderContract(
            provider_id="teacher_semantics",
            provider_kind="teacher_runtime",
            provider_family="teacher",
            availability="planning_only",
            provider_truth_class="advisory",
        )
        d = contract.to_dict()
        assert d["provider_id"] == "teacher_semantics"
        assert d["provider_truth_class"] == "advisory"


# -------------------------------------------------------------------------
# Promotion / demotion tests
# -------------------------------------------------------------------------


class TestGraphTransformerPromotion:
    def test_disabled_returns_heuristic_fallback(self):
        result = resolve_graph_transformer_helper(
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert result["helper_active"] is False
        assert result["promotion_stage"] == "heuristic_fallback"

    def test_auto_not_ready_returns_heuristic(self):
        result = resolve_graph_transformer_helper(
            loading_posture="auto",
            benchmark_signals={"ready": False},
        )
        assert result["helper_active"] is False
        assert result["promotion_stage"] == "heuristic_fallback"

    def test_auto_ready_returns_promoted(self):
        result = resolve_graph_transformer_helper(
            loading_posture="auto",
            benchmark_signals={"ready": True},
        )
        assert result["helper_active"] is True
        assert result["promotion_stage"] == "promoted"
        assert result["helper_weight"] == 1.0

    def test_demotion_on_evidence_failure(self):
        result = resolve_graph_transformer_helper(
            loading_posture="auto",
            benchmark_signals={"ready": True},
            evidence_signals={"evidence_failure": True},
        )
        assert result["promotion_stage"] == "demoted_to_shadow"
        assert result["helper_weight"] == 0.25
        assert result["demotion_reason"] == "evidence_failure"

    def test_demotion_on_failure_rate(self):
        result = resolve_graph_transformer_helper(
            loading_posture="auto",
            benchmark_signals={"ready": True, "benchmark_gate": {"demotion_failure_threshold": 0.3}},
            evidence_signals={"recent_failure_rate": 0.5},
        )
        assert result["promotion_stage"] == "demoted_to_shadow"

    def test_no_demotion_without_evidence(self):
        result = resolve_graph_transformer_helper(
            loading_posture="auto",
            benchmark_signals={"ready": True},
        )
        assert result["promotion_stage"] == "promoted"

    def test_required_not_ready(self):
        result = resolve_graph_transformer_helper(
            loading_posture="required",
            benchmark_signals={"ready": False},
        )
        assert result["promotion_stage"] == "required_but_not_ready"


class TestTemporalGroundingPromotion:
    def test_disabled(self):
        result = resolve_temporal_grounding_helper(
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert result["helper_active"] is False

    def test_auto_promoted(self):
        result = resolve_temporal_grounding_helper(
            loading_posture="auto",
            benchmark_signals={"ready": True},
        )
        assert result["promotion_stage"] == "promoted"

    def test_demotion(self):
        result = resolve_temporal_grounding_helper(
            loading_posture="required",
            benchmark_signals={"ready": True},
            evidence_signals={"benchmark_gate_revoked": True},
        )
        assert result["promotion_stage"] == "demoted_to_shadow"


class TestEvidenceFusionPromotion:
    def test_disabled(self):
        result = resolve_evidence_fusion_helper(
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert result["helper_active"] is False

    def test_auto_promoted(self):
        result = resolve_evidence_fusion_helper(
            loading_posture="auto",
            benchmark_signals={"benchmark_eligible": True},
        )
        assert result["promotion_stage"] == "promoted"


class TestSemanticBridgePromotion:
    def test_disabled(self):
        result = resolve_semantic_bridge_helper(
            bridge_kind="sim_synth",
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert result["bridge_kind"] == "sim_synth"
        assert result["helper_active"] is False

    def test_auto_promoted(self):
        result = resolve_semantic_bridge_helper(
            bridge_kind="annotation",
            loading_posture="auto",
            benchmark_signals={"ready": True},
        )
        assert result["promotion_stage"] == "promoted"
        assert result["helper_weight"] == 1.0

    def test_required_demoted(self):
        result = resolve_semantic_bridge_helper(
            bridge_kind="economic",
            loading_posture="required",
            benchmark_signals={"ready": True},
            evidence_signals={"evidence_failure": True},
        )
        assert result["promotion_stage"] == "demoted_to_shadow"
        assert result["demotion_reason"] == "evidence_failure"


class TestSemanticVLA:
    def test_scaffolding_posture(self):
        analyzer = SemanticVLA()
        assert analyzer.status == SEMANTIC_VLA_STATUS
        assert SEMANTIC_VLA_SUCCESSOR == "distributed_semantic_bridge_family"

    def test_output_carries_successor_metadata(self):
        analyzer = SemanticVLA()
        result = analyzer.analyze_episode({"semantic_tags": ["cup", "drawer"]})
        assert result["_semantic_vla_status"] == "scaffolding_only"
        assert result["_semantic_vla_successor"] == "distributed_semantic_bridge_family"
        assert result["semantic_tags"] == ["cup", "drawer"]
