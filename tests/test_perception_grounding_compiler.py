from __future__ import annotations

import numpy as np
import torch

from src.evidence.belief_state import BeliefState
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.interfaces import VisionFrame
from src.world_model.perception_grounding import (
    AnnotationBridgeProjectionSeam,
    EvidenceFusionSeam,
    PerceptionCompilationResult,
    VisionBackboneProjectionSeam,
    VJEPATemporalAlignmentSeam,
    compile_perception_grounding_with_receipts,
    compile_perception_grounding_world_state,
    encode_provider_features,
    export_annotation_record,
)
from src.world_model.perception_grounding.benchmark_evidence import (
    build_perception_benchmark_evidence,
    write_perception_benchmark_evidence,
)
from src.world_model.perception_grounding.receipts import EvidenceFusionReceipt
from src.world_model.semantic_coverage_graph import (
    CoverageEdge,
    CoverageNode,
    SemanticCoverageGraph,
)
from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state


def _scene_tracks_payload() -> dict[str, np.ndarray]:
    poses_r = np.stack([np.stack([np.eye(3), np.eye(3)]), np.stack([np.eye(3), np.eye(3)])]).astype(np.float32)
    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(["drawer_track", "vase_track"], dtype="U32"),
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
        timestamp="2026-04-03T10:00:00Z",
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


def _provider_adapter_benchmark_evidence() -> dict[str, object]:
    return {
        "benchmark_evidence_present": True,
        "evidence_source_provisional": False,
        "annotation_supervision_score": 0.86,
        "held_out_label_agreement": 0.82,
        "downstream_usefulness_score": 0.74,
        "receipt_consistency": 0.9,
        "gate_score": 0.84,
        "promotion_eligible": True,
    }


def _coverage_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("skill:open_drawer", "skill", "open_drawer"),
            CoverageNode("prim:grasp_handle", "env_primitive", "grasp_handle"),
        ],
        edges=[
            CoverageEdge(
                "skill:open_drawer",
                "prim:grasp_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.7,
                trust_priority=0.4,
                promotion_readiness=0.5,
            )
        ],
    )


def test_compile_perception_grounding_world_state_builds_functional_shadow_state() -> None:
    state = compile_perception_grounding_world_state(
        episode_id="ep_test",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    assert state.maturity_stage == "shadow_runtime"
    assert state.scene_graph is not None
    assert state.scene_graph.object_count >= 2
    assert state.evidence_routing is not None
    assert state.evidence_routing.fusion_confidence > 0.0
    assert state.semantic_bridge_registry is not None
    assert state.semantic_bridge_registry.sim_synth_bridge is not None
    assert state.semantic_bridge_registry.annotation_bridge is not None
    assert "object_preservation" in state.semantic_bridge_registry.sim_synth_bridge.contact_topology_summary["bridge_preconditions"]
    assert state.semantic_bridge_registry.annotation_bridge.object_class_labels


def test_compiled_perception_state_feeds_sim_synth_semantic_context() -> None:
    perception_state = compile_perception_grounding_world_state(
        episode_id="ep_test",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    sim_state = compile_sim_synth_physics_world_state(
        _coverage_graph(),
        limit=1,
        perception_grounding_state=perception_state,
    )

    semantic_ctx = sim_state.input_context["semantic"]
    assert semantic_ctx["perception_grounding_state_id"] == perception_state.state_id
    assert semantic_ctx["scene_object_count"] == perception_state.scene_graph.object_count
    assert semantic_ctx["sim_synth_bridge_ready"] is True
    assert semantic_ctx["sim_synth_branch_relevance_mean"] > 0.0
    assert (
        semantic_ctx["inferential_learnability_summary"]["mean_signal_yield_score"] > 0.0
    )


def test_evidence_fusion_seam_forward_pass() -> None:
    """The neural seam runs a forward pass and produces valid weights + confidence."""
    seam = EvidenceFusionSeam.heuristic_init(d_model=32, n_heads=2, d_ff=64)
    assert seam.param_count() > 0

    features = encode_provider_features(
        provider_ids=["scene_tracks", "vla_semantic_evidence", "vision_backbone_stub"],
        provider_kinds={
            "scene_tracks": "scene_tracks",
            "vla_semantic_evidence": "teacher_semantics",
            "vision_backbone_stub": "vision_backbone",
        },
        provider_availability={
            "scene_tracks": "available",
            "vla_semantic_evidence": "available",
            "vision_backbone_stub": "available",
        },
        provider_truth_class={
            "scene_tracks": "provider_backed",
            "vla_semantic_evidence": "advisory_evidence",
            "vision_backbone_stub": "stub_smoke_only",
        },
        semantic_quality=0.7,
        coverage=0.8,
        disagreement=0.1,
        object_count_norm=0.5,
        edge_count_norm=0.25,
    )

    assert features.shape == (3, 12)

    with torch.no_grad():
        weights, confidence = seam(features)

    assert weights.shape == (3,)
    assert abs(weights.sum().item() - 1.0) < 1e-5
    assert all(w >= 0.0 for w in weights.tolist())
    assert 0.0 <= confidence.item() <= 1.0


def test_evidence_fusion_seam_batched() -> None:
    """The neural seam handles batched input correctly."""
    seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
    batch_features = torch.randn(4, 3, 12)

    with torch.no_grad():
        weights, confidence = seam(batch_features)

    assert weights.shape == (4, 3)
    assert confidence.shape == (4,)
    # Each row sums to 1
    for i in range(4):
        assert abs(weights[i].sum().item() - 1.0) < 1e-5


def test_evidence_fusion_seam_with_mask() -> None:
    """Masked providers get zero weight."""
    seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
    features = torch.randn(3, 12)
    mask = torch.tensor([False, False, True])  # third provider masked

    with torch.no_grad():
        weights, confidence = seam(features, provider_mask=mask)

    assert weights.shape == (3,)
    assert weights[2].item() < 1e-6  # masked provider gets ~0 weight
    assert abs(weights.sum().item() - 1.0) < 1e-5


def test_compiler_backward_compat_without_seam() -> None:
    """Compiler still works identically when no neural seam is provided."""
    state = compile_perception_grounding_world_state(
        episode_id="ep_compat",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    assert state.maturity_stage == "shadow_runtime"
    assert state.evidence_routing is not None
    assert state.evidence_routing.fusion_method == "semantic_world_model_heuristic_fusion"
    # Receipt should be present in metadata
    assert "evidence_fusion_receipt" in state.metadata
    receipt_dict = state.metadata["evidence_fusion_receipt"]
    assert receipt_dict["fusion_method"] == "semantic_world_model_heuristic_fusion"
    assert receipt_dict["metadata"]["neural_seam_used"] is False


def test_annotation_bridge_shadow_receipt_carries_benchmark_evidence() -> None:
    seam = AnnotationBridgeProjectionSeam(
        d_token=128,
        d_hidden=64,
        n_categories=8,
        n_affordances=4,
    )

    state = compile_perception_grounding_world_state(
        episode_id="ep_annotation_bridge",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"benchmark_eligible": True},
        annotation_bridge_projection_seam=seam,
        annotation_bridge_benchmark_evidence={
            "benchmark_evidence_present": True,
            "evidence_source_provisional": True,
            "annotation_supervision_score": 0.91,
            "held_out_label_agreement": 0.87,
            "downstream_usefulness_score": 0.72,
            "receipt_consistency": 0.88,
            "gate_score": 0.86,
            "promotion_eligible": True,
        },
    )

    receipt = state.metadata["annotation_bridge_shadow_receipt"]
    assert receipt is not None
    assert receipt["benchmark_evidence_present"] is True
    assert receipt["evidence_source_provisional"] is True
    assert receipt["annotation_supervision_score"] == 0.91
    assert receipt["held_out_label_agreement"] == 0.87
    assert receipt["gate_score"] == 0.86
    assert receipt["promotion_eligible"] is False


def test_annotation_export_prefers_provider_backed_benchmark_tokens() -> None:
    base_state = compile_perception_grounding_world_state(
        episode_id="ep_export_tokens_base",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )
    n_objects = base_state.scene_graph.object_count
    provider_tokens = [
        [0.25] * 128 if idx % 2 == 0 else [0.75] * 128
        for idx in range(n_objects)
    ]
    state = compile_perception_grounding_world_state(
        episode_id="ep_export_tokens",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_object_tokens=provider_tokens,
        benchmark_object_token_source={
            "source_kind": "explicit_provider_object_tokens",
            "truth_class": "provider_backed",
            "provider_id": "backbone_provider",
            "provider_kind": "vision_backbone_projection",
            "evidence_source_provisional": False,
        },
    )

    record = export_annotation_record(state)
    assert record is not None
    assert record.object_tokens == provider_tokens
    assert record.object_token_source_kind == "explicit_provider_object_tokens"
    assert record.object_token_truth_class == "provider_backed"
    assert record.object_token_evidence_provisional is False


def test_runtime_vision_backbone_projection_tokens_feed_annotation_export() -> None:
    base_state = compile_perception_grounding_world_state(
        episode_id="ep_runtime_backbone_tokens_base",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )
    n_objects = base_state.scene_graph.object_count
    seam = VisionBackboneProjectionSeam(
        d_backbone=4,
        d_hidden=8,
        d_out=128,
        dropout=0.0,
    )
    seam.eval()

    state = compile_perception_grounding_world_state(
        episode_id="ep_runtime_backbone_tokens",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"benchmark_eligible": True},
        vision_backbone_projection_seam=seam,
        backbone_features=torch.ones((n_objects, 4), dtype=torch.float32),
        provider_adapter_benchmark_evidence={
            "vision_backbone_projection": _provider_adapter_benchmark_evidence()
        },
    )

    source = state.metadata["benchmark_object_token_source"]
    assert source["source_kind"] == "vision_backbone_projection"
    assert source["truth_class"] == "provider_backed"
    assert source["provider_id"] == "dinov2_vit_l_14"
    assert source["provider_invocation_status"] == "success"
    assert source["provider_output_token_count"] == state.scene_graph.object_count
    assert source["evidence_source_provisional"] is False
    assert "provider_invocation_receipt_id" in source

    provider_surface = state.provider_surface
    assert "dinov2_vit_l_14" in provider_surface.provider_ids
    assert "vision_backbone_stub" not in provider_surface.provider_ids
    assert provider_surface.provider_truth_class["dinov2_vit_l_14"] == "provider_backed"
    assert (
        provider_surface.metadata["runtime_provider_inputs"][
            "vision_backbone_projection"
        ]["seam_present"]
        is True
    )

    receipt = next(
        r
        for r in state.metadata["provider_adapter_receipts"]
        if r["provider_kind"] == "vision_backbone_projection"
    )
    assert receipt["invocation_status"] == "success"
    assert receipt["fallback_used"] is False
    assert receipt["metadata"]["runtime_provider_backed"] is True

    record = export_annotation_record(state)
    assert record is not None
    assert record.object_token_source_kind == "vision_backbone_projection"
    assert record.object_token_truth_class == "provider_backed"
    assert record.object_token_evidence_provisional is False


def test_runtime_vjepa_alignment_tokens_feed_annotation_export_without_explicit_wm_tokens() -> None:
    seam = VJEPATemporalAlignmentSeam(
        d_vjepa=4,
        d_wm_token=16,
        d_model=16,
        d_out=128,
        n_heads=4,
        d_ff=32,
        n_temporal_steps=2,
        dropout=0.0,
    )
    seam.eval()

    state = compile_perception_grounding_world_state(
        episode_id="ep_runtime_vjepa_tokens",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"benchmark_eligible": True},
        vjepa_temporal_alignment_seam=seam,
        vjepa_tokens=torch.ones((2, 3, 4), dtype=torch.float32),
        provider_adapter_benchmark_evidence={
            "vjepa_temporal_alignment": _provider_adapter_benchmark_evidence()
        },
    )

    source = state.metadata["benchmark_object_token_source"]
    assert source["source_kind"] == "vjepa_temporal_alignment"
    assert source["truth_class"] == "provider_backed"
    assert source["provider_id"] == "vjepa2"
    assert source["provider_invocation_status"] == "success"
    assert source["provider_output_token_count"] == state.scene_graph.object_count
    assert source["evidence_source_provisional"] is False
    assert "vjepa2" in state.provider_surface.provider_ids

    record = export_annotation_record(state)
    assert record is not None
    assert record.object_token_source_kind == "vjepa_temporal_alignment"
    assert record.object_token_truth_class == "provider_backed"
    assert record.object_token_evidence_provisional is False


def test_failed_runtime_provider_projection_does_not_claim_provider_backed_tokens() -> None:
    class BrokenVisionProjection:
        def __call__(self, backbone_features: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("projection failed")

    base_state = compile_perception_grounding_world_state(
        episode_id="ep_runtime_provider_failure_base",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )
    n_objects = base_state.scene_graph.object_count
    state = compile_perception_grounding_world_state(
        episode_id="ep_runtime_provider_failure",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"benchmark_eligible": True},
        vision_backbone_projection_seam=BrokenVisionProjection(),
        backbone_features=torch.ones((n_objects, 4), dtype=torch.float32),
        provider_adapter_benchmark_evidence={
            "vision_backbone_projection": _provider_adapter_benchmark_evidence()
        },
    )

    source = state.metadata["benchmark_object_token_source"]
    assert source["source_kind"] == "heuristic_scene_graph"
    assert source["truth_class"] == "heuristic_derived"
    assert source["evidence_source_provisional"] is True

    receipt = next(
        r
        for r in state.metadata["provider_adapter_receipts"]
        if r["provider_kind"] == "vision_backbone_projection"
    )
    assert receipt["invocation_status"] == "error"
    assert receipt["fallback_used"] is True
    assert receipt["metadata"]["runtime_provider_backed"] is False
    assert "vision_backbone_projection" not in state.metadata[
        "provider_adapter_outputs_available"
    ]


def test_graph_transformer_shadow_receipt_loads_persistent_benchmark_evidence(tmp_path) -> None:
    evidence = build_perception_benchmark_evidence(
        subsystem_key="scene_graph_transformer",
        metrics={
            "benchmark_evidence_present": True,
            "evidence_source_provisional": False,
            "evidence_truth_class": "provider_backed",
            "token_source_kind": "explicit_provider_object_tokens",
            "annotation_supervision_score": 0.84,
            "held_out_label_agreement": 0.81,
            "downstream_usefulness_score": 0.69,
            "receipt_consistency": 0.9,
            "gate_score": 0.82,
            "promotion_eligible": True,
        },
        source_record_count=10,
    )
    evidence_path = tmp_path / "graph_transformer_benchmark_evidence.json"
    write_perception_benchmark_evidence(evidence_path, evidence)

    from src.world_model.perception_grounding import SceneGraphTransformerSeam

    base_state = compile_perception_grounding_world_state(
        episode_id="ep_graph_evidence_base",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )
    n_objects = base_state.scene_graph.object_count
    graph_seam = SceneGraphTransformerSeam(
        d_token=128,
        d_model=64,
        d_out=64,
        n_heads=2,
        d_ff=128,
        n_layers=1,
    )
    state = compile_perception_grounding_world_state(
        episode_id="ep_graph_evidence",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"benchmark_eligible": True},
        scene_graph_transformer_seam=graph_seam,
        scene_graph_transformer_benchmark_evidence=str(evidence_path),
        benchmark_object_tokens=[
            [0.1] * 128 if idx % 2 == 0 else [0.2] * 128
            for idx in range(n_objects)
        ],
        benchmark_object_token_source={
            "source_kind": "explicit_provider_object_tokens",
            "truth_class": "provider_backed",
            "evidence_source_provisional": False,
        },
    )

    receipt = state.metadata["graph_transformer_shadow_receipt"]
    assert receipt is not None
    assert receipt["version"] == "graph_transformer_shadow_receipt_v3"
    assert receipt["benchmark_evidence_present"] is True
    assert receipt["evidence_source_provisional"] is False
    assert receipt["promotion_eligible"] is True
    assert receipt["metadata"]["token_source_kind"] == "explicit_provider_object_tokens"


def test_compiler_with_neural_seam_promoted() -> None:
    """When benchmark_signals promote and seam is provided, neural path runs."""
    seam = EvidenceFusionSeam.heuristic_init()

    state = compile_perception_grounding_world_state(
        episode_id="ep_neural",
        task_id="drawer_vase",
        semantic_tags=["drawer", "fragile"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"ready": True},
        evidence_fusion_seam=seam,
    )

    assert state.evidence_routing is not None
    assert state.evidence_routing.fusion_method == "neural_evidence_fusion_seam"
    assert state.evidence_routing.metadata["neural_seam_used"] is True
    assert state.evidence_routing.fusion_confidence > 0.0
    # Weights should still sum to ~1.0
    total = sum(state.evidence_routing.provider_contributions.values())
    assert abs(total - 1.0) < 1e-4


def test_compiler_neural_seam_fallback_without_benchmark() -> None:
    """Without benchmark_signals, seam is provided but heuristic runs."""
    seam = EvidenceFusionSeam.heuristic_init()

    state = compile_perception_grounding_world_state(
        episode_id="ep_no_bench",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        evidence_fusion_seam=seam,
    )

    # No benchmark = heuristic_fallback, seam not used
    assert state.evidence_routing.fusion_method == "semantic_world_model_heuristic_fusion"
    assert state.evidence_routing.metadata["neural_seam_used"] is False


def test_compile_with_receipts_returns_typed_result() -> None:
    """compile_perception_grounding_with_receipts returns structured result."""
    result = compile_perception_grounding_with_receipts(
        episode_id="ep_receipts",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
    )

    assert isinstance(result, PerceptionCompilationResult)
    assert result.state.maturity_stage == "shadow_runtime"
    assert len(result.receipts) >= 1
    assert isinstance(result.receipts[0], EvidenceFusionReceipt)
    assert result.receipts[0].fusion_method == "semantic_world_model_heuristic_fusion"
    assert result.receipts[0].provider_ids
    assert result.receipts[0].provider_weights


def test_compile_with_receipts_neural_seam() -> None:
    """compile_with_receipts captures neural seam receipt."""
    seam = EvidenceFusionSeam.heuristic_init()

    result = compile_perception_grounding_with_receipts(
        episode_id="ep_neural_receipts",
        task_id="drawer_vase",
        semantic_tags=["drawer"],
        belief_state=_belief_state(),
        scene_tracks_payload=_scene_tracks_payload(),
        benchmark_signals={"ready": True},
        evidence_fusion_seam=seam,
    )

    assert len(result.receipts) >= 1
    receipt = result.receipts[0]
    assert receipt.fusion_method == "neural_evidence_fusion_seam"
    assert receipt.metadata["neural_seam_used"] is True


def test_evidence_fusion_seam_describe() -> None:
    """Seam describe() returns useful metadata for logging/receipts."""
    seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
    desc = seam.describe()

    assert desc["seam_type"] == "evidence_fusion"
    assert desc["d_model"] == 32
    assert desc["n_heads"] == 2
    assert desc["param_count"] > 0
    assert desc["module_class"] == "EvidenceFusionSeam"


def test_vision_backbone_stub_exposes_typed_provider_posture() -> None:
    stub = VisionBackboneStub(model_name="dino-stub", latent_dim=8)
    contract = stub.describe_provider_contract()

    latent = stub.encode_frame(
        VisionFrame(
            backend="cpu",
            task_id="drawer_vase",
            episode_id="ep_test",
            timestep=0,
            metadata={"state": {"foo": "bar"}},
        )
    )

    assert contract.provider_id == "vision_backbone_stub"
    assert contract.provider_truth_class == "stub_smoke_only"
    assert contract.metadata["advisory_only"] is True
    assert latent.metadata["advisory_only"] is True
    assert latent.metadata["provider_contract"]["provider_id"] == "vision_backbone_stub"
