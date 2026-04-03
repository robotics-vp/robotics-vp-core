from __future__ import annotations

import numpy as np

from src.evidence.belief_state import BeliefState
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.interfaces import VisionFrame
from src.world_model.perception_grounding import (
    compile_perception_grounding_world_state,
)
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
