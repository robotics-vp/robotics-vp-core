from src.evidence import EvidenceBus, EvidenceRecord, belief_state_from_evidence_bus
from src.evidence.teacher_trace import TeacherTrace
from src.semantic.runtime_backbone import SemanticRuntimeBackbone
from src.world_model import GovernedVideoWorldModel, SemanticWorldModelBuilder
import numpy as np


def test_semantic_world_model_builder_and_backbone_bridge_stage1_state() -> None:
    evidence_bus = EvidenceBus(
        [
            EvidenceRecord.from_components(
                episode_id="ep_semantic_world_001",
                timestamp="2026-03-22T00:00:00+00:00",
                source="map_first",
                kind="map_first_semantics",
                confidence=0.85,
                disagreement=0.12,
                metrics={"map_first_quality_score": 0.85},
            ),
            EvidenceRecord.from_components(
                episode_id="ep_semantic_world_001",
                timestamp="2026-03-22T00:00:01+00:00",
                source="openvla",
                kind="teacher_trace",
                confidence=0.6,
                disagreement=0.04,
                metrics={"teacher_confidence_mean": 0.6},
            ),
        ]
    )
    belief_state = belief_state_from_evidence_bus(
        evidence_bus=evidence_bus,
        episode_id="ep_semantic_world_001",
        timestamp="2026-03-22T00:00:02+00:00",
        semantic_tags=["drawer", "vase", "fragile", "safety", "error_recovery"],
        extra_state={"geometry_quality": 0.92, "semantic_quality": 0.81},
    )
    video_model = GovernedVideoWorldModel()
    video_snapshot = video_model.build_state_snapshot(
        episode_id="ep_semantic_world_001",
        timestamp="2026-03-22T00:00:03+00:00",
        belief_state=belief_state,
        objective_preset="safety",
        semantic_tags=["drawer", "vase", "fragile", "mode:recovery"],
        media_refs=["artifact://demo.mp4"],
    )
    hypotheses = video_model.propose_hypotheses(
        snapshot=video_snapshot,
        constraint_set={"hard_bounds": {"clearance_m": {"min": 0.05}}},
    )

    scene_tracks_payload = {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": np.array(["drawer_track", "vase_track"], dtype="U32"),
        "scene_tracks_v1/entity_types": np.array([0, 0], dtype=np.int32),
        "scene_tracks_v1/class_ids": np.array([0, 1], dtype=np.int32),
        "scene_tracks_v1/class_names": np.array(["drawer", "vase"], dtype="U32"),
        "scene_tracks_v1/poses_R": np.tile(np.eye(3, dtype=np.float32), (3, 2, 1, 1)),
        "scene_tracks_v1/poses_t": np.array(
            [
                [[0.1, 0.0, 0.5], [0.18, 0.03, 0.5]],
                [[0.12, 0.0, 0.5], [0.18, 0.03, 0.5]],
                [[0.14, 0.0, 0.5], [0.18, 0.03, 0.5]],
            ],
            dtype=np.float32,
        ),
        "scene_tracks_v1/scales": np.ones((3, 2), dtype=np.float32),
        "scene_tracks_v1/visibility": np.ones((3, 2), dtype=np.float32) * 0.95,
        "scene_tracks_v1/occlusion": np.zeros((3, 2), dtype=np.float32),
        "scene_tracks_v1/ir_loss": np.zeros((3, 2), dtype=np.float32) + 0.05,
        "scene_tracks_v1/converged": np.ones((3, 2), dtype=bool),
        "scene_tracks_v1/summary_json": np.array(
            ['{"quality_score": 0.9, "training_eligible": true}'],
            dtype="U256",
        ),
    }
    teacher_trace = TeacherTrace.from_vla_action(
        episode_id="ep_semantic_world_001",
        instruction="Open the drawer without touching the fragile vase.",
        semantic_tags=["drawer", "fragile", "vase"],
        action={"vla_available": True, "confidence": 0.7},
    )
    vla_semantic_evidence = {
        "vla_semantic_evidence_v1/version": np.array(["v1"], dtype="U8"),
        "vla_semantic_evidence_v1/class_probs": np.array(
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.85, 0.15], [0.3, 0.7]],
                [[0.88, 0.12], [0.25, 0.75]],
            ],
            dtype=np.float32,
        ),
        "vla_semantic_evidence_v1/confidence": np.ones((3, 2), dtype=np.float32) * 0.8,
        "vla_semantic_evidence_v1/track_ids": np.array(["drawer_track", "vase_track"], dtype="U32"),
    }

    builder = SemanticWorldModelBuilder()
    semantic_world_model = builder.build_from_stage1(
        video_ref={
            "episode_id": "ep_semantic_world_001",
            "task_type": "drawer_vase",
            "instruction": "Open the drawer without hitting the vase.",
            "metadata": {"success": True, "duration_s": 12.0},
        },
        belief_state=belief_state,
        video_state_snapshot=video_snapshot,
        hypotheses=hypotheses,
        constraint_set={"hard_bounds": {"clearance_m": {"min": 0.05}}},
        objective_preset="safety",
        semantic_tags=["drawer", "vase", "fragile", "safety", "error_recovery"],
        scene_tracks_payload=scene_tracks_payload,
        teacher_trace=teacher_trace,
        vla_semantic_evidence=vla_semantic_evidence,
    )

    object_ids = {item.object_id for item in semantic_world_model.objects}
    meta_node_types = {item.node_type for item in semantic_world_model.meta_nodes}
    relation_types = {item.relation_type for item in semantic_world_model.relations}
    assert {"drawer", "vase", "robot_arm", "gripper", "track:drawer_track", "track:vase_track"} <= object_ids
    assert "risk_triage" in meta_node_types
    assert {"inside", "near"} <= relation_types
    assert semantic_world_model.capability_scores["meta_node_orchestration"] > 0.0
    assert semantic_world_model.topology["grounded_track_object_count"] >= 2
    assert semantic_world_model.metadata["grounded_scene"]["grounding_mode"] == "scene_tracks"

    backbone = SemanticRuntimeBackbone({"write_to_file": False})
    result = backbone.build(
        task_id="drawer_vase",
        objective_preset="safety",
        semantic_world_model=semantic_world_model,
        runtime_metrics={"avg_mpl_units_per_hour": 4.0, "avg_error_rate": 0.1},
        frontier_episodes=["ep_semantic_world_001"],
        metadata={"source_stage": "test"},
        backends=["governed_video"],
    )

    assert result.semantic_snapshot.semantic_world_model is not None
    assert result.semantic_snapshot.metadata["semantic_world_model_summary"]["topology"]["object_count"] >= 4
    assert result.orchestrator_advisory.meta_node_weights["risk_triage"] > 0.0
