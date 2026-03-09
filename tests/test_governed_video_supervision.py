from src.evidence import EvidenceBus, EvidenceRecord, belief_state_from_evidence_bus
from src.world_model import GovernedVideoWorldModel
from src.world_model.governed_video_supervision import build_governed_video_supervision_bundle


def test_governed_video_supervision_bundle_emits_runtime_and_value_artifacts() -> None:
    evidence_bus = EvidenceBus(
        [
            EvidenceRecord.from_components(
                episode_id="ep_supervision_001",
                timestamp="2026-03-09T00:00:00+00:00",
                source="map_first",
                kind="map_first_semantics",
                confidence=0.85,
                disagreement=0.08,
                metrics={"map_first_quality_score": 0.9},
            ),
            EvidenceRecord.from_components(
                episode_id="ep_supervision_001",
                timestamp="2026-03-09T00:00:01+00:00",
                source="teacher_runtime",
                kind="teacher_action",
                confidence=0.25,
                disagreement=0.0,
                metrics={"teacher_confidence_mean": 0.25},
            ),
        ]
    )
    belief_state = belief_state_from_evidence_bus(
        evidence_bus=evidence_bus,
        episode_id="ep_supervision_001",
        timestamp="2026-03-09T00:00:02+00:00",
        semantic_tags=["drawer", "fragile", "safety"],
        extra_state={"geometry_quality": 0.9, "semantic_quality": 0.82},
    )
    model = GovernedVideoWorldModel()
    snapshot = model.build_state_snapshot(
        episode_id="ep_supervision_001",
        timestamp="2026-03-09T00:00:03+00:00",
        belief_state=belief_state,
        objective_preset="safety",
        semantic_tags=["drawer", "fragile", "safety"],
        media_refs=["artifact://video.mp4"],
    )
    hypotheses = model.propose_hypotheses(
        snapshot=snapshot,
        constraint_set={"hard_bounds": {"clearance_m": {"min": 0.05}}},
    )

    bundle = build_governed_video_supervision_bundle(
        run_id="stage1_governed_ep_supervision_001",
        video_ref={
            "episode_id": "ep_supervision_001",
            "task_type": "drawer_vase",
            "source_type": "video_manifest",
            "metadata": {"duration_s": 15.0},
        },
        semantic_tags=["drawer", "fragile", "safety"],
        belief_state=belief_state,
        snapshot=snapshot,
        hypotheses=hypotheses,
        objective_preset="safety",
        constraint_set={"hard_bounds": {"clearance_m": {"min": 0.05}}},
        sidecar_refs={"belief_state_path": "/tmp/belief_state.json"},
        timestamp="2026-03-09T00:00:04+00:00",
    )

    assert bundle.runtime_packet.packet_id
    assert bundle.branch_evaluations
    assert len(bundle.events) == len(bundle.branch_evaluations)
    assert len(bundle.decisions) == len(bundle.branch_evaluations)
    assert len(bundle.governance_traces) == len(bundle.branch_evaluations)
    assert bundle.counterfactual_eval.recommended_action
    assert bundle.value_target_pack.targets
    assert bundle.value_ledger_receipt.ledger_event_id
    assert bundle.value_ledger_receipt.receipt_hash
