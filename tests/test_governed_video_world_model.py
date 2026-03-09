from src.evidence import EvidenceBus, EvidenceRecord, belief_state_from_evidence_bus
from src.world_model import GovernedVideoWorldModel


def test_governed_video_world_model_emits_ranked_hypotheses() -> None:
    evidence_bus = EvidenceBus(
        [
            EvidenceRecord.from_components(
                episode_id="ep_video_001",
                timestamp="2026-03-09T00:00:00+00:00",
                source="map_first",
                kind="map_first_semantics",
                confidence=0.8,
                disagreement=0.15,
                metrics={"map_first_quality_score": 0.8},
            ),
            EvidenceRecord.from_components(
                episode_id="ep_video_001",
                timestamp="2026-03-09T00:00:01+00:00",
                source="openvla",
                kind="teacher_trace",
                confidence=0.45,
                disagreement=0.02,
                metrics={"teacher_confidence_mean": 0.45},
            ),
        ]
    )
    belief_state = belief_state_from_evidence_bus(
        evidence_bus=evidence_bus,
        episode_id="ep_video_001",
        timestamp="2026-03-09T00:00:02+00:00",
        semantic_tags=["fragile", "drawer", "safety"],
        extra_state={"geometry_quality": 0.9, "semantic_quality": 0.85},
    )

    model = GovernedVideoWorldModel()
    snapshot = model.build_state_snapshot(
        episode_id="ep_video_001",
        timestamp="2026-03-09T00:00:03+00:00",
        belief_state=belief_state,
        objective_preset="safety",
        semantic_tags=["fragile", "drawer"],
        media_refs=["artifact://video.mp4"],
    )
    hypotheses = model.propose_hypotheses(
        snapshot=snapshot,
        constraint_set={"hard_bounds": {"clearance_m": {"min": 0.05}}},
    )

    assert len(snapshot.token_vector) == 128
    assert hypotheses
    assert hypotheses[0].scores["render_priority"] >= hypotheses[-1].scores["render_priority"]
    assert any(hypothesis.mode == "fragile_object_preservation" for hypothesis in hypotheses)
