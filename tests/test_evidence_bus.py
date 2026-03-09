from src.evidence import BeliefState, EvidenceBus, EvidenceRecord, belief_state_from_evidence_bus


def test_evidence_bus_and_belief_state_round_trip() -> None:
    bus = EvidenceBus(
        [
            EvidenceRecord.from_components(
                episode_id="ep_evidence_001",
                timestamp="2026-03-09T00:00:00+00:00",
                source="map_first",
                kind="map_first_semantics",
                confidence=0.82,
                disagreement=0.08,
                metrics={"map_first_quality_score": 0.82},
                artifact_refs={"map_first_path": "artifact://map_first"},
            ),
            EvidenceRecord.from_components(
                episode_id="ep_evidence_001",
                timestamp="2026-03-09T00:00:01+00:00",
                source="openvla",
                kind="teacher_trace",
                confidence=0.41,
                disagreement=0.03,
                metrics={"teacher_confidence_mean": 0.41},
                artifact_refs={"teacher_trace_path": "artifact://teacher_trace"},
            ),
        ]
    )

    payload = bus.to_dict()
    restored_bus = EvidenceBus.from_dict(payload)
    assert restored_bus.to_dict() == payload

    belief_state = belief_state_from_evidence_bus(
        evidence_bus=restored_bus,
        episode_id="ep_evidence_001",
        timestamp="2026-03-09T00:00:02+00:00",
        semantic_tags=["fragile", "drawer"],
        extra_state={"geometry_quality": 0.9},
    )

    assert belief_state.state_vector["evidence_confidence_mean"] > 0.5
    assert belief_state.state_vector["teacher_alignment"] > 0.0
    assert belief_state.state_vector["geometry_quality"] == 0.9
    assert BeliefState.from_dict(belief_state.to_dict()).to_dict() == belief_state.to_dict()
