from src.diffusion.real_video_diffusion_stub import VideoDiffusionStub


def test_stub_prefers_governed_hypotheses_over_tag_only_fallback() -> None:
    stub = VideoDiffusionStub()
    proposals = stub.propose_augmented_clips(
        episode_id="ep1",
        media_refs=["ref1"],
        semantic_tags=["fragile", "safety"],
        objective_preset="safety",
        routing_context={
            "routing_source": "coverage_gap_graph",
            "benchmark_gate_ready": False,
            "semantic_grounding_mode": "coverage_gap_pending",
            "coverage_gap_score": 0.9,
            "economic_priority_score": 0.8,
            "trust_priority_score": 0.7,
            "risk_family_targets": ["collision"],
            "governed_hypotheses": [
                {
                    "hypothesis_id": "hyp_low",
                    "mode": "geometry_guarded_continuation",
                    "scores": {
                        "render_priority": 0.25,
                        "plausibility": 0.5,
                        "novelty": 0.2,
                        "economic_priority": 0.2,
                        "trust_priority": 0.3,
                    },
                    "render_intent": {"should_render": True},
                },
                {
                    "hypothesis_id": "hyp_high",
                    "mode": "fragile_object_preservation",
                    "scores": {
                        "render_priority": 0.75,
                        "plausibility": 0.82,
                        "novelty": 0.55,
                        "economic_priority": 0.8,
                        "trust_priority": 0.7,
                    },
                    "render_intent": {"should_render": True},
                },
            ],
        },
        num_proposals=1,
    )

    assert len(proposals) == 1
    assert proposals[0].augmentation_type == "fragile_object_preservation"
    assert proposals[0].routing_source == "coverage_gap_graph"
    assert proposals[0].source_hypothesis_id == "hyp_high"
    assert proposals[0].benchmark_gate_ready is False


def test_stub_fallback_clamps_unready_semantic_routing() -> None:
    stub = VideoDiffusionStub()
    proposals = stub.propose_augmented_clips(
        episode_id="ep2",
        media_refs=["ref2"],
        semantic_tags=["energy_efficient"],
        objective_preset="energy_saver",
        routing_context={
            "routing_source": "semantic_tag_fallback",
            "benchmark_gate_ready": False,
            "semantic_grounding_mode": "heuristic_fallback",
            "missing_env_primitives": ["Locate Handle"],
        },
        num_proposals=2,
    )

    assert proposals
    assert all(p.routing_source == "semantic_tag_fallback" for p in proposals)
    assert all(p.benchmark_gate_ready is False for p in proposals)
    assert all(p.confidence <= 0.55 for p in proposals)
