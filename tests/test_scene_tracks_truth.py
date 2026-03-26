from src.evidence.scene_tracks_truth import normalize_scene_tracks_truth


def test_passthrough_does_not_count_as_non_stub_or_non_heuristic() -> None:
    truth = normalize_scene_tracks_truth(
        backend="passthrough",
        explicit_non_stub=True,
        semantic_grounding_ready=True,
        training_eligible=True,
        explicit_non_heuristic=True,
    )

    assert truth["scene_tracks_non_stub"] is False
    assert truth["semantic_grounding_non_heuristic"] is False
    assert truth["semantic_grounding_ready"] is False
    assert truth["scene_tracks_training_eligible"] is False


def test_real_backend_keeps_grounded_truth_flags() -> None:
    truth = normalize_scene_tracks_truth(
        backend="real",
        explicit_non_stub=True,
        semantic_grounding_ready=True,
        training_eligible=True,
        explicit_non_heuristic=True,
    )

    assert truth["scene_tracks_non_stub"] is True
    assert truth["semantic_grounding_non_heuristic"] is True
    assert truth["semantic_grounding_ready"] is True
    assert truth["scene_tracks_training_eligible"] is True
