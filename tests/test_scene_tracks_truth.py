from src.evidence.scene_tracks_truth import (
    normalize_scene_tracks_truth,
    resolve_scene_tracks_backend,
    scene_tracks_truth_from_metadata,
)


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


def test_resolve_backend_prefers_runner_selected_passthrough() -> None:
    backend = resolve_scene_tracks_backend(
        {
            "scene_tracks_metadata": {
                "runner": {
                    "run_config": {
                        "backend_selected": "passthrough",
                        "use_stub_adapters": False,
                        "zero_inference_passthrough": True,
                    }
                }
            }
        }
    )

    assert backend == "passthrough"


def test_unknown_artifact_presence_does_not_count_as_real_without_truth_flags() -> None:
    truth = scene_tracks_truth_from_metadata(
        {
            "scene_tracks_path": "/tmp/scene_tracks_v1.npz",
        }
    )

    assert truth["scene_tracks_backend"] == "artifact_present_unknown"
    assert truth["scene_tracks_non_stub"] is False
    assert truth["semantic_grounding_non_heuristic"] is False
