from src.evidence.scene_tracks_truth import (
    build_scene_tracks_provider_truth,
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


def test_build_scene_tracks_provider_truth_marks_grounding_class() -> None:
    provider_truth = build_scene_tracks_provider_truth(
        {
            "scene_tracks_backend": "real",
            "camera_name": "front",
            "scene_tracks_non_stub": True,
            "semantic_grounding_non_heuristic": True,
            "semantic_grounding_ready": True,
            "scene_ir_quality": 0.8,
            "scene_tracks_training_eligible": True,
        }
    )

    assert provider_truth["authority_class"] == "canonical_metadata"
    assert provider_truth["grounding_class"] == "non_heuristic_grounded"
    assert provider_truth["calibration_class"] == "camera_params_present"


def test_explicit_provider_truth_controls_scene_tracks_truth() -> None:
    truth = scene_tracks_truth_from_metadata(
        {
            "scene_tracks_provider_truth": {
                "provider_id": "scene_tracks",
                "provider_kind": "scene_tracks_runtime",
                "provider_name": "scene_tracks",
                "advisory_only": True,
                "available": True,
                "backend_selected": "passthrough",
                "fallback_mode": "passthrough",
                "availability_class": "passthrough_backend",
                "calibration_class": "camera_params_present",
                "grounding_class": "passthrough",
                "confidence": 0.2,
                "authority_class": "canonical_metadata",
                "decision_scope": "external_provider_status",
                "reward_math_mutation": False,
            }
        }
    )

    assert truth["scene_tracks_backend"] == "passthrough"
    assert truth["scene_tracks_non_stub"] is False
    assert truth["semantic_grounding_non_heuristic"] is False
