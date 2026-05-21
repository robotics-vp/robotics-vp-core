import json

from src.vision.reconstruction import (
    build_four_d_reconstruction_sidecar,
    build_reconstruction_grounding_report,
    load_four_d_reconstruction_sidecar,
    load_reconstruction_grounding_report,
    save_four_d_reconstruction_sidecar,
    save_reconstruction_grounding_report,
)


def test_four_d_reconstruction_sidecar_round_trip(tmp_path) -> None:
    sidecar = build_four_d_reconstruction_sidecar(
        episode_id="ep_recon_001",
        source_type="video_manifest",
        media_refs=["/tmp/video.mp4", "/tmp/depth.npy"],
        sensor_bundle_meta={
            "cameras": ["front"],
            "intrinsics": {"front": "intrinsics://front"},
            "extrinsics": {"front": "extrinsics://front"},
            "depth_unit": "mm",
        },
        frame_count=120,
        frame_range=[0, 119],
        geometry_refs={"video_state_path": "/tmp/video_state.json"},
        evidence_refs={"belief_state_path": "/tmp/belief_state.json"},
        quality={"geometry_quality": 0.82},
        metadata={"task_type": "drawer_vase"},
    )

    path = tmp_path / "reconstruction.json"
    save_four_d_reconstruction_sidecar(path, sidecar)
    loaded = load_four_d_reconstruction_sidecar(path)

    assert loaded.version == "four_d_reconstruction_sidecar_v1"
    assert loaded.calibrations[0].calibrated is True
    assert loaded.quality["calibration_score"] == 1.0
    assert loaded.quality["grounding_completeness"] > 0.0
    payload = json.loads(path.read_text())
    assert payload["episode_id"] == "ep_recon_001"


def test_reconstruction_grounding_report_classifies_calibration_and_training(
    tmp_path,
) -> None:
    sidecar = build_four_d_reconstruction_sidecar(
        episode_id="ep_recon_grounded",
        source_type="video_manifest",
        media_refs=["/tmp/video.mp4"],
        sensor_bundle_meta={
            "cameras": ["front"],
            "intrinsics": {"front": "intrinsics://front"},
            "extrinsics": {"front": "extrinsics://front"},
        },
        frame_count=8,
        geometry_refs={
            "video_state_path": "/tmp/video_state.json",
            "hypotheses_path": "/tmp/hypotheses.json",
            "scene_tracks_path": "/tmp/scene_tracks.npz",
        },
        evidence_refs={
            "belief_state_path": "/tmp/belief.json",
            "evidence_bus_path": "/tmp/evidence.json",
        },
    )
    report = build_reconstruction_grounding_report(
        sidecar=sidecar,
        sidecar_path=tmp_path / "reconstruction.json",
        scene_tracks_backend="real",
        semantic_grounding_mode="non_heuristic",
        vision_backbone_selected="real",
    )

    path = tmp_path / "grounding_report.json"
    save_reconstruction_grounding_report(path, report)
    loaded = load_reconstruction_grounding_report(path)

    assert loaded.version == "reconstruction_grounding_report_v1"
    assert loaded.calibration_class == "camera_calibrated"
    assert loaded.grounding_class == "real_scene_tracks_joined"
    assert loaded.training_eligible is True
    assert loaded.benchmark_ready is True


def test_reconstruction_grounding_report_marks_missing_calibration() -> None:
    sidecar = build_four_d_reconstruction_sidecar(
        episode_id="ep_recon_missing_calibration",
        source_type="video_manifest",
        sensor_bundle_meta={"cameras": ["front"], "intrinsics": {}, "extrinsics": {}},
        geometry_refs={
            "video_state_path": "/tmp/video_state.json",
            "hypotheses_path": "/tmp/hypotheses.json",
        },
        evidence_refs={"belief_state_path": "/tmp/belief.json"},
    )
    report = build_reconstruction_grounding_report(
        sidecar=sidecar,
        scene_tracks_backend="unavailable",
        semantic_grounding_mode="heuristic_fallback",
        vision_backbone_selected="unavailable",
    )

    assert report.calibration_class == "camera_missing"
    assert report.training_eligible is False
    assert "scene_tracks_path" in report.missing_refs
