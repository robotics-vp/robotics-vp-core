import json

from src.vision.reconstruction import (
    build_four_d_reconstruction_sidecar,
    load_four_d_reconstruction_sidecar,
    save_four_d_reconstruction_sidecar,
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

