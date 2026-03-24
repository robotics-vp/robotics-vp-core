from __future__ import annotations

import numpy as np


def test_zero_inference_passthrough_reconstructs_object_without_depth() -> None:
    from src.vision.nag.types import CameraParams
    from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig

    tracker = SceneIRTracker(
        SceneIRTrackerConfig(
            device="cpu",
            use_stub_adapters=False,
            zero_inference_passthrough=True,
        )
    )
    camera = CameraParams.from_single_pose(
        position=(0.0, -1.5, 1.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=60.0,
        width=32,
        height=32,
        camera_id="front",
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 12:22] = True

    result = tracker.process_episode(
        frames=[frame],
        instance_masks=[{"1": mask}],
        camera=camera,
        class_labels=[{"1": "bin"}],
        object_refs=[{"1": "bin_a"}],
    )

    assert result.num_frames == 1
    assert len(result.frames[0]) == 1
    entity = result.frames[0][0]
    assert entity.entity_type == "object"
    assert entity.class_name == "bin"
    assert entity.source_object_id == "bin_a"
    assert entity.label_source == "explicit_segmentation_map"
    assert np.isfinite(entity.position).all()
    assert entity.scale > 0.0
    assert result.config_used["output_frame"] == "world"
    assert tracker.adapter_status()["overall_mode"] == "passthrough"


def test_zero_inference_passthrough_uses_depth_when_available() -> None:
    from src.vision.nag.types import CameraParams
    from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig

    tracker = SceneIRTracker(
        SceneIRTrackerConfig(
            device="cpu",
            use_stub_adapters=False,
            zero_inference_passthrough=True,
        )
    )
    camera = CameraParams.from_single_pose(
        position=(0.0, 0.0, 0.0),
        look_at=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=16,
        height=16,
        camera_id="front",
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 5:9] = True
    depth = np.zeros((16, 16), dtype=np.float32)
    depth[mask] = 2.0

    result = tracker.process_episode(
        frames=[frame],
        instance_masks=[{"7": mask}],
        camera=camera,
        class_labels=[{"7": "tray"}],
        object_refs=[{"7": "tray_1"}],
        depth_frames=[depth],
    )

    entity = result.frames[0][0]
    assert np.isfinite(entity.position).all()
    assert abs(float(entity.position[2]) - 2.0) < 0.5
    assert entity.source_object_id == "tray_1"
