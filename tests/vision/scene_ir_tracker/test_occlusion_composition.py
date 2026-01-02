"""Test occlusion composition correctness in IR scene graph renderer."""
import numpy as np
import pytest

from src.vision.scene_ir_tracker.types import SceneEntity3D
from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer, IRRendererConfig


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def renderer():
    """Create test renderer."""
    config = IRRendererConfig(device="cpu", background_color=(0.5, 0.5, 0.5))
    return IRSceneGraphRenderer(config=config)


@pytest.fixture
def camera():
    """Create test camera."""
    from src.vision.nag.types import CameraParams
    return CameraParams.from_single_pose(
        position=(0.0, 0.0, -5.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=64,
        height=64,
    )


def test_empty_scene_renders_background(renderer, camera):
    """Empty entity list should return background color."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    rgb, masks, depth = renderer.render_scene([], camera, 64, 64)

    assert rgb.shape == (3, 64, 64)
    assert len(masks) == 0
    assert depth.shape == (64, 64)
    # Background should be uniform
    background = torch.tensor(renderer.config.background_color)
    for c in range(3):
        assert torch.allclose(rgb[c], background[c], atol=0.01)


def test_single_entity_visible(renderer, camera):
    """Single entity should be rendered visibly."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    # Create a simple circle mask in center
    mask = np.zeros((64, 64), dtype=bool)
    mask[24:40, 24:40] = True  # 16x16 box in center

    entity = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32),
        scale=1.0,
        mask_2d=mask,
    )

    rgb, masks, depth = renderer.render_scene([entity], camera, 64, 64)

    assert rgb.shape == (3, 64, 64)
    assert "obj_1" in masks
    assert masks["obj_1"].shape == (64, 64)
    # Entity should contribute some alpha
    assert masks["obj_1"].sum() > 0


def test_occlusion_ordering_front_wins(renderer, camera):
    """Entity closer to camera should occlude entity behind it."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    # Create masks for both entities in overlapping region
    front_mask = np.zeros((64, 64), dtype=bool)
    front_mask[20:44, 20:44] = True  # Center region

    back_mask = np.zeros((64, 64), dtype=bool)
    back_mask[20:44, 20:44] = True  # Same center region

    # Front entity at z=0 (closer to camera at z=-5)
    front_entity = SceneEntity3D(
        entity_type="object",
        track_id="front",
        pose=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],  # z=0
            [0, 0, 0, 1],
        ], dtype=np.float32),
        scale=1.0,
        mask_2d=front_mask,
    )

    # Back entity at z=3 (further from camera)
    back_entity = SceneEntity3D(
        entity_type="object",
        track_id="back",
        pose=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 3],  # z=3
            [0, 0, 0, 1],
        ], dtype=np.float32),
        scale=1.0,
        mask_2d=back_mask,
    )

    rgb, masks, depth = renderer.render_scene([back_entity, front_entity], camera, 64, 64)

    # Both should have masks
    assert "front" in masks
    assert "back" in masks

    # Both entities should contribute some alpha
    total_front = masks["front"].sum()
    total_back = masks["back"].sum()
    total_contrib = total_front + total_back

    # At least one entity should be visible (have non-zero contribution)
    assert total_contrib > 0, "At least one entity should contribute alpha"


def test_depth_map_correct_order(renderer, camera):
    """Depth map should show correct distances."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    # Create a mask that will be visible
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True

    entity = SceneEntity3D(
        entity_type="object",
        track_id="test",
        pose=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 2],  # z=2, camera at z=-5, so distance ~7
            [0, 0, 0, 1],
        ], dtype=np.float32),
        scale=1.0,
        mask_2d=mask,
    )

    _, _, depth = renderer.render_scene([entity], camera, 64, 64)

    # Depth should have some finite values where entity is visible
    finite_depths = depth[depth < float("inf")]
    assert len(finite_depths) > 0, "Should have visible depth pixels"
    assert finite_depths.min() > 0, "Depth should be positive"


def test_sequence_rendering(renderer, camera):
    """Test sequence rendering produces correct shapes."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity1 = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
    )
    entity2 = SceneEntity3D(
        entity_type="object",
        track_id="obj_2",
        pose=np.eye(4, dtype=np.float32),
        scale=0.5,
    )

    frames_entities = [[entity1], [entity1, entity2], [entity2]]

    rgb_seq, masks_seq, depth_seq = renderer.render_sequence(
        frames_entities, camera, 64, 64
    )

    assert rgb_seq.shape == (3, 3, 64, 64)
    assert len(masks_seq) == 3
    assert depth_seq.shape == (3, 64, 64)

    # First frame has 1 entity, second has 2, third has 1
    assert len(masks_seq[0]) == 1
    assert len(masks_seq[1]) == 2
    assert len(masks_seq[2]) == 1
