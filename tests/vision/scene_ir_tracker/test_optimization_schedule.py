"""Test optimization schedule sanity in IR refiner."""
import numpy as np
import pytest

from src.vision.scene_ir_tracker.config import IRRefinerConfig
from src.vision.scene_ir_tracker.ir_refiner import IRRefiner, IRRefinementResult
from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer, IRRendererConfig
from src.vision.scene_ir_tracker.types import SceneEntity3D


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def refiner():
    """Create test refiner with small iteration counts."""
    config = IRRefinerConfig(
        num_texture_iters=5,
        num_pose_iters=3,
        num_shape_iters=2,
        lr_texture=1e-3,
        lr_pose=1e-4,
        lr_shape=1e-4,
    )
    renderer = IRSceneGraphRenderer(IRRendererConfig(device="cpu"))
    return IRRefiner(config=config, renderer=renderer, device="cpu")


@pytest.fixture
def camera():
    """Create test camera."""
    from src.vision.nag.types import CameraParams
    return CameraParams.from_single_pose(
        position=(0.0, 0.0, -5.0),
        look_at=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=32,
        height=32,
    )


def test_empty_entities_refine(refiner, camera):
    """Refining empty entity list should return empty result."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    target_rgb = torch.rand(3, 32, 32)
    refined, result = refiner.refine([], target_rgb, {}, camera)

    assert len(refined) == 0
    assert result.converged is True
    assert result.final_loss == 0.0


def test_single_entity_refine(refiner, camera):
    """Single entity refinement should produce results."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        z_shape=np.random.randn(64).astype(np.float32),
        z_tex=np.random.randn(64).astype(np.float32),
    )

    target_rgb = torch.rand(3, 32, 32)
    target_masks = {"obj_1": torch.rand(32, 32)}

    refined, result = refiner.refine([entity], target_rgb, target_masks, camera)

    assert len(refined) == 1
    assert refined[0].track_id == "obj_1"
    assert result.iterations_run == 5 + 3 + 2  # texture + pose + shape
    assert len(result.loss_curve) >= 3  # At least one per phase
    assert "texture" in result.per_phase_losses
    assert "pose" in result.per_phase_losses
    assert "shape" in result.per_phase_losses


def test_loss_is_finite(refiner, camera):
    """Losses should be finite throughout optimization."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity = SceneEntity3D(
        entity_type="body",
        track_id="body_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        joints_3d={"pelvis": np.zeros(3, dtype=np.float32)},
    )

    target_rgb = torch.rand(3, 32, 32)
    _, result = refiner.refine([entity], target_rgb, {}, camera)

    for loss in result.loss_curve:
        assert np.isfinite(loss), f"Loss should be finite, got {loss}"

    assert np.isfinite(result.final_loss)


def test_entity_ir_loss_updated(refiner, camera):
    """Entity ir_loss field should be updated after refinement."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        ir_loss=0.0,
    )

    target_rgb = torch.rand(3, 32, 32)
    refined, result = refiner.refine([entity], target_rgb, {}, camera)

    # Entity should have ir_loss set
    assert refined[0].ir_loss >= 0


def test_per_phase_losses_order(refiner, camera):
    """Per-phase losses should be computed in correct order."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
        z_shape=np.zeros(64, dtype=np.float32),
        z_tex=np.zeros(64, dtype=np.float32),
    )

    target_rgb = torch.zeros(3, 32, 32)  # Black target
    _, result = refiner.refine([entity], target_rgb, {}, camera)

    # All phases should have losses
    assert len(result.per_phase_losses) == 3
    for phase in ["texture", "pose", "shape"]:
        assert phase in result.per_phase_losses


def test_refine_sequence(refiner, camera):
    """Test sequence refinement."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    entity = SceneEntity3D(
        entity_type="object",
        track_id="obj_1",
        pose=np.eye(4, dtype=np.float32),
        scale=1.0,
    )

    frames_entities = [[entity], [entity]]
    target_rgb_seq = torch.rand(2, 3, 32, 32)
    target_masks_seq = [{}, {}]

    refined_frames, results = refiner.refine_sequence(
        frames_entities, target_rgb_seq, target_masks_seq, camera
    )

    assert len(refined_frames) == 2
    assert len(results) == 2
    for result in results:
        assert isinstance(result, IRRefinementResult)
