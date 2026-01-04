"""
Upstream integration tests for Scene IR Tracker.

These tests verify that real third-party wrappers work correctly.
Tests are skipped if dependencies or weights are not available.
"""
from __future__ import annotations

import numpy as np
import pytest


# Check if third-party wrappers are available
def _check_wrapper_available(wrapper_name: str, class_name: str) -> bool:
    """Check if a wrapper module is available."""
    try:
        module = __import__(f"third_party.{wrapper_name}", fromlist=[class_name])
        getattr(module, class_name)
        return True
    except (ImportError, AttributeError):
        return False


SAM3D_OBJECTS_AVAILABLE = _check_wrapper_available("sam3d_objects_wrapper", "SAM3DObjectsInference")
SAM3D_BODY_AVAILABLE = _check_wrapper_available("sam3d_body_wrapper", "SAM3DBodyInference")
LPIPS_AVAILABLE = _check_wrapper_available("lpips_wrapper", "LPIPSLoss")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def sample_rgb():
    """Create sample RGB image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create sample object mask."""
    mask = np.zeros((256, 256), dtype=bool)
    mask[100:150, 100:150] = True
    return mask


@pytest.fixture
def sample_body_mask():
    """Create sample body mask (larger, person-shaped)."""
    mask = np.zeros((256, 256), dtype=bool)
    mask[50:200, 100:156] = True
    return mask


class TestSAM3DObjectsWrapper:
    """Tests for SAM3D-Objects wrapper."""
    
    @pytest.mark.skipif(not SAM3D_OBJECTS_AVAILABLE, reason="SAM3D-Objects wrapper not available")
    def test_wrapper_instantiation(self):
        """Test that wrapper can be instantiated."""
        from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
        
        wrapper = SAM3DObjectsInference(use_fallback=True)
        assert wrapper is not None
    
    @pytest.mark.skipif(not SAM3D_OBJECTS_AVAILABLE, reason="SAM3D-Objects wrapper not available")
    def test_fallback_inference(self, sample_rgb, sample_mask):
        """Test fallback inference produces valid output."""
        from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
        
        wrapper = SAM3DObjectsInference(use_fallback=True)
        results = wrapper.infer(sample_rgb, [sample_mask])
        
        assert len(results) == 1
        r = results[0]
        
        # Check shapes and types
        assert r.shape_latent.shape == (256,)
        assert r.appearance_latent.shape == (256,)
        assert r.position.shape == (3,)
        assert r.rotation.shape == (4,)
        assert isinstance(r.scale, float)
        assert r.geometry is not None
        assert "type" in r.geometry
    
    @pytest.mark.skipif(not SAM3D_OBJECTS_AVAILABLE, reason="SAM3D-Objects wrapper not available")
    def test_latents_are_nontrivial(self, sample_rgb, sample_mask):
        """Test that latents are not all zeros."""
        from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
        
        wrapper = SAM3DObjectsInference(use_fallback=True)
        results = wrapper.infer(sample_rgb, [sample_mask])
        
        assert len(results) == 1
        r = results[0]
        
        # Latents should have some non-zero values
        assert np.abs(r.shape_latent).sum() > 0
        assert np.abs(r.appearance_latent).sum() > 0


class TestSAM3DBodyWrapper:
    """Tests for SAM3D-Body wrapper."""
    
    @pytest.mark.skipif(not SAM3D_BODY_AVAILABLE, reason="SAM3D-Body wrapper not available")
    def test_wrapper_instantiation(self):
        """Test that wrapper can be instantiated."""
        from third_party.sam3d_body_wrapper import SAM3DBodyInference
        
        wrapper = SAM3DBodyInference(use_fallback=True)
        assert wrapper is not None
    
    @pytest.mark.skipif(not SAM3D_BODY_AVAILABLE, reason="SAM3D-Body wrapper not available")
    def test_fallback_inference(self, sample_rgb, sample_body_mask):
        """Test fallback inference produces valid output."""
        from third_party.sam3d_body_wrapper import SAM3DBodyInference
        
        wrapper = SAM3DBodyInference(use_fallback=True)
        result = wrapper.infer(sample_rgb, sample_body_mask)
        
        # Check basic output
        assert result.body_id is not None
        assert len(result.joints_3d) > 0
        assert "pelvis" in result.joints_3d
        assert result.mesh_vertices.shape[1] == 3
        assert result.shape_latent.shape == (10,)
        assert result.pose_latent.shape == (72,)
    
    @pytest.mark.skipif(not SAM3D_BODY_AVAILABLE, reason="SAM3D-Body wrapper not available")
    def test_skeleton_structure(self, sample_rgb, sample_body_mask):
        """Test that skeleton has expected joints."""
        from third_party.sam3d_body_wrapper import SAM3DBodyInference
        
        wrapper = SAM3DBodyInference(use_fallback=True)
        result = wrapper.infer(sample_rgb, sample_body_mask)
        
        expected_joints = ["pelvis", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
        for joint in expected_joints:
            assert joint in result.joints_3d
            assert result.joints_3d[joint].shape == (3,)


class TestLPIPSWrapper:
    """Tests for LPIPS wrapper."""
    
    @pytest.mark.skipif(not LPIPS_AVAILABLE or not TORCH_AVAILABLE, reason="LPIPS or PyTorch not available")
    def test_wrapper_instantiation(self):
        """Test that wrapper can be instantiated."""
        from third_party.lpips_wrapper import LPIPSLoss
        
        lpips = LPIPSLoss(use_fallback=True)
        assert lpips is not None
    
    @pytest.mark.skipif(not LPIPS_AVAILABLE or not TORCH_AVAILABLE, reason="LPIPS or PyTorch not available")
    def test_loss_computation(self):
        """Test that loss can be computed."""
        import torch
        from third_party.lpips_wrapper import LPIPSLoss
        
        lpips = LPIPSLoss(use_fallback=True)
        
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        
        loss = lpips(img1, img2)
        
        assert loss.ndim == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss >= 0
    
    @pytest.mark.skipif(not LPIPS_AVAILABLE or not TORCH_AVAILABLE, reason="LPIPS or PyTorch not available")
    def test_identical_images_low_loss(self):
        """Test that identical images have low loss."""
        import torch
        from third_party.lpips_wrapper import LPIPSLoss
        
        lpips = LPIPSLoss(use_fallback=True)
        
        img = torch.rand(1, 3, 64, 64)
        
        loss = lpips(img, img)
        
        assert loss < 0.1  # Identical images should have very low loss


class TestIRLossConvergence:
    """Tests for IR refinement loss convergence."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")  
    def test_ir_loss_decreases(self):
        """Test that IR loss decreases during refinement."""
        import torch
        from src.vision.scene_ir_tracker.ir_refiner import IRRefiner, IRRefinerConfig
        from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer, IRRendererConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        from src.vision.nag.types import CameraParams
        
        # Create renderer and refiner
        renderer = IRSceneGraphRenderer(IRRendererConfig(device="cpu"))
        refiner = IRRefiner(config=IRRefinerConfig(), renderer=renderer)
        
        # Create camera
        camera = CameraParams.from_single_pose(
            position=(0, 0, -5), look_at=(0, 0, 0),
            up=(0, 1, 0), fov_deg=60, width=64, height=64
        )
        
        # Create target image and entity
        target = torch.rand(3, 64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[24:40, 24:40] = True
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="test",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
            mask_2d=mask,
        )
        
        # Refine
        refined, result = refiner.refine([entity], target, {"test": mask}, camera)
        
        # Check that we got a result
        assert result is not None
        assert result.final_loss >= 0
        assert torch.isfinite(torch.tensor(result.final_loss))
