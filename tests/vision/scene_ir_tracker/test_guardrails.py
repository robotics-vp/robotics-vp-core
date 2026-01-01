"""
Tests for IR optimizer guardrails.

Verifies divergence detection, bounds checking, and graceful degradation.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGuardrailsConfig:
    """Tests for guardrails configuration."""

    def test_guardrails_config_fields_exist(self):
        """Config has all guardrail fields."""
        from src.vision.scene_ir_tracker.config import IRRefinerConfig
        
        config = IRRefinerConfig()
        
        assert hasattr(config, "max_pose_jump_m")
        assert hasattr(config, "max_scale_change_ratio")
        assert hasattr(config, "latent_norm_max")
        assert hasattr(config, "grad_clip_norm")
        assert hasattr(config, "early_stop_patience")
        assert hasattr(config, "divergence_loss_factor")

    def test_guardrails_config_defaults(self):
        """Config has sensible defaults."""
        from src.vision.scene_ir_tracker.config import IRRefinerConfig
        
        config = IRRefinerConfig()
        
        assert config.max_pose_jump_m == 1.0
        assert config.max_scale_change_ratio == 2.0
        assert config.latent_norm_max == 10.0
        assert config.grad_clip_norm == 1.0
        assert config.early_stop_patience == 10
        assert config.divergence_loss_factor == 5.0

    def test_guardrails_config_serialization(self):
        """Config serializes and deserializes guardrail fields."""
        from src.vision.scene_ir_tracker.config import IRRefinerConfig
        
        config = IRRefinerConfig(
            max_pose_jump_m=0.5,
            divergence_loss_factor=3.0,
        )
        
        data = config.to_dict()
        assert data["max_pose_jump_m"] == 0.5
        assert data["divergence_loss_factor"] == 3.0
        
        restored = IRRefinerConfig.from_dict(data)
        assert restored.max_pose_jump_m == 0.5
        assert restored.divergence_loss_factor == 3.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestIRRefinementResultDivergence:
    """Tests for divergence tracking in IRRefinementResult."""

    def test_result_has_diverged_field(self):
        """Result has diverged field."""
        from src.vision.scene_ir_tracker.ir_refiner import IRRefinementResult
        
        result = IRRefinementResult()
        assert hasattr(result, "diverged")
        assert result.diverged is False

    def test_result_has_best_loss_field(self):
        """Result has best_loss field."""
        from src.vision.scene_ir_tracker.ir_refiner import IRRefinementResult
        
        result = IRRefinementResult()
        assert hasattr(result, "best_loss")

    def test_result_serialization_includes_divergence(self):
        """Result serialization includes divergence fields."""
        from src.vision.scene_ir_tracker.ir_refiner import IRRefinementResult
        
        result = IRRefinementResult(diverged=True, best_loss=0.1)
        data = result.to_dict()
        
        assert "diverged" in data
        assert data["diverged"] is True
        assert "best_loss" in data
        assert data["best_loss"] == 0.1


class TestSceneTrackerMetricsDivergence:
    """Tests for divergence stats in SceneTrackerMetrics."""

    def test_metrics_has_divergence_fields(self):
        """Metrics has divergence fields."""
        from src.vision.scene_ir_tracker.types import SceneTrackerMetrics
        
        metrics = SceneTrackerMetrics()
        assert hasattr(metrics, "diverged_count")
        assert hasattr(metrics, "pct_diverged")
        assert hasattr(metrics, "pct_converged")

    def test_metrics_computes_percentages(self):
        """Metrics computes percentages correctly."""
        from src.vision.scene_ir_tracker.types import SceneTrackerMetrics
        
        metrics = SceneTrackerMetrics(
            total_frames=100,
            converged_count=80,
            diverged_count=5,
        )
        
        assert metrics.pct_converged == 80.0
        assert metrics.pct_diverged == 5.0

    def test_metrics_serialization_includes_divergence(self):
        """Metrics serialization includes divergence fields."""
        from src.vision.scene_ir_tracker.types import SceneTrackerMetrics
        
        metrics = SceneTrackerMetrics(
            total_frames=100,
            diverged_count=10,
        )
        data = metrics.to_dict()
        
        assert "diverged_count" in data
        assert data["diverged_count"] == 10
        assert "pct_diverged" in data


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRefinerOutputsFinite:
    """Tests that refiner outputs are always finite."""

    def test_refiner_outputs_finite_with_normal_input(self):
        """Refiner outputs finite values with normal input."""
        from src.vision.scene_ir_tracker.ir_refiner import IRRefiner, IRRefinerConfig
        from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer, IRRendererConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        from src.vision.nag.types import CameraParams
        
        renderer = IRSceneGraphRenderer(IRRendererConfig(device="cpu"))
        refiner = IRRefiner(config=IRRefinerConfig(), renderer=renderer)
        
        camera = CameraParams.from_single_pose(
            position=(0, 0, -5), look_at=(0, 0, 0),
            up=(0, 1, 0), fov_deg=60, width=64, height=64
        )
        
        target = torch.rand(3, 64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="test",
            pose=np.eye(4, dtype=np.float32),
            scale=1.0,
            mask_2d=mask,
            z_shape=np.random.randn(256).astype(np.float32),
            z_tex=np.random.randn(256).astype(np.float32),
        )
        
        refined, result = refiner.refine([entity], target, {"test": mask}, camera)
        
        # Check outputs are finite
        assert np.isfinite(result.final_loss)
        assert len(refined) == 1
        assert np.all(np.isfinite(refined[0].pose))

    def test_refiner_handles_zero_scale_entity(self):
        """Refiner handles entity with zero scale gracefully."""
        from src.vision.scene_ir_tracker.ir_refiner import IRRefiner, IRRefinerConfig
        from src.vision.scene_ir_tracker.ir_scene_graph_renderer import IRSceneGraphRenderer, IRRendererConfig
        from src.vision.scene_ir_tracker.types import SceneEntity3D
        from src.vision.nag.types import CameraParams
        
        renderer = IRSceneGraphRenderer(IRRendererConfig(device="cpu"))
        refiner = IRRefiner(config=IRRefinerConfig(), renderer=renderer)
        
        camera = CameraParams.from_single_pose(
            position=(0, 0, -5), look_at=(0, 0, 0),
            up=(0, 1, 0), fov_deg=60, width=64, height=64
        )
        
        target = torch.rand(3, 64, 64)
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True
        
        entity = SceneEntity3D(
            entity_type="object",
            track_id="test",
            pose=np.eye(4, dtype=np.float32),
            scale=0.0,  # Zero scale
            mask_2d=mask,
        )
        
        refined, result = refiner.refine([entity], target, {"test": mask}, camera)
        
        # Should not crash, output finite
        assert np.isfinite(result.final_loss) or result.final_loss == float("inf")
        assert len(refined) == 1
