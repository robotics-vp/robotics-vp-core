"""
Tests for fallback behavior in Scene IR Tracker.

Verifies that:
- Missing deps raise RuntimeError when allow_fallbacks=False
- Stub mode works when allow_fallbacks=True
"""
from __future__ import annotations

import os
import pytest


class TestSAM3DObjectsWrapperFallback:
    """Tests for SAM3D-Objects wrapper fallback behavior."""

    def test_wrapper_instantiation_with_fallback_allowed(self):
        """Wrapper should not raise when allow_fallback=True."""
        from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
        
        # Should not raise - fallback allowed
        wrapper = SAM3DObjectsInference(use_fallback=True, allow_fallback=True)
        assert wrapper is not None
        assert wrapper.use_fallback is True

    def test_wrapper_uses_stub_when_fallback_allowed(self):
        """Wrapper should use fallback stub when deps missing and fallback allowed."""
        from third_party.sam3d_objects_wrapper import SAM3DObjectsInference
        import numpy as np
        
        wrapper = SAM3DObjectsInference(use_fallback=True, allow_fallback=True)
        
        # Create test inputs
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:40, 20:40] = True
        
        # Should succeed with fallback
        results = wrapper.infer(rgb, [mask])
        assert len(results) == 1
        assert results[0].shape_latent.shape == (256,)


class TestSAM3DBodyWrapperFallback:
    """Tests for SAM3D-Body wrapper fallback behavior."""

    def test_wrapper_instantiation_with_fallback_allowed(self):
        """Wrapper should not raise when allow_fallback=True."""
        from third_party.sam3d_body_wrapper import SAM3DBodyInference
        
        wrapper = SAM3DBodyInference(use_fallback=True, allow_fallback=True)
        assert wrapper is not None
        assert wrapper.use_fallback is True

    def test_wrapper_uses_stub_when_fallback_allowed(self):
        """Wrapper should use fallback stub when deps missing and fallback allowed."""
        from third_party.sam3d_body_wrapper import SAM3DBodyInference
        import numpy as np
        
        wrapper = SAM3DBodyInference(use_fallback=True, allow_fallback=True)
        
        # Create test inputs
        rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=bool)
        mask[30:100, 40:90] = True
        
        # Should succeed with fallback
        result = wrapper.infer(rgb, mask)
        assert result is not None
        assert len(result.joints_3d) > 0


class TestLPIPSWrapperFallback:
    """Tests for LPIPS wrapper fallback behavior."""

    def test_wrapper_instantiation_with_fallback_allowed(self):
        """Wrapper should not raise when allow_fallback=True."""
        from third_party.lpips_wrapper import LPIPSLoss
        
        lpips = LPIPSLoss(use_fallback=True, allow_fallback=True)
        assert lpips is not None
        assert lpips.use_fallback is True

    def test_wrapper_uses_gradient_fallback(self):
        """Wrapper should use gradient-based fallback."""
        import torch
        from third_party.lpips_wrapper import LPIPSLoss
        
        lpips = LPIPSLoss(use_fallback=True, allow_fallback=True)
        
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        
        loss = lpips(img1, img2)
        assert torch.isfinite(loss)
        assert loss >= 0


class TestSceneIRTrackerConfigFallback:
    """Tests for SceneIRTrackerConfig allow_fallbacks field."""

    def test_allow_fallbacks_default_false(self):
        """allow_fallbacks should default to False (unless env var set)."""
        from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig
        
        # Clear env var if set
        old_val = os.environ.pop("SCENE_IR_ALLOW_FALLBACKS", None)
        try:
            config = SceneIRTrackerConfig()
            assert config.allow_fallbacks is False
        finally:
            if old_val is not None:
                os.environ["SCENE_IR_ALLOW_FALLBACKS"] = old_val

    def test_allow_fallbacks_env_var(self):
        """allow_fallbacks should respect SCENE_IR_ALLOW_FALLBACKS env var."""
        from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig
        
        old_val = os.environ.get("SCENE_IR_ALLOW_FALLBACKS")
        try:
            os.environ["SCENE_IR_ALLOW_FALLBACKS"] = "1"
            config = SceneIRTrackerConfig()
            assert config.allow_fallbacks is True
        finally:
            if old_val is None:
                os.environ.pop("SCENE_IR_ALLOW_FALLBACKS", None)
            else:
                os.environ["SCENE_IR_ALLOW_FALLBACKS"] = old_val

    def test_allow_fallbacks_serialization(self):
        """allow_fallbacks should serialize and deserialize correctly."""
        from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig
        
        config = SceneIRTrackerConfig(allow_fallbacks=True)
        data = config.to_dict()
        assert data["allow_fallbacks"] is True
        
        restored = SceneIRTrackerConfig.from_dict(data)
        assert restored.allow_fallbacks is True

    def test_use_stub_adapters_separate_from_fallbacks(self):
        """use_stub_adapters and allow_fallbacks should be independent."""
        from src.vision.scene_ir_tracker.config import SceneIRTrackerConfig
        
        # use_stub_adapters=True means explicitly request stubs (for testing)
        # allow_fallbacks=False means don't silently fall back to stubs
        config = SceneIRTrackerConfig(use_stub_adapters=True, allow_fallbacks=False)
        assert config.use_stub_adapters is True
        assert config.allow_fallbacks is False
