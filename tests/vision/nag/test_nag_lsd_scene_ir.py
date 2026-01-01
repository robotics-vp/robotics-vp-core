"""
Tests for NAG-LSD integration with Scene IR Tracker.
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


class TestNAGFromLSDConfigSceneIR:
    """Tests for scene_ir config fields in NAGFromLSDConfig."""

    def test_config_has_scene_ir_fields(self):
        """Config has scene_ir fields."""
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig
        
        config = NAGFromLSDConfig()
        assert hasattr(config, "enable_scene_ir_filter")
        assert hasattr(config, "scene_ir_max_mean_loss")
        assert hasattr(config, "scene_ir_min_quality_score")
        assert hasattr(config, "scene_ir_max_id_switch_rate")
        assert hasattr(config, "scene_ir_max_occlusion_rate")

    def test_config_defaults(self):
        """Config has sensible defaults."""
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig
        
        config = NAGFromLSDConfig()
        assert config.enable_scene_ir_filter is False
        assert config.scene_ir_max_mean_loss == 0.5
        assert config.scene_ir_min_quality_score == 0.3


class TestNAGDatapackSceneIR:
    """Tests for scene_ir fields in NAGDatapack."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_datapack_has_scene_ir_fields(self):
        """Datapack has scene_ir fields."""
        from src.vision.nag.integration_lsd_backend import NAGDatapack
        
        datapack = NAGDatapack(
            base_episode_id="test",
            counterfactual_id="test_cf0",
            frames=np.zeros((10, 3, 64, 64), dtype=np.float32),
        )
        
        assert hasattr(datapack, "scene_ir_summary")
        assert hasattr(datapack, "scene_ir_quality_score")
        assert hasattr(datapack, "scene_ir_flags")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_datapack_to_dict_includes_scene_ir(self):
        """Datapack serialization includes scene_ir fields."""
        from src.vision.nag.integration_lsd_backend import NAGDatapack
        
        datapack = NAGDatapack(
            base_episode_id="test",
            counterfactual_id="test_cf0",
            frames=np.zeros((10, 3, 64, 64), dtype=np.float32),
            scene_ir_summary={"ir_loss_mean": 0.1},
            scene_ir_quality_score=0.8,
            scene_ir_flags={"is_plausible": True, "reason": "pass"},
        )
        
        payload = datapack.to_dict()
        
        assert "scene_ir_summary" in payload
        assert payload["scene_ir_summary"]["ir_loss_mean"] == 0.1
        assert "scene_ir_quality_score" in payload
        assert payload["scene_ir_quality_score"] == 0.8
        assert "scene_ir_flags" in payload
        assert payload["scene_ir_flags"]["is_plausible"] is True


class TestSceneIRFlagsComputation:
    """Tests for scene_ir flags computation logic."""

    def test_flags_pass_when_within_thresholds(self):
        """Flags show plausible when within thresholds."""
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig
        
        config = NAGFromLSDConfig(
            enable_scene_ir_filter=True,
            scene_ir_max_mean_loss=0.5,
            scene_ir_min_quality_score=0.3,
            scene_ir_max_id_switch_rate=10.0,
            scene_ir_max_occlusion_rate=0.5,
        )
        
        # Simulate threshold check
        mean_loss = 0.2
        quality = 0.7
        id_switch_rate = 5.0
        occlusion_rate = 0.3
        
        is_plausible = (
            mean_loss <= config.scene_ir_max_mean_loss and
            quality >= config.scene_ir_min_quality_score and
            id_switch_rate <= config.scene_ir_max_id_switch_rate and
            occlusion_rate <= config.scene_ir_max_occlusion_rate
        )
        
        assert is_plausible is True

    def test_flags_fail_when_loss_too_high(self):
        """Flags show not plausible when loss too high."""
        from src.vision.nag.integration_lsd_backend import NAGFromLSDConfig
        
        config = NAGFromLSDConfig(
            enable_scene_ir_filter=True,
            scene_ir_max_mean_loss=0.5,
        )
        
        mean_loss = 0.8  # Above threshold
        quality = 0.7
        id_switch_rate = 5.0
        occlusion_rate = 0.3
        
        is_plausible = (
            mean_loss <= config.scene_ir_max_mean_loss and
            quality >= config.scene_ir_min_quality_score and
            id_switch_rate <= config.scene_ir_max_id_switch_rate and
            occlusion_rate <= config.scene_ir_max_occlusion_rate
        )
        
        assert is_plausible is False
