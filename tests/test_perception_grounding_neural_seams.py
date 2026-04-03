"""Tests for Perception / Grounding WM neural seams and seam registry.

Tests cover:
- Forward pass correctness for all seam types
- Shape handling (batched and unbatched)
- Parameter counts within expected bounds
- Seam registry operations (register, load, save, unload)
- Promotion resolver integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.world_model.perception_grounding import (
    DepthMetricCalibrationSeam,
    EvidenceFusionSeam,
    PerceptionSeamRegistry,
    SAMCalibrationSeam,
    SeamDescriptor,
    VisionBackboneProjectionSeam,
    VJEPATemporalAlignmentSeam,
    create_default_registry,
    encode_provider_features,
    resolve_provider_adapter_helper,
)


# ---------------------------------------------------------------------------
# EvidenceFusionSeam tests
# ---------------------------------------------------------------------------


class TestEvidenceFusionSeam:
    def test_forward_unbatched(self):
        seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
        provider_features = encode_provider_features(
            provider_ids=["sam", "dinov2", "depth"],
            provider_kinds={"sam": "scene_tracks", "dinov2": "vision_backbone", "depth": "scene_tracks"},
            provider_availability={"sam": "available", "dinov2": "available", "depth": "unavailable"},
            provider_truth_class={"sam": "provider_backed", "dinov2": "provider_backed", "depth": "unavailable"},
        )
        weights, confidence = seam(provider_features)
        assert weights.shape == (3,)
        assert confidence.shape == ()
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
        assert 0.0 <= confidence.item() <= 1.0

    def test_forward_batched(self):
        seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
        batch_size = 4
        n_providers = 3
        features = torch.randn(batch_size, n_providers, seam.D_PROVIDER_RAW)
        weights, confidence = seam(features)
        assert weights.shape == (batch_size, n_providers)
        assert confidence.shape == (batch_size,)

    def test_param_count(self):
        seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
        param_count = seam.param_count()
        assert 5_000 < param_count < 100_000

    def test_describe(self):
        seam = EvidenceFusionSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "evidence_fusion"
        assert "param_count" in desc


# ---------------------------------------------------------------------------
# SAMCalibrationSeam tests
# ---------------------------------------------------------------------------


class TestSAMCalibrationSeam:
    def test_forward_unbatched(self):
        seam = SAMCalibrationSeam(d_mask=256, d_model=128, n_heads=4, d_ff=256)
        n_masks = 8
        mask_features = torch.randn(n_masks, 256)
        raw_confidence = torch.rand(n_masks)
        result = seam(mask_features, raw_confidence)
        assert "calibrated_confidence" in result
        assert "epistemic_uncertainty" in result
        assert "prompt_satisfaction" in result
        assert result["calibrated_confidence"].shape == (n_masks,)
        assert result["calibrated_confidence"].min().item() >= 0.0
        assert result["calibrated_confidence"].max().item() <= 1.0

    def test_forward_batched(self):
        seam = SAMCalibrationSeam(d_mask=256, d_model=128, n_heads=4, d_ff=256)
        batch_size = 2
        n_masks = 8
        mask_features = torch.randn(batch_size, n_masks, 256)
        raw_confidence = torch.rand(batch_size, n_masks)
        mask_valid = torch.ones(batch_size, n_masks, dtype=torch.bool)
        mask_valid[:, -2:] = False  # Last 2 masks invalid
        result = seam(mask_features, raw_confidence, mask_valid)
        assert result["calibrated_confidence"].shape == (batch_size, n_masks)

    def test_param_count_range(self):
        seam = SAMCalibrationSeam(d_mask=256, d_model=128, n_heads=4, d_ff=256)
        param_count = seam.param_count()
        # Expected: 500K-2M params
        assert 100_000 < param_count < 3_000_000

    def test_describe(self):
        seam = SAMCalibrationSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "sam_calibration"


# ---------------------------------------------------------------------------
# VisionBackboneProjectionSeam tests
# ---------------------------------------------------------------------------


class TestVisionBackboneProjectionSeam:
    def test_forward_unbatched(self):
        seam = VisionBackboneProjectionSeam(d_backbone=1024, d_hidden=512, d_out=128)
        n_tokens = 196  # 14x14 patches
        backbone_features = torch.randn(n_tokens, 1024)
        projected = seam(backbone_features)
        assert projected.shape == (n_tokens, 128)

    def test_forward_batched(self):
        seam = VisionBackboneProjectionSeam(d_backbone=1024, d_hidden=512, d_out=128)
        batch_size = 4
        n_tokens = 196
        backbone_features = torch.randn(batch_size, n_tokens, 1024)
        projected = seam(backbone_features)
        assert projected.shape == (batch_size, n_tokens, 128)

    def test_param_count_range(self):
        seam = VisionBackboneProjectionSeam(d_backbone=1024, d_hidden=512, d_out=128)
        param_count = seam.param_count()
        # Expected: ~1-5M params (1024*512 + 512*128 + norms ≈ 590K)
        assert 500_000 < param_count < 6_000_000

    def test_describe(self):
        seam = VisionBackboneProjectionSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "vision_backbone_projection"
        assert desc["d_backbone"] == 1024


# ---------------------------------------------------------------------------
# DepthMetricCalibrationSeam tests
# ---------------------------------------------------------------------------


class TestDepthMetricCalibrationSeam:
    def test_forward_unbatched(self):
        seam = DepthMetricCalibrationSeam(d_depth=1, d_hidden=128)
        depth_map = torch.rand(1, 64, 64)  # (C, H, W)
        camera_intrinsics = torch.tensor([500.0, 500.0, 32.0, 32.0])  # fx, fy, cx, cy
        result = seam(depth_map, camera_intrinsics)
        assert "metric_depth" in result
        assert "scale" in result
        assert "shift" in result
        assert "uncertainty" in result
        assert result["metric_depth"].shape == (1, 64, 64)
        assert result["uncertainty"].shape == (1, 64, 64)

    def test_forward_batched(self):
        seam = DepthMetricCalibrationSeam(d_depth=1, d_hidden=128)
        batch_size = 4
        depth_map = torch.rand(batch_size, 1, 64, 64)
        camera_intrinsics = torch.tensor([[500.0, 500.0, 32.0, 32.0]] * batch_size)
        result = seam(depth_map, camera_intrinsics)
        assert result["metric_depth"].shape == (batch_size, 1, 64, 64)
        assert result["scale"].shape == (batch_size,)

    def test_forward_default_intrinsics(self):
        seam = DepthMetricCalibrationSeam()
        depth_map = torch.rand(1, 64, 64)
        result = seam(depth_map)  # No intrinsics
        assert result["metric_depth"].shape == (1, 64, 64)

    def test_param_count_range(self):
        seam = DepthMetricCalibrationSeam(d_depth=1, d_hidden=128)
        param_count = seam.param_count()
        # Expected: 500K-1M params
        assert 50_000 < param_count < 1_500_000

    def test_describe(self):
        seam = DepthMetricCalibrationSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "depth_metric_calibration"


# ---------------------------------------------------------------------------
# VJEPATemporalAlignmentSeam tests
# ---------------------------------------------------------------------------


class TestVJEPATemporalAlignmentSeam:
    def test_forward_unbatched(self):
        seam = VJEPATemporalAlignmentSeam(
            d_vjepa=1024, d_wm_token=128, d_model=256, d_out=128, n_heads=8
        )
        T = 4  # temporal steps
        n_vjepa = 196  # V-JEPA tokens
        n_obj = 10  # WM objects
        vjepa_tokens = torch.randn(T, n_vjepa, 1024)
        wm_object_tokens = torch.randn(n_obj, 128)
        result = seam(vjepa_tokens, wm_object_tokens)
        assert "temporal_aligned" in result
        assert "temporal_confidence" in result
        assert result["temporal_aligned"].shape == (T, n_obj, 128)
        assert result["temporal_confidence"].shape == (T,)

    def test_forward_batched(self):
        seam = VJEPATemporalAlignmentSeam(
            d_vjepa=1024, d_wm_token=128, d_model=256, d_out=128, n_heads=8
        )
        batch_size = 2
        T = 4
        n_vjepa = 196
        n_obj = 10
        vjepa_tokens = torch.randn(batch_size, T, n_vjepa, 1024)
        wm_object_tokens = torch.randn(batch_size, n_obj, 128)
        result = seam(vjepa_tokens, wm_object_tokens)
        assert result["temporal_aligned"].shape == (batch_size, T, n_obj, 128)
        assert result["temporal_confidence"].shape == (batch_size, T)

    def test_param_count_range(self):
        seam = VJEPATemporalAlignmentSeam(
            d_vjepa=1024, d_wm_token=128, d_model=256, d_out=128, n_heads=8
        )
        param_count = seam.param_count()
        # Expected: 2-5M params
        assert 1_000_000 < param_count < 10_000_000

    def test_describe(self):
        seam = VJEPATemporalAlignmentSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "vjepa_temporal_alignment"
        assert desc["n_temporal_steps"] == 4


# ---------------------------------------------------------------------------
# PerceptionSeamRegistry tests
# ---------------------------------------------------------------------------


class TestPerceptionSeamRegistry:
    def test_register_seam(self):
        registry = PerceptionSeamRegistry()
        desc = registry.register_seam(
            "evidence_fusion", "test_fusion", posture="auto"
        )
        assert desc.seam_type == "evidence_fusion"
        assert desc.seam_id == "test_fusion"
        assert desc.posture == "auto"
        assert desc.promotion_stage == "registered"
        assert not desc.loaded

    def test_load_seam(self):
        registry = PerceptionSeamRegistry()
        registry.register_seam("evidence_fusion", "test_fusion")
        seam = registry.load_seam("test_fusion")
        assert isinstance(seam, EvidenceFusionSeam)
        desc = registry.get_descriptor("test_fusion")
        assert desc.loaded
        assert desc.param_count > 0

    def test_get_seam(self):
        registry = PerceptionSeamRegistry()
        registry.register_seam("evidence_fusion", "test_fusion")
        assert registry.get_seam("test_fusion") is None
        registry.load_seam("test_fusion")
        assert registry.get_seam("test_fusion") is not None

    def test_save_and_reload_seam(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PerceptionSeamRegistry(checkpoint_dir=tmpdir)
            registry.register_seam("evidence_fusion", "test_fusion")
            seam = registry.load_seam("test_fusion")

            # Modify weights
            with torch.no_grad():
                seam.input_proj.weight.fill_(0.123)

            # Save
            save_path = registry.save_seam("test_fusion")
            assert save_path is not None
            assert save_path.exists()

            # Reload in new registry
            registry2 = PerceptionSeamRegistry(checkpoint_dir=tmpdir)
            registry2.register_seam("evidence_fusion", "test_fusion")
            seam2 = registry2.load_seam("test_fusion")
            assert torch.allclose(
                seam2.input_proj.weight,
                torch.full_like(seam2.input_proj.weight, 0.123),
            )

    def test_unload_seam(self):
        registry = PerceptionSeamRegistry()
        registry.register_seam("evidence_fusion", "test_fusion")
        registry.load_seam("test_fusion")
        assert registry.get_seam("test_fusion") is not None
        registry.unload_seam("test_fusion")
        assert registry.get_seam("test_fusion") is None
        desc = registry.get_descriptor("test_fusion")
        assert not desc.loaded

    def test_list_seams(self):
        registry = PerceptionSeamRegistry()
        registry.register_seam("evidence_fusion", "fusion_1")
        registry.register_seam("sam_calibration", "sam_1")
        seams = registry.list_seams()
        assert len(seams) == 2
        assert "fusion_1" in seams
        assert "sam_1" in seams

    def test_summary(self):
        registry = PerceptionSeamRegistry()
        registry.register_seam("evidence_fusion", "fusion_1")
        registry.load_seam("fusion_1")
        summary = registry.summary()
        assert summary["registered_count"] == 1
        assert summary["loaded_count"] == 1
        assert summary["total_params"] > 0

    def test_invalid_seam_type(self):
        registry = PerceptionSeamRegistry()
        with pytest.raises(ValueError, match="Unknown seam type"):
            registry.register_seam("invalid_type", "test")

    def test_load_unregistered_seam(self):
        registry = PerceptionSeamRegistry()
        with pytest.raises(KeyError):
            registry.load_seam("nonexistent")


class TestCreateDefaultRegistry:
    def test_creates_all_seam_types(self):
        registry = create_default_registry()
        seams = registry.list_seams()
        assert len(seams) == 5
        seam_types = {d.seam_type for d in seams.values()}
        assert "evidence_fusion" in seam_types
        assert "sam_calibration" in seam_types
        assert "vision_backbone_projection" in seam_types
        assert "depth_metric_calibration" in seam_types
        assert "vjepa_temporal_alignment" in seam_types


# ---------------------------------------------------------------------------
# Promotion resolver tests
# ---------------------------------------------------------------------------


class TestResolveProviderAdapterHelper:
    def test_disabled_posture(self):
        result = resolve_provider_adapter_helper(
            provider_kind="sam_calibration",
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert not result["helper_active"]
        assert result["promotion_stage"] == "raw_provider_output"
        assert result["posture"] == "disabled"

    def test_auto_not_ready(self):
        result = resolve_provider_adapter_helper(
            provider_kind="sam_calibration",
            loading_posture="auto",
            benchmark_signals={"ready": False},
        )
        assert not result["helper_active"]
        assert result["promotion_stage"] == "raw_provider_output"

    def test_auto_promoted(self):
        result = resolve_provider_adapter_helper(
            provider_kind="sam_calibration",
            loading_posture="auto",
            benchmark_signals={"ready": True},
        )
        assert result["helper_active"]
        assert result["promotion_stage"] == "promoted"
        assert result["helper_weight"] == 1.0

    def test_required_not_ready(self):
        result = resolve_provider_adapter_helper(
            provider_kind="sam_calibration",
            loading_posture="required",
            benchmark_signals={"ready": False},
        )
        assert not result["helper_active"]
        assert result["promotion_stage"] == "required_but_not_ready"

    def test_required_promoted(self):
        result = resolve_provider_adapter_helper(
            provider_kind="vision_backbone_projection",
            loading_posture="required",
            benchmark_signals={"ready": True},
        )
        assert result["helper_active"]
        assert result["promotion_stage"] == "promoted"

    def test_demotion_on_failure(self):
        result = resolve_provider_adapter_helper(
            provider_kind="depth_metric_calibration",
            loading_posture="auto",
            benchmark_signals={
                "ready": True,
                "benchmark_gate": {"demotion_failure_threshold": 0.3},
            },
            evidence_signals={"recent_failure_rate": 0.5},
        )
        assert result["helper_active"]
        assert result["promotion_stage"] == "demoted_to_shadow"
        assert result["helper_weight"] == 0.25
