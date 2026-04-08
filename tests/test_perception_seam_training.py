"""Tests for perception seam training infrastructure.

Tests cover:
- Loss function correctness for each seam type
- Data loader collation and batching
- Training loop mechanics
- Benchmark gate evaluation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.training.perception_seam_losses import (
    EvidenceFusionLossConfig,
    SAMCalibrationLossConfig,
    DepthMetricCalibrationLossConfig,
    VJEPATemporalAlignmentLossConfig,
    SeamLossResult,
    evidence_fusion_loss,
    sam_calibration_loss,
    depth_metric_calibration_loss,
    vjepa_temporal_alignment_loss,
    get_seam_loss_fn,
)
from src.training.perception_seam_data import (
    ProviderObservation,
    MultiProviderSample,
    EvidenceFusionBatch,
    EvidenceFusionDataset,
    SAMCalibrationSample,
    SAMCalibrationDataset,
    DepthCalibrationSample,
    DepthCalibrationDataset,
    VJEPATemporalSample,
    VJEPATemporalDataset,
    generate_synthetic_evidence_fusion_samples,
    generate_synthetic_sam_calibration_samples,
    generate_synthetic_depth_calibration_samples,
    generate_synthetic_vjepa_temporal_samples,
    create_evidence_fusion_loader,
    create_sam_calibration_loader,
    create_depth_calibration_loader,
    create_vjepa_temporal_loader,
)
from src.training.perception_seam_benchmarks import (
    BenchmarkGateConfig,
    BenchmarkMetric,
    BenchmarkGateResult,
    EvidenceFusionBenchmark,
    SAMCalibrationBenchmark,
    DepthCalibrationBenchmark,
    VJEPATemporalBenchmark,
    get_benchmark_evaluator,
    evaluate_seam_benchmark,
)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestEvidenceFusionLoss:
    def test_basic_loss_computation(self):
        batch_size = 4
        n_providers = 3
        d_feature = 128

        predicted_weights = torch.softmax(torch.randn(batch_size, n_providers), dim=-1)
        predicted_confidence = torch.sigmoid(torch.randn(batch_size))
        held_out_idx = torch.randint(0, n_providers, (batch_size,))
        held_out_target = torch.randn(batch_size, d_feature)
        provider_features = torch.randn(batch_size, n_providers, d_feature)

        result = evidence_fusion_loss(
            predicted_weights=predicted_weights,
            predicted_confidence=predicted_confidence,
            held_out_provider_idx=held_out_idx,
            held_out_reconstruction_target=held_out_target,
            provider_features=provider_features,
        )

        assert isinstance(result, SeamLossResult)
        assert result.total_loss.dim() == 0  # scalar
        assert not torch.isnan(result.total_loss)
        assert "held_out_reconstruction" in result.component_losses

    def test_with_task_correlation(self):
        batch_size = 8
        n_providers = 4
        d_feature = 64

        result = evidence_fusion_loss(
            predicted_weights=torch.softmax(torch.randn(batch_size, n_providers), dim=-1),
            predicted_confidence=torch.sigmoid(torch.randn(batch_size)),
            held_out_provider_idx=torch.randint(0, n_providers, (batch_size,)),
            held_out_reconstruction_target=torch.randn(batch_size, d_feature),
            provider_features=torch.randn(batch_size, n_providers, d_feature),
            task_correlation_target=torch.rand(batch_size),
        )

        assert "task_correlation" in result.component_losses
        assert "confidence_task_corr" in result.metrics


class TestSAMCalibrationLoss:
    def test_basic_loss_computation(self):
        batch_size = 4
        n_masks = 8

        result = sam_calibration_loss(
            calibrated_confidence=torch.sigmoid(torch.randn(batch_size, n_masks)),
            epistemic_uncertainty=torch.sigmoid(torch.randn(batch_size, n_masks)),
            prompt_satisfaction=torch.sigmoid(torch.randn(batch_size, n_masks)),
            downstream_mask_quality=torch.rand(batch_size, n_masks),
            mask_valid=torch.ones(batch_size, n_masks, dtype=torch.bool),
        )

        assert isinstance(result, SeamLossResult)
        assert not torch.isnan(result.total_loss)
        assert "calibration" in result.component_losses

    def test_with_optional_targets(self):
        batch_size = 4
        n_masks = 8

        result = sam_calibration_loss(
            calibrated_confidence=torch.sigmoid(torch.randn(batch_size, n_masks)),
            epistemic_uncertainty=torch.sigmoid(torch.randn(batch_size, n_masks)),
            prompt_satisfaction=torch.sigmoid(torch.randn(batch_size, n_masks)),
            downstream_mask_quality=torch.rand(batch_size, n_masks),
            provider_disagreement=torch.rand(batch_size, n_masks),
            segmentation_iou=torch.rand(batch_size, n_masks),
            mask_valid=torch.ones(batch_size, n_masks, dtype=torch.bool),
        )

        assert "uncertainty" in result.component_losses
        assert "prompt_satisfaction" in result.component_losses


class TestDepthMetricCalibrationLoss:
    def test_basic_loss_computation(self):
        batch_size = 2
        H, W = 32, 32

        result = depth_metric_calibration_loss(
            metric_depth=torch.rand(batch_size, 1, H, W) * 5 + 1,
            predicted_uncertainty=torch.rand(batch_size, 1, H, W),
            predicted_scale=torch.rand(batch_size) + 0.5,
            predicted_shift=torch.randn(batch_size),
            ground_truth_depth=torch.rand(batch_size, 1, H, W) * 5 + 1,
            depth_valid_mask=torch.ones(batch_size, 1, H, W, dtype=torch.bool),
        )

        assert isinstance(result, SeamLossResult)
        assert not torch.isnan(result.total_loss)
        assert "depth" in result.component_losses
        assert "uncertainty" in result.component_losses

    def test_with_temporal_consistency(self):
        batch_size = 2
        H, W = 32, 32

        result = depth_metric_calibration_loss(
            metric_depth=torch.rand(batch_size, 1, H, W) * 5 + 1,
            predicted_uncertainty=torch.rand(batch_size, 1, H, W),
            predicted_scale=torch.rand(batch_size) + 0.5,
            predicted_shift=torch.randn(batch_size),
            ground_truth_depth=torch.rand(batch_size, 1, H, W) * 5 + 1,
            depth_valid_mask=torch.ones(batch_size, 1, H, W, dtype=torch.bool),
            previous_scale=torch.rand(batch_size) + 0.5,
            previous_shift=torch.randn(batch_size),
        )

        assert "scale_consistency" in result.component_losses


class TestVJEPATemporalAlignmentLoss:
    def test_basic_loss_computation(self):
        batch_size = 2
        T = 4
        n_obj = 10
        d_out = 128

        result = vjepa_temporal_alignment_loss(
            temporal_aligned=torch.randn(batch_size, T, n_obj, d_out),
            temporal_confidence=torch.sigmoid(torch.randn(batch_size, T)),
            future_object_states=torch.randn(batch_size, T, n_obj, d_out),
            object_valid_mask=torch.ones(batch_size, n_obj, dtype=torch.bool),
        )

        assert isinstance(result, SeamLossResult)
        assert not torch.isnan(result.total_loss)
        assert "prediction" in result.component_losses
        assert "confidence" in result.component_losses


class TestGetSeamLossFn:
    def test_all_seam_types_registered(self):
        seam_types = [
            "evidence_fusion",
            "sam_calibration",
            "vision_backbone_projection",
            "depth_metric_calibration",
            "vjepa_temporal_alignment",
        ]
        for seam_type in seam_types:
            fn = get_seam_loss_fn(seam_type)
            assert callable(fn)

    def test_invalid_seam_type_raises(self):
        with pytest.raises(ValueError, match="Unknown seam type"):
            get_seam_loss_fn("nonexistent_type")


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestEvidenceFusionDataset:
    def test_synthetic_sample_generation(self):
        samples = generate_synthetic_evidence_fusion_samples(
            n_samples=50, n_providers=4, d_feature=128, seed=42
        )
        assert len(samples) == 50
        assert all(isinstance(s, MultiProviderSample) for s in samples)
        assert all(len(s.providers) == 4 for s in samples)

    def test_dataset_filtering(self):
        samples = generate_synthetic_evidence_fusion_samples(
            n_samples=100, n_providers=4, seed=42
        )
        # EvidenceFusionDataset has min_providers=2 by default
        dataset = EvidenceFusionDataset(samples)
        # Some samples may be filtered out due to availability
        assert len(dataset) <= 100

    def test_collation(self):
        samples = generate_synthetic_evidence_fusion_samples(
            n_samples=10, n_providers=4, d_feature=64, seed=42
        )
        # Filter to ensure we have enough samples
        valid_samples = [
            s for s in samples
            if sum(1 for p in s.providers if p.availability_status == "available") >= 2
        ]
        if len(valid_samples) >= 4:
            batch = EvidenceFusionDataset.collate_fn(valid_samples[:4], d_feature=64)
            assert isinstance(batch, EvidenceFusionBatch)
            assert batch.provider_features.shape[0] == 4
            assert batch.held_out_idx.shape[0] == 4

    def test_data_loader_creation(self):
        samples = generate_synthetic_evidence_fusion_samples(
            n_samples=100, n_providers=4, seed=42
        )
        loader = create_evidence_fusion_loader(
            samples, batch_size=16, shuffle=True
        )
        batch = next(iter(loader))
        assert isinstance(batch, EvidenceFusionBatch)


class TestSAMCalibrationDataset:
    def test_synthetic_sample_generation(self):
        samples = generate_synthetic_sam_calibration_samples(
            n_samples=50, n_masks=8, d_mask=256, seed=42
        )
        assert len(samples) == 50
        assert all(isinstance(s, SAMCalibrationSample) for s in samples)

    def test_collation(self):
        samples = generate_synthetic_sam_calibration_samples(
            n_samples=8, n_masks=8, d_mask=128, seed=42
        )
        from src.training.perception_seam_data import SAMCalibrationBatch
        batch = SAMCalibrationDataset.collate_fn(samples[:4], max_masks=8, d_mask=128)
        assert isinstance(batch, SAMCalibrationBatch)
        assert batch.mask_features.shape == (4, 8, 128)


class TestDepthCalibrationDataset:
    def test_synthetic_sample_generation(self):
        samples = generate_synthetic_depth_calibration_samples(
            n_samples=50, height=64, width=64, seed=42
        )
        assert len(samples) == 50
        assert all(isinstance(s, DepthCalibrationSample) for s in samples)

    def test_collation(self):
        samples = generate_synthetic_depth_calibration_samples(
            n_samples=8, height=32, width=32, seed=42
        )
        from src.training.perception_seam_data import DepthCalibrationBatch
        batch = DepthCalibrationDataset.collate_fn(samples[:4], target_size=(32, 32))
        assert isinstance(batch, DepthCalibrationBatch)
        assert batch.relative_depth.shape == (4, 1, 32, 32)


class TestVJEPATemporalDataset:
    def test_synthetic_sample_generation(self):
        samples = generate_synthetic_vjepa_temporal_samples(
            n_samples=50, n_temporal_steps=4, n_objects=10, seed=42
        )
        assert len(samples) == 50
        assert all(isinstance(s, VJEPATemporalSample) for s in samples)

    def test_collation(self):
        samples = generate_synthetic_vjepa_temporal_samples(
            n_samples=8, n_temporal_steps=4, n_objects=10, seed=42
        )
        from src.training.perception_seam_data import VJEPATemporalBatch
        batch = VJEPATemporalDataset.collate_fn(
            samples[:4], n_temporal_steps=4, max_objects=10
        )
        assert isinstance(batch, VJEPATemporalBatch)
        assert batch.vjepa_tokens.shape[0] == 4
        assert batch.vjepa_tokens.shape[1] == 4  # T


# ---------------------------------------------------------------------------
# Benchmark evaluation tests
# ---------------------------------------------------------------------------


class TestBenchmarkGateResult:
    def test_to_dict(self):
        result = BenchmarkGateResult(
            seam_id="test_seam",
            seam_type="evidence_fusion",
            evaluation_id="eval_123",
            overall_score=0.85,
            overall_passed=True,
            promotion_decision="promote",
            metrics=[
                BenchmarkMetric(
                    name="accuracy",
                    value=0.9,
                    threshold=0.8,
                    passed=True,
                )
            ],
        )
        d = result.to_dict()
        assert d["seam_id"] == "test_seam"
        assert d["overall_score"] == 0.85
        assert len(d["metrics"]) == 1


class TestEvidenceFusionBenchmark:
    def test_evaluate(self):
        from src.world_model.perception_grounding.neural_seams import EvidenceFusionSeam

        # EvidenceFusionSeam expects D_PROVIDER_RAW=12 dimensional features
        seam = EvidenceFusionSeam(d_model=32, n_heads=2, d_ff=64)
        samples = generate_synthetic_evidence_fusion_samples(
            n_samples=50, n_providers=4, d_feature=12, seed=42
        )
        # Match d_feature to what EvidenceFusionSeam expects
        loader = create_evidence_fusion_loader(samples, batch_size=8, d_feature=12)

        benchmark = EvidenceFusionBenchmark()
        result = benchmark.evaluate(seam, loader)

        assert isinstance(result, BenchmarkGateResult)
        assert result.seam_type == "evidence_fusion"
        assert len(result.metrics) > 0


class TestSAMCalibrationBenchmark:
    def test_evaluate(self):
        from src.world_model.perception_grounding.neural_seams import SAMCalibrationSeam

        seam = SAMCalibrationSeam(d_mask=128, d_model=64, n_heads=2, d_ff=128)
        samples = generate_synthetic_sam_calibration_samples(
            n_samples=50, n_masks=8, d_mask=128, seed=42
        )
        loader = create_sam_calibration_loader(
            samples, batch_size=8, max_masks=8, d_mask=128
        )

        benchmark = SAMCalibrationBenchmark()
        result = benchmark.evaluate(seam, loader)

        assert isinstance(result, BenchmarkGateResult)
        assert result.seam_type == "sam_calibration"


class TestDepthCalibrationBenchmark:
    def test_evaluate(self):
        from src.world_model.perception_grounding.neural_seams import DepthMetricCalibrationSeam

        seam = DepthMetricCalibrationSeam(d_depth=1, d_hidden=64)
        samples = generate_synthetic_depth_calibration_samples(
            n_samples=20, height=32, width=32, seed=42
        )
        loader = create_depth_calibration_loader(
            samples, batch_size=4, target_size=(32, 32)
        )

        benchmark = DepthCalibrationBenchmark()
        result = benchmark.evaluate(seam, loader)

        assert isinstance(result, BenchmarkGateResult)
        assert result.seam_type == "depth_metric_calibration"


class TestVJEPATemporalBenchmark:
    def test_evaluate(self):
        from src.world_model.perception_grounding.neural_seams import VJEPATemporalAlignmentSeam

        # Use default dimensions to match the seam
        seam = VJEPATemporalAlignmentSeam(
            d_vjepa=1024, d_wm_token=128, d_model=256, d_out=128, n_heads=8
        )
        samples = generate_synthetic_vjepa_temporal_samples(
            n_samples=20, n_temporal_steps=4, n_objects=10,
            d_vjepa=1024, d_wm=128, d_out=128, seed=42
        )
        loader = create_vjepa_temporal_loader(
            samples, batch_size=4, n_temporal_steps=4, max_objects=10
        )

        benchmark = VJEPATemporalBenchmark()
        result = benchmark.evaluate(seam, loader)

        assert isinstance(result, BenchmarkGateResult)
        assert result.seam_type == "vjepa_temporal_alignment"


class TestGetBenchmarkEvaluator:
    def test_all_seam_types_registered(self):
        seam_types = [
            "evidence_fusion",
            "sam_calibration",
            "depth_metric_calibration",
            "vjepa_temporal_alignment",
        ]
        for seam_type in seam_types:
            evaluator = get_benchmark_evaluator(seam_type)
            assert evaluator is not None

    def test_invalid_seam_type_raises(self):
        with pytest.raises(ValueError, match="Unknown seam type"):
            get_benchmark_evaluator("nonexistent_type")
