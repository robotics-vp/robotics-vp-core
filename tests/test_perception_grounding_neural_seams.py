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
    SceneGraphTransformerSeam,
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
        assert len(seams) == 6
        seam_types = {d.seam_type for d in seams.values()}
        assert "evidence_fusion" in seam_types
        assert "sam_calibration" in seam_types
        assert "vision_backbone_projection" in seam_types
        assert "depth_metric_calibration" in seam_types
        assert "vjepa_temporal_alignment" in seam_types
        assert "scene_graph_transformer" in seam_types


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


# ---------------------------------------------------------------------------
# Graph Transformer shadow path tests
# ---------------------------------------------------------------------------


class TestGraphTransformerShadowPath:
    """Test the shadow execution of SceneGraphTransformerSeam in the compiler."""

    def test_compiler_shadow_receipt_emitted(self):
        """Shadow receipt appears in metadata when seam is provided."""
        from src.world_model.perception_grounding.compiler import (
            compile_perception_grounding_world_state,
        )

        seam = SceneGraphTransformerSeam(d_token=128, d_model=64, d_out=64, n_heads=2, d_ff=128, n_layers=1)
        state = compile_perception_grounding_world_state(
            episode_id="test_shadow",
            task_id="shadow_test",
            frame_index=0,
            scene_graph_transformer_seam=seam,
        )
        shadow_receipt = state.metadata.get("graph_transformer_shadow_receipt")
        assert shadow_receipt is not None
        assert shadow_receipt["seam_id"] == "scene_graph_transformer_default"
        assert shadow_receipt["version"] == "graph_transformer_shadow_receipt_v2"
        assert "graph_confidence" in shadow_receipt
        assert "edge_overlap_fraction" in shadow_receipt
        assert "node_token_cosine_similarity" in shadow_receipt
        assert "gate_score" in shadow_receipt
        assert "promotion_eligible" in shadow_receipt
        assert shadow_receipt["latency_ms"] >= 0.0

    def test_compiler_no_shadow_without_seam(self):
        """No shadow receipt when no seam is provided."""
        from src.world_model.perception_grounding.compiler import (
            compile_perception_grounding_world_state,
        )

        state = compile_perception_grounding_world_state(
            episode_id="test_no_shadow",
            task_id="no_shadow_test",
        )
        shadow_receipt = state.metadata.get("graph_transformer_shadow_receipt")
        assert shadow_receipt is None

    def test_shadow_receipt_in_compilation_result(self):
        """Shadow receipt is extracted into typed receipt list."""
        from src.world_model.perception_grounding.compiler import (
            compile_perception_grounding_with_receipts,
        )
        from src.world_model.perception_grounding.receipts import (
            GraphTransformerShadowReceipt,
        )

        seam = SceneGraphTransformerSeam(d_token=128, d_model=64, d_out=64, n_heads=2, d_ff=128, n_layers=1)
        result = compile_perception_grounding_with_receipts(
            episode_id="test_receipt_extraction",
            task_id="receipt_test",
            scene_graph_transformer_seam=seam,
        )
        shadow_receipts = [
            r for r in result.receipts
            if isinstance(r, GraphTransformerShadowReceipt)
        ]
        assert len(shadow_receipts) == 1
        assert shadow_receipts[0].seam_id == "scene_graph_transformer_default"
        assert 0.0 <= shadow_receipts[0].gate_score <= 1.0

    def test_shadow_receipt_fields_valid(self):
        """All comparison fields are numerically valid."""
        from src.world_model.perception_grounding.compiler import (
            compile_perception_grounding_world_state,
        )

        seam = SceneGraphTransformerSeam(d_token=128, d_model=64, d_out=64, n_heads=2, d_ff=128, n_layers=1)
        state = compile_perception_grounding_world_state(
            episode_id="test_shadow_fields",
            task_id="field_test",
            scene_graph_transformer_seam=seam,
        )
        r = state.metadata["graph_transformer_shadow_receipt"]
        assert 0.0 <= r["graph_confidence"] <= 1.0
        assert 0.0 <= r["mean_edge_weight"] <= 1.0
        assert 0.0 <= r["edge_overlap_fraction"] <= 1.0
        assert -1.0 <= r["node_token_cosine_similarity"] <= 1.0
        assert 0.0 <= r["gate_score"] <= 1.0
        assert r["node_count"] >= 0
        assert r["param_count"] > 0

    def test_promotion_requires_benchmark_evidence(self):
        """Promotion is never granted from heuristic-similarity alone.

        The plasticity gating discipline requires benchmark evidence
        (annotation-export supervision, held-out label agreement,
        downstream usefulness) before promotion_eligible can be True.
        Shadow comparison metrics are diagnostic only.
        """
        from src.world_model.perception_grounding.compiler import (
            compile_perception_grounding_world_state,
        )

        seam = SceneGraphTransformerSeam(d_token=128, d_model=64, d_out=64, n_heads=2, d_ff=128, n_layers=1)
        state = compile_perception_grounding_world_state(
            episode_id="test_promotion_gate",
            task_id="promotion_gate_test",
            scene_graph_transformer_seam=seam,
        )
        r = state.metadata["graph_transformer_shadow_receipt"]
        # Without benchmark evidence, promotion must be denied
        assert r["benchmark_evidence_present"] is False
        assert r["promotion_eligible"] is False
        # Shadow comparison metrics exist but are diagnostic only
        assert "node_token_cosine_similarity" in r
        assert "edge_overlap_fraction" in r
        assert "edge_weight_correlation" in r
        # gate_score reflects intrinsic quality only (graph_confidence)
        assert 0.0 <= r["gate_score"] <= 1.0

    def test_shadow_receipt_separates_comparison_from_promotion(self):
        """Receipt explicitly separates shadow comparison from promotion evidence."""
        from src.world_model.perception_grounding.receipts import (
            GraphTransformerShadowReceipt,
        )

        import dataclasses
        field_names = {f.name for f in dataclasses.fields(GraphTransformerShadowReceipt)}
        # Promotion evidence fields must exist
        assert "benchmark_evidence_present" in field_names
        assert "annotation_supervision_score" in field_names
        assert "held_out_label_agreement" in field_names
        assert "downstream_usefulness_score" in field_names
        assert "receipt_consistency" in field_names
        # Shadow comparison fields must exist separately
        assert "node_token_cosine_similarity" in field_names
        assert "edge_overlap_fraction" in field_names
        assert "edge_weight_correlation" in field_names


# ---------------------------------------------------------------------------
# Annotation export → SceneGraphSample converter tests
# ---------------------------------------------------------------------------


class TestAnnotationExportToSceneGraphSamples:
    """Test the real-data unlock converter."""

    def _make_annotation_record(self, n_objects=4, n_edges=3):
        """Create a minimal AnnotationExportRecord for testing."""
        from src.world_model.perception_grounding.annotation_export import (
            AnnotationExportRecord,
        )

        all_categories = ["cup", "plate", "fork", "cup"]
        all_confidences = [0.9, 0.8, 0.7, 0.85]
        track_ids = [f"track_{i}" for i in range(n_objects)]

        # Build edges that only reference valid track ids
        edge_pairs = [(0, 1), (1, 2), (2, 3)]
        edge_src = []
        edge_tgt = []
        edge_types_list = ["contact", "spatial_adjacency", "containment"]
        edge_confs = [0.9, 0.7, 0.8]
        for idx in range(min(n_edges, len(edge_pairs))):
            s, t = edge_pairs[idx]
            if s < n_objects and t < n_objects:
                edge_src.append(track_ids[s])
                edge_tgt.append(track_ids[t])
        actual_edges = len(edge_src)

        return AnnotationExportRecord(
            record_id="test_annot_001",
            scene_graph_id="graph_test",
            episode_id="ep_test",
            frame_index=0,
            object_track_ids=track_ids,
            object_tokens=[[float(j) / 10 for j in range(8)] for _ in range(n_objects)],
            object_categories=all_categories[:n_objects],
            object_confidences=all_confidences[:n_objects],
            edge_source_ids=edge_src,
            edge_target_ids=edge_tgt,
            edge_types=edge_types_list[:actual_edges],
            edge_confidences=edge_confs[:actual_edges],
            edge_features=[[0.5, 0.3, 0.0, 0.0] for _ in range(actual_edges)],
            object_class_labels={tid: cat for tid, cat in zip(track_ids, all_categories[:n_objects])},
            object_annotation_confidences={tid: 0.85 for tid in track_ids},
            teacher_alignment_score=0.75,
            annotation_quality_score=0.8,
        )

    def test_basic_conversion(self):
        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )

        records = [self._make_annotation_record()]
        samples = annotation_export_to_scene_graph_samples(records)
        assert len(samples) == 1
        s = samples[0]
        assert s.node_features.shape[0] == 4
        assert s.node_features.shape[1] == 128  # padded to d_token
        assert s.edge_index.shape[0] == 3
        assert s.edge_type.shape[0] == 3
        assert s.edge_features.shape == (3, 64)  # padded to d_edge
        assert s.node_labels is not None
        assert s.edge_importance is not None
        assert s.node_confidence_target is not None

    def test_filters_too_few_objects(self):
        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )

        records = [self._make_annotation_record(n_objects=1, n_edges=0)]
        samples = annotation_export_to_scene_graph_samples(records, min_objects=2)
        assert len(samples) == 0

    def test_dict_input(self):
        """Converter works with dict records (JSON-loaded)."""
        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )

        record = self._make_annotation_record()
        records = [record.to_dict()]
        samples = annotation_export_to_scene_graph_samples(records)
        assert len(samples) == 1

    def test_category_vocab_auto_built(self):
        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )

        records = [self._make_annotation_record()]
        samples = annotation_export_to_scene_graph_samples(records)
        # Should have distinct labels for cup, plate, fork
        labels = samples[0].node_labels.tolist()
        assert len(set(labels)) >= 2  # at least 2 distinct categories

    def test_edge_types_mapped(self):
        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )
        from src.world_model.perception_grounding.neural_seams import EDGE_TYPE_VOCAB

        records = [self._make_annotation_record()]
        samples = annotation_export_to_scene_graph_samples(records)
        # "contact" should map to EDGE_TYPE_VOCAB["contact"]
        assert samples[0].edge_type[0].item() == EDGE_TYPE_VOCAB["contact"]

    def test_roundtrip_json(self):
        """Records saved/loaded as JSON still convert correctly."""
        import json

        from src.training.perception_seam_data import (
            annotation_export_to_scene_graph_samples,
        )

        record = self._make_annotation_record()
        json_str = json.dumps(record.to_dict())
        loaded = json.loads(json_str)
        samples = annotation_export_to_scene_graph_samples([loaded])
        assert len(samples) == 1
        assert samples[0].node_features.shape == (4, 128)
