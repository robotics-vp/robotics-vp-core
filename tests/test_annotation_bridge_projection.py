"""Tests for the annotation-bridge projection lane.

Covers:
- AnnotationBridgeProjectionSeam forward pass and shapes
- annotation_bridge_projection_loss computation
- evaluate_seam_on_annotations with provisional-evidence gating
- AnnotationBridgeShadowReceipt provisional enforcement
- resolve_annotation_bridge_helper promotion blocking on provisional evidence
- Seam registry integration
"""

from __future__ import annotations

import torch

from src.world_model.perception_grounding.neural_seams import (
    AnnotationBridgeProjectionSeam,
)
from src.world_model.perception_grounding.receipts import (
    AnnotationBridgeShadowReceipt,
)
from src.world_model.perception_grounding.promotion import (
    resolve_annotation_bridge_helper,
)
from src.world_model.perception_grounding.seam_registry import (
    SEAM_TYPES,
    SEAM_DEFAULTS,
)
from src.training.perception_seam_losses import (
    SeamLossResult,
    annotation_bridge_projection_loss,
    get_seam_loss_fn,
)
from src.training.perception_seam_data import (
    evaluate_seam_on_annotations,
)


# ---------------------------------------------------------------------------
# Seam module tests
# ---------------------------------------------------------------------------


class TestAnnotationBridgeProjectionSeam:
    def test_forward_batched(self):
        seam = AnnotationBridgeProjectionSeam(d_token=64, d_hidden=128, n_categories=8, n_affordances=4)
        x = torch.randn(2, 5, 64)  # batch=2, N=5, d_token=64
        result = seam(x)

        assert result["class_logits"].shape == (2, 5, 8)
        assert result["confidence"].shape == (2, 5)
        assert result["affordance_scores"].shape == (2, 5, 4)
        # confidence in [0,1]
        assert result["confidence"].min() >= 0.0
        assert result["confidence"].max() <= 1.0
        # affordances in [0,1]
        assert result["affordance_scores"].min() >= 0.0
        assert result["affordance_scores"].max() <= 1.0

    def test_forward_unbatched(self):
        seam = AnnotationBridgeProjectionSeam(d_token=32, n_categories=4, n_affordances=2)
        x = torch.randn(3, 32)  # N=3, d_token=32 (no batch dim)
        result = seam(x)

        assert result["class_logits"].shape == (3, 4)
        assert result["confidence"].shape == (3,)
        assert result["affordance_scores"].shape == (3, 2)

    def test_param_count_and_describe(self):
        seam = AnnotationBridgeProjectionSeam()
        desc = seam.describe()
        assert desc["seam_type"] == "annotation_bridge_projection"
        assert desc["param_count"] > 0
        assert desc["param_count"] == seam.param_count()

    def test_gradients_flow(self):
        seam = AnnotationBridgeProjectionSeam(d_token=32, n_categories=4)
        x = torch.randn(2, 3, 32, requires_grad=True)
        result = seam(x)
        loss = result["class_logits"].sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestAnnotationBridgeProjectionLoss:
    def test_basic_loss(self):
        B, N, C = 2, 5, 8
        result = annotation_bridge_projection_loss(
            class_logits=torch.randn(B, N, C),
            confidence=torch.sigmoid(torch.randn(B, N)),
            affordance_scores=torch.sigmoid(torch.randn(B, N, 4)),
            class_labels=torch.randint(0, C, (B, N)),
        )
        assert isinstance(result, SeamLossResult)
        assert result.total_loss.dim() == 0
        assert not torch.isnan(result.total_loss)
        assert "class_prediction" in result.component_losses
        assert "class_accuracy" in result.metrics

    def test_with_all_targets(self):
        B, N, C = 4, 6, 16
        result = annotation_bridge_projection_loss(
            class_logits=torch.randn(B, N, C),
            confidence=torch.sigmoid(torch.randn(B, N)),
            affordance_scores=torch.sigmoid(torch.randn(B, N, 8)),
            class_labels=torch.randint(0, C, (B, N)),
            confidence_targets=torch.rand(B, N),
            affordance_targets=torch.rand(B, N, 8),
        )
        assert "confidence_calibration" in result.component_losses
        assert "affordance_prediction" in result.component_losses
        assert "confidence_mae" in result.metrics

    def test_with_node_mask(self):
        B, N, C = 2, 8, 4
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :5] = True
        result = annotation_bridge_projection_loss(
            class_logits=torch.randn(B, N, C),
            confidence=torch.sigmoid(torch.randn(B, N)),
            affordance_scores=torch.sigmoid(torch.randn(B, N, 2)),
            class_labels=torch.randint(0, C, (B, N)),
            node_mask=mask,
        )
        assert not torch.isnan(result.total_loss)

    def test_empty_valid_nodes(self):
        B, N, C = 2, 4, 8
        mask = torch.zeros(B, N, dtype=torch.bool)  # no valid nodes
        result = annotation_bridge_projection_loss(
            class_logits=torch.randn(B, N, C),
            confidence=torch.sigmoid(torch.randn(B, N)),
            affordance_scores=torch.sigmoid(torch.randn(B, N, 2)),
            class_labels=torch.randint(0, C, (B, N)),
            node_mask=mask,
        )
        assert result.total_loss.item() == 0.0

    def test_registry_lookup(self):
        fn = get_seam_loss_fn("annotation_bridge_projection")
        assert fn is annotation_bridge_projection_loss


# ---------------------------------------------------------------------------
# Shadow receipt tests
# ---------------------------------------------------------------------------


class TestAnnotationBridgeShadowReceipt:
    def test_provisional_blocks_promotion(self):
        """Provisional evidence MUST NOT produce promotion_eligible=True in to_dict."""
        receipt = AnnotationBridgeShadowReceipt(
            receipt_id="test",
            seam_id="test_seam",
            promotion_stage="benchmark_gated",
            posture="auto",
            benchmark_evidence_present=True,
            evidence_source_provisional=True,  # <-- heuristic tokens
            annotation_supervision_score=0.95,
            held_out_label_agreement=0.90,
            gate_score=0.9,
            promotion_eligible=True,  # even if set True in raw field...
        )
        d = receipt.to_dict()
        # ...to_dict enforces provisional blocks promotion
        assert d["promotion_eligible"] is False
        assert d["evidence_source_provisional"] is True

    def test_non_provisional_allows_promotion(self):
        receipt = AnnotationBridgeShadowReceipt(
            receipt_id="test",
            seam_id="test_seam",
            promotion_stage="benchmark_gated",
            posture="auto",
            benchmark_evidence_present=True,
            evidence_source_provisional=False,
            gate_score=0.9,
            promotion_eligible=True,
        )
        d = receipt.to_dict()
        assert d["promotion_eligible"] is True

    def test_no_benchmark_blocks_promotion(self):
        receipt = AnnotationBridgeShadowReceipt(
            receipt_id="test",
            seam_id="test_seam",
            promotion_stage="heuristic_fallback",
            posture="auto",
            benchmark_evidence_present=False,
            evidence_source_provisional=False,
            promotion_eligible=True,
        )
        d = receipt.to_dict()
        assert d["promotion_eligible"] is False


# ---------------------------------------------------------------------------
# Promotion resolver tests
# ---------------------------------------------------------------------------


class TestResolveAnnotationBridgeHelper:
    def test_disabled_posture(self):
        result = resolve_annotation_bridge_helper(
            loading_posture="disabled",
            benchmark_signals={},
        )
        assert result["helper_active"] is False
        assert result["promotion_stage"] == "heuristic_fallback"

    def test_auto_provisional_blocks_promotion(self):
        """Even with benchmark_eligible=True, provisional evidence stays in shadow."""
        result = resolve_annotation_bridge_helper(
            loading_posture="auto",
            benchmark_signals={"benchmark_eligible": True},
            evidence_source_provisional=True,
        )
        assert result["promotion_stage"] == "shadow_monitoring"
        assert result["helper_weight"] == 0.0

    def test_auto_non_provisional_promotes(self):
        result = resolve_annotation_bridge_helper(
            loading_posture="auto",
            benchmark_signals={"benchmark_eligible": True},
            evidence_source_provisional=False,
        )
        assert result["promotion_stage"] == "promoted"
        assert result["helper_weight"] == 1.0

    def test_required_provisional_stays_shadow(self):
        result = resolve_annotation_bridge_helper(
            loading_posture="required",
            benchmark_signals={"benchmark_eligible": True},
            evidence_source_provisional=True,
        )
        assert result["promotion_stage"] == "shadow_monitoring"


# ---------------------------------------------------------------------------
# Seam registry integration
# ---------------------------------------------------------------------------


class TestSeamRegistryIntegration:
    def test_annotation_bridge_in_seam_types(self):
        assert "annotation_bridge_projection" in SEAM_TYPES
        assert SEAM_TYPES["annotation_bridge_projection"] is AnnotationBridgeProjectionSeam

    def test_annotation_bridge_in_defaults(self):
        assert "annotation_bridge_projection" in SEAM_DEFAULTS
        defaults = SEAM_DEFAULTS["annotation_bridge_projection"]
        assert "d_token" in defaults
        assert "n_categories" in defaults


# ---------------------------------------------------------------------------
# Benchmark evaluation with provisional gating
# ---------------------------------------------------------------------------


class TestEvaluateSeamOnAnnotations:
    def _make_annotation_records(self, n: int = 10):
        """Create minimal annotation records for testing."""
        records = []
        for i in range(n):
            records.append({
                "episode_id": f"ep_{i}",
                "frame_index": i,
                "objects": [
                    {
                        "track_id": f"obj_{j}",
                        "class_label": f"class_{j % 4}",
                        "bbox": [0.1, 0.1, 0.5, 0.5],
                        "confidence": 0.9,
                    }
                    for j in range(3)
                ],
                "relations": [
                    {
                        "source_track_id": "obj_0",
                        "target_track_id": "obj_1",
                        "relation_type": "near",
                        "confidence": 0.8,
                    }
                ],
            })
        return records

    def test_provisional_flag_in_output(self):
        """Default evidence_source_provisional=True must appear in result."""
        seam = AnnotationBridgeProjectionSeam(d_token=128, n_categories=16)
        records = self._make_annotation_records(10)
        result = evaluate_seam_on_annotations(
            seam=seam,
            seam_type="annotation_bridge_projection",
            annotation_records=records,
        )
        assert result["evidence_source_provisional"] is True
        assert result["promotion_eligible"] is False

    def test_insufficient_data_returns_not_present(self):
        seam = AnnotationBridgeProjectionSeam(d_token=128, n_categories=16)
        result = evaluate_seam_on_annotations(
            seam=seam,
            seam_type="annotation_bridge_projection",
            annotation_records=[],  # empty
        )
        assert result["benchmark_evidence_present"] is False
        assert result["promotion_eligible"] is False

    def test_non_provisional_allows_promotion_eligible(self):
        """With evidence_source_provisional=False, promotion_eligible is gate-scored."""
        seam = AnnotationBridgeProjectionSeam(d_token=128, n_categories=16)
        records = self._make_annotation_records(10)
        result = evaluate_seam_on_annotations(
            seam=seam,
            seam_type="annotation_bridge_projection",
            annotation_records=records,
            evidence_source_provisional=False,
        )
        assert result["evidence_source_provisional"] is False
        assert "gate_score" in result
        # promotion_eligible depends on gate_score — either True or False is valid
        assert "promotion_eligible" in result
