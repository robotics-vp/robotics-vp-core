"""Benchmark gate evaluation infrastructure for perception seams.

This module provides benchmark evaluation capabilities for determining
when perception seams should be promoted from heuristic fallback to
learned inference.

Promotion gates
---------------
Each seam type has specific benchmark criteria:

**EvidenceFusionSeam**:
- Held-out provider reconstruction accuracy
- Task correlation improvement over heuristic
- Robustness to provider dropout

**SAMCalibrationSeam**:
- Calibration error (ECE) on held-out masks
- Uncertainty correlation with true error
- Downstream mask selection improvement

**VisionBackboneProjectionSeam**:
- Object identity retrieval accuracy
- Cross-provider alignment score
- Scene classification accuracy

**DepthMetricCalibrationSeam**:
- Absolute relative error vs ground truth
- Uncertainty calibration quality
- Scale consistency across frames

**VJEPATemporalAlignmentSeam**:
- Future state prediction accuracy
- Temporal confidence calibration
- Object tracking improvement

Promotion thresholds
--------------------
Default thresholds are conservative:
- Promotion requires consistent improvement over heuristic baseline
- Demotion triggered by degradation or excessive failure rate
- Shadow mode allows gradual evaluation before full promotion
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkGateConfig:
    """Configuration for benchmark gate evaluation."""

    # Promotion thresholds
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.5
    shadow_threshold: float = 0.6

    # Evaluation settings
    n_eval_samples: int = 100
    n_bootstrap_rounds: int = 10
    confidence_level: float = 0.95

    # Robustness testing
    test_provider_dropout: bool = True
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    # Comparison to heuristic
    require_heuristic_improvement: bool = True
    heuristic_improvement_margin: float = 0.05


# ---------------------------------------------------------------------------
# Benchmark result schemas
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkMetric:
    """Single benchmark metric result."""

    name: str
    value: float
    threshold: float
    passed: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "threshold": float(self.threshold),
            "passed": bool(self.passed),
            "confidence_interval": (
                [float(self.confidence_interval[0]), float(self.confidence_interval[1])]
                if self.confidence_interval else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass
class BenchmarkGateResult:
    """Result of benchmark gate evaluation."""

    seam_id: str
    seam_type: str
    evaluation_id: str
    overall_score: float
    overall_passed: bool
    promotion_decision: str  # "promote" | "demote" | "maintain" | "shadow"
    metrics: List[BenchmarkMetric]
    heuristic_comparison: Optional[Dict[str, float]] = None
    robustness_scores: Optional[Dict[str, float]] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seam_id": self.seam_id,
            "seam_type": self.seam_type,
            "evaluation_id": self.evaluation_id,
            "overall_score": float(self.overall_score),
            "overall_passed": bool(self.overall_passed),
            "promotion_decision": self.promotion_decision,
            "metrics": [m.to_dict() for m in self.metrics],
            "heuristic_comparison": (
                {k: float(v) for k, v in self.heuristic_comparison.items()}
                if self.heuristic_comparison else None
            ),
            "robustness_scores": (
                {k: float(v) for k, v in self.robustness_scores.items()}
                if self.robustness_scores else None
            ),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Seam-specific benchmark evaluators
# ---------------------------------------------------------------------------


class EvidenceFusionBenchmark:
    """Benchmark evaluator for EvidenceFusionSeam."""

    def __init__(self, config: Optional[BenchmarkGateConfig] = None):
        self.config = config or BenchmarkGateConfig()

    def evaluate(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
        *,
        heuristic_baseline: Optional[Callable] = None,
    ) -> BenchmarkGateResult:
        """Evaluate evidence fusion seam against benchmark criteria."""
        seam.eval()
        metrics: List[BenchmarkMetric] = []

        # Collect predictions and targets
        all_weights = []
        all_confidences = []
        all_held_out_errors = []
        all_task_correlations = []

        with torch.no_grad():
            for batch in eval_loader:
                # Seam takes 12-dim encoded metadata; raw features used for reconstruction
                seam_in = batch.seam_input_features if hasattr(batch, "seam_input_features") else batch.provider_features
                weights, confidence = seam(seam_in)
                all_weights.append(weights)
                all_confidences.append(confidence)

                # Held-out reconstruction error (uses raw provider features)
                held_out_pred = (weights.unsqueeze(-1) * batch.provider_features).sum(dim=1)
                held_out_error = (held_out_pred - batch.held_out_features).pow(2).mean(dim=-1)
                all_held_out_errors.append(held_out_error)

                # Task correlation
                if batch.task_correlation_target is not None:
                    all_task_correlations.append(
                        (confidence, batch.task_correlation_target)
                    )

        # Compute metrics
        # 1. Held-out reconstruction accuracy
        all_errors = torch.cat(all_held_out_errors)
        reconstruction_accuracy = 1.0 - all_errors.mean().item()
        metrics.append(BenchmarkMetric(
            name="held_out_reconstruction_accuracy",
            value=reconstruction_accuracy,
            threshold=0.7,
            passed=reconstruction_accuracy >= 0.7,
        ))

        # 2. Confidence-task correlation
        if all_task_correlations:
            confs = torch.cat([c for c, _ in all_task_correlations])
            targets = torch.cat([t for _, t in all_task_correlations])
            if confs.numel() > 1:
                correlation = torch.corrcoef(torch.stack([confs, targets]))[0, 1].item()
            else:
                correlation = 0.0
            metrics.append(BenchmarkMetric(
                name="confidence_task_correlation",
                value=correlation,
                threshold=0.3,
                passed=correlation >= 0.3,
            ))

        # 3. Weight entropy (measure of decisiveness)
        all_weights_cat = torch.cat(all_weights)
        weight_entropy = -(
            all_weights_cat * (all_weights_cat + 1e-8).log()
        ).sum(dim=-1).mean().item()
        n_providers = all_weights_cat.size(-1)
        max_entropy = torch.log(torch.tensor(n_providers)).item()
        normalized_entropy = weight_entropy / max_entropy
        metrics.append(BenchmarkMetric(
            name="weight_decisiveness",
            value=1.0 - normalized_entropy,
            threshold=0.3,
            passed=(1.0 - normalized_entropy) >= 0.3,
        ))

        # Robustness testing (provider dropout)
        robustness_scores = {}
        if self.config.test_provider_dropout:
            for dropout_rate in self.config.dropout_rates:
                dropout_score = self._evaluate_dropout_robustness(
                    seam, eval_loader, dropout_rate
                )
                robustness_scores[f"dropout_{dropout_rate:.1f}"] = dropout_score

        # Heuristic comparison
        heuristic_comparison = None
        if heuristic_baseline is not None:
            heuristic_comparison = self._compare_to_heuristic(
                seam, heuristic_baseline, eval_loader
            )

        # Overall score
        metric_scores = [m.value for m in metrics if m.value >= 0]
        overall_score = sum(metric_scores) / max(1, len(metric_scores))
        overall_passed = all(m.passed for m in metrics)

        # Promotion decision
        if overall_score >= self.config.promotion_threshold and overall_passed:
            promotion_decision = "promote"
        elif overall_score < self.config.demotion_threshold:
            promotion_decision = "demote"
        elif overall_score >= self.config.shadow_threshold:
            promotion_decision = "shadow"
        else:
            promotion_decision = "maintain"

        return BenchmarkGateResult(
            seam_id="",  # Set by caller
            seam_type="evidence_fusion",
            evaluation_id=f"bench_{uuid.uuid4().hex[:12]}",
            overall_score=overall_score,
            overall_passed=overall_passed,
            promotion_decision=promotion_decision,
            metrics=metrics,
            heuristic_comparison=heuristic_comparison,
            robustness_scores=robustness_scores,
        )

    def _evaluate_dropout_robustness(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
        dropout_rate: float,
    ) -> float:
        """Evaluate seam robustness to provider dropout."""
        seam.eval()
        original_errors = []
        dropout_errors = []

        with torch.no_grad():
            for batch in eval_loader:
                seam_in = batch.seam_input_features if hasattr(batch, "seam_input_features") else batch.provider_features
                # Original prediction
                weights, _ = seam(seam_in)
                held_out_pred = (weights.unsqueeze(-1) * batch.provider_features).sum(dim=1)
                original_error = (held_out_pred - batch.held_out_features).pow(2).mean()
                original_errors.append(original_error)

                # Dropout prediction — zero out both seam input and raw features
                dropout_mask = torch.rand_like(batch.provider_availability.float()) > dropout_rate
                masked_seam_in = seam_in * dropout_mask.unsqueeze(-1)
                masked_features = batch.provider_features * dropout_mask.unsqueeze(-1)
                weights_drop, _ = seam(masked_seam_in)
                held_out_pred_drop = (weights_drop.unsqueeze(-1) * masked_features).sum(dim=1)
                dropout_error = (held_out_pred_drop - batch.held_out_features).pow(2).mean()
                dropout_errors.append(dropout_error)

        original_mean = torch.stack(original_errors).mean().item()
        dropout_mean = torch.stack(dropout_errors).mean().item()

        # Robustness = how much error increases with dropout (lower is better)
        if original_mean > 0:
            degradation = (dropout_mean - original_mean) / original_mean
            robustness = max(0.0, 1.0 - degradation)
        else:
            robustness = 1.0

        return robustness

    def _compare_to_heuristic(
        self,
        seam: nn.Module,
        heuristic: Callable,
        eval_loader: DataLoader,
    ) -> Dict[str, float]:
        """Compare seam performance to heuristic baseline."""
        seam_errors = []
        heuristic_errors = []

        with torch.no_grad():
            for batch in eval_loader:
                seam_in = batch.seam_input_features if hasattr(batch, "seam_input_features") else batch.provider_features
                # Seam prediction
                weights, _ = seam(seam_in)
                seam_pred = (weights.unsqueeze(-1) * batch.provider_features).sum(dim=1)
                seam_error = (seam_pred - batch.held_out_features).pow(2).mean()
                seam_errors.append(seam_error)

                # Heuristic prediction (operates on raw features)
                heuristic_weights = heuristic(batch.provider_features)
                heuristic_pred = (heuristic_weights.unsqueeze(-1) * batch.provider_features).sum(dim=1)
                heuristic_error = (heuristic_pred - batch.held_out_features).pow(2).mean()
                heuristic_errors.append(heuristic_error)

        seam_mean = torch.stack(seam_errors).mean().item()
        heuristic_mean = torch.stack(heuristic_errors).mean().item()

        return {
            "seam_error": seam_mean,
            "heuristic_error": heuristic_mean,
            "improvement": (heuristic_mean - seam_mean) / max(heuristic_mean, 1e-8),
            "beats_heuristic": seam_mean < heuristic_mean,
        }


class SAMCalibrationBenchmark:
    """Benchmark evaluator for SAMCalibrationSeam."""

    def __init__(self, config: Optional[BenchmarkGateConfig] = None):
        self.config = config or BenchmarkGateConfig()

    def evaluate(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
        *,
        n_bins: int = 10,
    ) -> BenchmarkGateResult:
        """Evaluate SAM calibration seam against benchmark criteria."""
        seam.eval()
        metrics: List[BenchmarkMetric] = []

        all_confidences = []
        all_uncertainties = []
        all_qualities = []
        all_disagreements = []

        with torch.no_grad():
            for batch in eval_loader:
                result = seam(batch.mask_features, batch.raw_confidence, batch.mask_valid)
                valid = batch.mask_valid

                all_confidences.append(result["calibrated_confidence"][valid])
                all_uncertainties.append(result["epistemic_uncertainty"][valid])
                all_qualities.append(batch.downstream_quality[valid])
                if batch.provider_disagreement is not None:
                    all_disagreements.append(batch.provider_disagreement[valid])

        # Compute metrics
        confidences = torch.cat(all_confidences)
        qualities = torch.cat(all_qualities)
        uncertainties = torch.cat(all_uncertainties)

        # 1. Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences, qualities, n_bins)
        metrics.append(BenchmarkMetric(
            name="expected_calibration_error",
            value=1.0 - ece,  # Convert to accuracy
            threshold=0.85,
            passed=ece <= 0.15,
        ))

        # 2. Uncertainty-error correlation
        errors = (confidences - qualities).abs()
        if uncertainties.numel() > 1 and errors.numel() > 1:
            unc_corr = torch.corrcoef(torch.stack([uncertainties, errors]))[0, 1].item()
        else:
            unc_corr = 0.0
        metrics.append(BenchmarkMetric(
            name="uncertainty_error_correlation",
            value=unc_corr,
            threshold=0.3,
            passed=unc_corr >= 0.3,
        ))

        # 3. Confidence-quality correlation
        if confidences.numel() > 1:
            conf_corr = torch.corrcoef(torch.stack([confidences, qualities]))[0, 1].item()
        else:
            conf_corr = 0.0
        metrics.append(BenchmarkMetric(
            name="confidence_quality_correlation",
            value=conf_corr,
            threshold=0.5,
            passed=conf_corr >= 0.5,
        ))

        # Overall score
        metric_scores = [m.value for m in metrics if not torch.isnan(torch.tensor(m.value))]
        overall_score = sum(metric_scores) / max(1, len(metric_scores))
        overall_passed = all(m.passed for m in metrics)

        # Promotion decision
        if overall_score >= self.config.promotion_threshold and overall_passed:
            promotion_decision = "promote"
        elif overall_score < self.config.demotion_threshold:
            promotion_decision = "demote"
        elif overall_score >= self.config.shadow_threshold:
            promotion_decision = "shadow"
        else:
            promotion_decision = "maintain"

        return BenchmarkGateResult(
            seam_id="",
            seam_type="sam_calibration",
            evaluation_id=f"bench_{uuid.uuid4().hex[:12]}",
            overall_score=overall_score,
            overall_passed=overall_passed,
            promotion_decision=promotion_decision,
            metrics=metrics,
        )

    def _compute_ece(
        self,
        confidences: torch.Tensor,
        accuracies: torch.Tensor,
        n_bins: int,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = confidences.numel()

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            n_in_bin = in_bin.sum().item()

            if n_in_bin > 0:
                avg_confidence = confidences[in_bin].mean().item()
                avg_accuracy = accuracies[in_bin].mean().item()
                ece += (n_in_bin / total_samples) * abs(avg_accuracy - avg_confidence)

        return ece


class VisionBackboneProjectionBenchmark:
    """Benchmark evaluator for VisionBackboneProjectionSeam."""

    def __init__(self, config: Optional[BenchmarkGateConfig] = None):
        self.config = config or BenchmarkGateConfig()

    @staticmethod
    def _centroid_accuracy(
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        valid_mask = labels >= 0
        if valid_mask.sum() <= 1:
            return 0.0
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        unique_labels = torch.unique(valid_labels)
        if unique_labels.numel() <= 0:
            return 0.0
        centroids = torch.stack(
            [
                valid_features[valid_labels == label].mean(dim=0)
                for label in unique_labels
            ],
            dim=0,
        )
        sims = torch.nn.functional.cosine_similarity(
            valid_features.unsqueeze(1),
            centroids.unsqueeze(0),
            dim=-1,
        )
        pred_labels = unique_labels[sims.argmax(dim=-1)]
        return float((pred_labels == valid_labels).float().mean().item())

    def evaluate(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
    ) -> BenchmarkGateResult:
        """Evaluate projection quality on identity, scene, and alignment signals."""
        seam.eval()
        projected_batches: List[torch.Tensor] = []
        label_batches: List[torch.Tensor] = []
        pooled_scene_features: List[torch.Tensor] = []
        scene_label_batches: List[torch.Tensor] = []
        alignment_scores: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in eval_loader:
                projected = seam(batch.backbone_features)
                projected_batches.append(projected.reshape(-1, projected.size(-1)))
                label_batches.append(batch.object_identity_labels.reshape(-1))

                if batch.scene_labels is not None:
                    pooled_scene_features.append(projected.mean(dim=1))
                    scene_label_batches.append(batch.scene_labels)

                if batch.cross_provider_embeddings is not None:
                    mse = torch.nn.functional.mse_loss(
                        projected,
                        batch.cross_provider_embeddings,
                        reduction="none",
                    ).mean(dim=(-1, -2))
                    alignment_scores.append(1.0 / (1.0 + mse))

        flat_projected = torch.cat(projected_batches, dim=0)
        flat_labels = torch.cat(label_batches, dim=0)
        identity_accuracy = self._centroid_accuracy(flat_projected, flat_labels)

        metrics: List[BenchmarkMetric] = [
            BenchmarkMetric(
                name="object_identity_retrieval_accuracy",
                value=identity_accuracy,
                threshold=0.6,
                passed=identity_accuracy >= 0.6,
            )
        ]

        if pooled_scene_features and scene_label_batches:
            pooled = torch.cat(pooled_scene_features, dim=0)
            scene_labels = torch.cat(scene_label_batches, dim=0)
            scene_accuracy = self._centroid_accuracy(pooled, scene_labels)
            metrics.append(
                BenchmarkMetric(
                    name="scene_retrieval_accuracy",
                    value=scene_accuracy,
                    threshold=0.6,
                    passed=scene_accuracy >= 0.6,
                )
            )

        if alignment_scores:
            alignment_score = torch.cat(alignment_scores).mean().item()
            metrics.append(
                BenchmarkMetric(
                    name="cross_provider_alignment_score",
                    value=alignment_score,
                    threshold=0.7,
                    passed=alignment_score >= 0.7,
                )
            )

        metric_scores = [metric.value for metric in metrics]
        overall_score = sum(metric_scores) / max(1, len(metric_scores))
        overall_passed = all(metric.passed for metric in metrics)

        if overall_score >= self.config.promotion_threshold and overall_passed:
            promotion_decision = "promote"
        elif overall_score < self.config.demotion_threshold:
            promotion_decision = "demote"
        elif overall_score >= self.config.shadow_threshold:
            promotion_decision = "shadow"
        else:
            promotion_decision = "maintain"

        return BenchmarkGateResult(
            seam_id="",
            seam_type="vision_backbone_projection",
            evaluation_id=f"bench_{uuid.uuid4().hex[:12]}",
            overall_score=overall_score,
            overall_passed=overall_passed,
            promotion_decision=promotion_decision,
            metrics=metrics,
        )


class DepthCalibrationBenchmark:
    """Benchmark evaluator for DepthMetricCalibrationSeam."""

    def __init__(self, config: Optional[BenchmarkGateConfig] = None):
        self.config = config or BenchmarkGateConfig()

    def evaluate(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
    ) -> BenchmarkGateResult:
        """Evaluate depth calibration seam against benchmark criteria."""
        seam.eval()
        metrics: List[BenchmarkMetric] = []

        all_abs_rel = []
        all_rmse = []
        all_delta1 = []
        all_unc_correlations = []

        with torch.no_grad():
            for batch in eval_loader:
                result = seam(batch.relative_depth, batch.camera_intrinsics)
                pred_depth = result["metric_depth"]
                uncertainty = result["uncertainty"]
                gt_depth = batch.ground_truth_depth
                valid = batch.depth_valid_mask

                # Mask valid pixels
                pred_valid = pred_depth[valid]
                gt_valid = gt_depth[valid]
                unc_valid = uncertainty[valid]

                if pred_valid.numel() == 0:
                    continue

                # Absolute relative error
                abs_rel = ((pred_valid - gt_valid).abs() / gt_valid.clamp(min=1e-8)).mean()
                all_abs_rel.append(abs_rel)

                # RMSE
                rmse = ((pred_valid - gt_valid).pow(2).mean()).sqrt()
                all_rmse.append(rmse)

                # Delta < 1.25
                ratio = torch.max(pred_valid / gt_valid.clamp(min=1e-8), gt_valid / pred_valid.clamp(min=1e-8))
                delta1 = (ratio < 1.25).float().mean()
                all_delta1.append(delta1)

                # Uncertainty-error correlation
                errors = (pred_valid - gt_valid).abs()
                if unc_valid.numel() > 1:
                    corr = torch.corrcoef(torch.stack([unc_valid.flatten()[:errors.numel()], errors.flatten()]))[0, 1]
                    if not torch.isnan(corr):
                        all_unc_correlations.append(corr)

        # Compute aggregate metrics
        if all_abs_rel:
            abs_rel = torch.stack(all_abs_rel).mean().item()
            metrics.append(BenchmarkMetric(
                name="absolute_relative_error",
                value=1.0 - min(abs_rel, 1.0),  # Convert to accuracy
                threshold=0.85,
                passed=abs_rel <= 0.15,
            ))

        if all_delta1:
            delta1 = torch.stack(all_delta1).mean().item()
            metrics.append(BenchmarkMetric(
                name="delta_1.25_accuracy",
                value=delta1,
                threshold=0.85,
                passed=delta1 >= 0.85,
            ))

        if all_unc_correlations:
            unc_corr = torch.stack(all_unc_correlations).mean().item()
            metrics.append(BenchmarkMetric(
                name="uncertainty_error_correlation",
                value=unc_corr,
                threshold=0.3,
                passed=unc_corr >= 0.3,
            ))

        # Overall score
        metric_scores = [m.value for m in metrics]
        overall_score = sum(metric_scores) / max(1, len(metric_scores)) if metric_scores else 0.0
        overall_passed = all(m.passed for m in metrics) if metrics else False

        # Promotion decision
        if overall_score >= self.config.promotion_threshold and overall_passed:
            promotion_decision = "promote"
        elif overall_score < self.config.demotion_threshold:
            promotion_decision = "demote"
        elif overall_score >= self.config.shadow_threshold:
            promotion_decision = "shadow"
        else:
            promotion_decision = "maintain"

        return BenchmarkGateResult(
            seam_id="",
            seam_type="depth_metric_calibration",
            evaluation_id=f"bench_{uuid.uuid4().hex[:12]}",
            overall_score=overall_score,
            overall_passed=overall_passed,
            promotion_decision=promotion_decision,
            metrics=metrics,
        )


class VJEPATemporalBenchmark:
    """Benchmark evaluator for VJEPATemporalAlignmentSeam."""

    def __init__(self, config: Optional[BenchmarkGateConfig] = None):
        self.config = config or BenchmarkGateConfig()

    def evaluate(
        self,
        seam: nn.Module,
        eval_loader: DataLoader,
    ) -> BenchmarkGateResult:
        """Evaluate V-JEPA temporal alignment seam against benchmark criteria."""
        seam.eval()
        metrics: List[BenchmarkMetric] = []

        all_pred_errors = []
        all_conf_correlations = []
        all_temporal_consistencies = []

        with torch.no_grad():
            for batch in eval_loader:
                result = seam(batch.vjepa_tokens, batch.wm_object_tokens)
                pred = result["temporal_aligned"]
                conf = result["temporal_confidence"]
                target = batch.future_object_states
                valid = batch.object_valid_mask

                # Prediction error per timestep
                valid_mask = valid.unsqueeze(1).unsqueeze(-1).expand_as(pred)
                errors = ((pred - target).pow(2) * valid_mask).sum(dim=(2, 3)) / valid_mask.sum(dim=(2, 3)).clamp(min=1.0)
                all_pred_errors.append(errors.mean())

                # Confidence-accuracy correlation
                per_t_accuracy = 1.0 - errors.clamp(max=1.0)
                if conf.numel() > 1:
                    corr = torch.corrcoef(torch.stack([conf.flatten(), per_t_accuracy.flatten()]))[0, 1]
                    if not torch.isnan(corr):
                        all_conf_correlations.append(corr)

                # Temporal consistency (smoothness)
                if pred.size(1) > 1:
                    temporal_diff = (pred[:, 1:] - pred[:, :-1]).pow(2).mean()
                    all_temporal_consistencies.append(temporal_diff)

        # Compute aggregate metrics
        if all_pred_errors:
            pred_error = torch.stack(all_pred_errors).mean().item()
            pred_accuracy = 1.0 - min(pred_error, 1.0)
            metrics.append(BenchmarkMetric(
                name="prediction_accuracy",
                value=pred_accuracy,
                threshold=0.6,
                passed=pred_accuracy >= 0.6,
            ))

        if all_conf_correlations:
            conf_corr = torch.stack(all_conf_correlations).mean().item()
            metrics.append(BenchmarkMetric(
                name="confidence_accuracy_correlation",
                value=conf_corr,
                threshold=0.3,
                passed=conf_corr >= 0.3,
            ))

        if all_temporal_consistencies:
            consistency = 1.0 - torch.stack(all_temporal_consistencies).mean().item()
            metrics.append(BenchmarkMetric(
                name="temporal_consistency",
                value=consistency,
                threshold=0.7,
                passed=consistency >= 0.7,
            ))

        # Overall score
        metric_scores = [m.value for m in metrics]
        overall_score = sum(metric_scores) / max(1, len(metric_scores)) if metric_scores else 0.0
        overall_passed = all(m.passed for m in metrics) if metrics else False

        # Promotion decision
        if overall_score >= self.config.promotion_threshold and overall_passed:
            promotion_decision = "promote"
        elif overall_score < self.config.demotion_threshold:
            promotion_decision = "demote"
        elif overall_score >= self.config.shadow_threshold:
            promotion_decision = "shadow"
        else:
            promotion_decision = "maintain"

        return BenchmarkGateResult(
            seam_id="",
            seam_type="vjepa_temporal_alignment",
            evaluation_id=f"bench_{uuid.uuid4().hex[:12]}",
            overall_score=overall_score,
            overall_passed=overall_passed,
            promotion_decision=promotion_decision,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------


BENCHMARK_REGISTRY: Dict[str, type] = {
    "evidence_fusion": EvidenceFusionBenchmark,
    "sam_calibration": SAMCalibrationBenchmark,
    "vision_backbone_projection": VisionBackboneProjectionBenchmark,
    "depth_metric_calibration": DepthCalibrationBenchmark,
    "vjepa_temporal_alignment": VJEPATemporalBenchmark,
}


def get_benchmark_evaluator(
    seam_type: str,
    config: Optional[BenchmarkGateConfig] = None,
):
    """Get benchmark evaluator for a seam type."""
    if seam_type not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown seam type: {seam_type}. "
            f"Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[seam_type](config)


def evaluate_seam_benchmark(
    seam: nn.Module,
    seam_type: str,
    eval_loader: DataLoader,
    *,
    seam_id: str = "",
    config: Optional[BenchmarkGateConfig] = None,
) -> BenchmarkGateResult:
    """Evaluate a seam against its benchmark criteria.

    Convenience function that gets the right evaluator and runs evaluation.
    """
    evaluator = get_benchmark_evaluator(seam_type, config)
    result = evaluator.evaluate(seam, eval_loader)

    # Set seam_id
    result = BenchmarkGateResult(
        seam_id=seam_id,
        seam_type=result.seam_type,
        evaluation_id=result.evaluation_id,
        overall_score=result.overall_score,
        overall_passed=result.overall_passed,
        promotion_decision=result.promotion_decision,
        metrics=result.metrics,
        heuristic_comparison=result.heuristic_comparison,
        robustness_scores=result.robustness_scores,
        timestamp=result.timestamp,
        metadata=result.metadata,
    )

    return result


__all__ = [
    # Config
    "BenchmarkGateConfig",
    # Results
    "BenchmarkMetric",
    "BenchmarkGateResult",
    # Evaluators
    "EvidenceFusionBenchmark",
    "SAMCalibrationBenchmark",
    "VisionBackboneProjectionBenchmark",
    "DepthCalibrationBenchmark",
    "VJEPATemporalBenchmark",
    # Registry
    "BENCHMARK_REGISTRY",
    "get_benchmark_evaluator",
    "evaluate_seam_benchmark",
]
