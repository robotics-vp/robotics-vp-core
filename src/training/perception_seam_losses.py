"""Loss functions for perception seam training.

This module provides supervised/contrastive/predictive loss functions for
training the perception neural seams.  These are NOT RL losses — the seams
are middleware that transforms provider outputs into WM-native form.

Training regime
---------------
All seams are trained on provider agreement data, cross-provider
disagreement signals, and downstream task correlation (as frozen target).
The training objectives are supervised/contrastive/predictive, never
direct RL on task reward.

Loss functions per seam type
----------------------------
See ``neural_seams.py`` docstring for full training objective documentation.

**EvidenceFusionSeam**:
- ``evidence_fusion_loss``: held-out provider reconstruction + task correlation

**SAMCalibrationSeam**:
- ``sam_calibration_loss``: calibrated confidence vs downstream mask quality

**VisionBackboneProjectionSeam**:
- ``vision_backbone_projection_loss``: object identity prediction + contrastive

**DepthMetricCalibrationSeam**:
- ``depth_metric_calibration_loss``: metric depth vs ground truth + uncertainty

**VJEPATemporalAlignmentSeam**:
- ``vjepa_temporal_alignment_loss``: future object state prediction + ordering
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceFusionLossConfig:
    """Configuration for evidence fusion loss computation."""

    held_out_weight: float = 1.0
    task_correlation_weight: float = 0.5
    availability_contrastive_weight: float = 0.2
    confidence_entropy_weight: float = 0.1
    label_smoothing: float = 0.1


@dataclass(frozen=True)
class SAMCalibrationLossConfig:
    """Configuration for SAM calibration loss computation."""

    calibration_weight: float = 1.0
    uncertainty_weight: float = 0.5
    prompt_satisfaction_weight: float = 0.3
    temperature: float = 1.0


@dataclass(frozen=True)
class VisionBackboneProjectionLossConfig:
    """Configuration for vision backbone projection loss computation."""

    identity_weight: float = 1.0
    contrastive_weight: float = 0.5
    alignment_weight: float = 0.3
    temperature: float = 0.07


@dataclass(frozen=True)
class DepthMetricCalibrationLossConfig:
    """Configuration for depth metric calibration loss computation."""

    depth_weight: float = 1.0
    uncertainty_weight: float = 0.5
    scale_consistency_weight: float = 0.2
    gradient_weight: float = 0.1


@dataclass(frozen=True)
class VJEPATemporalAlignmentLossConfig:
    """Configuration for V-JEPA temporal alignment loss computation."""

    prediction_weight: float = 1.0
    confidence_weight: float = 0.5
    ordering_weight: float = 0.3
    smoothness_weight: float = 0.1


@dataclass(frozen=True)
class AnnotationBridgeProjectionLossConfig:
    """Configuration for annotation bridge projection loss computation."""

    class_weight: float = 1.0
    confidence_weight: float = 0.5
    affordance_weight: float = 0.3
    label_smoothing: float = 0.1


# ---------------------------------------------------------------------------
# Loss result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SeamLossResult:
    """Result of computing a seam loss.

    Provides total loss plus component breakdown for logging/debugging.
    """

    total_loss: torch.Tensor
    component_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": float(self.total_loss.item()),
            "component_losses": {
                k: float(v.item()) for k, v in self.component_losses.items()
            },
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------------
# Evidence Fusion Seam Loss
# ---------------------------------------------------------------------------


def evidence_fusion_loss(
    *,
    predicted_weights: torch.Tensor,
    predicted_confidence: torch.Tensor,
    held_out_provider_idx: torch.Tensor,
    held_out_reconstruction_target: torch.Tensor,
    provider_features: torch.Tensor,
    task_correlation_target: Optional[torch.Tensor] = None,
    provider_availability_mask: Optional[torch.Tensor] = None,
    config: Optional[EvidenceFusionLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for EvidenceFusionSeam training.

    Primary objective: the fused state (minus held-out provider) should
    reconstruct the held-out provider's contribution when re-weighted.

    Secondary objective: predicted confidence should correlate with
    downstream task success (frozen target, no gradient to task).

    Auxiliary: contrastive loss on provider availability patterns to
    encourage robust fusion when providers are missing.

    Args:
        predicted_weights: ``(batch, N_providers)`` — softmax fusion weights.
        predicted_confidence: ``(batch,)`` — fusion confidence in [0, 1].
        held_out_provider_idx: ``(batch,)`` — index of held-out provider.
        held_out_reconstruction_target: ``(batch, d_feature)`` — target
            features from the held-out provider.
        provider_features: ``(batch, N_providers, d_feature)`` — raw
            provider features for reconstruction.
        task_correlation_target: ``(batch,)`` — downstream task success
            score to correlate with confidence (optional, frozen target).
        provider_availability_mask: ``(batch, N_providers)`` — True if
            provider was available (for contrastive auxiliary).
        config: Loss weights and hyperparameters.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = EvidenceFusionLossConfig()

    batch_size = predicted_weights.size(0)
    device = predicted_weights.device

    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Primary loss: held-out provider reconstruction
    # Mask out the held-out provider and predict its features from the rest
    held_out_mask = torch.zeros_like(predicted_weights, dtype=torch.bool)
    held_out_mask.scatter_(1, held_out_provider_idx.unsqueeze(1), True)

    # Re-normalize weights without held-out
    masked_weights = predicted_weights.clone()
    masked_weights[held_out_mask] = 0.0
    weight_sum = masked_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    renorm_weights = masked_weights / weight_sum  # (batch, N)

    # Weighted sum of provider features (excluding held-out)
    fused_features = (renorm_weights.unsqueeze(-1) * provider_features).sum(dim=1)

    # L2 reconstruction loss against held-out
    held_out_loss = F.mse_loss(fused_features, held_out_reconstruction_target)
    component_losses["held_out_reconstruction"] = held_out_loss

    # Secondary loss: confidence-task correlation
    task_corr_loss = torch.tensor(0.0, device=device)
    if task_correlation_target is not None:
        # Encourage confidence to correlate with task success
        # Use MSE between predicted confidence and task success (as soft target)
        task_corr_loss = F.mse_loss(predicted_confidence, task_correlation_target)
        component_losses["task_correlation"] = task_corr_loss
        metrics["confidence_task_corr"] = float(
            torch.corrcoef(
                torch.stack([predicted_confidence, task_correlation_target])
            )[0, 1].item()
            if batch_size > 1
            else 0.0
        )

    # Auxiliary: availability contrastive loss
    avail_contrastive_loss = torch.tensor(0.0, device=device)
    if provider_availability_mask is not None:
        # Encourage similar weight patterns for similar availability patterns
        # Simple approach: entropy on weights (lower entropy = more decisive)
        weight_entropy = (
            -(predicted_weights * (predicted_weights + 1e-8).log()).sum(dim=-1).mean()
        )
        avail_contrastive_loss = -weight_entropy  # Maximize entropy for robustness
        component_losses["availability_contrastive"] = avail_contrastive_loss

    # Confidence entropy regularization (prevent overconfidence)
    conf_entropy = -(
        predicted_confidence * (predicted_confidence + 1e-8).log()
        + (1 - predicted_confidence) * (1 - predicted_confidence + 1e-8).log()
    ).mean()
    component_losses["confidence_entropy"] = -conf_entropy

    # Total loss
    total_loss = (
        config.held_out_weight * held_out_loss
        + config.task_correlation_weight * task_corr_loss
        + config.availability_contrastive_weight * avail_contrastive_loss
        + config.confidence_entropy_weight * (-conf_entropy)
    )

    # Metrics
    metrics["mean_confidence"] = float(predicted_confidence.mean().item())
    metrics["weight_entropy"] = float(
        -(predicted_weights * (predicted_weights + 1e-8).log())
        .sum(dim=-1)
        .mean()
        .item()
    )

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# SAM Calibration Seam Loss
# ---------------------------------------------------------------------------


def sam_calibration_loss(
    *,
    calibrated_confidence: torch.Tensor,
    epistemic_uncertainty: torch.Tensor,
    prompt_satisfaction: torch.Tensor,
    downstream_mask_quality: torch.Tensor,
    provider_disagreement: Optional[torch.Tensor] = None,
    segmentation_iou: Optional[torch.Tensor] = None,
    mask_valid: Optional[torch.Tensor] = None,
    config: Optional[SAMCalibrationLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for SAMCalibrationSeam training.

    Primary: calibrated confidence should match downstream mask quality
    (e.g., from held-out evaluator or human annotation).

    Secondary: epistemic uncertainty should correlate with cross-provider
    disagreement on the same mask.

    Auxiliary: prompt satisfaction should correlate with segmentation IoU.

    Args:
        calibrated_confidence: ``(batch, N_masks)`` — predicted calibrated
            confidence.
        epistemic_uncertainty: ``(batch, N_masks)`` — predicted epistemic
            uncertainty.
        prompt_satisfaction: ``(batch, N_masks)`` — predicted prompt
            satisfaction.
        downstream_mask_quality: ``(batch, N_masks)`` — ground truth
            quality score (from evaluator or human label).
        provider_disagreement: ``(batch, N_masks)`` — cross-provider
            disagreement score (optional).
        segmentation_iou: ``(batch, N_masks)`` — IoU with ground truth
            segmentation (optional).
        mask_valid: ``(batch, N_masks)`` — True if mask is valid.
        config: Loss weights and hyperparameters.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = SAMCalibrationLossConfig()

    device = calibrated_confidence.device
    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Apply mask
    if mask_valid is not None:
        valid_mask = mask_valid.float()
        n_valid = valid_mask.sum().clamp(min=1.0)
    else:
        valid_mask = torch.ones_like(calibrated_confidence)
        n_valid = torch.tensor(
            calibrated_confidence.numel(), device=device, dtype=torch.float
        )

    # Primary: calibration loss (calibrated confidence vs mask quality)
    calibration_loss = (
        F.mse_loss(
            calibrated_confidence * valid_mask,
            downstream_mask_quality * valid_mask,
            reduction="sum",
        )
        / n_valid
    )
    component_losses["calibration"] = calibration_loss

    # Secondary: uncertainty correlation with disagreement
    uncertainty_loss = torch.tensor(0.0, device=device)
    if provider_disagreement is not None:
        # Uncertainty should be high where disagreement is high
        uncertainty_loss = (
            F.mse_loss(
                epistemic_uncertainty * valid_mask,
                provider_disagreement * valid_mask,
                reduction="sum",
            )
            / n_valid
        )
        component_losses["uncertainty"] = uncertainty_loss

    # Auxiliary: prompt satisfaction correlation with IoU
    prompt_sat_loss = torch.tensor(0.0, device=device)
    if segmentation_iou is not None:
        prompt_sat_loss = (
            F.mse_loss(
                prompt_satisfaction * valid_mask,
                segmentation_iou * valid_mask,
                reduction="sum",
            )
            / n_valid
        )
        component_losses["prompt_satisfaction"] = prompt_sat_loss

    # Total loss
    total_loss = (
        config.calibration_weight * calibration_loss
        + config.uncertainty_weight * uncertainty_loss
        + config.prompt_satisfaction_weight * prompt_sat_loss
    )

    # Metrics
    valid_conf = (
        calibrated_confidence[mask_valid]
        if mask_valid is not None
        else calibrated_confidence
    )
    metrics["mean_calibrated_confidence"] = float(valid_conf.mean().item())
    metrics["calibration_mse"] = float(calibration_loss.item())

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Vision Backbone Projection Seam Loss
# ---------------------------------------------------------------------------


def vision_backbone_projection_loss(
    *,
    projected_features: torch.Tensor,
    object_identity_labels: torch.Tensor,
    scene_labels: Optional[torch.Tensor] = None,
    cross_provider_embeddings: Optional[torch.Tensor] = None,
    config: Optional[VisionBackboneProjectionLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for VisionBackboneProjectionSeam training.

    Primary: projected features should predict downstream object identity
    (supervised classification or retrieval).

    Secondary: contrastive loss between projected features and scene-level
    labels (attract same-scene, repel different-scene).

    Auxiliary: alignment with other provider embeddings (e.g., V-JEPA
    spatial tokens, depth features).

    Args:
        projected_features: ``(batch, N_tokens, d_out)`` — projected
            backbone features.
        object_identity_labels: ``(batch, N_tokens)`` — object identity
            class labels (-1 for background).
        scene_labels: ``(batch,)`` — scene-level labels for contrastive
            (optional).
        cross_provider_embeddings: ``(batch, N_tokens, d_out)`` — embeddings
            from other providers to align with (optional).
        config: Loss weights and hyperparameters.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = VisionBackboneProjectionLossConfig()

    device = projected_features.device
    batch_size = projected_features.size(0)
    d_out = projected_features.size(2)

    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Primary: object identity prediction (proxy: same-identity tokens should be similar)
    # Group tokens by identity and compute within-group vs across-group distances
    identity_loss = torch.tensor(0.0, device=device)

    # Flatten for easier processing
    flat_features = projected_features.reshape(-1, d_out)  # (batch*N, d_out)
    flat_labels = object_identity_labels.reshape(-1)  # (batch*N,)

    # Filter valid labels (not background)
    valid_mask = flat_labels >= 0
    if valid_mask.sum() > 1:
        valid_features = flat_features[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # Compute pairwise similarities
        valid_features_norm = F.normalize(valid_features, dim=-1)
        similarity = torch.mm(valid_features_norm, valid_features_norm.t())

        # Create label match matrix
        label_match = (valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)).float()

        # InfoNCE-style loss: pull same-identity, push different-identity
        # Log-sum-exp over negatives
        n_valid = valid_features.size(0)
        if n_valid > 1:
            # Mask out self-similarity
            mask_self = torch.eye(n_valid, device=device, dtype=torch.bool)
            similarity_masked = similarity.masked_fill(mask_self, -1e9)

            # Positive mask (same identity, not self)
            pos_mask = label_match.bool() & ~mask_self

            # For each anchor, compute contrastive loss
            # Simplified: cross-entropy with positives as targets
            if pos_mask.any():
                logits = similarity_masked / config.temperature
                # Soft labels based on identity match
                targets = label_match.masked_fill(mask_self, 0)
                targets = targets / targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
                identity_loss = F.cross_entropy(logits, targets, reduction="mean")

    component_losses["identity"] = identity_loss

    # Secondary: scene-level contrastive
    scene_contrastive_loss = torch.tensor(0.0, device=device)
    if scene_labels is not None and batch_size > 1:
        # Pool features per scene
        pooled = projected_features.mean(dim=1)  # (batch, d_out)
        pooled_norm = F.normalize(pooled, dim=-1)

        # Pairwise scene similarity
        scene_sim = torch.mm(pooled_norm, pooled_norm.t())
        scene_match = (scene_labels.unsqueeze(0) == scene_labels.unsqueeze(1)).float()

        # Contrastive: same scene should be similar
        scene_mask_self = torch.eye(batch_size, device=device, dtype=torch.bool)
        scene_logits = scene_sim.masked_fill(scene_mask_self, -1e9) / config.temperature
        scene_targets = scene_match.masked_fill(scene_mask_self, 0)
        scene_targets = scene_targets / scene_targets.sum(dim=1, keepdim=True).clamp(
            min=1e-8
        )

        if not scene_targets.isnan().any():
            scene_contrastive_loss = F.cross_entropy(
                scene_logits, scene_targets, reduction="mean"
            )

    component_losses["contrastive"] = scene_contrastive_loss

    # Auxiliary: cross-provider alignment
    alignment_loss = torch.tensor(0.0, device=device)
    if cross_provider_embeddings is not None:
        # L2 alignment between projected features and cross-provider embeddings
        alignment_loss = F.mse_loss(projected_features, cross_provider_embeddings)
    component_losses["alignment"] = alignment_loss

    # Total loss
    total_loss = (
        config.identity_weight * identity_loss
        + config.contrastive_weight * scene_contrastive_loss
        + config.alignment_weight * alignment_loss
    )

    # Metrics
    metrics["mean_feature_norm"] = float(projected_features.norm(dim=-1).mean().item())

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Depth Metric Calibration Seam Loss
# ---------------------------------------------------------------------------


def depth_metric_calibration_loss(
    *,
    metric_depth: torch.Tensor,
    predicted_uncertainty: torch.Tensor,
    predicted_scale: torch.Tensor,
    predicted_shift: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    depth_valid_mask: Optional[torch.Tensor] = None,
    previous_scale: Optional[torch.Tensor] = None,
    previous_shift: Optional[torch.Tensor] = None,
    config: Optional[DepthMetricCalibrationLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for DepthMetricCalibrationSeam training.

    Primary: metric depth should match available ground truth (LiDAR,
    stereo disparity, structured light).

    Secondary: uncertainty should correlate with depth estimation error.

    Auxiliary: scale/shift consistency across consecutive frames.

    Args:
        metric_depth: ``(batch, 1, H, W)`` — predicted metric depth.
        predicted_uncertainty: ``(batch, 1, H, W)`` — predicted per-pixel
            uncertainty.
        predicted_scale: ``(batch,)`` — predicted scale factor.
        predicted_shift: ``(batch,)`` — predicted shift factor.
        ground_truth_depth: ``(batch, 1, H, W)`` — ground truth metric
            depth (sparse or dense).
        depth_valid_mask: ``(batch, 1, H, W)`` — True where ground truth
            is valid (optional).
        previous_scale: ``(batch,)`` — scale from previous frame for
            consistency (optional).
        previous_shift: ``(batch,)`` — shift from previous frame (optional).
        config: Loss weights and hyperparameters.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = DepthMetricCalibrationLossConfig()

    device = metric_depth.device
    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Apply validity mask
    if depth_valid_mask is not None:
        valid_mask = depth_valid_mask.float()
        n_valid = valid_mask.sum().clamp(min=1.0)
    else:
        valid_mask = (ground_truth_depth > 0).float()  # Assume 0 is invalid
        n_valid = valid_mask.sum().clamp(min=1.0)

    # Primary: depth error (scale-invariant or absolute)
    depth_error = (metric_depth - ground_truth_depth).abs() * valid_mask
    depth_loss = depth_error.sum() / n_valid
    component_losses["depth"] = depth_loss

    # Secondary: uncertainty calibration
    # Uncertainty should be high where error is high
    uncertainty_target = depth_error.detach() / (depth_error.detach().max() + 1e-8)
    uncertainty_loss = (
        F.mse_loss(
            predicted_uncertainty * valid_mask,
            uncertainty_target * valid_mask,
            reduction="sum",
        )
        / n_valid
    )
    component_losses["uncertainty"] = uncertainty_loss

    # Gradient loss for edge preservation
    def gradient_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        mask_dx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        mask_dy = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]

        loss_dx = ((pred_dx - target_dx).abs() * mask_dx).sum() / mask_dx.sum().clamp(
            min=1.0
        )
        loss_dy = ((pred_dy - target_dy).abs() * mask_dy).sum() / mask_dy.sum().clamp(
            min=1.0
        )
        return loss_dx + loss_dy

    gradient_loss = gradient_loss_fn(metric_depth, ground_truth_depth)
    component_losses["gradient"] = gradient_loss

    # Auxiliary: scale/shift consistency
    scale_consistency_loss = torch.tensor(0.0, device=device)
    if previous_scale is not None and previous_shift is not None:
        scale_consistency_loss = F.mse_loss(
            predicted_scale, previous_scale
        ) + F.mse_loss(predicted_shift, previous_shift)
    component_losses["scale_consistency"] = scale_consistency_loss

    # Total loss
    total_loss = (
        config.depth_weight * depth_loss
        + config.uncertainty_weight * uncertainty_loss
        + config.gradient_weight * gradient_loss
        + config.scale_consistency_weight * scale_consistency_loss
    )

    # Metrics
    metrics["mean_depth_error"] = float(depth_error.sum().item() / n_valid.item())
    metrics["mean_scale"] = float(predicted_scale.mean().item())
    metrics["mean_shift"] = float(predicted_shift.mean().item())

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# V-JEPA Temporal Alignment Seam Loss
# ---------------------------------------------------------------------------


def vjepa_temporal_alignment_loss(
    *,
    temporal_aligned: torch.Tensor,
    temporal_confidence: torch.Tensor,
    future_object_states: torch.Tensor,
    object_valid_mask: Optional[torch.Tensor] = None,
    temporal_ordering_labels: Optional[torch.Tensor] = None,
    config: Optional[VJEPATemporalAlignmentLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for VJEPATemporalAlignmentSeam training.

    Primary: aligned temporal state should predict actual future object
    state (from ground truth or next-frame observation).

    Secondary: temporal confidence should correlate with prediction
    accuracy per timestep.

    Auxiliary: contrastive loss on temporal ordering (distinguish t from t+1).

    Args:
        temporal_aligned: ``(batch, T, N_obj, d_out)`` — aligned temporal
            states from seam forward pass.
        temporal_confidence: ``(batch, T)`` — per-timestep confidence.
        future_object_states: ``(batch, T, N_obj, d_out)`` — ground truth
            future object states.
        object_valid_mask: ``(batch, N_obj)`` — True if object is valid.
        temporal_ordering_labels: ``(batch, T)`` — temporal step indices
            for ordering loss (optional).
        config: Loss weights and hyperparameters.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = VJEPATemporalAlignmentLossConfig()

    device = temporal_aligned.device
    batch_size, T, N_obj, d_out = temporal_aligned.shape

    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Apply object mask
    if object_valid_mask is not None:
        valid_mask = (
            object_valid_mask.float().unsqueeze(1).unsqueeze(-1)
        )  # (batch, 1, N_obj, 1)
        n_valid = object_valid_mask.sum().clamp(min=1.0) * T
    else:
        valid_mask = torch.ones(batch_size, 1, N_obj, 1, device=device)
        n_valid = torch.tensor(batch_size * T * N_obj, device=device, dtype=torch.float)

    # Primary: future state prediction loss
    prediction_error = (temporal_aligned - future_object_states).pow(2) * valid_mask
    prediction_loss = prediction_error.sum() / n_valid
    component_losses["prediction"] = prediction_loss

    # Secondary: confidence calibration
    # Confidence should correlate with per-timestep prediction accuracy
    per_timestep_error = prediction_error.sum(dim=(2, 3)) / (
        valid_mask.sum(dim=(2, 3)).clamp(min=1.0)
    )  # (batch, T)
    per_timestep_accuracy = 1.0 - per_timestep_error.clamp(max=1.0).detach()
    confidence_loss = F.mse_loss(temporal_confidence, per_timestep_accuracy)
    component_losses["confidence"] = confidence_loss

    # Auxiliary: temporal ordering contrastive
    ordering_loss = torch.tensor(0.0, device=device)
    if T > 1:
        # Pool features per timestep and compute ordering loss
        pooled = (temporal_aligned * valid_mask).sum(dim=2) / valid_mask.sum(
            dim=2
        ).clamp(min=1.0)
        # (batch, T, d_out)

        # Contrastive: adjacent timesteps should be more similar than distant
        # Simple approach: predict timestep from features
        pooled_flat = pooled.reshape(batch_size * T, d_out)  # (batch*T, d_out)
        pooled_norm = F.normalize(pooled_flat, dim=-1)
        sim_matrix = torch.mm(pooled_norm, pooled_norm.t())  # (batch*T, batch*T)

        # Create temporal distance matrix
        timestep_idx = (
            torch.arange(T, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        timestep_flat = timestep_idx.reshape(-1)  # (batch*T,)
        temporal_dist = (
            (timestep_flat.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs().float()
        )

        # Soft target: closer timesteps should be more similar
        temporal_sim_target = torch.exp(-temporal_dist / max(1, T // 2))

        # MSE between similarity and target
        ordering_loss = F.mse_loss(sim_matrix, temporal_sim_target)

    component_losses["ordering"] = ordering_loss

    # Smoothness loss: temporal transitions should be smooth
    if T > 1:
        temporal_diff = temporal_aligned[:, 1:, :, :] - temporal_aligned[:, :-1, :, :]
        smoothness_loss = (temporal_diff.pow(2) * valid_mask[:, :, :, :]).mean()
    else:
        smoothness_loss = torch.tensor(0.0, device=device)
    component_losses["smoothness"] = smoothness_loss

    # Total loss
    total_loss = (
        config.prediction_weight * prediction_loss
        + config.confidence_weight * confidence_loss
        + config.ordering_weight * ordering_loss
        + config.smoothness_weight * smoothness_loss
    )

    # Metrics
    metrics["mean_prediction_error"] = float(prediction_loss.item())
    metrics["mean_temporal_confidence"] = float(temporal_confidence.mean().item())
    metrics["temporal_steps"] = T

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Scene Graph Transformer Seam Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneGraphTransformerLossConfig:
    """Configuration for scene graph transformer loss computation."""

    node_classification_weight: float = 1.0
    edge_importance_weight: float = 0.5
    node_confidence_weight: float = 0.3
    token_reconstruction_weight: float = 0.2
    label_smoothing: float = 0.1


def scene_graph_transformer_loss(
    *,
    refined_tokens: torch.Tensor,
    edge_weights: torch.Tensor,
    graph_confidence: torch.Tensor,
    input_tokens: torch.Tensor,
    node_labels: Optional[torch.Tensor] = None,
    edge_importance_target: Optional[torch.Tensor] = None,
    node_confidence_target: Optional[torch.Tensor] = None,
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
    n_categories: int = 8,
    config: Optional[SceneGraphTransformerLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for SceneGraphTransformerSeam training.

    Primary: refined tokens should preserve and improve input representation
    (token reconstruction / refinement loss).

    Secondary: if node classification labels are available, refined tokens
    should be linearly separable by category.

    Auxiliary: edge weights should match importance targets; graph confidence
    should correlate with actual graph quality.

    Args:
        refined_tokens: ``(batch, N, d_out)`` — refined node features.
        edge_weights: ``(batch, E)`` — predicted edge importance.
        graph_confidence: ``(batch,)`` — predicted graph quality.
        input_tokens: ``(batch, N, d_token)`` — original node features.
        node_labels: ``(batch, N)`` int — category labels (optional).
        edge_importance_target: ``(batch, E)`` float — target edge weights.
        node_confidence_target: ``(batch, N)`` float — target node confidence.
        node_mask: ``(batch, N)`` bool — True for valid nodes.
        edge_mask: ``(batch, E)`` bool — True for valid edges.
        n_categories: Number of node categories for classification.
        config: Loss weights.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = SceneGraphTransformerLossConfig()

    device = refined_tokens.device
    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # Token reconstruction / refinement loss
    # Refined tokens should be close to input (regularization) but
    # with improved structure for downstream tasks.
    if node_mask is not None:
        mask_f = node_mask.unsqueeze(-1).float()
        recon_loss = (
            (
                (refined_tokens - input_tokens[..., : refined_tokens.size(-1)]) ** 2
                * mask_f
            ).sum()
            / mask_f.sum().clamp(min=1.0)
            / refined_tokens.size(-1)
        )
    else:
        recon_loss = F.mse_loss(
            refined_tokens, input_tokens[..., : refined_tokens.size(-1)]
        )
    component_losses["token_reconstruction"] = recon_loss

    # Node classification loss (if labels available)
    node_cls_loss = torch.tensor(0.0, device=device)
    if node_labels is not None:
        # Use a simple linear probe on refined tokens
        # (This evaluates representation quality, not used for direct classification)
        B, N, D = refined_tokens.shape
        flat_tokens = refined_tokens.reshape(B * N, D)
        flat_labels = node_labels.reshape(B * N)

        # Compute per-class centroids and use distance-based classification
        # Simple approach: prototype-based classification loss
        valid_mask = (
            node_mask.reshape(B * N)
            if node_mask is not None
            else torch.ones(B * N, dtype=torch.bool, device=device)
        )
        valid_tokens = flat_tokens[valid_mask]
        valid_labels = flat_labels[valid_mask]

        if valid_tokens.size(0) > 0:
            # Compute class centroids
            centroids = torch.zeros(n_categories, D, device=device)
            counts = torch.zeros(n_categories, device=device)
            for c in range(n_categories):
                c_mask = valid_labels == c
                if c_mask.any():
                    centroids[c] = valid_tokens[c_mask].mean(dim=0)
                    counts[c] = c_mask.sum().float()

            # Distance to centroids (prototype loss)
            dists = torch.cdist(valid_tokens, centroids)  # (valid, n_categories)
            # Soft cross-entropy on negative distances
            logits = -dists
            node_cls_loss = F.cross_entropy(
                logits,
                valid_labels.clamp(0, n_categories - 1),
                label_smoothing=config.label_smoothing,
            )
            # Accuracy metric
            preds = logits.argmax(dim=-1)
            accuracy = (preds == valid_labels).float().mean().item()
            metrics["node_classification_accuracy"] = accuracy

        component_losses["node_classification"] = node_cls_loss

    # Edge importance loss
    edge_imp_loss = torch.tensor(0.0, device=device)
    if edge_importance_target is not None:
        if edge_mask is not None:
            mask_e = edge_mask.float()
            edge_imp_loss = (
                (edge_weights - edge_importance_target) ** 2 * mask_e
            ).sum() / mask_e.sum().clamp(min=1.0)
        else:
            edge_imp_loss = F.mse_loss(edge_weights, edge_importance_target)
        component_losses["edge_importance"] = edge_imp_loss
        metrics["edge_weight_mae"] = float(
            (edge_weights - edge_importance_target).abs().mean().item()
        )

    # Node confidence loss
    node_conf_loss = torch.tensor(0.0, device=device)
    if node_confidence_target is not None:
        if node_mask is not None:
            mask_n = node_mask.float()
            conf_pred = graph_confidence.unsqueeze(-1).expand_as(node_confidence_target)
            node_conf_loss = (
                (conf_pred - node_confidence_target) ** 2 * mask_n
            ).sum() / mask_n.sum().clamp(min=1.0)
        else:
            node_conf_loss = F.mse_loss(
                graph_confidence.unsqueeze(-1).expand_as(node_confidence_target),
                node_confidence_target,
            )
        component_losses["node_confidence"] = node_conf_loss

    # Total loss
    total_loss = (
        config.token_reconstruction_weight * recon_loss
        + config.node_classification_weight * node_cls_loss
        + config.edge_importance_weight * edge_imp_loss
        + config.node_confidence_weight * node_conf_loss
    )

    metrics["graph_confidence_mean"] = float(graph_confidence.mean().item())
    metrics["edge_weight_mean"] = float(edge_weights.mean().item())

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Annotation Bridge Projection Seam Loss
# ---------------------------------------------------------------------------


def annotation_bridge_projection_loss(
    *,
    class_logits: torch.Tensor,
    confidence: torch.Tensor,
    affordance_scores: torch.Tensor,
    class_labels: torch.Tensor,
    confidence_targets: Optional[torch.Tensor] = None,
    affordance_targets: Optional[torch.Tensor] = None,
    node_mask: Optional[torch.Tensor] = None,
    config: Optional[AnnotationBridgeProjectionLossConfig] = None,
) -> SeamLossResult:
    """Compute loss for AnnotationBridgeProjectionSeam training.

    Primary: class prediction accuracy (cross-entropy on class_logits).
    Secondary: confidence calibration (BCE on confidence vs annotation quality).
    Auxiliary: affordance prediction (BCE on affordance_scores vs targets).

    Args:
        class_logits: ``(batch, N, n_categories)`` — predicted class logits.
        confidence: ``(batch, N)`` — predicted confidence in [0, 1].
        affordance_scores: ``(batch, N, n_affordances)`` — predicted affordances.
        class_labels: ``(batch, N)`` int — ground truth category indices.
        confidence_targets: ``(batch, N)`` float — annotation quality targets.
        affordance_targets: ``(batch, N, n_affordances)`` float — affordance targets.
        node_mask: ``(batch, N)`` bool — True for valid nodes.
        config: Loss weights.

    Returns:
        SeamLossResult with total loss and component breakdown.
    """
    if config is None:
        config = AnnotationBridgeProjectionLossConfig()

    device = class_logits.device
    component_losses: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    B, N, C = class_logits.shape

    # Flatten for loss computation
    flat_logits = class_logits.reshape(B * N, C)
    flat_labels = class_labels.reshape(B * N)

    if node_mask is not None:
        valid = node_mask.reshape(B * N)
        flat_logits = flat_logits[valid]
        flat_labels = flat_labels[valid]

    # Class prediction loss
    if flat_logits.size(0) > 0:
        class_loss = F.cross_entropy(
            flat_logits,
            flat_labels.clamp(0, C - 1),
            label_smoothing=config.label_smoothing,
        )
        preds = flat_logits.argmax(dim=-1)
        accuracy = (preds == flat_labels).float().mean().item()
        metrics["class_accuracy"] = accuracy
    else:
        class_loss = torch.tensor(0.0, device=device)
    component_losses["class_prediction"] = class_loss

    # Confidence calibration loss
    conf_loss = torch.tensor(0.0, device=device)
    if confidence_targets is not None:
        flat_conf = confidence.reshape(B * N)
        flat_conf_tgt = confidence_targets.reshape(B * N)
        if node_mask is not None:
            flat_conf = flat_conf[valid]
            flat_conf_tgt = flat_conf_tgt[valid]
        if flat_conf.size(0) > 0:
            conf_loss = F.binary_cross_entropy(
                flat_conf.clamp(1e-6, 1 - 1e-6),
                flat_conf_tgt.clamp(0, 1),
            )
            metrics["confidence_mae"] = float(
                (flat_conf - flat_conf_tgt).abs().mean().item()
            )
    component_losses["confidence_calibration"] = conf_loss

    # Affordance prediction loss
    aff_loss = torch.tensor(0.0, device=device)
    if affordance_targets is not None:
        flat_aff = affordance_scores.reshape(B * N, -1)
        flat_aff_tgt = affordance_targets.reshape(B * N, -1)
        if node_mask is not None:
            flat_aff = flat_aff[valid]
            flat_aff_tgt = flat_aff_tgt[valid]
        if flat_aff.size(0) > 0:
            aff_loss = F.binary_cross_entropy(
                flat_aff.clamp(1e-6, 1 - 1e-6),
                flat_aff_tgt.clamp(0, 1),
            )
    component_losses["affordance_prediction"] = aff_loss

    total_loss = (
        config.class_weight * class_loss
        + config.confidence_weight * conf_loss
        + config.affordance_weight * aff_loss
    )

    return SeamLossResult(
        total_loss=total_loss,
        component_losses=component_losses,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------


SEAM_LOSS_REGISTRY: Dict[str, Any] = {
    "annotation_bridge_projection": annotation_bridge_projection_loss,
    "evidence_fusion": evidence_fusion_loss,
    "sam_calibration": sam_calibration_loss,
    "vision_backbone_projection": vision_backbone_projection_loss,
    "depth_metric_calibration": depth_metric_calibration_loss,
    "vjepa_temporal_alignment": vjepa_temporal_alignment_loss,
    "scene_graph_transformer": scene_graph_transformer_loss,
}


def get_seam_loss_fn(seam_type: str):
    """Get the loss function for a seam type."""
    if seam_type not in SEAM_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown seam type: {seam_type}. "
            f"Available: {list(SEAM_LOSS_REGISTRY.keys())}"
        )
    return SEAM_LOSS_REGISTRY[seam_type]


__all__ = [
    # Config classes
    "AnnotationBridgeProjectionLossConfig",
    "DepthMetricCalibrationLossConfig",
    "EvidenceFusionLossConfig",
    "SAMCalibrationLossConfig",
    "SceneGraphTransformerLossConfig",
    "VisionBackboneProjectionLossConfig",
    "VJEPATemporalAlignmentLossConfig",
    # Result
    "SeamLossResult",
    # Loss functions
    "annotation_bridge_projection_loss",
    "depth_metric_calibration_loss",
    "evidence_fusion_loss",
    "get_seam_loss_fn",
    "sam_calibration_loss",
    "scene_graph_transformer_loss",
    "SEAM_LOSS_REGISTRY",
    "vision_backbone_projection_loss",
    "vjepa_temporal_alignment_loss",
]
