"""Data loading infrastructure for perception seam training.

This module provides dataset classes and data loaders for training
perception neural seams.  The primary data source is provider agreement
records: multi-provider observations where we can measure cross-provider
disagreement and calibrate seams against held-out providers.

Data sources
------------
1. **Provider Agreement Corpus**: Multi-provider observations from replay
   or live collection.  Each sample includes outputs from 2+ providers
   on the same scene, enabling held-out reconstruction training.

2. **Ground Truth Annotations**: Sparse human annotations or LiDAR/stereo
   ground truth for supervised calibration (depth, segmentation quality).

3. **Cross-Provider Disagreement**: Computed disagreement scores between
   providers, used as uncertainty calibration targets.

4. **Downstream Task Correlation**: Frozen task success signals correlated
   with seam outputs (not used for gradient, only as target).

Dataset classes
---------------
- ``ProviderAgreementDataset``: Base dataset for multi-provider observations
- ``EvidenceFusionDataset``: Dataset for evidence fusion seam training
- ``SAMCalibrationDataset``: Dataset for SAM calibration seam training
- ``DepthCalibrationDataset``: Dataset for depth calibration seam training
- ``VJEPATemporalDataset``: Dataset for V-JEPA temporal alignment training
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# ---------------------------------------------------------------------------
# Data record schemas
# ---------------------------------------------------------------------------


@dataclass
class ProviderObservation:
    """Single provider's observation on a scene."""

    provider_id: str
    provider_kind: str
    availability_status: str
    truth_class: str
    features: torch.Tensor  # Provider-specific features
    confidence: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiProviderSample:
    """Multi-provider observation on a single scene.

    This is the fundamental training sample for seam training: multiple
    providers' outputs on the same scene, enabling cross-provider
    agreement/disagreement measurement.
    """

    sample_id: str
    scene_id: str
    frame_idx: int
    providers: List[ProviderObservation]
    ground_truth: Optional[Dict[str, torch.Tensor]] = None
    downstream_task_success: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def provider_ids(self) -> List[str]:
        return [p.provider_id for p in self.providers]

    @property
    def n_providers(self) -> int:
        return len(self.providers)


# ---------------------------------------------------------------------------
# Base Provider Agreement Dataset
# ---------------------------------------------------------------------------


class ProviderAgreementDataset(Dataset[MultiProviderSample]):
    """Base dataset for multi-provider observations.

    Subclasses implement specific loading logic for different seam types.
    This class provides the common infrastructure for iterating over
    multi-provider samples.
    """

    def __init__(
        self,
        samples: Sequence[MultiProviderSample],
        *,
        min_providers: int = 2,
        required_provider_kinds: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[MultiProviderSample], MultiProviderSample]] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            samples: Sequence of multi-provider samples.
            min_providers: Minimum number of available providers per sample.
            required_provider_kinds: Provider kinds that must be present.
            transform: Optional transform to apply to each sample.
        """
        self.min_providers = min_providers
        self.required_provider_kinds = set(required_provider_kinds or [])
        self.transform = transform

        # Filter samples by provider requirements
        self.samples = [
            s for s in samples
            if self._sample_valid(s)
        ]

    def _sample_valid(self, sample: MultiProviderSample) -> bool:
        """Check if sample meets provider requirements."""
        available = [
            p for p in sample.providers
            if p.availability_status == "available"
        ]
        if len(available) < self.min_providers:
            return False

        if self.required_provider_kinds:
            available_kinds = {p.provider_kind for p in available}
            if not self.required_provider_kinds.issubset(available_kinds):
                return False

        return True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultiProviderSample:
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Evidence Fusion Dataset
# ---------------------------------------------------------------------------


@dataclass
class EvidenceFusionBatch:
    """Collated batch for evidence fusion seam training."""

    provider_features: torch.Tensor  # (batch, N_providers, d_feature) raw features for loss
    seam_input_features: torch.Tensor  # (batch, N_providers, 12) encoded metadata for seam forward
    provider_availability: torch.Tensor  # (batch, N_providers) bool
    held_out_idx: torch.Tensor  # (batch,) int
    held_out_features: torch.Tensor  # (batch, d_feature)
    task_correlation_target: Optional[torch.Tensor] = None  # (batch,)
    sample_ids: Optional[List[str]] = None


class EvidenceFusionDataset(ProviderAgreementDataset):
    """Dataset for evidence fusion seam training.

    Each sample is a multi-provider observation.  During training, one
    provider is held out and the seam learns to reconstruct its
    contribution from the others.
    """

    def __init__(
        self,
        samples: Sequence[MultiProviderSample],
        *,
        d_feature: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(samples, min_providers=2, **kwargs)
        self.d_feature = d_feature

    @staticmethod
    def collate_fn(
        samples: List[MultiProviderSample],
        d_feature: int = 128,
    ) -> EvidenceFusionBatch:
        """Collate samples into a batch for training.

        Randomly selects one provider per sample to hold out.
        Produces both raw features (for loss) and 12-dim encoded
        metadata (for seam forward pass).
        """
        from src.world_model.perception_grounding.neural_seams import (
            PROVIDER_KIND_VOCAB,
            NUM_PROVIDER_KINDS,
            TRUTH_CLASS_SCORES,
        )

        batch_size = len(samples)

        # Determine max providers across batch
        max_providers = max(s.n_providers for s in samples)
        d_seam_input = NUM_PROVIDER_KINDS + 8  # 12

        # Initialize tensors
        provider_features = torch.zeros(batch_size, max_providers, d_feature)
        seam_input_features = torch.zeros(batch_size, max_providers, d_seam_input)
        provider_availability = torch.zeros(batch_size, max_providers, dtype=torch.bool)
        held_out_idx = torch.zeros(batch_size, dtype=torch.long)
        held_out_features = torch.zeros(batch_size, d_feature)
        task_targets = []

        for i, sample in enumerate(samples):
            available_providers = [
                (j, p) for j, p in enumerate(sample.providers)
                if p.availability_status == "available"
            ]

            # Randomly select held-out provider
            hold_out_local_idx = torch.randint(len(available_providers), (1,)).item()
            hold_out_global_idx, hold_out_provider = available_providers[hold_out_local_idx]

            for j, provider in enumerate(sample.providers):
                if j < max_providers:
                    # Raw features for loss
                    feat = provider.features
                    if feat.shape[-1] < d_feature:
                        feat = torch.nn.functional.pad(
                            feat.flatten(), (0, d_feature - feat.numel())
                        )
                    elif feat.numel() > d_feature:
                        feat = feat.flatten()[:d_feature]
                    else:
                        feat = feat.flatten()

                    provider_features[i, j] = feat
                    provider_availability[i, j] = provider.availability_status == "available"

                    # 12-dim encoded metadata for seam
                    kind_idx = PROVIDER_KIND_VOCAB.get(provider.provider_kind, -1)
                    kind_onehot = [0.0] * NUM_PROVIDER_KINDS
                    if 0 <= kind_idx < NUM_PROVIDER_KINDS:
                        kind_onehot[kind_idx] = 1.0
                    avail = 1.0 if provider.availability_status == "available" else 0.0
                    truth = TRUTH_CLASS_SCORES.get(provider.truth_class, 0.1)
                    conf_val = float(provider.confidence.item()) if provider.confidence is not None else 0.0
                    seam_input_features[i, j] = torch.tensor(
                        kind_onehot + [avail, truth, conf_val, 0.0, 0.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32,
                    )

            held_out_idx[i] = hold_out_global_idx
            held_out_features[i] = provider_features[i, hold_out_global_idx]

            if sample.downstream_task_success is not None:
                task_targets.append(sample.downstream_task_success)

        task_target_tensor = None
        if task_targets and len(task_targets) == batch_size:
            task_target_tensor = torch.tensor(task_targets, dtype=torch.float32)

        return EvidenceFusionBatch(
            provider_features=provider_features,
            seam_input_features=seam_input_features,
            provider_availability=provider_availability,
            held_out_idx=held_out_idx,
            held_out_features=held_out_features,
            task_correlation_target=task_target_tensor,
            sample_ids=[s.sample_id for s in samples],
        )


# ---------------------------------------------------------------------------
# SAM Calibration Dataset
# ---------------------------------------------------------------------------


@dataclass
class SAMCalibrationSample:
    """Sample for SAM calibration seam training."""

    sample_id: str
    mask_features: torch.Tensor  # (N_masks, d_mask)
    raw_confidence: torch.Tensor  # (N_masks,)
    mask_valid: torch.Tensor  # (N_masks,) bool
    downstream_quality: torch.Tensor  # (N_masks,) — ground truth quality
    provider_disagreement: Optional[torch.Tensor] = None  # (N_masks,)
    segmentation_iou: Optional[torch.Tensor] = None  # (N_masks,)


@dataclass
class SAMCalibrationBatch:
    """Collated batch for SAM calibration seam training."""

    mask_features: torch.Tensor  # (batch, N_masks, d_mask)
    raw_confidence: torch.Tensor  # (batch, N_masks)
    mask_valid: torch.Tensor  # (batch, N_masks) bool
    downstream_quality: torch.Tensor  # (batch, N_masks)
    provider_disagreement: Optional[torch.Tensor] = None
    segmentation_iou: Optional[torch.Tensor] = None


class SAMCalibrationDataset(Dataset[SAMCalibrationSample]):
    """Dataset for SAM calibration seam training.

    Each sample contains SAM mask outputs with ground truth quality
    annotations for calibration learning.
    """

    def __init__(
        self,
        samples: Sequence[SAMCalibrationSample],
        *,
        max_masks: int = 32,
        d_mask: int = 256,
    ) -> None:
        self.samples = list(samples)
        self.max_masks = max_masks
        self.d_mask = d_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SAMCalibrationSample:
        return self.samples[idx]

    @staticmethod
    def collate_fn(
        samples: List[SAMCalibrationSample],
        max_masks: int = 32,
        d_mask: int = 256,
    ) -> SAMCalibrationBatch:
        """Collate samples into a batch."""
        batch_size = len(samples)

        mask_features = torch.zeros(batch_size, max_masks, d_mask)
        raw_confidence = torch.zeros(batch_size, max_masks)
        mask_valid = torch.zeros(batch_size, max_masks, dtype=torch.bool)
        downstream_quality = torch.zeros(batch_size, max_masks)
        provider_disagreement = torch.zeros(batch_size, max_masks)
        segmentation_iou = torch.zeros(batch_size, max_masks)

        has_disagreement = False
        has_iou = False

        for i, sample in enumerate(samples):
            n_masks = min(sample.mask_features.size(0), max_masks)

            # Handle feature dimension
            feat = sample.mask_features[:n_masks]
            if feat.size(-1) < d_mask:
                feat = torch.nn.functional.pad(feat, (0, d_mask - feat.size(-1)))
            elif feat.size(-1) > d_mask:
                feat = feat[:, :d_mask]

            mask_features[i, :n_masks] = feat
            raw_confidence[i, :n_masks] = sample.raw_confidence[:n_masks]
            mask_valid[i, :n_masks] = sample.mask_valid[:n_masks]
            downstream_quality[i, :n_masks] = sample.downstream_quality[:n_masks]

            if sample.provider_disagreement is not None:
                provider_disagreement[i, :n_masks] = sample.provider_disagreement[:n_masks]
                has_disagreement = True
            if sample.segmentation_iou is not None:
                segmentation_iou[i, :n_masks] = sample.segmentation_iou[:n_masks]
                has_iou = True

        return SAMCalibrationBatch(
            mask_features=mask_features,
            raw_confidence=raw_confidence,
            mask_valid=mask_valid,
            downstream_quality=downstream_quality,
            provider_disagreement=provider_disagreement if has_disagreement else None,
            segmentation_iou=segmentation_iou if has_iou else None,
        )


# ---------------------------------------------------------------------------
# Depth Calibration Dataset
# ---------------------------------------------------------------------------


@dataclass
class DepthCalibrationSample:
    """Sample for depth metric calibration seam training."""

    sample_id: str
    relative_depth: torch.Tensor  # (1, H, W)
    camera_intrinsics: torch.Tensor  # (4,) — fx, fy, cx, cy
    ground_truth_depth: torch.Tensor  # (1, H, W)
    depth_valid_mask: torch.Tensor  # (1, H, W) bool
    previous_scale: Optional[float] = None
    previous_shift: Optional[float] = None


@dataclass
class DepthCalibrationBatch:
    """Collated batch for depth calibration seam training."""

    relative_depth: torch.Tensor  # (batch, 1, H, W)
    camera_intrinsics: torch.Tensor  # (batch, 4)
    ground_truth_depth: torch.Tensor  # (batch, 1, H, W)
    depth_valid_mask: torch.Tensor  # (batch, 1, H, W) bool
    previous_scale: Optional[torch.Tensor] = None
    previous_shift: Optional[torch.Tensor] = None


class DepthCalibrationDataset(Dataset[DepthCalibrationSample]):
    """Dataset for depth metric calibration seam training.

    Each sample contains monocular depth predictions with sparse or
    dense ground truth from LiDAR/stereo.
    """

    def __init__(
        self,
        samples: Sequence[DepthCalibrationSample],
        *,
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.samples = list(samples)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DepthCalibrationSample:
        return self.samples[idx]

    @staticmethod
    def collate_fn(
        samples: List[DepthCalibrationSample],
        target_size: Tuple[int, int] = (256, 256),
    ) -> DepthCalibrationBatch:
        """Collate samples into a batch."""
        batch_size = len(samples)
        H, W = target_size

        relative_depth = torch.zeros(batch_size, 1, H, W)
        camera_intrinsics = torch.zeros(batch_size, 4)
        ground_truth_depth = torch.zeros(batch_size, 1, H, W)
        depth_valid_mask = torch.zeros(batch_size, 1, H, W, dtype=torch.bool)

        prev_scales = []
        prev_shifts = []
        has_prev = False

        for i, sample in enumerate(samples):
            # Resize if needed
            rel = sample.relative_depth
            if rel.shape[-2:] != (H, W):
                rel = torch.nn.functional.interpolate(
                    rel.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)
            relative_depth[i] = rel

            camera_intrinsics[i] = sample.camera_intrinsics

            gt = sample.ground_truth_depth
            if gt.shape[-2:] != (H, W):
                gt = torch.nn.functional.interpolate(
                    gt.unsqueeze(0), size=(H, W), mode="nearest"
                ).squeeze(0)
            ground_truth_depth[i] = gt

            mask = sample.depth_valid_mask
            if mask.shape[-2:] != (H, W):
                mask = torch.nn.functional.interpolate(
                    mask.float().unsqueeze(0), size=(H, W), mode="nearest"
                ).squeeze(0).bool()
            depth_valid_mask[i] = mask

            if sample.previous_scale is not None:
                prev_scales.append(sample.previous_scale)
                prev_shifts.append(sample.previous_shift or 0.0)
                has_prev = True

        prev_scale_tensor = None
        prev_shift_tensor = None
        if has_prev and len(prev_scales) == batch_size:
            prev_scale_tensor = torch.tensor(prev_scales, dtype=torch.float32)
            prev_shift_tensor = torch.tensor(prev_shifts, dtype=torch.float32)

        return DepthCalibrationBatch(
            relative_depth=relative_depth,
            camera_intrinsics=camera_intrinsics,
            ground_truth_depth=ground_truth_depth,
            depth_valid_mask=depth_valid_mask,
            previous_scale=prev_scale_tensor,
            previous_shift=prev_shift_tensor,
        )


# ---------------------------------------------------------------------------
# V-JEPA Temporal Dataset
# ---------------------------------------------------------------------------


@dataclass
class VJEPATemporalSample:
    """Sample for V-JEPA temporal alignment seam training."""

    sample_id: str
    vjepa_tokens: torch.Tensor  # (T, N_vjepa, d_vjepa)
    wm_object_tokens: torch.Tensor  # (N_obj, d_wm)
    future_object_states: torch.Tensor  # (T, N_obj, d_out)
    object_valid_mask: torch.Tensor  # (N_obj,) bool
    temporal_ordering_labels: Optional[torch.Tensor] = None  # (T,)


@dataclass
class VJEPATemporalBatch:
    """Collated batch for V-JEPA temporal alignment training."""

    vjepa_tokens: torch.Tensor  # (batch, T, N_vjepa, d_vjepa)
    wm_object_tokens: torch.Tensor  # (batch, N_obj, d_wm)
    future_object_states: torch.Tensor  # (batch, T, N_obj, d_out)
    object_valid_mask: torch.Tensor  # (batch, N_obj) bool
    temporal_ordering_labels: Optional[torch.Tensor] = None


class VJEPATemporalDataset(Dataset[VJEPATemporalSample]):
    """Dataset for V-JEPA temporal alignment seam training.

    Each sample contains V-JEPA temporal predictions and ground truth
    future object states for alignment learning.
    """

    def __init__(
        self,
        samples: Sequence[VJEPATemporalSample],
        *,
        n_temporal_steps: int = 4,
        max_objects: int = 32,
    ) -> None:
        self.samples = list(samples)
        self.n_temporal_steps = n_temporal_steps
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VJEPATemporalSample:
        return self.samples[idx]

    @staticmethod
    def collate_fn(
        samples: List[VJEPATemporalSample],
        n_temporal_steps: int = 4,
        max_objects: int = 32,
        max_vjepa_tokens: int = 196,
        d_vjepa: int = 1024,
        d_wm: int = 128,
        d_out: int = 128,
    ) -> VJEPATemporalBatch:
        """Collate samples into a batch."""
        batch_size = len(samples)
        T = n_temporal_steps

        vjepa_tokens = torch.zeros(batch_size, T, max_vjepa_tokens, d_vjepa)
        wm_object_tokens = torch.zeros(batch_size, max_objects, d_wm)
        future_object_states = torch.zeros(batch_size, T, max_objects, d_out)
        object_valid_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
        temporal_ordering = torch.zeros(batch_size, T, dtype=torch.long)

        has_ordering = False

        for i, sample in enumerate(samples):
            # V-JEPA tokens
            n_t = min(sample.vjepa_tokens.size(0), T)
            n_v = min(sample.vjepa_tokens.size(1), max_vjepa_tokens)
            d_v = min(sample.vjepa_tokens.size(2), d_vjepa)
            vjepa_tokens[i, :n_t, :n_v, :d_v] = sample.vjepa_tokens[:n_t, :n_v, :d_v]

            # WM object tokens
            n_obj = min(sample.wm_object_tokens.size(0), max_objects)
            d_w = min(sample.wm_object_tokens.size(1), d_wm)
            wm_object_tokens[i, :n_obj, :d_w] = sample.wm_object_tokens[:n_obj, :d_w]

            # Future states
            d_o = min(sample.future_object_states.size(2), d_out)
            future_object_states[i, :n_t, :n_obj, :d_o] = sample.future_object_states[:n_t, :n_obj, :d_o]

            # Valid mask
            n_valid = min(sample.object_valid_mask.size(0), max_objects)
            object_valid_mask[i, :n_valid] = sample.object_valid_mask[:n_valid]

            # Ordering labels
            if sample.temporal_ordering_labels is not None:
                temporal_ordering[i, :n_t] = sample.temporal_ordering_labels[:n_t]
                has_ordering = True

        return VJEPATemporalBatch(
            vjepa_tokens=vjepa_tokens,
            wm_object_tokens=wm_object_tokens,
            future_object_states=future_object_states,
            object_valid_mask=object_valid_mask,
            temporal_ordering_labels=temporal_ordering if has_ordering else None,
        )


# ---------------------------------------------------------------------------
# Synthetic data generation (for testing / bootstrapping)
# ---------------------------------------------------------------------------


def generate_synthetic_evidence_fusion_samples(
    n_samples: int = 100,
    n_providers: int = 4,
    d_feature: int = 128,
    *,
    seed: Optional[int] = None,
) -> List[MultiProviderSample]:
    """Generate synthetic multi-provider samples for testing.

    Creates random provider observations with controlled agreement/
    disagreement patterns for seam training development.
    """
    if seed is not None:
        torch.manual_seed(seed)

    samples = []
    provider_kinds = ["scene_tracks", "vision_backbone", "teacher_semantics", "teacher_trace"]

    for i in range(n_samples):
        providers = []
        for j in range(n_providers):
            # Random availability (80% available)
            available = torch.rand(1).item() > 0.2

            providers.append(ProviderObservation(
                provider_id=f"provider_{j}",
                provider_kind=provider_kinds[j % len(provider_kinds)],
                availability_status="available" if available else "unavailable",
                truth_class="provider_backed" if available else "unavailable",
                features=torch.randn(d_feature),
                confidence=torch.rand(1) if available else None,
            ))

        # Synthetic task success (correlated with provider availability)
        n_available = sum(1 for p in providers if p.availability_status == "available")
        task_success = min(1.0, 0.3 + 0.2 * n_available + 0.1 * torch.rand(1).item())

        samples.append(MultiProviderSample(
            sample_id=f"synthetic_{i:04d}",
            scene_id=f"scene_{i // 10}",
            frame_idx=i % 10,
            providers=providers,
            downstream_task_success=task_success,
        ))

    return samples


def generate_synthetic_sam_calibration_samples(
    n_samples: int = 100,
    n_masks: int = 8,
    d_mask: int = 256,
    *,
    seed: Optional[int] = None,
) -> List[SAMCalibrationSample]:
    """Generate synthetic SAM calibration samples for testing."""
    if seed is not None:
        torch.manual_seed(seed)

    samples = []
    for i in range(n_samples):
        # Random mask features
        mask_features = torch.randn(n_masks, d_mask)

        # Raw confidence (from SAM)
        raw_confidence = torch.sigmoid(torch.randn(n_masks))

        # Mask validity (90% valid)
        mask_valid = torch.rand(n_masks) > 0.1

        # Downstream quality (somewhat correlated with raw confidence)
        noise = torch.randn(n_masks) * 0.2
        downstream_quality = torch.clamp(raw_confidence + noise, 0, 1)

        # Provider disagreement (inversely correlated with quality)
        disagreement = torch.clamp(1.0 - downstream_quality + torch.randn(n_masks) * 0.1, 0, 1)

        # IoU (correlated with quality)
        iou = torch.clamp(downstream_quality + torch.randn(n_masks) * 0.15, 0, 1)

        samples.append(SAMCalibrationSample(
            sample_id=f"sam_synthetic_{i:04d}",
            mask_features=mask_features,
            raw_confidence=raw_confidence,
            mask_valid=mask_valid,
            downstream_quality=downstream_quality,
            provider_disagreement=disagreement,
            segmentation_iou=iou,
        ))

    return samples


def generate_synthetic_depth_calibration_samples(
    n_samples: int = 100,
    height: int = 64,
    width: int = 64,
    *,
    seed: Optional[int] = None,
) -> List[DepthCalibrationSample]:
    """Generate synthetic depth calibration samples for testing."""
    if seed is not None:
        torch.manual_seed(seed)

    samples = []
    for i in range(n_samples):
        # Ground truth depth (smooth random surface)
        gt_depth = torch.abs(torch.randn(1, height, width)) * 5 + 1  # 1-6m range

        # Apply some smoothing
        gt_depth = torch.nn.functional.avg_pool2d(
            gt_depth.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)

        # Relative depth (scaled/shifted version of GT)
        true_scale = 0.5 + torch.rand(1).item()
        true_shift = -1.0 + 2.0 * torch.rand(1).item()
        relative_depth = (gt_depth - true_shift) / true_scale + torch.randn_like(gt_depth) * 0.1

        # Valid mask (80% valid, sparse for LiDAR simulation)
        depth_valid = torch.rand(1, height, width) > 0.2

        # Camera intrinsics (normalized)
        intrinsics = torch.tensor([1.0, 1.0, 0.5, 0.5])

        # Previous scale/shift (for temporal consistency)
        prev_scale = true_scale + torch.randn(1).item() * 0.1 if i > 0 else None
        prev_shift = true_shift + torch.randn(1).item() * 0.1 if i > 0 else None

        samples.append(DepthCalibrationSample(
            sample_id=f"depth_synthetic_{i:04d}",
            relative_depth=relative_depth,
            camera_intrinsics=intrinsics,
            ground_truth_depth=gt_depth,
            depth_valid_mask=depth_valid,
            previous_scale=prev_scale,
            previous_shift=prev_shift,
        ))

    return samples


def generate_synthetic_vjepa_temporal_samples(
    n_samples: int = 100,
    n_temporal_steps: int = 4,
    n_vjepa_tokens: int = 196,
    n_objects: int = 10,
    d_vjepa: int = 1024,
    d_wm: int = 128,
    d_out: int = 128,
    *,
    seed: Optional[int] = None,
) -> List[VJEPATemporalSample]:
    """Generate synthetic V-JEPA temporal samples for testing."""
    if seed is not None:
        torch.manual_seed(seed)

    samples = []
    for i in range(n_samples):
        # V-JEPA tokens (temporal predictions)
        vjepa_tokens = torch.randn(n_temporal_steps, n_vjepa_tokens, d_vjepa)

        # Current WM object tokens
        wm_tokens = torch.randn(n_objects, d_wm)

        # Future object states (somewhat correlated with V-JEPA predictions)
        # In practice, these would come from actual future observations
        base_future = torch.randn(n_temporal_steps, n_objects, d_out)

        # Add temporal smoothness
        for t in range(1, n_temporal_steps):
            base_future[t] = 0.7 * base_future[t-1] + 0.3 * base_future[t]

        # Object validity (80% valid)
        object_valid = torch.rand(n_objects) > 0.2

        # Temporal ordering
        ordering = torch.arange(n_temporal_steps)

        samples.append(VJEPATemporalSample(
            sample_id=f"vjepa_synthetic_{i:04d}",
            vjepa_tokens=vjepa_tokens,
            wm_object_tokens=wm_tokens,
            future_object_states=base_future,
            object_valid_mask=object_valid,
            temporal_ordering_labels=ordering,
        ))

    return samples


# ---------------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------------


def create_evidence_fusion_loader(
    samples: Sequence[MultiProviderSample],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    d_feature: int = 128,
) -> DataLoader[EvidenceFusionBatch]:
    """Create data loader for evidence fusion seam training."""
    dataset = EvidenceFusionDataset(samples, d_feature=d_feature)

    def collate(batch: List[MultiProviderSample]) -> EvidenceFusionBatch:
        return EvidenceFusionDataset.collate_fn(batch, d_feature=d_feature)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def create_sam_calibration_loader(
    samples: Sequence[SAMCalibrationSample],
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_masks: int = 32,
    d_mask: int = 256,
) -> DataLoader[SAMCalibrationBatch]:
    """Create data loader for SAM calibration seam training."""
    dataset = SAMCalibrationDataset(samples, max_masks=max_masks, d_mask=d_mask)

    def collate(batch: List[SAMCalibrationSample]) -> SAMCalibrationBatch:
        return SAMCalibrationDataset.collate_fn(batch, max_masks=max_masks, d_mask=d_mask)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def create_depth_calibration_loader(
    samples: Sequence[DepthCalibrationSample],
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (256, 256),
) -> DataLoader[DepthCalibrationBatch]:
    """Create data loader for depth calibration seam training."""
    dataset = DepthCalibrationDataset(samples, target_size=target_size)

    def collate(batch: List[DepthCalibrationSample]) -> DepthCalibrationBatch:
        return DepthCalibrationDataset.collate_fn(batch, target_size=target_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def create_vjepa_temporal_loader(
    samples: Sequence[VJEPATemporalSample],
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    n_temporal_steps: int = 4,
    max_objects: int = 32,
) -> DataLoader[VJEPATemporalBatch]:
    """Create data loader for V-JEPA temporal alignment training."""
    dataset = VJEPATemporalDataset(
        samples, n_temporal_steps=n_temporal_steps, max_objects=max_objects
    )

    def collate(batch: List[VJEPATemporalSample]) -> VJEPATemporalBatch:
        return VJEPATemporalDataset.collate_fn(
            batch, n_temporal_steps=n_temporal_steps, max_objects=max_objects
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


__all__ = [
    # Records
    "ProviderObservation",
    "MultiProviderSample",
    # Evidence fusion
    "EvidenceFusionBatch",
    "EvidenceFusionDataset",
    "ProviderAgreementDataset",
    # SAM calibration
    "SAMCalibrationSample",
    "SAMCalibrationBatch",
    "SAMCalibrationDataset",
    # Depth calibration
    "DepthCalibrationSample",
    "DepthCalibrationBatch",
    "DepthCalibrationDataset",
    # V-JEPA temporal
    "VJEPATemporalSample",
    "VJEPATemporalBatch",
    "VJEPATemporalDataset",
    # Synthetic data
    "generate_synthetic_depth_calibration_samples",
    "generate_synthetic_evidence_fusion_samples",
    "generate_synthetic_sam_calibration_samples",
    "generate_synthetic_vjepa_temporal_samples",
    # Loaders
    "create_depth_calibration_loader",
    "create_evidence_fusion_loader",
    "create_sam_calibration_loader",
    "create_vjepa_temporal_loader",
]
