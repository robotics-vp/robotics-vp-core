"""Bounded neural seams for the Perception / Grounding WM.

This module contains the first real ``torch.nn.Module`` implementations
behind the ``disabled|auto|required`` promotion posture.  These are not
heuristic helpers — they are bounded neural components that can be
trained, evaluated, and promoted/demoted via the standard machinery.

Anti-heuristic-without-neuralization: heuristic fusion is the
transitional fallback.  These seams are the successor codepath.

Current seams
-------------
- ``EvidenceFusionSeam``: tiny set-attention module (~10-50K params) that
  fuses heterogeneous provider evidence into calibrated provider weights
  and fusion confidence.

- ``SAMCalibrationSeam``: calibration head (~500K-2M params) for SAM 3/3.1
  mask outputs.  Produces calibrated confidence, epistemic uncertainty,
  and prompt satisfaction score from raw SAM mask features.

- ``VisionBackboneProjectionSeam``: 2-layer MLP (~1M params) projecting
  frozen DINOv2/SigLIP backbone features to WM object token space (d=128).

- ``DepthMetricCalibrationSeam``: metric calibration head (~500K-1M params)
  for monocular depth.  Learns scene-global scale/shift correction with
  per-pixel uncertainty.

- ``VJEPATemporalAlignmentSeam``: cross-attention module (~2-5M params)
  aligning V-JEPA 2 temporal latent predictions to WM object tokens.

Capacity principles
-------------------
All seams are deliberately bounded:
- Small enough to train with limited data (perception-provider agreement)
- Large enough to demonstrate non-trivial learned behavior
- Promotion-gated so heuristic fallback remains available

Training objectives (supervised, not RL)
----------------------------------------
These seams are middleware: they transform external provider outputs
into WM-native form.  Training uses supervised/contrastive/predictive
objectives, NOT direct RL on task reward.

**EvidenceFusionSeam**:
- Primary: minimize disagreement between fused state and held-out provider
- Secondary: maximize downstream task success correlation (as frozen target)
- Auxiliary: contrastive loss on provider availability patterns

**SAMCalibrationSeam**:
- Primary: calibrated confidence should match downstream mask quality
- Secondary: epistemic uncertainty should correlate with provider disagreement
- Auxiliary: prompt satisfaction should correlate with segmentation IoU

**VisionBackboneProjectionSeam**:
- Primary: projected features should predict downstream object identity
- Secondary: contrastive loss between projected features and scene-level labels
- Auxiliary: alignment with other provider embeddings (V-JEPA, depth)

**DepthMetricCalibrationSeam**:
- Primary: metric depth should match available ground truth (LiDAR, stereo)
- Secondary: uncertainty should correlate with depth estimation error
- Auxiliary: scale/shift consistency across frames

**VJEPATemporalAlignmentSeam**:
- Primary: aligned temporal state should predict actual future object state
- Secondary: temporal confidence should correlate with prediction accuracy
- Auxiliary: contrastive loss on temporal ordering

Checkpoint governance
---------------------
Seam checkpoints are managed by ``PerceptionSeamRegistry`` in
``seam_registry.py``.  Checkpoints are loaded/saved per seam_id,
enabling independent promotion/demotion of individual seams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Provider kind vocabulary
# ---------------------------------------------------------------------------

PROVIDER_KIND_VOCAB: Dict[str, int] = {
    "scene_tracks": 0,
    "teacher_semantics": 1,
    "teacher_trace": 2,
    "vision_backbone": 3,
}
NUM_PROVIDER_KINDS: int = len(PROVIDER_KIND_VOCAB)

TRUTH_CLASS_SCORES: Dict[str, float] = {
    "stub_smoke_only": 0.0,
    "unavailable": 0.1,
    "advisory_evidence": 0.5,
    "provider_backed": 1.0,
}


# ---------------------------------------------------------------------------
# Feature encoding helper
# ---------------------------------------------------------------------------


def encode_provider_features(
    *,
    provider_ids: List[str],
    provider_kinds: Dict[str, str],
    provider_availability: Dict[str, str],
    provider_truth_class: Dict[str, str],
    semantic_quality: float = 0.0,
    coverage: float = 0.0,
    disagreement: float = 0.0,
    object_count_norm: float = 0.0,
    edge_count_norm: float = 0.0,
) -> torch.Tensor:
    """Encode provider metadata into a feature tensor for the fusion seam.

    Per-provider feature layout (d=12):
        [kind_onehot(4), available(1), truth_score(1),
         sem_quality(1), coverage(1), disagreement(1),
         obj_norm(1), edge_norm(1), pad(1)]

    Returns:
        Tensor of shape ``(N_providers, 12)``.
    """
    features: List[List[float]] = []
    for pid in provider_ids:
        kind = provider_kinds.get(pid, "unknown")
        kind_idx = PROVIDER_KIND_VOCAB.get(kind, -1)
        kind_onehot = [0.0] * NUM_PROVIDER_KINDS
        if 0 <= kind_idx < NUM_PROVIDER_KINDS:
            kind_onehot[kind_idx] = 1.0

        avail = (
            1.0
            if provider_availability.get(pid, "unavailable") == "available"
            else 0.0
        )
        truth = TRUTH_CLASS_SCORES.get(
            provider_truth_class.get(pid, "unavailable"), 0.1
        )

        row = [
            *kind_onehot,
            avail,
            truth,
            float(semantic_quality),
            float(coverage),
            float(disagreement),
            float(object_count_norm),
            float(edge_count_norm),
            0.0,  # padding / future use
        ]
        features.append(row)

    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Evidence Fusion Seam
# ---------------------------------------------------------------------------


class EvidenceFusionSeam(nn.Module):
    """Tiny set-attention module for evidence fusion.

    Replaces the hardcoded weighted fusion in the heuristic path with
    a learned fusion that considers provider kind, availability, truth
    class, and belief-state context signals.

    Architecture::

        provider_features (N, 12)
            → input_proj (12 → d_model)
            → self-attention (n_heads, d_model)
            → layer-norm + FFN + layer-norm
            → weight_head (d_model → 1) per provider → softmax → weights
            → mean-pool → confidence_head → sigmoid → confidence

    The self-attention lets providers "see" each other's availability and
    kind, so the model can upweight scene_tracks when vla_semantic is
    unavailable, etc.

    Capacity: ~10-50K params (d_model=32, n_heads=2, d_ff=64).
    Deliberately tiny.  This is the smallest useful neural seam.

    Promotion posture: ``disabled|auto|required`` via standard machinery.
    When ``disabled``, the heuristic path runs instead.  When ``promoted``
    (benchmark-gated), this module's forward pass produces the weights.
    """

    D_PROVIDER_RAW: int = NUM_PROVIDER_KINDS + 8  # = 12

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.input_proj = nn.Linear(self.D_PROVIDER_RAW, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.weight_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(1, d_model // 2), 1),
        )

    def forward(
        self,
        provider_features: torch.Tensor,
        provider_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse provider evidence into weights and confidence.

        Args:
            provider_features: ``(batch, N, D_PROVIDER_RAW)`` or
                ``(N, D_PROVIDER_RAW)`` for unbatched single-sample.
            provider_mask: ``(batch, N)`` bool tensor where ``True``
                means the provider is masked/unavailable.

        Returns:
            weights: ``(batch, N)`` or ``(N,)`` — softmax-normalized
                provider contribution weights.
            confidence: ``(batch,)`` or scalar — fusion confidence
                in ``[0, 1]``.
        """
        unbatched = provider_features.dim() == 2
        if unbatched:
            provider_features = provider_features.unsqueeze(0)
            if provider_mask is not None:
                provider_mask = provider_mask.unsqueeze(0)

        # Input projection
        x = self.input_proj(provider_features)

        # Self-attention across providers
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=provider_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)

        # Per-provider weight logits → softmax
        weight_logits = self.weight_head(x).squeeze(-1)  # (batch, N)
        if provider_mask is not None:
            weight_logits = weight_logits.masked_fill(provider_mask, float("-inf"))
        weights = F.softmax(weight_logits, dim=-1)

        # Pooled confidence
        if provider_mask is not None:
            mask_f = (~provider_mask).unsqueeze(-1).float()
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)
        confidence = torch.sigmoid(
            self.confidence_head(pooled)
        ).squeeze(-1)  # (batch,) or scalar

        if unbatched:
            weights = weights.squeeze(0)
            confidence = confidence.squeeze(0)

        return weights, confidence

    @classmethod
    def heuristic_init(
        cls,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
    ) -> "EvidenceFusionSeam":
        """Create a seam with small-weight initialization.

        Weights start near zero so the initial output is approximately
        uniform softmax (conservative fusion).  Training will specialize
        the seam toward the empirically best fusion strategy.
        """
        model = cls(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.05)
                else:
                    nn.init.zeros_(p)
        return model

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        """Metadata for receipts and logging."""
        return {
            "seam_type": "evidence_fusion",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# SAM Calibration Seam
# ---------------------------------------------------------------------------


class SAMCalibrationSeam(nn.Module):
    """Calibration head for SAM 3/3.1 mask outputs.

    Takes raw SAM mask features (per-mask embeddings + raw confidence) and
    produces calibrated confidence, epistemic uncertainty, and prompt
    satisfaction score.

    Architecture::

        mask_features (N_masks, d_mask)
            → input_proj (d_mask → d_model)
            → self-attention (cross-mask reasoning)
            → layer-norm + FFN + layer-norm
            → calibration_head → (calibrated_conf, epistemic_unc, prompt_sat)

    Capacity: ~500K-2M params (d_model=128, n_heads=4, d_ff=256).
    Governed by Perception/Grounding WM, not SAM.

    Promotion posture: ``disabled|auto|required`` via standard machinery.
    When disabled, raw SAM confidences pass through uncalibrated.
    """

    def __init__(
        self,
        d_mask: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_mask = d_mask
        self.d_model = d_model
        self.n_heads = n_heads

        self.input_proj = nn.Linear(d_mask, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Three calibration outputs per mask
        self.calibration_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 3),  # conf, epistemic_unc, prompt_sat
        )

    def forward(
        self,
        mask_features: torch.Tensor,
        raw_confidence: torch.Tensor,
        mask_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calibrate SAM mask outputs.

        Args:
            mask_features: ``(batch, N_masks, d_mask)`` or ``(N_masks, d_mask)``.
            raw_confidence: ``(batch, N_masks)`` or ``(N_masks,)`` — SAM's
                raw per-mask confidence scores.
            mask_valid: ``(batch, N_masks)`` bool — True if mask is valid.

        Returns:
            Dict with:
                - calibrated_confidence: ``(batch, N_masks)``
                - epistemic_uncertainty: ``(batch, N_masks)``
                - prompt_satisfaction: ``(batch, N_masks)``
        """
        unbatched = mask_features.dim() == 2
        if unbatched:
            mask_features = mask_features.unsqueeze(0)
            raw_confidence = raw_confidence.unsqueeze(0)
            if mask_valid is not None:
                mask_valid = mask_valid.unsqueeze(0)

        # Concatenate raw confidence as additional feature
        conf_feat = raw_confidence.unsqueeze(-1)  # (batch, N, 1)
        x = torch.cat([mask_features, conf_feat], dim=-1)

        # Project (handle dimension mismatch gracefully)
        if x.size(-1) != self.d_mask:
            # Pad or truncate to expected d_mask
            if x.size(-1) < self.d_mask:
                pad = torch.zeros(
                    *x.shape[:-1], self.d_mask - x.size(-1),
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :self.d_mask]

        x = self.input_proj(x)

        # Self-attention across masks
        key_padding_mask = ~mask_valid if mask_valid is not None else None
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)

        # Calibration outputs
        calibration = self.calibration_head(x)  # (batch, N, 3)
        calibrated_conf = torch.sigmoid(calibration[..., 0])
        epistemic_unc = torch.sigmoid(calibration[..., 1])
        prompt_sat = torch.sigmoid(calibration[..., 2])

        result = {
            "calibrated_confidence": calibrated_conf,
            "epistemic_uncertainty": epistemic_unc,
            "prompt_satisfaction": prompt_sat,
        }

        if unbatched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {
            "seam_type": "sam_calibration",
            "d_mask": self.d_mask,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Vision Backbone Projection Seam
# ---------------------------------------------------------------------------


class VisionBackboneProjectionSeam(nn.Module):
    """Projection head for DINOv2/SigLIP backbone features.

    2-layer MLP that projects frozen backbone features to the canonical
    WM object token dimension.  Backbone weights are always frozen;
    only this projection head is trained.

    Architecture::

        backbone_features (H*W, d_backbone)
            → fc1 (d_backbone → d_hidden)
            → GELU + dropout
            → fc2 (d_hidden → d_out)
            → layer-norm

    Capacity: ~1-5M params depending on backbone dimension.
    d_backbone=1024 (DINOv2-L), d_hidden=512, d_out=128 → ~1.1M params.

    Promotion posture: ``disabled|auto|required``.
    When disabled, backbone features are either unavailable or passed
    through identity (if dimensions match).
    """

    def __init__(
        self,
        d_backbone: int = 1024,
        d_hidden: int = 512,
        d_out: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_backbone = d_backbone
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.fc1 = nn.Linear(d_backbone, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        backbone_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project backbone features to WM token space.

        Args:
            backbone_features: ``(batch, N_tokens, d_backbone)`` or
                ``(N_tokens, d_backbone)`` — ViT patch tokens or pooled
                features from frozen backbone.

        Returns:
            Projected features ``(batch, N_tokens, d_out)`` or
            ``(N_tokens, d_out)``.
        """
        unbatched = backbone_features.dim() == 2
        if unbatched:
            backbone_features = backbone_features.unsqueeze(0)

        x = self.fc1(backbone_features)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)

        if unbatched:
            x = x.squeeze(0)

        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {
            "seam_type": "vision_backbone_projection",
            "d_backbone": self.d_backbone,
            "d_hidden": self.d_hidden,
            "d_out": self.d_out,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Depth Metric Calibration Seam
# ---------------------------------------------------------------------------


class DepthMetricCalibrationSeam(nn.Module):
    """Metric calibration head for monocular depth estimation.

    Takes relative depth predictions from DepthAnythingV2/UniDepth and
    produces metric-calibrated depth via learned scale and shift
    corrections.  Also estimates per-pixel uncertainty.

    Architecture::

        depth_features (H, W, d_depth) + camera_intrinsics
            → spatial_encoder (conv stack)
            → global_pool + intrinsics_embed
            → scale_shift_head → (scale, shift)
            → uncertainty_head → per-pixel uncertainty

    The scale/shift are scene-global; uncertainty is per-pixel.

    Capacity: ~500K-1M params.

    Promotion posture: ``disabled|auto|required``.
    When disabled, raw relative depth passes through (not metric).
    """

    def __init__(
        self,
        d_depth: int = 1,
        d_hidden: int = 128,
        intrinsic_dim: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_depth = d_depth
        self.d_hidden = d_hidden

        # Spatial encoder: lightweight conv stack
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(d_depth, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(64, d_hidden, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Intrinsics embedding (fx, fy, cx, cy)
        self.intrinsics_embed = nn.Linear(intrinsic_dim, d_hidden)

        # Scale and shift prediction
        self.scale_shift_head = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),  # scale, shift
        )

        # Per-pixel uncertainty (operates on original resolution features)
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(d_depth, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        depth_map: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Calibrate relative depth to metric depth.

        Args:
            depth_map: ``(batch, 1, H, W)`` or ``(1, H, W)`` — relative
                depth from frozen depth model.
            camera_intrinsics: ``(batch, 4)`` or ``(4,)`` — [fx, fy, cx, cy].
                If None, uses default intrinsics.

        Returns:
            Dict with:
                - metric_depth: ``(batch, 1, H, W)`` — calibrated depth
                - scale: ``(batch,)`` — learned scale factor
                - shift: ``(batch,)`` — learned shift factor
                - uncertainty: ``(batch, 1, H, W)`` — per-pixel uncertainty
        """
        unbatched = depth_map.dim() == 3
        if unbatched:
            depth_map = depth_map.unsqueeze(0)
            if camera_intrinsics is not None:
                camera_intrinsics = camera_intrinsics.unsqueeze(0)

        batch_size = depth_map.size(0)
        device = depth_map.device
        dtype = depth_map.dtype

        # Default intrinsics if not provided (normalized focal length)
        if camera_intrinsics is None:
            camera_intrinsics = torch.tensor(
                [[1.0, 1.0, 0.5, 0.5]], device=device, dtype=dtype
            ).expand(batch_size, -1)

        # Global features from depth
        global_feat = self.spatial_encoder(depth_map).flatten(1)  # (batch, d_hidden)

        # Intrinsics embedding
        intrinsic_feat = self.intrinsics_embed(camera_intrinsics)  # (batch, d_hidden)

        # Predict scale and shift
        combined = torch.cat([global_feat, intrinsic_feat], dim=-1)
        scale_shift = self.scale_shift_head(combined)
        scale = F.softplus(scale_shift[:, 0]) + 0.1  # Ensure positive scale
        shift = scale_shift[:, 1]

        # Apply calibration
        metric_depth = depth_map * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)

        # Per-pixel uncertainty
        uncertainty = self.uncertainty_conv(depth_map)

        result = {
            "metric_depth": metric_depth,
            "scale": scale,
            "shift": shift,
            "uncertainty": uncertainty,
        }

        if unbatched:
            result = {
                k: v.squeeze(0) if v.dim() > 0 else v
                for k, v in result.items()
            }

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {
            "seam_type": "depth_metric_calibration",
            "d_depth": self.d_depth,
            "d_hidden": self.d_hidden,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# V-JEPA Temporal Alignment Seam
# ---------------------------------------------------------------------------


class VJEPATemporalAlignmentSeam(nn.Module):
    """Temporal alignment head for V-JEPA 2 latent predictions.

    Cross-attention module that aligns V-JEPA latent tokens (future
    predictions) to WM object tokens.  Enables temporal state reasoning
    without collapsing V-JEPA's rich latent space.

    Architecture::

        vjepa_tokens (T, N_vjepa, d_vjepa)
            → input_proj (d_vjepa → d_model)
            → cross-attn(query=wm_tokens, key/value=vjepa_tokens)
            → layer-norm + FFN + layer-norm
            → output_proj (d_model → d_out)

    The WM queries V-JEPA for temporal state; V-JEPA provides evidence.

    Capacity: ~2-5M params (d_model=256, n_heads=8).

    Promotion posture: ``disabled|auto|required``.
    When disabled, temporal state relies on scene_tracks extrapolation only.
    """

    def __init__(
        self,
        d_vjepa: int = 1024,
        d_wm_token: int = 128,
        d_model: int = 256,
        d_out: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        n_temporal_steps: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_vjepa = d_vjepa
        self.d_wm_token = d_wm_token
        self.d_model = d_model
        self.d_out = d_out
        self.n_heads = n_heads
        self.n_temporal_steps = n_temporal_steps

        # Project V-JEPA tokens to model dimension
        self.vjepa_proj = nn.Linear(d_vjepa, d_model)

        # Project WM tokens to query dimension
        self.wm_query_proj = nn.Linear(d_wm_token, d_model)

        # Temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, n_temporal_steps, 1, d_model) * 0.02
        )

        # Cross-attention: WM queries V-JEPA
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention among aligned tokens
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, d_out)

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        vjepa_tokens: torch.Tensor,
        wm_object_tokens: torch.Tensor,
        vjepa_mask: Optional[torch.Tensor] = None,
        wm_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Align V-JEPA temporal predictions to WM object tokens.

        Args:
            vjepa_tokens: ``(batch, T, N_vjepa, d_vjepa)`` — V-JEPA latent
                predictions for T future timesteps.
            wm_object_tokens: ``(batch, N_obj, d_wm_token)`` — current WM
                object tokens that will query temporal state.
            vjepa_mask: ``(batch, T, N_vjepa)`` — True if token is masked.
            wm_mask: ``(batch, N_obj)`` — True if object is masked.

        Returns:
            Dict with:
                - temporal_aligned: ``(batch, T, N_obj, d_out)`` — aligned
                    temporal state per object per timestep
                - temporal_confidence: ``(batch, T)`` — per-timestep
                    alignment confidence
        """
        unbatched = vjepa_tokens.dim() == 3
        if unbatched:
            vjepa_tokens = vjepa_tokens.unsqueeze(0)
            wm_object_tokens = wm_object_tokens.unsqueeze(0)
            if vjepa_mask is not None:
                vjepa_mask = vjepa_mask.unsqueeze(0)
            if wm_mask is not None:
                wm_mask = wm_mask.unsqueeze(0)

        batch_size = vjepa_tokens.size(0)
        T = vjepa_tokens.size(1)
        N_vjepa = vjepa_tokens.size(2)
        N_obj = wm_object_tokens.size(1)

        # Project inputs
        vjepa_proj = self.vjepa_proj(vjepa_tokens)  # (batch, T, N_vjepa, d_model)
        wm_query = self.wm_query_proj(wm_object_tokens)  # (batch, N_obj, d_model)

        # Add temporal position embedding
        if T <= self.n_temporal_steps:
            vjepa_proj = vjepa_proj + self.temporal_pos_embed[:, :T, :, :]
        else:
            # Interpolate position embeddings for longer sequences
            pos = F.interpolate(
                self.temporal_pos_embed.permute(0, 3, 1, 2),
                size=(T, 1),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            vjepa_proj = vjepa_proj + pos

        # Process each timestep
        temporal_aligned = []
        temporal_confidence = []

        for t in range(T):
            # V-JEPA keys/values for this timestep
            kv = vjepa_proj[:, t, :, :]  # (batch, N_vjepa, d_model)

            # Cross-attention: WM objects query V-JEPA
            t_mask = vjepa_mask[:, t, :] if vjepa_mask is not None else None
            cross_out, _ = self.cross_attn(
                wm_query, kv, kv, key_padding_mask=t_mask
            )
            x = self.norm1(wm_query + cross_out)

            # Self-attention among objects
            self_out, _ = self.self_attn(x, x, x, key_padding_mask=wm_mask)
            x = self.norm2(x + self_out)

            # Feed-forward
            ff_out = self.ffn(x)
            x = self.norm3(x + ff_out)

            # Output projection
            aligned = self.output_proj(x)  # (batch, N_obj, d_out)
            temporal_aligned.append(aligned)

            # Timestep confidence from pooled features
            if wm_mask is not None:
                mask_f = (~wm_mask).unsqueeze(-1).float()
                pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)
            conf = self.confidence_head(pooled).squeeze(-1)
            temporal_confidence.append(conf)

        # Stack outputs
        temporal_aligned = torch.stack(temporal_aligned, dim=1)  # (batch, T, N_obj, d_out)
        temporal_confidence = torch.stack(temporal_confidence, dim=1)  # (batch, T)

        result = {
            "temporal_aligned": temporal_aligned,
            "temporal_confidence": temporal_confidence,
        }

        if unbatched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {
            "seam_type": "vjepa_temporal_alignment",
            "d_vjepa": self.d_vjepa,
            "d_wm_token": self.d_wm_token,
            "d_model": self.d_model,
            "d_out": self.d_out,
            "n_heads": self.n_heads,
            "n_temporal_steps": self.n_temporal_steps,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


# ---------------------------------------------------------------------------
# Scene Graph Transformer Seam
# ---------------------------------------------------------------------------

# Edge type vocabulary for typed graph attention
EDGE_TYPE_VOCAB: Dict[str, int] = {
    "spatial_adjacency": 0,
    "contact": 1,
    "containment": 2,
    "occlusion": 3,
    "temporal_co_occurrence": 4,
    "affordance_relation": 5,
}
NUM_EDGE_TYPES: int = len(EDGE_TYPE_VOCAB)


class SceneGraphTransformerSeam(nn.Module):
    """Message-passing GNN seam for scene graph refinement.

    Replaces heuristic scene graph construction (SceneTracks Kalman
    filter + rule-based edges) with a learned message-passing module
    that refines object tokens and edge weights using typed graph
    attention.

    This is the canonical neural successor for Scene Graph / Relation
    State (subsystem #2 in the Perception / Grounding WM).

    Architecture::

        object_tokens (N, d_token=128) + edge_index (E, 2) + edge_type (E,)
            → node_proj (d_token → d_model)
            → edge_embed (num_edge_types → d_edge)
            → L layers of:
                edge-conditioned multihead attention (message passing)
                + layer-norm + FFN + layer-norm
            → node_out_proj (d_model → d_out)
            → edge_weight_head (d_model*2 + d_edge → 1) per edge → sigmoid
            → mean-pool → graph_confidence_head → sigmoid

    The edge-conditioned attention lets each object attend to neighbors
    with attention bias from edge type and features, so the model can
    learn that contact edges matter more than occlusion for physics
    planning, etc.

    Capacity: ~500K-2M params (d_model=128, n_heads=4, n_layers=2).
    Bounded: small enough for limited scene-graph supervision data,
    large enough for non-trivial relational reasoning.

    Promotion posture: ``disabled|auto|required`` via standard machinery.
    When ``disabled``, the heuristic SceneTracks scene graph is used.
    When promoted, this module refines the heuristic graph.
    """

    def __init__(
        self,
        d_token: int = 128,
        d_model: int = 128,
        d_out: int = 128,
        d_edge: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_token = d_token
        self.d_model = d_model
        self.d_out = d_out
        self.d_edge = d_edge
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_edge_types = num_edge_types

        # Input projections
        self.node_proj = nn.Linear(d_token, d_model)
        self.edge_type_embed = nn.Embedding(num_edge_types, d_edge)
        self.edge_feat_proj = nn.Linear(d_edge, d_edge)

        # Message-passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(_GraphMessagePassingLayer(
                d_model=d_model,
                d_edge=d_edge,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            ))

        # Output projections
        self.node_out_proj = nn.Linear(d_model, d_out)
        self.edge_weight_head = nn.Sequential(
            nn.Linear(d_model * 2 + d_edge, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        self.graph_confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine scene graph via message passing.

        Args:
            node_features: ``(batch, N, d_token)`` or ``(N, d_token)``
                — object token features from heuristic scene graph.
            edge_index: ``(batch, E, 2)`` or ``(E, 2)`` — source/target
                indices for each edge.
            edge_type: ``(batch, E)`` or ``(E,)`` — edge type indices.
            edge_features: ``(batch, E, d_edge)`` or ``(E, d_edge)``
                — optional continuous edge features.
            node_mask: ``(batch, N)`` — True if node is valid/present.

        Returns:
            Dict with:
                - refined_tokens: ``(batch, N, d_out)`` — refined node features
                - edge_weights: ``(batch, E)`` — learned edge importance in [0,1]
                - graph_confidence: ``(batch,)`` — overall graph quality in [0,1]
        """
        unbatched = node_features.dim() == 2
        if unbatched:
            node_features = node_features.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
            edge_type = edge_type.unsqueeze(0)
            if edge_features is not None:
                edge_features = edge_features.unsqueeze(0)
            if node_mask is not None:
                node_mask = node_mask.unsqueeze(0)

        batch_size = node_features.size(0)
        N = node_features.size(1)
        E = edge_index.size(1)

        # Project nodes
        x = self.node_proj(node_features)  # (B, N, d_model)

        # Compute edge embeddings
        edge_type_clamped = edge_type.clamp(0, self.num_edge_types - 1)
        e_type = self.edge_type_embed(edge_type_clamped)  # (B, E, d_edge)
        if edge_features is not None:
            e_feat = self.edge_feat_proj(edge_features)
            e = e_type + e_feat
        else:
            e = e_type

        # Message-passing layers
        for layer in self.mp_layers:
            x = layer(x, edge_index, e, node_mask)

        # Output node features
        refined_tokens = self.node_out_proj(x)  # (B, N, d_out)

        # Edge weight prediction
        src_idx = edge_index[..., 0].unsqueeze(-1).expand(-1, -1, x.size(-1))
        tgt_idx = edge_index[..., 1].unsqueeze(-1).expand(-1, -1, x.size(-1))
        src_feat = torch.gather(x, 1, src_idx)  # (B, E, d_model)
        tgt_feat = torch.gather(x, 1, tgt_idx)  # (B, E, d_model)
        edge_cat = torch.cat([src_feat, tgt_feat, e], dim=-1)
        edge_weights = torch.sigmoid(
            self.edge_weight_head(edge_cat).squeeze(-1)
        )  # (B, E)

        # Graph-level confidence from mean-pooled node features
        if node_mask is not None:
            mask_f = (~node_mask).unsqueeze(-1).float()
            # node_mask True = valid, so we want to pool valid nodes
            valid_f = node_mask.unsqueeze(-1).float()
            pooled = (x * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)
        graph_confidence = torch.sigmoid(
            self.graph_confidence_head(pooled).squeeze(-1)
        )  # (B,)

        result = {
            "refined_tokens": refined_tokens,
            "edge_weights": edge_weights,
            "graph_confidence": graph_confidence,
        }

        if unbatched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> Dict[str, Any]:
        return {
            "seam_type": "scene_graph_transformer",
            "d_token": self.d_token,
            "d_model": self.d_model,
            "d_out": self.d_out,
            "d_edge": self.d_edge,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "num_edge_types": self.num_edge_types,
            "param_count": self.param_count(),
            "module_class": self.__class__.__name__,
        }


class _GraphMessagePassingLayer(nn.Module):
    """Single message-passing layer with edge-conditioned attention.

    Implements a simplified edge-conditioned graph attention mechanism:
    1. For each node, aggregate neighbor features weighted by attention
       scores that incorporate edge embeddings.
    2. Apply layer-norm + FFN + layer-norm (standard transformer post-attn).

    This avoids requiring sparse attention libraries by computing
    attention via dense gather/scatter on the edge index.
    """

    def __init__(
        self,
        d_model: int,
        d_edge: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Query/key/value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Edge bias projection (d_edge → n_heads)
        self.edge_bias_proj = nn.Linear(d_edge, n_heads)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embed: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Message passing with edge-conditioned attention.

        Args:
            x: (B, N, d_model)
            edge_index: (B, E, 2) — [src, tgt] per edge
            edge_embed: (B, E, d_edge)
            node_mask: (B, N) — True if node is valid

        Returns:
            Updated x: (B, N, d_model)
        """
        B, N, D = x.shape
        E = edge_index.size(1)
        H = self.n_heads
        d_h = self.d_head

        # Compute Q, K, V for all nodes
        Q = self.q_proj(x).view(B, N, H, d_h)  # (B, N, H, d_h)
        K = self.k_proj(x).view(B, N, H, d_h)
        V = self.v_proj(x).view(B, N, H, d_h)

        # Gather source and target features via edge index
        src_idx = edge_index[..., 0]  # (B, E)
        tgt_idx = edge_index[..., 1]  # (B, E)

        # Expand indices for gathering: (B, E) → (B, E, H, d_h)
        src_expand = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, E, H, d_h)
        tgt_expand = tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(B, E, H, d_h)

        K_src = torch.gather(K, 1, src_expand)  # (B, E, H, d_h)
        V_src = torch.gather(V, 1, src_expand)  # (B, E, H, d_h)
        Q_tgt = torch.gather(Q, 1, tgt_expand)  # (B, E, H, d_h)

        # Attention scores: dot product + edge bias
        attn_logits = (Q_tgt * K_src).sum(dim=-1) / (d_h ** 0.5)  # (B, E, H)
        edge_bias = self.edge_bias_proj(edge_embed)  # (B, E, H)
        attn_logits = attn_logits + edge_bias

        # Softmax over edges targeting same node
        # Use scatter-based softmax: for each target node, softmax over
        # all edges pointing to it.
        # For bounded graphs this is tractable; we use a simple approach:
        # group by target node and apply softmax per group.
        attn_weights = _edge_softmax(attn_logits, tgt_idx, N)  # (B, E, H)

        # Weighted aggregation of values
        weighted_v = attn_weights.unsqueeze(-1) * V_src  # (B, E, H, d_h)

        # Scatter-add to target nodes
        agg = torch.zeros(B, N, H, d_h, device=x.device, dtype=x.dtype)
        tgt_scatter = tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(B, E, H, d_h)
        agg.scatter_add_(1, tgt_scatter, weighted_v)

        # Reshape and project
        agg = agg.reshape(B, N, D)
        agg = self.out_proj(agg)
        agg = self.dropout(agg)

        # Residual + norm
        x = self.norm1(x + agg)

        # FFN block
        ff_out = self.ffn(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)

        return x


def _edge_softmax(
    logits: torch.Tensor,
    tgt_idx: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Compute softmax over edges grouped by target node.

    Args:
        logits: (B, E, H) — raw attention logits per edge per head.
        tgt_idx: (B, E) — target node index for each edge.
        N: Total number of nodes.

    Returns:
        Attention weights: (B, E, H) — softmax-normalized per target node.
    """
    B, E, H = logits.shape

    # For numerical stability, subtract max per target node
    tgt_expand = tgt_idx.unsqueeze(-1).expand(B, E, H)
    max_logits = torch.full((B, N, H), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_logits.scatter_reduce_(1, tgt_expand, logits, reduce="amax", include_self=False)
    max_per_edge = torch.gather(max_logits, 1, tgt_expand)
    logits_shifted = logits - max_per_edge

    # Exponentiate
    exp_logits = logits_shifted.exp()

    # Sum per target node
    sum_exp = torch.zeros(B, N, H, device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(1, tgt_expand, exp_logits)
    sum_per_edge = torch.gather(sum_exp, 1, tgt_expand).clamp(min=1e-8)

    return exp_logits / sum_per_edge


__all__ = [
    "DepthMetricCalibrationSeam",
    "EDGE_TYPE_VOCAB",
    "EvidenceFusionSeam",
    "NUM_EDGE_TYPES",
    "PROVIDER_KIND_VOCAB",
    "SAMCalibrationSeam",
    "SceneGraphTransformerSeam",
    "TRUTH_CLASS_SCORES",
    "VisionBackboneProjectionSeam",
    "VJEPATemporalAlignmentSeam",
    "encode_provider_features",
]
