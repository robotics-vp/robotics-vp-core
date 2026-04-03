"""Bounded neural seams for the Perception / Grounding WM.

This module contains the first real ``torch.nn.Module`` implementations
behind the ``disabled|auto|required`` promotion posture.  These are not
heuristic helpers — they are bounded neural components that can be
trained, evaluated, and promoted/demoted via the standard machinery.

Anti-heuristic-without-neuralization: heuristic fusion is the
transitional fallback.  These seams are the successor codepath.

Current seams
-------------
- ``EvidenceFusionSeam``: tiny set-attention module that fuses
  heterogeneous provider evidence into calibrated provider weights
  and fusion confidence.  Replaces the hardcoded 0.55/0.25/0.15/0.05
  weighted fusion at the ``promoted`` promotion stage.

Capacity
--------
``EvidenceFusionSeam`` is deliberately tiny (10-50K params, d_model=32,
2 attention heads).  This is the smallest useful neural component that
can replace the heuristic fusion and demonstrate that the promotion
posture is real — not doctrinal aspiration.

Training objective (future)
---------------------------
Supervised on downstream task quality and evidence agreement.  Not
direct RL on task reward.  Bridges and fusion seams are middleware:
supervised/contrastive/predictive training is correct for structural
fidelity.
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


__all__ = [
    "PROVIDER_KIND_VOCAB",
    "TRUTH_CLASS_SCORES",
    "EvidenceFusionSeam",
    "encode_provider_features",
]
