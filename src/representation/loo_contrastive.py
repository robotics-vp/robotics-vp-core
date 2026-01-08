"""Leave-one-out contrastive loss for channel embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn.functional as F


@dataclass
class LOOContrastiveConfig:
    temperature: float = 0.1


def _pool_tokens(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    if mask is None:
        return tokens.mean(dim=1)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(-1).to(dtype=tokens.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denom


def compute_loo_contrastive_loss(
    tokens_by_channel: Dict[str, torch.Tensor],
    mask_by_channel: Dict[str, torch.Tensor] | None = None,
    config: LOOContrastiveConfig | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Compute leave-one-out contrastive loss across channels.

    Args:
        tokens_by_channel: channel -> tokens (B, T, D) or (T, D)
        mask_by_channel: optional channel -> mask (B, T)
        config: LOO contrastive config

    Returns:
        (loss, metrics) where metrics include per-channel losses and cosine similarity.
    """
    cfg = config or LOOContrastiveConfig()
    pooled: Dict[str, torch.Tensor] = {}
    for name, tokens in tokens_by_channel.items():
        if tokens is None:
            continue
        mask = mask_by_channel.get(name) if mask_by_channel else None
        pooled[name] = _pool_tokens(tokens, mask)

    metrics: Dict[str, Any] = {
        "per_channel": {},
        "missing_rates": {},
    }
    if not pooled:
        return torch.tensor(0.0), metrics

    total_loss = 0.0
    count = 0
    for name, z_m in pooled.items():
        other = [z for c, z in pooled.items() if c != name]
        if not other:
            metrics["missing_rates"][name] = 1.0
            continue
        z_minus = torch.stack(other, dim=0).mean(dim=0)

        z_m_norm = F.normalize(z_m, dim=-1)
        z_minus_norm = F.normalize(z_minus, dim=-1)
        logits = z_m_norm @ z_minus_norm.T / max(cfg.temperature, 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, labels)
        cos = (z_m_norm * z_minus_norm).sum(dim=-1).mean()

        metrics["per_channel"][name] = {
            "loss": float(loss.detach().cpu().item()),
            "cosine": float(cos.detach().cpu().item()),
        }
        metrics["missing_rates"][name] = 0.0
        total_loss = total_loss + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0), metrics
    return total_loss / count, metrics


__all__ = ["LOOContrastiveConfig", "compute_loo_contrastive_loss"]
