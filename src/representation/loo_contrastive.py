"""Leave-one-out contrastive loss for channel embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LOOContrastiveConfig:
    temperature: float = 0.1
    proj_dim: int = 64
    hidden_dim: Optional[int] = None
    queue_size: int = 0
    min_channel_weight: float = 0.05


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


class LOOContrastive(nn.Module):
    """Multi-positive leave-one-out contrastive loss with per-channel heads."""

    def __init__(self, config: Optional[LOOContrastiveConfig] = None) -> None:
        super().__init__()
        self.config = config or LOOContrastiveConfig()
        self._heads = nn.ModuleDict()
        self._queues: Dict[str, torch.Tensor] = {}

    def _get_head(self, channel: str, in_dim: int) -> nn.Module:
        if channel in self._heads:
            return self._heads[channel]
        hidden_dim = self.config.hidden_dim or self.config.proj_dim
        head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.config.proj_dim),
        )
        self._heads[channel] = head
        return head

    def _channel_weight(
        self,
        channel: str,
        mask_by_channel: Dict[str, torch.Tensor] | None,
        channel_weights: Dict[str, float] | None,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        if channel_weights and channel in channel_weights:
            value = float(channel_weights[channel])
            weight = torch.full((batch_size,), value, device=device)
            return weight.clamp_min(self.config.min_channel_weight)
        if mask_by_channel and channel in mask_by_channel:
            mask = mask_by_channel[channel]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            weight = mask.float().mean(dim=1)
            return weight.clamp_min(self.config.min_channel_weight)
        return torch.full((batch_size,), 1.0, device=device)

    def forward(
        self,
        tokens_by_channel: Dict[str, torch.Tensor],
        mask_by_channel: Dict[str, torch.Tensor] | None = None,
        channel_weights: Dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        cfg = self.config
        pooled: Dict[str, torch.Tensor] = {}
        weights: Dict[str, torch.Tensor] = {}
        device = None

        for name, tokens in tokens_by_channel.items():
            if tokens is None:
                continue
            mask = mask_by_channel.get(name) if mask_by_channel else None
            pooled[name] = _pool_tokens(tokens, mask)
            if device is None:
                device = pooled[name].device

        metrics: Dict[str, Any] = {
            "per_channel": {},
            "missing_rates": {},
        }
        if not pooled:
            return torch.tensor(0.0), metrics

        projected: Dict[str, torch.Tensor] = {}
        for name, z in pooled.items():
            head = self._get_head(name, z.shape[-1]).to(z.device)
            proj = head(z)
            projected[name] = F.normalize(proj, dim=-1)
            weights[name] = self._channel_weight(name, mask_by_channel, channel_weights, z.device, z.shape[0])

        metrics["weight_stats"] = {}
        for name, weight in weights.items():
            metrics["weight_stats"][name] = {
                "mean": float(weight.mean().detach().cpu().item()),
                "min": float(weight.min().detach().cpu().item()),
                "max": float(weight.max().detach().cpu().item()),
            }

        temperature = max(cfg.temperature, 1e-6)
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)

        for anchor_name, anchor in projected.items():
            pos_names = [name for name in projected.keys() if name != anchor_name]
            if not pos_names:
                metrics["missing_rates"][anchor_name] = 1.0
                continue

            numer = None
            denom = None
            cos_terms = []
            for pos_name in pos_names:
                pos = projected[pos_name]
                logits = anchor @ pos.T / temperature
                pos_logit = torch.diagonal(logits)
                pos_exp = torch.exp(pos_logit)
                denom_exp = torch.exp(logits).sum(dim=1)

                queue = self._queues.get(pos_name)
                if queue is not None and queue.numel() > 0:
                    logits_q = anchor @ queue.T / temperature
                    denom_exp = denom_exp + torch.exp(logits_q).sum(dim=1)

                numer = pos_exp if numer is None else numer + pos_exp
                denom = denom_exp if denom is None else denom + denom_exp
                cos_terms.append(float(pos_logit.mean().detach().cpu().item()))

            loss_vec = -torch.log((numer + 1e-6) / (denom + 1e-6))
            anchor_weight = weights.get(anchor_name, torch.ones_like(loss_vec, device=device))
            pos_weight = torch.stack([weights[name] for name in pos_names], dim=0).mean(dim=0)
            weight_vec = anchor_weight * pos_weight
            channel_weight_sum = weight_vec.sum().clamp_min(1e-6)
            loss = (loss_vec * weight_vec).sum() / channel_weight_sum

            metrics["per_channel"][anchor_name] = {
                "loss": float(loss.detach().cpu().item()),
                "cosine": float(sum(cos_terms) / max(1, len(cos_terms))),
                "weight": float(weight_vec.mean().detach().cpu().item()),
            }
            metrics["missing_rates"][anchor_name] = 0.0
            total_loss = total_loss + loss * channel_weight_sum
            total_weight = total_weight + channel_weight_sum

        if cfg.queue_size > 0 and self.training:
            for name, emb in projected.items():
                queue = self._queues.get(name)
                emb_detached = emb.detach()
                if queue is None or queue.numel() == 0:
                    new_queue = emb_detached[-cfg.queue_size :]
                else:
                    new_queue = torch.cat([queue, emb_detached], dim=0)[-cfg.queue_size :]
                self._queues[name] = new_queue.to(emb.device)

        if total_weight.item() == 0:
            return torch.tensor(0.0, device=device), metrics
        return total_loss / total_weight, metrics


def compute_loo_contrastive_loss(
    tokens_by_channel: Dict[str, torch.Tensor],
    mask_by_channel: Dict[str, torch.Tensor] | None = None,
    config: LOOContrastiveConfig | None = None,
    channel_weights: Dict[str, float] | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Stateless helper for LOO-CL loss (use LOOContrastive for persistent heads/queue)."""
    module = LOOContrastive(config=config)
    return module(tokens_by_channel, mask_by_channel=mask_by_channel, channel_weights=channel_weights)


__all__ = ["LOOContrastiveConfig", "LOOContrastive", "compute_loo_contrastive_loss"]
