"""Geometry SSL contrastive loss for BEV tokens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeometrySSLContrastiveConfig:
    temperature: float = 0.1
    proj_dim: int = 64
    hidden_dim: int = 128
    queue_size: int = 0
    aug_translation_cells: int = 2
    aug_rotation: bool = True
    aug_dropout_prob: float = 0.1
    aug_mask_prob: float = 0.1
    jitter_std: float = 0.01
    use_temporal_positive: bool = False
    occlusion_cutout_frac: float = 0.0


class GeometrySSLContrastive(nn.Module):
    """DepthContrast-style contrastive loss for BEV grids or BEV tokens."""

    def __init__(self, config: Optional[GeometrySSLContrastiveConfig] = None) -> None:
        super().__init__()
        self.config = config or GeometrySSLContrastiveConfig()
        self._grid_encoder: Optional[nn.Module] = None
        self._grid_in_channels: Optional[int] = None
        self._token_head: Optional[nn.Module] = None
        self._token_in_features: Optional[int] = None
        self._queue: Optional[torch.Tensor] = None

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        bev_grid: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if bev_grid is None and tokens is None:
            return torch.tensor(0.0), {"skipped": True}

        if bev_grid is not None:
            z1, z2 = self._encode_grid_views(bev_grid)
            grid_used = True
        else:
            z1, z2 = self._encode_token_views(tokens)
            grid_used = False

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss, metrics = _multi_positive_infonce(
            z1,
            z2,
            temperature=self.config.temperature,
            use_temporal_positive=self.config.use_temporal_positive,
            sample_weights=sample_weights,
            queue=self._queue if self.config.queue_size > 0 else None,
        )

        metrics["grid_used"] = grid_used
        metrics["embed_norm_mean"] = float(z1.norm(dim=-1).mean().detach().cpu().item())
        metrics["embed_norm_std"] = float(z1.norm(dim=-1).std().detach().cpu().item())

        if self.config.queue_size > 0 and self.training:
            self._update_queue(z2)
        if self.config.queue_size > 0:
            queue_len = 0 if self._queue is None else int(self._queue.shape[0])
            metrics["queue_len"] = queue_len
            metrics["queue_capacity"] = int(self.config.queue_size)

        return loss, metrics

    def _encode_grid_views(self, bev_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, C = bev_grid.shape
        grid = bev_grid.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        if self._grid_encoder is None or self._grid_in_channels != C:
            self._grid_encoder = _make_grid_encoder(C, self.config.hidden_dim, self.config.proj_dim).to(grid.device)
            self._grid_in_channels = C
        view_a = _augment_grid(grid, self.config)
        view_b = _augment_grid(grid, self.config)
        z1 = self._grid_encoder(view_a)
        z2 = self._grid_encoder(view_b)
        return z1, z2

    def _encode_token_views(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens is None:
            raise ValueError("tokens required for token-view contrastive loss")
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        B, T, D = tokens.shape
        flat = tokens.reshape(B * T, D)
        if self._token_head is None or self._token_in_features != D:
            self._token_head = _make_token_head(D, self.config.hidden_dim, self.config.proj_dim).to(flat.device)
            self._token_in_features = D
        view_a = _augment_tokens(flat, self.config)
        view_b = _augment_tokens(flat, self.config)
        z1 = self._token_head(view_a)
        z2 = self._token_head(view_b)
        return z1, z2

    def _update_queue(self, embeddings: torch.Tensor) -> None:
        emb = embeddings.detach()
        if self._queue is None or self._queue.numel() == 0:
            self._queue = emb[-self.config.queue_size :].detach()
            return
        self._queue = torch.cat([self._queue, emb], dim=0)[-self.config.queue_size :]


def _make_grid_encoder(in_channels: int, hidden_dim: int, proj_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(hidden_dim, proj_dim),
    )


def _make_token_head(in_features: int, hidden_dim: int, proj_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, proj_dim),
    )


def _augment_grid(grid: torch.Tensor, cfg: GeometrySSLContrastiveConfig) -> torch.Tensor:
    out = grid.clone()
    if cfg.aug_rotation:
        k = int(torch.randint(0, 4, (1,), device=out.device).item())
        out = torch.rot90(out, k, dims=(2, 3))
    if cfg.aug_translation_cells > 0:
        span = cfg.aug_translation_cells * 2 + 1
        shift_x = int(torch.randint(0, span, (1,), device=out.device)) - cfg.aug_translation_cells
        shift_y = int(torch.randint(0, span, (1,), device=out.device)) - cfg.aug_translation_cells
        out = torch.roll(out, shifts=(shift_y, shift_x), dims=(2, 3))
    if cfg.occlusion_cutout_frac > 0:
        _, _, H, W = out.shape
        cutout = max(1, int(min(H, W) * cfg.occlusion_cutout_frac))
        y0 = int(torch.randint(0, max(1, H - cutout + 1), (1,), device=out.device))
        x0 = int(torch.randint(0, max(1, W - cutout + 1), (1,), device=out.device))
        out[:, :, y0 : y0 + cutout, x0 : x0 + cutout] = 0.0
    if cfg.aug_mask_prob > 0:
        mask = torch.rand_like(out[:, :1, :, :]) > cfg.aug_mask_prob
        out = out * mask
    if cfg.aug_dropout_prob > 0:
        drop = torch.rand_like(out[:, :1, :, :]) > cfg.aug_dropout_prob
        out = out * drop
    if cfg.jitter_std > 0:
        out = out + torch.randn_like(out) * cfg.jitter_std
    return out


def _augment_tokens(tokens: torch.Tensor, cfg: GeometrySSLContrastiveConfig) -> torch.Tensor:
    out = tokens.clone()
    if cfg.aug_dropout_prob > 0:
        drop = torch.rand_like(out) > cfg.aug_dropout_prob
        out = out * drop
    if cfg.jitter_std > 0:
        out = out + torch.randn_like(out) * cfg.jitter_std
    return out


def _multi_positive_infonce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    use_temporal_positive: bool,
    sample_weights: Optional[torch.Tensor],
    queue: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    temp = max(float(temperature), 1e-6)
    logits = z1 @ z2.T / temp
    pos_logit = torch.diagonal(logits)
    numer = torch.exp(pos_logit)
    if use_temporal_positive:
        idx = torch.arange(z1.shape[0], device=z1.device)
        if sample_weights is not None and sample_weights.dim() == 2:
            B, T = sample_weights.shape
            t = idx % T
            neighbor = idx + 1
            neighbor = torch.where(t + 1 < T, neighbor, idx)
        else:
            neighbor = idx + 1
            neighbor = torch.where(neighbor < z2.shape[0], neighbor, idx)
        numer = numer + torch.exp(logits[idx, neighbor])
    denom = torch.exp(logits).sum(dim=1)
    if queue is not None and queue.numel() > 0:
        denom = denom + torch.exp(z1 @ queue.T / temp).sum(dim=1)
    loss_vec = -torch.log((numer + 1e-6) / (denom + 1e-6))

    if sample_weights is not None:
        weights = sample_weights.reshape(-1).to(loss_vec.device)
        weights = weights[: loss_vec.shape[0]]
        weight_sum = weights.sum().clamp_min(1e-6)
        loss = (loss_vec * weights).sum() / weight_sum
        weight_mean = float(weights.mean().detach().cpu().item())
    else:
        loss = loss_vec.mean()
        weight_mean = 1.0

    if logits.numel() > 0 and logits.shape[0] > 1:
        neg_mean = float((logits.sum() - pos_logit.sum()).div(logits.numel() - pos_logit.numel()).detach().cpu().item())
    else:
        neg_mean = 0.0

    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "weight_mean": weight_mean,
        "pos_logit_mean": float(pos_logit.mean().detach().cpu().item()),
        "neg_logit_mean": neg_mean,
    }
    return loss, metrics


__all__ = ["GeometrySSLContrastiveConfig", "GeometrySSLContrastive"]
