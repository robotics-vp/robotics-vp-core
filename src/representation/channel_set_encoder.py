"""Channel-set encoder with permutation-invariant pooling."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChannelSetEncoderConfig:
    d_model: int = 128
    num_heads: int = 4
    pma_k: int = 1
    dropout: float = 0.0
    layernorm: bool = True


@dataclass
class ChannelSetEncoderOutput:
    canonical_tokens: torch.Tensor
    availability_mask: torch.Tensor
    channel_contributions: Dict[str, float]
    attn_weights: Optional[torch.Tensor] = None
    projected_tokens: Optional[Dict[str, torch.Tensor]] = None


class ChannelSetEncoder(nn.Module):
    """Permutation-invariant encoder over channel token sets per timestep."""

    def __init__(
        self,
        channel_names: List[str],
        config: Optional[ChannelSetEncoderConfig] = None,
    ) -> None:
        super().__init__()
        self.channel_names = list(channel_names)
        self.channel_to_idx = {name: idx for idx, name in enumerate(self.channel_names)}
        self.config = config or ChannelSetEncoderConfig()

        self.channel_embed = nn.Embedding(len(self.channel_names), self.config.d_model)
        self.projections = nn.ModuleDict()

        self.pma_queries = nn.Parameter(torch.randn(self.config.pma_k, self.config.d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.config.d_model) if self.config.layernorm else nn.Identity()

    def _project_channel_tokens(self, tokens: torch.Tensor, channel_name: str) -> torch.Tensor:
        if channel_name not in self.projections:
            layer = nn.Linear(tokens.shape[-1], self.config.d_model)
            _init_projection(layer, channel_name)
            self.projections[channel_name] = layer.to(tokens.device)
        proj = self.projections[channel_name](tokens)
        channel_idx = self.channel_to_idx[channel_name]
        embed = self.channel_embed.weight[channel_idx]
        return proj + embed

    def project_tokens(
        self,
        tokens_by_channel: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        projected: Dict[str, torch.Tensor] = {}
        for name in self.channel_names:
            if name not in tokens_by_channel:
                continue
            tokens = tokens_by_channel[name]
            if tokens is None:
                continue
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(0)
            projected[name] = self._project_channel_tokens(tokens, name)
        return projected

    def forward(
        self,
        tokens_by_channel: Dict[str, torch.Tensor],
        return_projected: bool = False,
    ) -> ChannelSetEncoderOutput:
        projected = self.project_tokens(tokens_by_channel)
        if not projected:
            raise ValueError("ChannelSetEncoder received no tokens to encode")

        channel_order = [name for name in self.channel_names if name in projected]
        stacked = torch.stack([projected[name] for name in channel_order], dim=2)
        # stacked: (B, T, C, D)
        B, T, C, D = stacked.shape
        flattened = stacked.reshape(B * T, C, D)

        query = self.pma_queries.unsqueeze(0).expand(B * T, -1, -1)
        attn_out, attn_weights = self.attn(query, flattened, flattened)
        attn_out = self.norm(attn_out)

        pooled = attn_out.reshape(B, T, self.config.pma_k, D)
        if self.config.pma_k == 1:
            canonical = pooled.squeeze(2)
        else:
            canonical = pooled

        availability_mask = torch.ones((B, T), device=canonical.device, dtype=torch.float32)
        contrib = _summarize_contributions(attn_weights, channel_order)

        return ChannelSetEncoderOutput(
            canonical_tokens=canonical,
            availability_mask=availability_mask,
            channel_contributions=contrib,
            attn_weights=attn_weights,
            projected_tokens=projected if return_projected else None,
        )


def _summarize_contributions(attn_weights: Optional[torch.Tensor], channel_order: List[str]) -> Dict[str, float]:
    if attn_weights is None or not channel_order:
        return {}
    # attn_weights: (B*T, K, C)
    with torch.no_grad():
        weights = attn_weights.mean(dim=(0, 1))
        weights = weights / (weights.sum() + 1e-6)
    return {name: float(weights[idx].item()) for idx, name in enumerate(channel_order)}


def _init_projection(layer: nn.Linear, channel_name: str) -> None:
    seed = _stable_seed(channel_name, layer.in_features, layer.out_features)
    gen = torch.Generator(device=layer.weight.device)
    gen.manual_seed(seed)
    std = 1.0 / max(1.0, float(layer.in_features)) ** 0.5
    nn.init.normal_(layer.weight, mean=0.0, std=std, generator=gen)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _stable_seed(channel_name: str, in_dim: int, out_dim: int) -> int:
    payload = f"{channel_name}:{in_dim}:{out_dim}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


__all__ = ["ChannelSetEncoderConfig", "ChannelSetEncoderOutput", "ChannelSetEncoder"]
