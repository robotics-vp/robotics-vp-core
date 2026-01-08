"""Pipeline to build canonical tokens via channel-set encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence

import torch

from src.representation.channel_groups import ChannelGroupSpec, validate_required_channels
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderOutput
from src.representation.loo_contrastive import LOOContrastiveConfig, compute_loo_contrastive_loss
from src.representation.token_providers import BaseTokenProvider


@dataclass
class ChannelSetPipelineConfig:
    use_channel_set_encoder: bool = True
    use_loo_cl_pretrain: bool = False
    aux_loss_loo_cl_weight: float = 0.1
    target_len: Optional[int] = None
    enforce_required_channels: bool = True


@dataclass
class ChannelSetPipelineOutput:
    canonical_tokens: Optional[torch.Tensor]
    channel_tokens: Dict[str, torch.Tensor]
    channel_masks: Dict[str, torch.Tensor]
    channel_contributions: Dict[str, float]
    loo_cl_loss: Optional[torch.Tensor]
    loo_cl_metrics: Dict[str, Any]
    encoder_output: Optional[ChannelSetEncoderOutput]


class ChannelSetPipeline:
    def __init__(
        self,
        channel_spec: ChannelGroupSpec,
        providers: Sequence[BaseTokenProvider],
        encoder: Optional[ChannelSetEncoder] = None,
        config: Optional[ChannelSetPipelineConfig] = None,
        loo_config: Optional[LOOContrastiveConfig] = None,
    ) -> None:
        self.channel_spec = channel_spec
        self.providers = list(providers)
        self.encoder = encoder
        self.config = config or ChannelSetPipelineConfig()
        self.loo_config = loo_config or LOOContrastiveConfig()

    def encode(
        self,
        episodes: Sequence[Any] | Any,
        mode: str = "eval",
        device: Optional[torch.device] = None,
    ) -> ChannelSetPipelineOutput:
        tokens_by_channel: Dict[str, torch.Tensor] = {}
        masks_by_channel: Dict[str, torch.Tensor] = {}

        for provider in self.providers:
            output = provider.provide(episodes, target_len=self.config.target_len, device=device)
            tokens_by_channel[output.channel_name] = output.tokens
            masks_by_channel[output.channel_name] = output.mask

        if tokens_by_channel:
            target_len = self.config.target_len or max(t.shape[1] for t in tokens_by_channel.values())
            for name, tokens in tokens_by_channel.items():
                if tokens.shape[1] != target_len:
                    tokens_by_channel[name] = _resample_tokens(tokens, target_len)
                    masks_by_channel[name] = _resample_masks(masks_by_channel[name], target_len)

        if self.config.enforce_required_channels and self.config.use_channel_set_encoder:
            validate_required_channels(tokens_by_channel, self.channel_spec, mode=mode)

        canonical_tokens = None
        channel_contributions: Dict[str, float] = {}
        encoder_output = None
        loo_loss = None
        loo_metrics: Dict[str, Any] = {}

        if self.config.use_channel_set_encoder:
            if self.encoder is None:
                raise ValueError("ChannelSetPipeline requires an encoder when enabled")
            encoder_output = self.encoder(tokens_by_channel, return_projected=self.config.use_loo_cl_pretrain)
            canonical_tokens = encoder_output.canonical_tokens
            channel_contributions = encoder_output.channel_contributions

            if self.config.use_loo_cl_pretrain and encoder_output.projected_tokens:
                loo_loss, loo_metrics = compute_loo_contrastive_loss(
                    encoder_output.projected_tokens,
                    mask_by_channel=masks_by_channel,
                    config=self.loo_config,
                )

        return ChannelSetPipelineOutput(
            canonical_tokens=canonical_tokens,
            channel_tokens=tokens_by_channel,
            channel_masks=masks_by_channel,
            channel_contributions=channel_contributions,
            loo_cl_loss=loo_loss,
            loo_cl_metrics=loo_metrics,
            encoder_output=encoder_output,
        )


def _resample_tokens(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[1] == target_len:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    resized = torch.nn.functional.interpolate(tokens_t, size=target_len, mode="linear", align_corners=False)
    return resized.transpose(1, 2)


def _resample_masks(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[1] == target_len:
        return mask
    mask_t = mask.unsqueeze(1)
    resized = torch.nn.functional.interpolate(mask_t, size=target_len, mode="nearest")
    return resized.squeeze(1)


__all__ = [
    "ChannelSetPipelineConfig",
    "ChannelSetPipelineOutput",
    "ChannelSetPipeline",
]
