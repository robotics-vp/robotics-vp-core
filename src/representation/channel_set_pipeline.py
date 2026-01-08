"""Pipeline to build canonical tokens via channel-set encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence

import torch

from src.representation.channel_groups import ChannelGroupSpec, validate_required_channels
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderOutput
from src.representation.loo_contrastive import LOOContrastiveConfig, LOOContrastive
from src.representation.geom_ssl_contrastive import GeometrySSLContrastiveConfig, GeometrySSLContrastive
from src.representation.token_providers import BaseTokenProvider


@dataclass
class ChannelSetPipelineConfig:
    use_channel_set_encoder: bool = True
    aux_loss_loo_cl_weight: float = 0.1
    aux_loss_geom_ssl_weight: float = 0.0
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
    geom_ssl_loss: Optional[torch.Tensor]
    geom_ssl_metrics: Dict[str, Any]
    encoder_output: Optional[ChannelSetEncoderOutput]


class ChannelSetPipeline:
    def __init__(
        self,
        channel_spec: ChannelGroupSpec,
        providers: Sequence[BaseTokenProvider],
        encoder: Optional[ChannelSetEncoder] = None,
        config: Optional[ChannelSetPipelineConfig] = None,
        loo_config: Optional[LOOContrastiveConfig] = None,
        geom_ssl_config: Optional[GeometrySSLContrastiveConfig] = None,
    ) -> None:
        self.channel_spec = channel_spec
        self.providers = list(providers)
        self.encoder = encoder
        self.config = config or ChannelSetPipelineConfig()
        self.loo_module = LOOContrastive(loo_config or LOOContrastiveConfig())
        self.geom_ssl_module = GeometrySSLContrastive(geom_ssl_config or GeometrySSLContrastiveConfig())

    def encode(
        self,
        episodes: Sequence[Any] | Any,
        mode: str = "eval",
        device: Optional[torch.device] = None,
    ) -> ChannelSetPipelineOutput:
        tokens_by_channel: Dict[str, torch.Tensor] = {}
        masks_by_channel: Dict[str, torch.Tensor] = {}
        metadata_by_channel: Dict[str, Dict[str, Any]] = {}

        for provider in self.providers:
            output = provider.provide(episodes, target_len=self.config.target_len, device=device)
            tokens_by_channel[output.channel_name] = output.tokens
            masks_by_channel[output.channel_name] = output.mask
            metadata_by_channel[output.channel_name] = output.metadata

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
        geom_loss = None
        geom_metrics: Dict[str, Any] = {}

        if self.config.use_channel_set_encoder:
            if self.encoder is None:
                raise ValueError("ChannelSetPipeline requires an encoder when enabled")
            encoder_output = self.encoder(tokens_by_channel)
            canonical_tokens = encoder_output.canonical_tokens
            channel_contributions = encoder_output.channel_contributions
            self.loo_module.train(self.encoder.training)
            if self.encoder.training and self.config.aux_loss_loo_cl_weight > 0:
                channel_weights = _compute_channel_weights(tokens_by_channel, masks_by_channel, metadata_by_channel)
                loo_loss, loo_metrics = self.loo_module(
                    tokens_by_channel,
                    mask_by_channel=masks_by_channel,
                    channel_weights=channel_weights,
                )
                if loo_loss is not None:
                    weight = float(self.config.aux_loss_loo_cl_weight)
                    loo_metrics["loss_weight"] = weight
                    loo_loss = loo_loss * weight
            self.geom_ssl_module.train(self.encoder.training)
            if self.encoder.training and self.config.aux_loss_geom_ssl_weight > 0:
                geom_tokens = tokens_by_channel.get("geometry_bev")
                geom_mask = masks_by_channel.get("geometry_bev")
                geom_meta = metadata_by_channel.get("geometry_bev", {})
                if geom_tokens is not None:
                    geom_loss, geom_metrics = self.geom_ssl_module(
                        tokens=geom_tokens,
                        bev_grid=geom_meta.get("bev_grid"),
                        sample_weights=geom_mask,
                    )
                    if geom_loss is not None:
                        weight = float(self.config.aux_loss_geom_ssl_weight)
                        geom_metrics["loss_weight"] = weight
                        geom_loss = geom_loss * weight
                        if "bev_stats" in geom_meta:
                            geom_metrics["bev_stats"] = geom_meta["bev_stats"]

        return ChannelSetPipelineOutput(
            canonical_tokens=canonical_tokens,
            channel_tokens=tokens_by_channel,
            channel_masks=masks_by_channel,
            channel_contributions=channel_contributions,
            loo_cl_loss=loo_loss,
            loo_cl_metrics=loo_metrics,
            geom_ssl_loss=geom_loss,
            geom_ssl_metrics=geom_metrics,
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


def _compute_channel_weights(
    tokens_by_channel: Dict[str, torch.Tensor],
    masks_by_channel: Dict[str, torch.Tensor],
    metadata_by_channel: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    emb_tokens = tokens_by_channel.get("embodiment")
    emb_meta = metadata_by_channel.get("embodiment")
    if emb_tokens is not None and emb_meta is not None:
        weight = _embodiment_confidence_weight(emb_tokens, emb_meta)
        if weight is not None:
            mask = masks_by_channel.get("embodiment")
            if mask is not None:
                with torch.no_grad():
                    weight = float(weight * mask.float().mean().detach().cpu().item())
            weights["embodiment"] = float(weight)
    return weights


def _embodiment_confidence_weight(tokens: torch.Tensor, metadata: Dict[str, Any]) -> Optional[float]:
    names = metadata.get("feature_names")
    if not names:
        return None
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    indices = []
    for key in ("contact_conf_mean", "visibility_mean", "w_embodiment"):
        idx = name_to_idx.get(key)
        if idx is not None:
            indices.append(("raw", idx))
    occ_idx = name_to_idx.get("occlusion_mean")
    if occ_idx is not None:
        indices.append(("invert", occ_idx))
    if not indices:
        return None
    with torch.no_grad():
        values = []
        for mode, idx in indices:
            value = tokens[..., idx].float()
            if mode == "invert":
                value = 1.0 - value
            values.append(value.mean())
        weight = torch.stack(values).mean().clamp(0.0, 1.0)
    return float(weight.detach().cpu().item())


__all__ = [
    "ChannelSetPipelineConfig",
    "ChannelSetPipelineOutput",
    "ChannelSetPipeline",
]
