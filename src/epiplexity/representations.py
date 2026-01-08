"""Representation function builders for epiplexity evaluation."""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Union

import torch

from src.representation.channel_groups import load_channel_groups
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.channel_set_pipeline import ChannelSetPipeline, ChannelSetPipelineConfig
from src.representation.token_providers import (
    EmbodimentTokenProvider,
    SceneGraphTokenProvider,
    RGBVisionTokenProvider,
    GaussianSceneTokenProvider,
    GeometryBEVProvider,
)

RepresentationFn = Callable[[Union[Sequence[Any], Any]], torch.Tensor]


def build_default_representation_fns(
    channel_groups_path: str,
    encoder_config: ChannelSetEncoderConfig | None = None,
    include_geometry_bev: bool = False,
    geometry_bev_config: Any | None = None,
) -> Dict[str, RepresentationFn]:
    spec = load_channel_groups(channel_groups_path)
    encoder = ChannelSetEncoder(list(spec.channels.keys()), encoder_config or ChannelSetEncoderConfig())
    encoder.eval()

    rgb_provider = RGBVisionTokenProvider(allow_synthetic=True)
    emb_provider = EmbodimentTokenProvider(allow_synthetic=True)
    scene_provider = SceneGraphTokenProvider()
    gaussian_provider = GaussianSceneTokenProvider()
    bev_provider = GeometryBEVProvider(config=geometry_bev_config, allow_synthetic=True)

    providers = [rgb_provider, emb_provider, scene_provider]
    if include_geometry_bev:
        providers.append(bev_provider)

    pipeline = ChannelSetPipeline(
        channel_spec=spec,
        providers=providers,
        encoder=encoder,
        config=ChannelSetPipelineConfig(use_channel_set_encoder=True),
    )

    return {
        "raw": lambda episodes: _tokens_from_key(episodes, "raw_tokens"),
        "vision_rgb": lambda episodes: rgb_provider.provide(episodes).tokens,
        "geometry_scene_graph": lambda episodes: scene_provider.provide(episodes).tokens,
        "geometry_bev": lambda episodes: bev_provider.provide(episodes).tokens,
        "embodiment": lambda episodes: emb_provider.provide(episodes).tokens,
        "canonical_tokens": lambda episodes: pipeline.encode(episodes).canonical_tokens,
        "homeomorphic": lambda episodes: _tokens_from_key(episodes, "homeomorphic_tokens"),
        "mhn_tokens": lambda episodes: _tokens_from_key(episodes, "mhn_tokens"),
        "geometry_gaussian_scene": lambda episodes: gaussian_provider.provide(episodes).tokens,
    }


def _tokens_from_key(episodes: Sequence[Any] | Any, key: str) -> torch.Tensor:
    eps = episodes if isinstance(episodes, (list, tuple)) else [episodes]
    collected = []
    for ep in eps:
        if isinstance(ep, dict) and key in ep:
            collected.append(torch.as_tensor(ep[key], dtype=torch.float32))
        else:
            raise ValueError(f"Episode missing '{key}' for representation")
    stacked = []
    for t in collected:
        if t.dim() == 2:
            stacked.append(t)
        elif t.dim() == 3:
            stacked.append(t.squeeze(0))
    return torch.stack(stacked, dim=0)


__all__ = ["build_default_representation_fns", "RepresentationFn"]
