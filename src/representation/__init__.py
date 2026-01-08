"""Representation utilities for channel-set encoding and token providers."""

from .channel_groups import ChannelSpec, ChannelGroupSpec, load_channel_groups, validate_required_channels
from .channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig, ChannelSetEncoderOutput
from .loo_contrastive import LOOContrastiveConfig, compute_loo_contrastive_loss
from .token_providers import (
    TokenProviderOutput,
    BaseTokenProvider,
    EmbodimentTokenProvider,
    SceneGraphTokenProvider,
    RGBVisionTokenProvider,
    GaussianSceneTokenProvider,
)
from .channel_set_pipeline import (
    ChannelSetPipelineConfig,
    ChannelSetPipelineOutput,
    ChannelSetPipeline,
)

__all__ = [
    "ChannelSpec",
    "ChannelGroupSpec",
    "load_channel_groups",
    "validate_required_channels",
    "ChannelSetEncoder",
    "ChannelSetEncoderConfig",
    "ChannelSetEncoderOutput",
    "LOOContrastiveConfig",
    "compute_loo_contrastive_loss",
    "TokenProviderOutput",
    "BaseTokenProvider",
    "EmbodimentTokenProvider",
    "SceneGraphTokenProvider",
    "RGBVisionTokenProvider",
    "GaussianSceneTokenProvider",
    "ChannelSetPipelineConfig",
    "ChannelSetPipelineOutput",
    "ChannelSetPipeline",
]
