import numpy as np
import pytest
import torch

from src.representation.channel_groups import ChannelGroupSpec, ChannelSpec, validate_required_channels, ChannelSpecError
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.loo_contrastive import compute_loo_contrastive_loss
from src.representation.token_providers import RGBVisionTokenProvider
from src.utils.determinism import maybe_enable_determinism_from_env


def _channel_spec():
    return ChannelGroupSpec(
        version="v1",
        channels={
            "vision_rgb": ChannelSpec(name="vision_rgb", required=True),
            "embodiment": ChannelSpec(name="embodiment", required=True),
            "geometry_scene_graph": ChannelSpec(name="geometry_scene_graph", required=True),
            "geometry_gaussian_scene": ChannelSpec(name="geometry_gaussian_scene", required=False),
        },
    )


def test_channel_set_permutation_invariance():
    maybe_enable_determinism_from_env(default_seed=0)
    torch.manual_seed(0)
    spec = _channel_spec()
    encoder = ChannelSetEncoder(list(spec.channels.keys()), ChannelSetEncoderConfig(d_model=32, num_heads=4))

    tokens_by_channel = {
        "vision_rgb": torch.randn(2, 5, 16),
        "embodiment": torch.randn(2, 5, 8),
        "geometry_scene_graph": torch.randn(2, 5, 12),
    }
    out_a = encoder(tokens_by_channel)
    out_b = encoder(dict(reversed(list(tokens_by_channel.items()))))

    max_diff = (out_a.canonical_tokens - out_b.canonical_tokens).abs().max().item()
    assert max_diff < 1e-6


def test_missing_optional_channel_ok():
    spec = _channel_spec()
    tokens_by_channel = {
        "vision_rgb": torch.randn(1, 3, 4),
        "embodiment": torch.randn(1, 3, 4),
        "geometry_scene_graph": torch.randn(1, 3, 4),
    }
    missing = validate_required_channels(tokens_by_channel, spec, mode="eval")
    assert missing == []


def test_missing_required_channel_raises():
    spec = _channel_spec()
    tokens_by_channel = {
        "vision_rgb": torch.randn(1, 3, 4),
        "geometry_scene_graph": torch.randn(1, 3, 4),
    }
    with pytest.raises(ChannelSpecError):
        validate_required_channels(tokens_by_channel, spec, mode="train")


def test_loo_cl_backprop():
    torch.manual_seed(0)
    tokens_by_channel = {
        "vision_rgb": torch.randn(4, 6, 8, requires_grad=True),
        "embodiment": torch.randn(4, 6, 8, requires_grad=True),
    }
    loss, _ = compute_loo_contrastive_loss(tokens_by_channel)
    loss.backward()
    assert tokens_by_channel["vision_rgb"].grad is not None
    assert tokens_by_channel["embodiment"].grad is not None


def test_rgb_provider_layouts_and_res():
    provider = RGBVisionTokenProvider(token_dim=16, allow_synthetic=True)
    frames_hwc = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    frames_chw = np.zeros((3, 3, 48, 48), dtype=np.uint8)

    out_hwc = provider.provide({"rgb_frames": frames_hwc})
    out_chw = provider.provide({"rgb_frames": frames_chw})

    assert out_hwc.tokens.shape[-1] == 16
    assert out_chw.tokens.shape[-1] == 16
    assert out_hwc.tokens.shape[1] == out_chw.tokens.shape[1] == 3
