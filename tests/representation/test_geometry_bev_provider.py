import numpy as np
import torch

from src.representation.token_providers import GeometryBEVProvider, GeometryBEVConfig
from src.representation.geom_ssl_contrastive import GeometrySSLContrastive


def _scene_tracks_payload(T: int = 4, K: int = 2) -> dict[str, np.ndarray]:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    for k in range(K):
        poses_t[:, k, 0] = np.linspace(0.1 * k, 0.4 * k, T)
        poses_t[:, k, 1] = 0.2 * k
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    ir_loss = np.zeros((T, K), dtype=np.float32)
    converged = np.ones((T, K), dtype=bool)
    return {
        "scene_tracks_v1/version": np.array(["v1"], dtype="U8"),
        "scene_tracks_v1/track_ids": track_ids,
        "scene_tracks_v1/entity_types": entity_types,
        "scene_tracks_v1/class_ids": class_ids,
        "scene_tracks_v1/poses_R": poses_R,
        "scene_tracks_v1/poses_t": poses_t,
        "scene_tracks_v1/scales": scales,
        "scene_tracks_v1/visibility": visibility,
        "scene_tracks_v1/occlusion": occlusion,
        "scene_tracks_v1/ir_loss": ir_loss,
        "scene_tracks_v1/converged": converged,
    }


def test_geometry_bev_deterministic():
    payload = _scene_tracks_payload()
    provider = GeometryBEVProvider(
        config=GeometryBEVConfig(resolution_m=0.5, extent_m=2.0, patch_size=2, embed_dim=32, seed=0),
    )
    out_a = provider.provide({"scene_tracks": payload})
    out_b = provider.provide({"scene_tracks": payload})
    max_diff = (out_a.tokens - out_b.tokens).abs().max().item()
    assert max_diff < 1e-6


def test_geometry_bev_target_len():
    payload = _scene_tracks_payload(T=3)
    provider = GeometryBEVProvider(
        config=GeometryBEVConfig(resolution_m=0.5, extent_m=2.0, patch_size=2, embed_dim=16, seed=0),
    )
    out = provider.provide({"scene_tracks": payload}, target_len=5)
    assert out.tokens.shape[1] == 5
    assert out.mask.shape[1] == 5


def test_geometry_ssl_backprop():
    torch.manual_seed(0)
    tokens = torch.randn(2, 4, 12, requires_grad=True)
    module = GeometrySSLContrastive()
    loss, _ = module(tokens=tokens)
    loss.backward()
    assert tokens.grad is not None
