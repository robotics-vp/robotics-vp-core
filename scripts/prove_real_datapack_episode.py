"""Run a real episode through channel-set encoding to validate alignment/determinism."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.embodiment.runner import run_embodiment_for_rollouts
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
from src.representation.channel_groups import load_channel_groups
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.channel_set_pipeline import ChannelSetPipeline, ChannelSetPipelineConfig
from src.representation.token_providers import (
    RGBVisionTokenProvider,
    EmbodimentTokenProvider,
    SceneGraphTokenProvider,
)
from src.utils.determinism import maybe_enable_determinism_from_env


DEFAULT_EPISODE_DIR = Path("results/workcell_demo/episode_000")


def _load_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_rgb_frames(path: Path) -> np.ndarray:
    data = np.load(path)
    if "frames" not in data:
        raise ValueError("rgb.npz missing 'frames' key")
    return data["frames"]


def _load_scene_tracks(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _scene_graphs_from_scene_tracks(scene_tracks_payload: Dict[str, np.ndarray]) -> List[SceneGraph]:
    scene_tracks = deserialize_scene_tracks_v1(scene_tracks_payload)
    poses_t = scene_tracks.poses_t
    T, K = poses_t.shape[:2]
    graphs = []

    base_node = SceneNode(
        id=0,
        polyline=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        node_type=NodeType.UNKNOWN,
    )
    for t in range(T):
        objects = []
        for k in range(K):
            pos = poses_t[t, k]
            obj = SceneObject(
                id=int(k),
                class_id=ObjectClass.UNKNOWN,
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
            )
            objects.append(obj)
        graphs.append(SceneGraph(nodes=[base_node], edges=[], objects=objects, metadata={"t": t}))
    return graphs


def _run_embodiment(episode_dir: Path) -> Dict[str, Any]:
    meta = _load_metadata(episode_dir / "metadata.json")
    episode_id = meta.get("metadata", {}).get("episode_id", "episode")
    trajectory_path = Path(meta.get("trajectory_path"))

    metadata = EpisodeMetadata(
        episode_id=episode_id,
        task_id=meta.get("metadata", {}).get("task_id", "task"),
        robot_family=meta.get("metadata", {}).get("robot_family"),
        seed=meta.get("metadata", {}).get("seed"),
        env_params=meta.get("metadata", {}).get("env_params", {}),
    )
    rollout = EpisodeRollout(metadata=metadata, trajectory_path=trajectory_path, metrics=meta.get("metrics", {}))
    bundle = RolloutBundle(scenario_id="workcell_demo", episodes=[rollout])

    summaries = run_embodiment_for_rollouts(bundle, output_dir=episode_dir)
    if not summaries:
        raise RuntimeError("Embodiment pipeline did not produce summaries")
    return summaries[0]


def _print_channel_shapes(tokens_by_channel: Dict[str, torch.Tensor], masks_by_channel: Dict[str, torch.Tensor]) -> None:
    for name, tokens in tokens_by_channel.items():
        mask = masks_by_channel.get(name)
        timesteps = int(mask[0].sum().item()) if mask is not None else tokens.shape[1]
        print(f"{name}: tokens={list(tokens.shape)} timesteps={timesteps}")


def main() -> None:
    episode_dir = DEFAULT_EPISODE_DIR
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode dir not found: {episode_dir}")

    seed = 0
    det_seed = maybe_enable_determinism_from_env(default_seed=seed)
    if det_seed is not None:
        seed = det_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = _load_metadata(episode_dir / "metadata.json")
    rgb_path = Path(meta.get("rgb_video_path"))
    scene_tracks_path = Path(meta.get("scene_tracks_path"))

    rgb_frames = _load_rgb_frames(rgb_path)
    scene_tracks_payload = _load_scene_tracks(scene_tracks_path)
    scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks_payload)

    embodiment_summary = _run_embodiment(episode_dir)

    episode = {
        "rgb_frames": rgb_frames,
        "scene_graphs": scene_graphs,
        "embodiment_profile": embodiment_summary,
    }

    spec = load_channel_groups("configs/channel_groups_robotics.json")
    d_model = 64

    torch.manual_seed(seed)
    rgb_provider = RGBVisionTokenProvider(seed=seed)
    emb_provider = EmbodimentTokenProvider()
    sg_provider = SceneGraphTokenProvider(hidden_dim=64, num_layers=2, num_heads=4)

    encoder = ChannelSetEncoder(
        channel_names=list(spec.channels.keys()),
        config=ChannelSetEncoderConfig(d_model=d_model, num_heads=4, dropout=0.0, pma_k=1),
    )
    encoder.eval()

    pipeline = ChannelSetPipeline(
        channel_spec=spec,
        providers=[rgb_provider, emb_provider, sg_provider],
        encoder=encoder,
        config=ChannelSetPipelineConfig(
            use_channel_set_encoder=True,
            use_loo_cl_pretrain=True,
            target_len=rgb_frames.shape[0],
        ),
    )

    output = pipeline.encode(episode, mode="eval")

    # Determinism checks for vision tokens
    torch.manual_seed(seed)
    rgb_provider_b = RGBVisionTokenProvider(seed=seed)
    out_a = rgb_provider.provide(episode)
    out_b = rgb_provider_b.provide(episode)
    rgb_diff = (out_a.tokens - out_b.tokens).abs().max().item()

    print("Per-channel token shapes + timesteps after resampling:")
    _print_channel_shapes(output.channel_tokens, output.channel_masks)
    timestep_set = {tokens.shape[1] for tokens in output.channel_tokens.values()}
    if len(timestep_set) != 1:
        raise RuntimeError(f"Channel timesteps mismatch: {timestep_set}")

    print(f"vision_rgb deterministic max diff: {rgb_diff:.6f}")

    # Canonical token norms
    canonical = output.canonical_tokens
    if canonical is None:
        raise RuntimeError("Canonical tokens not produced")

    norms = torch.linalg.norm(canonical[0], dim=-1)
    print("Canonical token norms (first 2 timesteps):", norms[:2].tolist())
    if torch.isnan(canonical).any():
        raise RuntimeError("NaNs detected in canonical tokens")

    # LOO-CL checks
    print("LOO-CL loss:", float(output.loo_cl_loss.detach().cpu().item()) if output.loo_cl_loss is not None else None)
    print("LOO-CL metrics:", json.dumps(output.loo_cl_metrics, indent=2))
    missing = output.loo_cl_metrics.get("missing_rates", {})
    if any(rate > 0 for rate in missing.values()):
        raise RuntimeError(f"LOO-CL skipped channels: {missing}")

    # Determinism of canonical tokens
    torch.manual_seed(seed)
    sg_provider_b = SceneGraphTokenProvider(hidden_dim=64, num_layers=2, num_heads=4)
    emb_provider_b = EmbodimentTokenProvider()
    encoder_b = ChannelSetEncoder(
        channel_names=list(spec.channels.keys()),
        config=ChannelSetEncoderConfig(d_model=d_model, num_heads=4, dropout=0.0, pma_k=1),
    )
    encoder_b.eval()
    pipeline_b = ChannelSetPipeline(
        channel_spec=spec,
        providers=[rgb_provider_b, emb_provider_b, sg_provider_b],
        encoder=encoder_b,
        config=ChannelSetPipelineConfig(
            use_channel_set_encoder=True,
            use_loo_cl_pretrain=True,
            target_len=rgb_frames.shape[0],
        ),
    )
    output_b = pipeline_b.encode(episode, mode="eval")
    max_diff = (output.canonical_tokens - output_b.canonical_tokens).abs().max().item()
    print(f"Canonical tokens deterministic max diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
