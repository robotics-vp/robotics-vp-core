"""Demo for channel-set encoder + LOO-CL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.representation.channel_groups import load_channel_groups
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.loo_contrastive import compute_loo_contrastive_loss
from src.representation.token_providers import (
    EmbodimentTokenProvider,
    SceneGraphTokenProvider,
    RGBVisionTokenProvider,
)
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, SceneEdge, NodeType, EdgeType, ObjectClass
from src.utils.determinism import maybe_enable_determinism_from_env
from src.valuation.datapack_repo import DataPackRepo


def _load_datapack_episode() -> Dict[str, Any] | None:
    base_dirs = ["data/datapacks/phase_c", "data/datapacks"]
    for base in base_dirs:
        repo = DataPackRepo(base_dir=base)
        tasks = repo.list_tasks()
        if not tasks:
            continue
        datapacks = repo.load_all(tasks[0])
        if datapacks:
            dp = datapacks[0]
            raw_path = getattr(dp, "raw_data_path", None)
            if raw_path and Path(raw_path).exists():
                return {"datapack": dp, "raw_data_path": raw_path}
    return None


def _synthetic_episode(T: int = 6) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)

    nodes = [
        SceneNode(id=0, polyline=np.array([[0.0, 0.0], [1.0, 0.0]]), node_type=NodeType.CORRIDOR),
        SceneNode(id=1, polyline=np.array([[1.0, 0.0], [1.0, 1.0]]), node_type=NodeType.DOORWAY),
    ]
    edges = [SceneEdge(src_id=0, dst_id=1, edge_type=EdgeType.ADJACENT)]
    objects = [SceneObject(id=0, class_id=ObjectClass.ROBOT, x=0.5, y=0.2)]
    graphs = [SceneGraph(nodes=nodes, edges=edges, objects=objects, metadata={"t": t}) for t in range(T)]

    return {
        "rgb_frames": rgb,
        "scene_graphs": graphs,
    }


def main() -> None:
    seed = 0
    det_seed = maybe_enable_determinism_from_env(default_seed=seed)
    if det_seed is not None:
        seed = det_seed
    torch.manual_seed(seed)
    spec = load_channel_groups("configs/channel_groups_robotics.json")
    episode = _load_datapack_episode() or _synthetic_episode()

    vision_provider = RGBVisionTokenProvider(allow_synthetic=True)
    other_providers = [
        EmbodimentTokenProvider(debug=True, allow_synthetic=True),
        SceneGraphTokenProvider(),
    ]

    tokens_by_channel: Dict[str, torch.Tensor] = {}
    vision_output = vision_provider.provide(episode)
    target_len = vision_output.tokens.shape[1]
    tokens_by_channel[vision_output.channel_name] = vision_output.tokens

    for provider in other_providers:
        output = provider.provide(episode, target_len=target_len)
        tokens_by_channel[output.channel_name] = output.tokens

    encoder = ChannelSetEncoder(
        channel_names=list(spec.channels.keys()),
        config=ChannelSetEncoderConfig(d_model=64, num_heads=4, pma_k=1),
    )

    out_a = encoder(tokens_by_channel, return_projected=True)
    out_b = encoder(dict(reversed(list(tokens_by_channel.items()))))

    diff = (out_a.canonical_tokens - out_b.canonical_tokens).abs().max().item()

    loo_loss, loo_metrics = compute_loo_contrastive_loss(out_a.projected_tokens or {})

    print("Channel tokens:")
    for name, tokens in tokens_by_channel.items():
        print(f"  {name}: {list(tokens.shape)}")

    print(f"Canonical tokens: {list(out_a.canonical_tokens.shape)}")
    print(f"Permutation invariance max diff: {diff:.6f}")
    print(f"LOO-CL loss: {float(loo_loss.detach().cpu().item()):.6f}")
    print("LOO-CL metrics:")
    print(json.dumps(loo_metrics, indent=2))


if __name__ == "__main__":
    main()
