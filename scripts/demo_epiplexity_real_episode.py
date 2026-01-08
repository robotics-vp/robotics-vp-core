"""Demo epiplexity on a real episode slice (vision vs geometry vs canonical)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
from src.embodiment.runner import run_embodiment_for_rollouts
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.representation.channel_groups import load_channel_groups
from src.representation.channel_set_encoder import ChannelSetEncoder, ChannelSetEncoderConfig
from src.representation.channel_set_pipeline import ChannelSetPipeline, ChannelSetPipelineConfig
from src.representation.token_providers import RGBVisionTokenProvider, EmbodimentTokenProvider, SceneGraphTokenProvider
from src.epiplexity.tracker import EpiplexityTracker, EpiplexityRunKey, ComputeBudget
from src.epiplexity.harness import TokenizerAblationHarness
from src.rl.episode_sampling import DataPackRLSampler
from src.utils.determinism import maybe_enable_determinism_from_env


EPISODE_DIR = Path("results/workcell_demo/episode_000")


def _load_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_rgb_frames(path: Path) -> np.ndarray:
    data = np.load(path)
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


def _window_tokens(tokens: torch.Tensor, window: int, stride: int) -> List[torch.Tensor]:
    windows = []
    T = tokens.shape[0]
    for start in range(0, T - window + 1, stride):
        windows.append(tokens[start : start + window])
    return windows


def _repr_fn(key: str):
    def fn(episodes):
        eps = episodes if isinstance(episodes, (list, tuple)) else [episodes]
        tensors = [torch.as_tensor(ep[key], dtype=torch.float32) for ep in eps]
        return torch.stack(tensors, dim=0)
    return fn


def main() -> None:
    seed = 0
    det_seed = maybe_enable_determinism_from_env(default_seed=seed)
    if det_seed is not None:
        seed = det_seed
    torch.manual_seed(seed)
    np.random.seed(0)

    meta = _load_metadata(EPISODE_DIR / "metadata.json")
    rgb_frames = _load_rgb_frames(Path(meta["rgb_video_path"]))
    scene_tracks_payload = _load_scene_tracks(Path(meta["scene_tracks_path"]))
    scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks_payload)

    embodiment_summary = _run_embodiment(EPISODE_DIR)

    episode = {
        "rgb_frames": rgb_frames,
        "scene_graphs": scene_graphs,
        "embodiment_profile": embodiment_summary,
    }

    spec = load_channel_groups("configs/channel_groups_robotics.json")
    encoder = ChannelSetEncoder(
        channel_names=list(spec.channels.keys()),
        config=ChannelSetEncoderConfig(d_model=64, num_heads=4, dropout=0.0, pma_k=1),
    )
    encoder.eval()

    pipeline = ChannelSetPipeline(
        channel_spec=spec,
        providers=[
            RGBVisionTokenProvider(seed=seed),
            EmbodimentTokenProvider(),
            SceneGraphTokenProvider(hidden_dim=64, num_layers=2, num_heads=4),
        ],
        encoder=encoder,
        config=ChannelSetPipelineConfig(use_channel_set_encoder=True, target_len=rgb_frames.shape[0]),
    )
    output = pipeline.encode(episode)

    vision_tokens = output.channel_tokens["vision_rgb"][0]
    geom_tokens = output.channel_tokens["geometry_scene_graph"][0]
    canonical_tokens = output.canonical_tokens[0]

    # Window the episode into multiple slices
    window = 2
    stride = 1
    vision_windows = _window_tokens(vision_tokens, window, stride)
    geom_windows = _window_tokens(geom_tokens, window, stride)
    canon_windows = _window_tokens(canonical_tokens, window, stride)

    episodes = []
    for idx in range(len(vision_windows)):
        episodes.append(
            {
                "episode_id": f"real_{idx}",
                "vision_rgb_tokens": vision_windows[idx].detach().cpu().numpy(),
                "geometry_scene_graph_tokens": geom_windows[idx].detach().cpu().numpy(),
                "canonical_tokens": canon_windows[idx].detach().cpu().numpy(),
            }
        )

    representation_fns = {
        "vision_rgb": _repr_fn("vision_rgb_tokens"),
        "geometry_scene_graph": _repr_fn("geometry_scene_graph_tokens"),
        "canonical_tokens": _repr_fn("canonical_tokens"),
    }

    harness = TokenizerAblationHarness(
        tracker=EpiplexityTracker(cache_dir="artifacts/epiplexity_cache_real"),
        representation_fns=representation_fns,
    )

    budgets = [ComputeBudget(max_steps=30, batch_size=8)]
    seeds = [0, 1, 2]

    leaderboard = harness.evaluate(
        episodes=episodes,
        repr_ids=list(representation_fns.keys()),
        budgets=budgets,
        seeds=seeds,
        baseline_repr="vision_rgb",
        dataset_slice_id="workcell_windows",
    )

    print("Epiplexity summary (mean):")
    for repr_id, budget_map in leaderboard.summaries.items():
        mean = budget_map[budgets[0].budget_id()]["mean"]
        print(
            f"  {repr_id}: S_T={mean['S_T_proxy']:.4f} H_T={mean['H_T_proxy']:.4f} "
            f"epi/step={mean['epi_per_flop']:.6f} delta={mean['delta_epi_vs_baseline']:.6f}"
        )

    # Sampling shift demo (per-episode w_epi from canonical vs vision)
    tracker = EpiplexityTracker(cache_dir="artifacts/epiplexity_cache_real")
    budget = budgets[0]
    epi_repr = "canonical_tokens"
    descriptors = []
    raw_w_epi: Dict[str, float] = {}
    for ep in episodes:
        base_key = EpiplexityRunKey(
            repr_id="vision_rgb",
            repr_version_hash="v1",
            tokenizer_version="v1",
            transform_chain_hash="v1",
            dataset_slice_id=ep["episode_id"],
            probe_model_id="probe",
            compute_budget_id=budget.budget_id(),
            seed=0,
        )
        alt_key = EpiplexityRunKey(
            repr_id=epi_repr,
            repr_version_hash="v1",
            tokenizer_version="v1",
            transform_chain_hash="v1",
            dataset_slice_id=ep["episode_id"],
            probe_model_id="probe",
            compute_budget_id=budget.budget_id(),
            seed=0,
        )
        base_tokens = torch.as_tensor(ep["vision_rgb_tokens"], dtype=torch.float32).unsqueeze(0)
        if epi_repr == "geometry_scene_graph":
            alt_tokens = torch.as_tensor(ep["geometry_scene_graph_tokens"], dtype=torch.float32).unsqueeze(0)
        else:
            alt_tokens = torch.as_tensor(ep["canonical_tokens"], dtype=torch.float32).unsqueeze(0)
        base_res = tracker.evaluate_tokens(base_tokens, base_key, budget)
        alt_res = tracker.evaluate_tokens(alt_tokens, alt_key, budget, baseline_result=base_res)
        w_epi = max(0.0, alt_res.delta_epi_vs_baseline)

        raw_w_epi[ep["episode_id"]] = w_epi
        descriptors.append(
            {
                "pack_id": ep["episode_id"],
                "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
                "env_name": "workcell_demo",
                "task_type": "workcell_demo",
                "engine_type": "workcell",
                "tier": 1,
                "trust_score": 0.5,
                "sampling_weight": 1.0,
                "w_epi": w_epi,
                "w_epi_raw": w_epi,
            }
        )

    if descriptors:
        values = [desc["w_epi"] for desc in descriptors]
        min_w, max_w = min(values), max(values)
        if max_w - min_w > 1e-6:
            for desc in descriptors:
                desc["w_epi"] = 0.1 + 0.9 * (desc["w_epi"] - min_w) / (max_w - min_w)
    print("w_epi raw per episode:", raw_w_epi)
    print("w_epi normalized per episode:", {desc["pack_id"]: desc["w_epi"] for desc in descriptors})
    sampler = DataPackRLSampler(existing_descriptors=descriptors, use_unified_quality=False)
    def sample_counts(strategy: str, draws: int = 200) -> Counter:
        counts = Counter()
        for seed in range(draws):
            batch = sampler.sample_batch(1, seed=seed, strategy=strategy)
            counts.update(item.get("pack_id") or item.get("episode_id") for item in batch)
        return counts

    print("Sampling counts (balanced):", dict(sample_counts("balanced")))
    print("Sampling counts (epiplexity_roi):", dict(sample_counts("epiplexity_roi")))


if __name__ == "__main__":
    main()
