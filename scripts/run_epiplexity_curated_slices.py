"""Run epiplexity eval on curated slices (occluded/dynamic/static)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.epiplexity import TokenizerAblationHarness, ComputeBudget, build_default_representation_fns
from src.epiplexity.slice_selector import SliceSelectorConfig, select_curated_slices
from src.representation.channel_set_encoder import ChannelSetEncoderConfig
from src.representation.token_providers import GeometryBEVConfig
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.portable_datapacks import coerce_scene_tracks_payload
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.utils.determinism import maybe_enable_determinism_from_env


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _load_rgb_frames(path: Path) -> np.ndarray:
    data = np.load(path)
    if "rgb_frames" in data:
        return data["rgb_frames"]
    if "frames" in data:
        return data["frames"]
    raise ValueError(f"RGB frames not found in {path}")


def _scene_graphs_from_scene_tracks(scene_tracks_payload: Dict[str, np.ndarray]) -> List[SceneGraph]:
    scene_tracks = deserialize_scene_tracks_v1(coerce_scene_tracks_payload(scene_tracks_payload))
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
            objects.append(
                SceneObject(
                    id=int(k),
                    class_id=ObjectClass.UNKNOWN,
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                )
            )
        graphs.append(SceneGraph(nodes=[base_node], edges=[], objects=objects, metadata={"t": t}))
    return graphs


def _extract_scene_tracks_payload(data: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if any(k.startswith("scene_tracks_v1/") for k in data):
        return {k: v for k, v in data.items() if k.startswith("scene_tracks_v1/")}
    if all(key in data for key in ("scene_tracks_v1/poses_t", "scene_tracks_v1/poses_R")):
        return data
    return None


def _load_episode_from_path(raw_path: Path) -> Optional[Dict[str, Any]]:
    if raw_path.is_dir():
        meta_path = raw_path / "metadata.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        rgb_path = Path(meta.get("rgb_video_path", ""))
        scene_tracks_path = Path(meta.get("scene_tracks_path", ""))
        if not rgb_path.exists() or not scene_tracks_path.exists():
            return None
        rgb_frames = _load_rgb_frames(rgb_path)
        scene_tracks_payload = _load_npz(scene_tracks_path)
    else:
        if raw_path.suffix != ".npz":
            return None
        data = _load_npz(raw_path)
        scene_tracks_payload = _extract_scene_tracks_payload(data)
        if scene_tracks_payload is None:
            return None
        if "rgb_frames" in data or "frames" in data:
            rgb_frames = data.get("rgb_frames") or data.get("frames")
        else:
            return None
    scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks_payload)
    return {
        "rgb_frames": rgb_frames,
        "scene_tracks": scene_tracks_payload,
        "scene_graphs": scene_graphs,
    }


def _load_episodes_from_repo(repo: DataPackRepo, task_name: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    datapacks = repo.load_all(task_name)
    episodes: List[Dict[str, Any]] = []
    stats = {
        "total": len(datapacks),
        "with_raw": 0,
        "missing_raw": 0,
        "missing_inputs": 0,
        "usable": 0,
        "portable": 0,
        "missing_portable": 0,
    }
    for dp in datapacks:
        if not dp.raw_data_path:
            stats["missing_raw"] += 1
            episode = _load_portable_episode(dp)
            if episode is None:
                stats["missing_portable"] += 1
                continue
            episodes.append(episode)
            stats["portable"] += 1
            stats["usable"] += 1
            continue
        stats["with_raw"] += 1
        raw_path = Path(dp.raw_data_path)
        episode = _load_episode_from_path(raw_path)
        if episode is None:
            stats["missing_inputs"] += 1
            continue
        episode["episode_id"] = dp.episode_id or dp.pack_id
        if dp.embodiment_profile is not None:
            episode["embodiment_profile"] = dp.embodiment_profile
        episodes.append(episode)
        stats["usable"] += 1
    return episodes, stats


def _load_portable_episode(dp: Any) -> Optional[Dict[str, Any]]:
    scene_tracks = getattr(dp, "scene_tracks_v1", None)
    rgb_features = getattr(dp, "rgb_features_v1", None)
    slice_labels = getattr(dp, "slice_labels_v1", None)
    if scene_tracks is None or rgb_features is None or slice_labels is None:
        return None
    scene_graphs = _scene_graphs_from_scene_tracks(coerce_scene_tracks_payload(scene_tracks))
    episode = {
        "episode_id": dp.episode_id or dp.pack_id,
        "scene_tracks": scene_tracks,
        "scene_graphs": scene_graphs,
        "rgb_features_v1": rgb_features,
        "slice_labels_v1": slice_labels,
    }
    if dp.embodiment_profile is not None:
        episode["embodiment_profile"] = dp.embodiment_profile
    return episode


def _load_token_only_episode(dp: Any) -> Optional[Dict[str, Any]]:
    """Load episode in token-only mode using stored repr_tokens."""
    repr_tokens = getattr(dp, "repr_tokens", None)
    slice_labels = getattr(dp, "slice_labels_v1", None)
    if repr_tokens is None or slice_labels is None:
        return None
    episode = {
        "episode_id": dp.episode_id or dp.pack_id,
        "repr_tokens": repr_tokens,
        "slice_labels_v1": slice_labels,
        "token_only": True,
    }
    if dp.embodiment_profile is not None:
        episode["embodiment_profile"] = dp.embodiment_profile
    return episode


def _load_episodes_token_only(repo: DataPackRepo, task_name: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load episodes in token-only mode (using stored repr_tokens)."""
    datapacks = repo.load_all(task_name)
    episodes: List[Dict[str, Any]] = []
    stats = {
        "total": len(datapacks),
        "with_repr_tokens": 0,
        "missing_repr_tokens": 0,
        "usable": 0,
    }
    for dp in datapacks:
        if dp.repr_tokens is None:
            stats["missing_repr_tokens"] += 1
            continue
        stats["with_repr_tokens"] += 1
        episode = _load_token_only_episode(dp)
        if episode is None:
            continue
        episodes.append(episode)
        stats["usable"] += 1
    return episodes, stats


def _build_token_only_leaderboard(
    episodes: List[Dict[str, Any]],
    repr_ids: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    """Build leaderboard summaries from stored repr_tokens.

    In token-only mode, we don't run the full prequential probe training.
    Instead, we compute simple statistics from stored features to produce
    comparable metrics that can be used for longitudinal tracking.
    """
    import numpy as np

    summaries: Dict[str, Any] = {}

    # Collect features per repr
    repr_features: Dict[str, List[np.ndarray]] = {r: [] for r in repr_ids}

    for ep in episodes:
        repr_tokens = ep.get("repr_tokens", {})
        for repr_id in repr_ids:
            token_data = repr_tokens.get(repr_id)
            if token_data is None or "error" in token_data:
                continue
            features = token_data.get("features")
            if features is not None:
                repr_features[repr_id].append(np.array(features))

    # Compute simple metrics per repr
    for repr_id in repr_ids:
        features_list = repr_features[repr_id]
        if not features_list:
            summaries[repr_id] = {
                "status": "no_data",
                "num_episodes": 0,
            }
            continue

        features_array = np.stack(features_list)
        # Compute variance as proxy for structure (higher variance = more structure)
        variance = float(np.var(features_array))
        mean_norm = float(np.mean(np.linalg.norm(features_array, axis=-1)))

        summaries[repr_id] = {
            "status": "ok",
            "num_episodes": len(features_list),
            "dim": int(features_array.shape[-1]) if features_array.ndim > 1 else 1,
            "variance": variance,
            "mean_norm": mean_norm,
            # Token-only mode doesn't run probes, so these are placeholders
            "S_T_proxy": variance,  # Structure proxy
            "H_T_proxy": 1.0 - variance,  # Residual entropy proxy (inverted)
            "mode": "token_only",
        }

    return summaries


def _select_curated_slices_from_labels(episodes: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    slices = {"occluded": [], "dynamic": [], "static": []}
    for ep in episodes:
        labels = ep.get("slice_labels_v1")
        if not isinstance(labels, dict):
            continue
        if labels.get("is_occluded"):
            slices["occluded"].append(ep)
        if labels.get("is_dynamic"):
            slices["dynamic"].append(ep)
        if labels.get("is_static"):
            slices["static"].append(ep)
    return slices


def _synthetic_scene_tracks(
    T: int,
    K: int,
    motion_scale: float,
    occlusion_rate: float,
) -> Dict[str, np.ndarray]:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    for k in range(K):
        poses_t[:, k, 0] = np.linspace(0.0, motion_scale, T) * (k + 1) / float(K)
        poses_t[:, k, 1] = 0.1 * k
    scales = np.ones((T, K), dtype=np.float32)
    visibility = np.ones((T, K), dtype=np.float32)
    occlusion = np.full((T, K), occlusion_rate, dtype=np.float32)
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


def _synthetic_curated_episodes(count_per_slice: int = 6, T: int = 6, K: int = 3) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    episodes: List[Dict[str, Any]] = []
    configs = {
        "occluded": {"motion_scale": 0.1, "occlusion_rate": 0.8},
        "dynamic": {"motion_scale": 2.0, "occlusion_rate": 0.1},
        "static": {"motion_scale": 0.02, "occlusion_rate": 0.05},
    }
    for slice_id, cfg in configs.items():
        for idx in range(count_per_slice):
            rgb_frames = rng.integers(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
            scene_tracks = _synthetic_scene_tracks(T, K, cfg["motion_scale"], cfg["occlusion_rate"])
            scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks)
            episodes.append(
                {
                    "episode_id": f"synthetic_{slice_id}_{idx}",
                    "rgb_frames": rgb_frames,
                    "scene_tracks": scene_tracks,
                    "scene_graphs": scene_graphs,
                }
            )
    return episodes


def main() -> None:
    maybe_enable_determinism_from_env(default_seed=0)
    parser = argparse.ArgumentParser(
        description="Run epiplexity on curated slices",
        epilog=(
            "Examples:\n"
            "  python -m scripts.run_epiplexity_curated_slices --synthetic\n"
            "  python -m scripts.run_epiplexity_curated_slices --datapack-dir /path/to/datapacks --task drawer_vase"
        ),
    )
    parser.add_argument(
        "--datapack-dir",
        type=str,
        default="data/datapacks",
        help="Datapack repository root (requires raw_data_path with rgb_frames + scene_tracks)",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name (defaults to first in repo)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic episodes instead of datapacks")
    parser.add_argument("--synthetic-count", type=int, default=6, help="Synthetic episodes per slice")
    parser.add_argument("--max-per-slice", type=int, default=12, help="Max episodes per slice")
    parser.add_argument("--occlusion-threshold", type=float, default=0.4, help="Occlusion threshold")
    parser.add_argument("--dynamic-threshold", type=float, default=0.15, help="Motion threshold for dynamic slice")
    parser.add_argument("--static-threshold", type=float, default=0.05, help="Motion threshold for static slice")
    parser.add_argument("--budget-steps", type=str, default="50", help="Comma-separated step budgets")
    parser.add_argument("--batch-size", type=int, default=16, help="Probe batch size")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument(
        "--channel-groups-path",
        type=str,
        default="configs/channel_groups_robotics.json",
        help="Channel group spec path",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/epiplexity_leaderboards", help="Output dir")
    parser.add_argument(
        "--token-only",
        action="store_true",
        help="Use stored repr_tokens instead of running encoders (requires datapacks with repr_tokens)",
    )
    args = parser.parse_args()

    stats = {"with_raw": 0, "portable": 0, "token_only": 0}
    is_token_only = args.token_only

    if args.synthetic:
        if is_token_only:
            raise RuntimeError("--token-only cannot be used with --synthetic")
        episodes = _synthetic_curated_episodes(args.synthetic_count)
    elif is_token_only:
        # Token-only mode: load using stored repr_tokens
        repo = DataPackRepo(base_dir=args.datapack_dir)
        tasks = repo.list_tasks()
        if not tasks:
            raise RuntimeError(f"No datapacks found in {args.datapack_dir}")
        task_name = args.task or tasks[0]
        episodes, stats = _load_episodes_token_only(repo, task_name)
        stats["token_only"] = stats.get("with_repr_tokens", 0)
        if not episodes:
            raise RuntimeError(
                "Cannot run token-only curated slices: no datapacks with repr_tokens found "
                f"(total={stats['total']}; with_repr_tokens={stats['with_repr_tokens']}; "
                f"missing_repr_tokens={stats['missing_repr_tokens']}). "
                "Run export_portable_datapacks with --include-repr-tokens first."
            )
    else:
        repo = DataPackRepo(base_dir=args.datapack_dir)
        tasks = repo.list_tasks()
        if not tasks:
            raise RuntimeError(f"No datapacks found in {args.datapack_dir}")
        task_name = args.task or tasks[0]
        episodes, stats = _load_episodes_from_repo(repo, task_name)
        if not episodes:
            if stats["with_raw"] == 0:
                raise RuntimeError(
                    "Cannot run curated slices: no raw_data_path present "
                    f"(total={stats['total']}; raw_data_path_nonnull={stats['with_raw']}; "
                    f"missing_raw={stats['missing_raw']}; usable={stats['usable']}); "
                    "required inputs missing (frames for vision_rgb; scene_tracks for geometry_bev). "
                    "Provide raw streams or embedded scene_tracks_v1 + rgb_features_v1 + slice_labels_v1."
                )
            raise RuntimeError(
                "Cannot run curated slices: raw_data_path present but required inputs missing "
                f"(total={stats['total']}; raw_data_path_nonnull={stats['with_raw']}; "
                f"missing_inputs={stats['missing_inputs']}; usable={stats['usable']}). "
                "Ensure raw_data_path points to data with rgb_frames and scene_tracks."
            )

    selector_cfg = SliceSelectorConfig(
        occlusion_threshold=args.occlusion_threshold,
        dynamic_motion_threshold=args.dynamic_threshold,
        static_motion_threshold=args.static_threshold,
        max_per_slice=args.max_per_slice,
    )

    # Select curated slices based on mode
    if is_token_only or (stats.get("with_raw", 0) == 0 and stats.get("portable", 0) > 0) or stats.get("token_only", 0) > 0:
        curated = _select_curated_slices_from_labels(episodes)
        if not any(curated.values()):
            raise RuntimeError("Datapacks missing slice_labels_v1; cannot compute curated slices.")
    else:
        curated = select_curated_slices(episodes, selector_cfg)

    budgets = [ComputeBudget(max_steps=int(x.strip()), batch_size=args.batch_size) for x in args.budget_steps.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    repr_ids = ["vision_rgb", "geometry_scene_graph", "geometry_bev", "canonical_tokens"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_token_only:
        # Token-only mode: use stored repr_tokens directly
        for slice_id, slice_episodes in curated.items():
            if not slice_episodes:
                continue
            dataset_slice_id = f"curated_{slice_id}"

            # Build leaderboard from stored repr_tokens
            slice_summaries = _build_token_only_leaderboard(slice_episodes, repr_ids, seeds)
            leaderboard_file = output_dir / f"{dataset_slice_id}.json"
            leaderboard_file.write_text(json.dumps({"summaries": slice_summaries}, indent=2))
            print(json.dumps({"slice": slice_id, "summaries": slice_summaries, "mode": "token_only"}, indent=2))
    else:
        # Normal mode: run harness with full encoding
        geometry_bev_cfg = GeometryBEVConfig()
        representation_fns = build_default_representation_fns(
            args.channel_groups_path,
            encoder_config=ChannelSetEncoderConfig(),
            include_geometry_bev=True,
            geometry_bev_config=geometry_bev_cfg,
        )
        harness = TokenizerAblationHarness(representation_fns=representation_fns, output_dir=args.output_dir)

        for slice_id, slice_episodes in curated.items():
            if not slice_episodes:
                continue
            dataset_slice_id = f"curated_{slice_id}"
            leaderboard = harness.evaluate(
                episodes=slice_episodes,
                repr_ids=repr_ids,
                budgets=budgets,
                seeds=seeds,
                baseline_repr="vision_rgb",
                dataset_slice_id=dataset_slice_id,
            )
            print(json.dumps({"slice": slice_id, "summaries": leaderboard.summaries}, indent=2))


if __name__ == "__main__":
    main()
