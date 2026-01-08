"""Validate BEV determinism across multiple episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.representation.token_providers import GeometryBEVProvider, GeometryBEVConfig
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
from src.valuation.datapack_repo import DataPackRepo
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
from src.utils.determinism import maybe_enable_determinism_from_env


def _hash_tensor(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
        scene_tracks_path = Path(meta.get("scene_tracks_path", ""))
        rgb_path = Path(meta.get("rgb_video_path", ""))
        if not scene_tracks_path.exists() or not rgb_path.exists():
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
            rgb_frames = None
    scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks_payload)
    return {
        "rgb_frames": rgb_frames,
        "scene_tracks": scene_tracks_payload,
        "scene_graphs": scene_graphs,
    }


def _load_episodes_from_repo(repo: DataPackRepo, task_name: str, max_episodes: int) -> List[Dict[str, Any]]:
    datapacks = repo.load_all(task_name)
    episodes: List[Dict[str, Any]] = []
    for dp in datapacks:
        if not dp.raw_data_path:
            continue
        raw_path = Path(dp.raw_data_path)
        episode = _load_episode_from_path(raw_path)
        if episode is None:
            continue
        episode["episode_id"] = dp.episode_id or dp.pack_id
        episodes.append(episode)
        if len(episodes) >= max_episodes:
            break
    return episodes


def _synthetic_scene_tracks(T: int, K: int, motion_scale: float) -> Dict[str, np.ndarray]:
    track_ids = np.array([f"track_{k}" for k in range(K)], dtype="U16")
    entity_types = np.zeros((K,), dtype=np.int32)
    class_ids = np.full((K,), -1, dtype=np.int32)
    poses_R = np.repeat(np.eye(3, dtype=np.float32)[None, None, ...], T, axis=0)
    poses_R = np.repeat(poses_R, K, axis=1)
    poses_t = np.zeros((T, K, 3), dtype=np.float32)
    for k in range(K):
        poses_t[:, k, 0] = np.linspace(0.0, motion_scale, T)
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


def _synthetic_episodes(count: int, T: int, K: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    episodes: List[Dict[str, Any]] = []
    for idx in range(count):
        rgb_frames = rng.integers(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
        scene_tracks = _synthetic_scene_tracks(T, K, motion_scale=0.5 + 0.1 * idx)
        scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks)
        episodes.append(
            {
                "episode_id": f"synthetic_{idx}",
                "rgb_frames": rgb_frames,
                "scene_tracks": scene_tracks,
                "scene_graphs": scene_graphs,
            }
        )
    return episodes


def main() -> None:
    maybe_enable_determinism_from_env(default_seed=0)
    parser = argparse.ArgumentParser(
        description="Validate BEV determinism across episodes",
        epilog=(
            "Examples:\n"
            "  python -m scripts.validate_bev_determinism --synthetic\n"
            "  python -m scripts.validate_bev_determinism --datapack-dir /path/to/datapacks --task drawer_vase"
        ),
    )
    parser.add_argument(
        "--datapack-dir",
        type=str,
        default="data/datapacks",
        help="Datapack repository root (requires raw_data_path with rgb_frames + scene_tracks)",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name (defaults to first in repo)")
    parser.add_argument("--max-episodes", type=int, default=50, help="Max episodes to scan")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic episodes")
    parser.add_argument("--resolution-m", type=float, default=0.2, help="BEV resolution (meters)")
    parser.add_argument("--extent-m", type=float, default=5.0, help="BEV extent (meters)")
    parser.add_argument("--patch-size", type=int, default=4, help="BEV patch size")
    parser.add_argument("--embed-dim", type=int, default=128, help="BEV embedding dim")
    parser.add_argument("--return-grid", action="store_true", help="Hash BEV grids instead of tokens")
    args = parser.parse_args()

    if args.synthetic:
        episodes = _synthetic_episodes(args.max_episodes, T=6, K=3)
    else:
        repo = DataPackRepo(base_dir=args.datapack_dir)
        tasks = repo.list_tasks()
        if not tasks:
            raise RuntimeError(f"No datapacks found in {args.datapack_dir}")
        task_name = args.task or tasks[0]
        episodes = _load_episodes_from_repo(repo, task_name, args.max_episodes)
        if not episodes:
            raise RuntimeError("No episodes available for BEV determinism check")

    cfg = GeometryBEVConfig(
        resolution_m=args.resolution_m,
        extent_m=args.extent_m,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        return_grid=args.return_grid,
    )

    provider_a = GeometryBEVProvider(config=cfg, allow_synthetic=True)
    provider_b = GeometryBEVProvider(config=cfg, allow_synthetic=True)

    mismatches = 0
    for ep in episodes:
        out_a = provider_a.provide(ep)
        out_b = provider_b.provide(ep)
        if args.return_grid:
            grid_a = out_a.metadata.get("bev_grid")
            grid_b = out_b.metadata.get("bev_grid")
            if grid_a is None or grid_b is None:
                raise RuntimeError("return_grid requested but BEV grid not returned")
            hash_a = _hash_tensor(grid_a.cpu().numpy())
            hash_b = _hash_tensor(grid_b.cpu().numpy())
        else:
            hash_a = _hash_tensor(out_a.tokens.cpu().numpy())
            hash_b = _hash_tensor(out_b.tokens.cpu().numpy())
        if hash_a != hash_b:
            mismatches += 1
            print(f"Mismatch: {ep.get('episode_id', 'unknown')}")

    total = len(episodes)
    if mismatches:
        raise RuntimeError(f"BEV determinism failed: {mismatches}/{total} mismatches")
    print(f"BEV determinism OK ({total} episodes)")


if __name__ == "__main__":
    main()
