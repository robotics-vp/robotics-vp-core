"""CLI for running epiplexity evaluation over representations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.epiplexity import TokenizerAblationHarness, ComputeBudget, build_default_representation_fns
from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, SceneEdge, NodeType, EdgeType, ObjectClass
from src.utils.determinism import maybe_enable_determinism_from_env


def _load_episode_jsonl(path: str) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(json.loads(line))
    return episodes


def _synthetic_episodes(count: int = 4, T: int = 6, D: int = 8) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    episodes: List[Dict[str, Any]] = []
    nodes = [
        SceneNode(id=0, polyline=np.array([[0.0, 0.0], [1.0, 0.0]]), node_type=NodeType.CORRIDOR),
        SceneNode(id=1, polyline=np.array([[1.0, 0.0], [1.0, 1.0]]), node_type=NodeType.DOORWAY),
    ]
    edges = [SceneEdge(src_id=0, dst_id=1, edge_type=EdgeType.ADJACENT)]
    objects = [SceneObject(id=0, class_id=ObjectClass.ROBOT, x=0.5, y=0.2)]

    for idx in range(count):
        raw_tokens = rng.standard_normal((T, D)).astype(np.float32)
        homeomorphic_tokens = (raw_tokens + 0.05 * rng.standard_normal((T, D))).astype(np.float32)
        rgb_frames = rng.integers(0, 255, size=(T, 64, 64, 3), dtype=np.uint8)
        graphs = [SceneGraph(nodes=nodes, edges=edges, objects=objects, metadata={"t": t}) for t in range(T)]
        episodes.append(
            {
                "episode_id": f"synthetic_{idx}",
                "raw_tokens": raw_tokens,
                "homeomorphic_tokens": homeomorphic_tokens,
                "rgb_frames": rgb_frames,
                "scene_graphs": graphs,
            }
        )
    return episodes


def main() -> None:
    maybe_enable_determinism_from_env(default_seed=0)
    parser = argparse.ArgumentParser(description="Run epiplexity eval on representations")
    parser.add_argument("--episode-jsonl", type=str, default=None, help="JSONL file with episode payloads")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic episodes")
    parser.add_argument("--repr", action="append", dest="repr_ids", default=[], help="Representation id to evaluate")
    parser.add_argument("--baseline-repr", type=str, default="raw", help="Baseline representation id")
    parser.add_argument("--budget-steps", type=str, default="50", help="Comma-separated step budgets")
    parser.add_argument("--batch-size", type=int, default=16, help="Probe batch size")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--dataset-slice-id", type=str, default="synthetic_slice", help="Dataset slice identifier")
    parser.add_argument(
        "--channel-groups-path",
        type=str,
        default="configs/channel_groups_robotics.json",
        help="Channel group spec path",
    )
    parser.add_argument(
        "--store-full-runs",
        action="store_true",
        help="Store per-seed epiplexity runs in datapack metadata (debug only).",
    )

    args = parser.parse_args()

    if args.episode_jsonl:
        episodes = _load_episode_jsonl(args.episode_jsonl)
    else:
        episodes = _synthetic_episodes() if args.synthetic or not args.episode_jsonl else []

    repr_ids = args.repr_ids or [
        "raw",
        "vision_rgb",
        "geometry_scene_graph",
        "embodiment",
        "canonical_tokens",
        "homeomorphic",
    ]

    budgets = [ComputeBudget(max_steps=int(x.strip()), batch_size=args.batch_size) for x in args.budget_steps.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    representation_fns = build_default_representation_fns(args.channel_groups_path)
    harness = TokenizerAblationHarness(representation_fns=representation_fns)

    leaderboard = harness.evaluate(
        episodes=episodes,
        repr_ids=repr_ids,
        budgets=budgets,
        seeds=seeds,
        baseline_repr=args.baseline_repr,
        dataset_slice_id=args.dataset_slice_id,
        store_full_runs=args.store_full_runs,
    )

    print(json.dumps({"summaries": leaderboard.summaries}, indent=2))


if __name__ == "__main__":
    main()
