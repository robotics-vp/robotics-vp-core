#!/usr/bin/env python3
"""Export portable datapacks with embedded tracks/features/slice labels."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DATAPACK_SCHEMA_VERSION_PORTABLE, DATAPACK_SCHEMA_VERSION_REPR
from src.valuation.portable_datapacks import (
    load_raw_episode_artifacts,
    compute_rgb_features_v1,
    compute_slice_labels_v1,
    compute_repr_tokens_v1,
)
from src.representation.token_providers import SceneGraphTokenProvider


def _scene_graphs_from_scene_tracks(scene_tracks_payload):
    """Convert scene tracks to scene graphs for geometry providers."""
    from src.valuation.portable_datapacks import coerce_scene_tracks_payload
    from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
    from src.scene.vector_scene.graph import SceneGraph, SceneNode, SceneObject, NodeType, ObjectClass
    import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export portable datapacks")
    parser.add_argument("--datapack-dir", type=str, default="data/datapacks", help="Source datapack repo root")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output datapack repo root (defaults to <datapack-dir>_portable)",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to export")
    parser.add_argument("--stride-seconds", type=float, default=1.0, help="RGB feature stride (seconds)")
    parser.add_argument("--source-fps", type=float, default=None, help="Source FPS (optional)")
    parser.add_argument("--store-temporal", action="store_true", help="Store temporal RGB features")
    parser.add_argument("--token-dim", type=int, default=64, help="vision_rgb token dim")
    parser.add_argument("--pool-size", type=int, nargs=2, default=(4, 4), help="vision_rgb pooling size")
    parser.add_argument(
        "--include-repr-tokens",
        type=str,
        default=None,
        help="Comma-separated repr names to compute and store (e.g., vision_rgb,geometry_bev)",
    )
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapack_dir)
    datapacks = repo.load_all(args.task)
    if not datapacks:
        raise RuntimeError(f"No datapacks found for task '{args.task}' in {args.datapack_dir}")

    output_dir = args.output_dir or f"{args.datapack_dir}_portable"
    out_repo = DataPackRepo(base_dir=output_dir)

    # Parse repr token names
    repr_names: Optional[List[str]] = None
    if args.include_repr_tokens:
        repr_names = [x.strip() for x in args.include_repr_tokens.split(",") if x.strip()]

    exported = 0
    skipped_missing_raw = 0
    skipped_missing_inputs = 0
    reused_existing = 0
    repr_tokens_computed = 0

    for dp in datapacks:
        if args.max_episodes is not None and exported >= args.max_episodes:
            break

        # Check if already has portable artifacts (reuse if no repr_tokens requested or already has them)
        already_portable = dp.scene_tracks_v1 and dp.rgb_features_v1 and dp.slice_labels_v1
        already_has_repr = dp.repr_tokens is not None
        if already_portable and (repr_names is None or already_has_repr):
            dp.schema_version = DATAPACK_SCHEMA_VERSION_REPR if dp.repr_tokens else DATAPACK_SCHEMA_VERSION_PORTABLE
            out_repo.append(dp)
            reused_existing += 1
            exported += 1
            continue

        if not dp.raw_data_path:
            skipped_missing_raw += 1
            continue

        rgb_frames, scene_tracks, _ = load_raw_episode_artifacts(Path(dp.raw_data_path))
        if rgb_frames is None or scene_tracks is None:
            skipped_missing_inputs += 1
            continue

        rgb_features = compute_rgb_features_v1(
            rgb_frames,
            token_dim=args.token_dim,
            pool_size=tuple(args.pool_size),
            stride_seconds=args.stride_seconds,
            source_fps=args.source_fps,
            store_temporal=args.store_temporal,
        )
        slice_labels = compute_slice_labels_v1(dp.condition, scene_tracks)
        dp.scene_tracks_v1 = scene_tracks
        dp.rgb_features_v1 = rgb_features
        dp.slice_labels_v1 = slice_labels

        # Compute repr tokens if requested
        if repr_names:
            scene_graphs = _scene_graphs_from_scene_tracks(scene_tracks)
            episode_artifacts = {
                "rgb_frames": rgb_frames,
                "scene_tracks": scene_tracks,
                "scene_graphs": scene_graphs,
            }
            dp.repr_tokens = compute_repr_tokens_v1(episode_artifacts, repr_names)
            dp.schema_version = DATAPACK_SCHEMA_VERSION_REPR
            repr_tokens_computed += 1
        else:
            dp.schema_version = DATAPACK_SCHEMA_VERSION_PORTABLE

        out_repo.append(dp)
        exported += 1

    print("Portable export summary")
    print(f"  task: {args.task}")
    print(f"  source: {args.datapack_dir}")
    print(f"  output: {output_dir}")
    print(f"  exported: {exported}")
    print(f"  reused_existing: {reused_existing}")
    print(f"  skipped_missing_raw: {skipped_missing_raw}")
    print(f"  skipped_missing_inputs: {skipped_missing_inputs}")
    if repr_names:
        print(f"  repr_tokens_computed: {repr_tokens_computed}")
        print(f"  repr_names: {repr_names}")


if __name__ == "__main__":
    main()

