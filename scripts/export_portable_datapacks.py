#!/usr/bin/env python3
"""Export portable datapacks with embedded tracks/features/slice labels."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DATAPACK_SCHEMA_VERSION_PORTABLE
from src.valuation.portable_datapacks import (
    load_raw_episode_artifacts,
    compute_rgb_features_v1,
    compute_slice_labels_v1,
)


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
    args = parser.parse_args()

    repo = DataPackRepo(base_dir=args.datapack_dir)
    datapacks = repo.load_all(args.task)
    if not datapacks:
        raise RuntimeError(f"No datapacks found for task '{args.task}' in {args.datapack_dir}")

    output_dir = args.output_dir or f"{args.datapack_dir}_portable"
    out_repo = DataPackRepo(base_dir=output_dir)

    exported = 0
    skipped_missing_raw = 0
    skipped_missing_inputs = 0
    reused_existing = 0

    for dp in datapacks:
        if args.max_episodes is not None and exported >= args.max_episodes:
            break
        if dp.scene_tracks_v1 and dp.rgb_features_v1 and dp.slice_labels_v1:
            dp.schema_version = DATAPACK_SCHEMA_VERSION_PORTABLE
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


if __name__ == "__main__":
    main()
