#!/usr/bin/env python3
"""Run Scene IR tracker on datapack frames and emit SceneTracks_v1."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.vision.scene_ir_tracker.io.scene_tracks_runner import (
    SceneTracksQualityError,
    run_scene_tracks,
)


def main() -> int:
    if sys.version_info < (3, 8):
        print("This script requires python3.", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Produce SceneTracks_v1 from datapacks")
    parser.add_argument("--datapack", required=True, help="Path to datapack directory or trajectory.npz")
    parser.add_argument("--out", required=True, help="Output directory or .npz path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic sampling")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--camera", type=str, default="front", help="Camera name (top/front/wrist)")
    parser.add_argument("--mode", type=str, default="rgb", choices=["rgb", "rgbd", "vector_proxy"])
    parser.add_argument("--ontology-root", type=str, default="data/ontology", help="Ontology store root")
    parser.add_argument("--min-quality", type=float, default=0.2, help="Minimum quality threshold")
    parser.add_argument("--allow-low-quality", action="store_true", help="Allow low quality outputs")

    args = parser.parse_args()

    try:
        result = run_scene_tracks(
            datapack_path=args.datapack,
            output_path=args.out,
            seed=args.seed,
            max_frames=args.max_frames,
            camera=args.camera,
            mode=args.mode,
            ontology_root=args.ontology_root,
            min_quality=args.min_quality,
            allow_low_quality=args.allow_low_quality,
        )
    except SceneTracksQualityError as exc:
        print(f"[run_scene_tracks] quality gate failed: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[run_scene_tracks] failed: {exc}", file=sys.stderr)
        return 1

    print(f"SceneTracks saved: {result.scene_tracks_path}")
    print(f"Quality score: {result.quality.quality_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
