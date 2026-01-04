#!/usr/bin/env python3
"""CLI runner for Map-First pseudo-supervision (use python3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.vision.map_first_supervision.config import MapFirstSupervisionConfig
from src.vision.map_first_supervision.node import MapFirstPseudoSupervisionNode


def _load_npz_scene_tracks(path: Path) -> Dict[str, np.ndarray]:
    data = dict(np.load(path, allow_pickle=False))
    return data


def _load_json_scene_tracks(payload: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    raw = payload.get("scene_tracks_v1")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {k: np.asarray(v) for k, v in raw.items()}
    return None


def _load_dataset_shard(shard_path: Path) -> List[Dict[str, Any]]:
    with shard_path.open("r") as f:
        return json.load(f)


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _run_on_npz_files(
    npz_paths: List[Path],
    output_dir: Path,
    config: MapFirstSupervisionConfig,
) -> List[Dict[str, Any]]:
    node = MapFirstPseudoSupervisionNode(config=config)
    summaries: List[Dict[str, Any]] = []
    for path in npz_paths:
        data = _load_npz_scene_tracks(path)
        out_path = output_dir / f"{path.stem}_map_first_v1.npz"
        result = node.run(data, episode_assets=None, output_path=str(out_path))
        summaries.append({"episode_id": path.stem, **result.summary.to_dict()})
    return summaries


def _run_on_dataset_shard(
    shard_path: Path,
    output_dir: Path,
    config: MapFirstSupervisionConfig,
    write_shard: Optional[Path],
) -> List[Dict[str, Any]]:
    node = MapFirstPseudoSupervisionNode(config=config)
    episodes = _load_dataset_shard(shard_path)
    summaries: List[Dict[str, Any]] = []

    for idx, episode in enumerate(episodes):
        scene_tracks = _load_json_scene_tracks(episode)
        if scene_tracks is None:
            scene_tracks_path = episode.get("scene_tracks_path") or episode.get("scene_tracks_npz")
            if scene_tracks_path:
                scene_tracks = _load_npz_scene_tracks(Path(scene_tracks_path))
        if scene_tracks is None:
            continue

        episode_id = episode.get("episode_id") or episode.get("scene_id") or f"episode_{idx:04d}"
        out_path = output_dir / f"{episode_id}_map_first_v1.npz"
        result = node.run(scene_tracks, episode_assets=episode, output_path=str(out_path))
        summaries.append({"episode_id": episode_id, **result.summary.to_dict()})

        if write_shard is not None:
            episode["map_first_summary"] = result.summary.to_dict()
            episode["map_first_quality_score"] = result.summary.map_first_quality_score

    if write_shard is not None:
        with write_shard.open("w") as f:
            json.dump(episodes, f)

    return summaries


def main() -> int:
    if sys.version_info < (3, 8):
        print("This CLI requires python3. Try: python3 -m src.vision.map_first_supervision.cli_run_map_first_supervision", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="Run Map-First pseudo-supervision")
    parser.add_argument("--input-npz", action="append", help="SceneTracks_v1 npz file (repeatable)")
    parser.add_argument("--input-dir", type=str, help="Directory containing npz files")
    parser.add_argument("--dataset-shard", type=str, help="LSD dataset shard JSON")
    parser.add_argument("--output-dir", type=str, default="map_first_outputs", help="Output directory for artifacts")
    parser.add_argument("--write-shard", type=str, help="Optional path to write updated shard JSON")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    config = MapFirstSupervisionConfig()

    summaries: List[Dict[str, Any]] = []

    if args.input_dir:
        npz_paths = list(Path(args.input_dir).glob("**/*.npz"))
        summaries.extend(_run_on_npz_files(npz_paths, output_dir, config))

    if args.input_npz:
        npz_paths = [Path(p) for p in args.input_npz]
        summaries.extend(_run_on_npz_files(npz_paths, output_dir, config))

    if args.dataset_shard:
        write_shard = Path(args.write_shard) if args.write_shard else None
        summaries.extend(_run_on_dataset_shard(Path(args.dataset_shard), output_dir, config, write_shard))

    if not summaries:
        print("No inputs processed", file=sys.stderr)
        return 1

    summary_path = output_dir / "map_first_summary.jsonl"
    with summary_path.open("w") as f:
        for entry in summaries:
            f.write(json.dumps(entry) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
