#!/usr/bin/env python3
"""Export Stage-1 governed-video admissions to replay, RLDS, and LeRobot rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.dataset_bridges.lerobot_bridge import lerobot_rows_from_replay
from src.dataset_bridges.rlds_bridge import rlds_episode_from_replay
from src.replay.dataset import ReplayDatasetBuilder, ReplayDatasetBundle


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _steps_by_episode(bundle: ReplayDatasetBundle) -> Dict[str, list[Any]]:
    grouped: Dict[str, list[Any]] = {}
    for step in bundle.steps:
        grouped.setdefault(step.episode_id, []).append(step)
    return grouped


def export_governed_video_stage1_bridges(
    *,
    admission_log_path: str | Path,
    output_dir: str | Path,
    replay_dir: str | Path | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """Build canonical replay plus lossy public bridge exports from Stage-1 admissions."""

    output_root = Path(output_dir)
    replay_root = (
        Path(replay_dir) if replay_dir is not None else output_root / "replay_dataset"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = (
        ReplayDatasetBuilder()
        .add_governed_video_admission_log(admission_log_path, run_id=run_id)
        .write(replay_root)
    )
    steps_by_episode = _steps_by_episode(bundle)
    rlds_rows = [
        rlds_episode_from_replay(episode, steps_by_episode.get(episode.episode_id, []))
        for episode in bundle.episodes
    ]
    lerobot_rows = [
        row
        for episode in bundle.episodes
        for row in lerobot_rows_from_replay(
            episode, steps_by_episode.get(episode.episode_id, [])
        )
    ]
    rlds_path = output_root / "rlds_episodes.jsonl"
    lerobot_path = output_root / "lerobot_rows.jsonl"
    manifest_path = output_root / "bridge_manifest.json"
    _write_jsonl(rlds_path, rlds_rows)
    _write_jsonl(lerobot_path, lerobot_rows)
    manifest = {
        "version": "governed_video_stage1_bridge_export_v1",
        "admission_log_path": str(admission_log_path),
        "replay_dataset_dir": str(replay_root),
        "rlds_episodes_path": str(rlds_path),
        "lerobot_rows_path": str(lerobot_path),
        "num_episodes": bundle.manifest.num_episodes,
        "num_steps": bundle.manifest.num_steps,
        "source_adapters": list(bundle.manifest.source_adapters),
        "dataset_digest": bundle.manifest.dataset_digest,
        "internal_sidecar_policy": "preserve_refs_lossy_public_rows",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Stage-1 governed-video admission logs to replay, RLDS, and LeRobot rows."
    )
    parser.add_argument(
        "--admission-log",
        required=True,
        help="Path to governed_video/proposal_admission_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for bridge exports"
    )
    parser.add_argument(
        "--replay-dir", default=None, help="Optional replay dataset output directory"
    )
    parser.add_argument("--run-id", default=None, help="Optional replay import run id")
    args = parser.parse_args()
    manifest = export_governed_video_stage1_bridges(
        admission_log_path=args.admission_log,
        output_dir=args.output_dir,
        replay_dir=args.replay_dir,
        run_id=args.run_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
