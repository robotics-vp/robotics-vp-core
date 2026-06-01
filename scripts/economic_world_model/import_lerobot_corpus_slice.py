#!/usr/bin/env python3
"""Download/import a tiny LeRobot corpus slice into repo-native artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.economic_world_model import import_lerobot_corpus_slice  # noqa: E402


def run_import_lerobot_corpus_slice(
    *,
    repo_id: str,
    output_dir: str | Path,
    source_root: Optional[str | Path] = None,
    max_episodes: int = 2,
    max_steps_per_episode: int = 200,
    include_videos: bool = False,
    max_video_files: int = 1,
    max_video_bytes: int = 25_000_000,
) -> dict[str, Any]:
    return import_lerobot_corpus_slice(
        repo_id=repo_id,
        output_dir=output_dir,
        source_root=source_root,
        download=source_root is None,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
        include_videos=include_videos,
        max_video_files=max_video_files,
        max_video_bytes=max_video_bytes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a LeRobot-format Parquet corpus slice"
    )
    parser.add_argument("--repo-id", default="lerobot/pusht_keypoints")
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/external_lerobot_import",
    )
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--max-video-files", type=int, default=1)
    parser.add_argument("--max-video-bytes", type=int, default=25_000_000)
    args = parser.parse_args()
    payload = run_import_lerobot_corpus_slice(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        source_root=args.source_root,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        include_videos=args.include_videos,
        max_video_files=args.max_video_files,
        max_video_bytes=args.max_video_bytes,
    )
    summary_keys = [
        "status",
        "dataset_id",
        "download_executed",
        "files_downloaded_count",
        "video_files_downloaded_count",
        "source_total_bytes",
        "video_total_bytes",
        "selected_episode_count",
        "selected_step_count",
        "replay_episode_count",
        "replay_step_count",
        "quality_receipt_count",
        "quality_passed_count",
        "label_gap_count",
        "governance_label_count",
        "ingestion_row_count",
        "ready_for_shadow_eval",
        "ready_for_training",
        "provider_executed",
        "gpu_training_executed",
        "unitree_hardware_truth",
        "promotion_eligible",
        "phase7_authority_granted",
        "image_video_modalities_imported",
    ]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
