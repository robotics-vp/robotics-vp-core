#!/usr/bin/env python3
"""Compile local LeRobot video receipt -> replay -> perception receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.dataset_bridges.lerobot_video_receipt_adapter import (  # noqa: E402
    build_fixture_lerobot_video_receipts,
    write_lerobot_video_receipt_bridge_artifacts,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def run_compile_lerobot_video_replay_perception_receipts(
    *,
    output_dir: str | Path,
    receipt_jsonl: str | Path | None = None,
    dataset_id: str = "fixture/lerobot_video_receipts",
) -> dict[str, Any]:
    receipts = (
        _load_jsonl(Path(receipt_jsonl))
        if receipt_jsonl is not None
        else build_fixture_lerobot_video_receipts()
    )
    return write_lerobot_video_receipt_bridge_artifacts(
        receipts,
        output_dir,
        dataset_id=dataset_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile local LeRobot video receipt replay/perception receipts"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/lerobot_video_replay_perception_receipts",
    )
    parser.add_argument("--receipt-jsonl", default=None)
    parser.add_argument("--dataset-id", default="fixture/lerobot_video_receipts")
    args = parser.parse_args()
    payload = run_compile_lerobot_video_replay_perception_receipts(
        output_dir=args.output_dir,
        receipt_jsonl=args.receipt_jsonl,
        dataset_id=args.dataset_id,
    )
    summary_keys = [
        "status",
        "dataset_id",
        "video_receipt_count",
        "replay_episode_count",
        "replay_step_count",
        "camera_key_count",
        "evidence_fusion_sample_count",
        "vjepa_temporal_sample_count",
        "vision_backbone_projection_sample_count",
        "provider_executed",
        "gpu_training_executed",
        "video_decoding_executed",
        "weights_downloaded",
        "unitree_hardware_truth",
        "promotion_eligible",
        "phase7_authority_granted",
    ]
    print(json.dumps({key: payload[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
