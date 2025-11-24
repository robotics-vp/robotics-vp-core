#!/usr/bin/env python3
"""
Smoke test for ROS bridge ingestion.

Validates:
- Deterministic episode IDs and rollout contents
- JSON-safe serialization
- Ingestion works on a tiny synthetic ROS-style log
"""
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.ros_bridge import ROSBridgeIngestor


def _make_synthetic_log(path: Path) -> None:
    log = {
        "topic_messages": {
            "/camera/rgb": [
                {
                    "stamp": 0.0,
                    "data": [
                        [[0, 0, 0], [255, 255, 255]],
                        [[128, 128, 128], [64, 64, 64]],
                    ],
                    "camera_intrinsics": {"resolution": [2, 2]},
                }
            ],
            "/camera/depth": [
                {
                    "stamp": 0.0,
                    "data": [
                        [1.0, 1.1],
                        [0.9, 1.0],
                    ],
                }
            ],
            "/joint_states": [
                {
                    "stamp": 0.0,
                    "position": [0.1, -0.1],
                    "velocity": [0.01, -0.02],
                    "effort": [1.0, 0.5],
                    "dt": 0.1,
                }
            ],
            "/tf": [
                {
                    "stamp": 0.0,
                    "transforms": [
                        {"child_frame_id": "base_link", "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
                    ],
                }
            ],
            "/action": [
                {
                    "stamp": 0.0,
                    "action": {"gripper": "close", "speed": 0.2},
                }
            ],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, sort_keys=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "synthetic_log.json"
        _make_synthetic_log(log_path)

        ingestor = ROSBridgeIngestor(output_root=str(Path(tmpdir) / "results"), run_timestamp=0)

        rollout_a = ingestor.ingest(str(log_path), task_id="synthetic_task")
        rollout_b = ingestor.ingest(str(log_path), task_id="synthetic_task")

        assert rollout_a.episode_id == rollout_b.episode_id, "Episode ID should be deterministic"
        assert rollout_a.to_dict() == rollout_b.to_dict(), "Rollout content should be deterministic"

        # JSON-safe serialization
        json.dumps(rollout_a.to_dict(), sort_keys=True)

        # Datapack written
        ts_label = time.strftime("%Y%m%dT%H%M%S", time.gmtime(0))
        datapack_path = Path(tmpdir) / "results" / ts_label / "datapacks.jsonl"
        assert datapack_path.exists(), "Datapack JSONL should be created"
        with open(datapack_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        assert lines, "Datapack JSONL should contain at least one entry"
        json.loads(lines[0])

        # Vision + proprio presence
        assert rollout_a.vision_frames, "Vision frames should be populated"
        assert rollout_a.proprio_frames, "Proprio frames should be populated"
        assert rollout_a.env_digests, "Env digests should be populated"

    print("[smoke_test_ros_ingestion] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
