#!/usr/bin/env python3
"""
Smoke test: ROS → Stage 2 pipeline with anomaly injection.

Validates:
- Pipeline determinism for fixed log
- JSON-safe artifacts
- OOD/Recovery tags emitted when anomalies are injected
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ros_to_stage2_pipeline import run_pipeline
from src.utils.json_safe import to_json_safe


def _write_synthetic_log(path: Path) -> Path:
    log = {
        "messages": [
            {"topic": "/camera/rgb", "stamp": 0.0, "data": [[0, 0], [0, 0]]},
            {"topic": "/depth", "stamp": 0.0, "data": [[1.0, 1.0], [1.0, 1.0]]},
            {"topic": "/joint_states", "stamp": 0.0, "position": [0.05], "velocity": [0.01], "effort": [0.2], "contacts": [0.0]},
            {"topic": "/joint_states", "stamp": 1.0, "position": [0.02], "velocity": [0.2], "effort": [1.5], "contacts": [1.0]},
            {"topic": "/action", "stamp": 0.0, "action": {"grasp": 0.1}},
            {"topic": "/tf", "stamp": 0.0, "transforms": [{"child_frame_id": "base", "translation": [0, 0, 0], "rotation": [0, 0, 0, 1]}]},
        ]
    }
    with path.open("w") as f:
        json.dump(log, f)
    return path


def _to_dict_list(objs):
    payload = []
    for obj in objs:
        if hasattr(obj, "to_dict"):
            try:
                payload.append(obj.to_dict())
                continue
            except Exception:
                pass
        payload.append(obj)
    return payload


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = _write_synthetic_log(Path(tmpdir) / "ros_log.json")
        out_dir_a = Path(tmpdir) / "ros_stage2_a"
        out_dir_b = Path(tmpdir) / "ros_stage2_b"

        outputs_a = run_pipeline(log_path, out_dir_a, inject_anomaly=True)
        outputs_b = run_pipeline(log_path, out_dir_b, inject_anomaly=True)

        # Determinism
        assert to_json_safe(outputs_a["trust_matrix"]) == to_json_safe(outputs_b["trust_matrix"]), "Trust matrix must be deterministic"
        assert _to_dict_list(outputs_a["primitives"]) == _to_dict_list(outputs_b["primitives"]), "Primitives should be deterministic"

        # JSON safety
        json.dumps(to_json_safe(outputs_a["rollout"]))
        json.dumps(to_json_safe(_to_dict_list(outputs_a["semantic_tags"])))

        tags = _to_dict_list(outputs_a["semantic_tags"])
        ood_present = any((t.get("ood_tags") or t.get("enrichment", {}).get("ood_tags") or []) for t in tags)
        recovery_present = any((t.get("recovery_tags") or t.get("enrichment", {}).get("recovery_tags") or []) for t in tags)
        assert ood_present, "OOD tags should be emitted when anomaly injected"
        assert recovery_present, "Recovery tags should be emitted when anomaly injected"

    print("[smoke_test_ros_to_stage2_pipeline] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
