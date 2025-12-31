"""Tests for semantic run logging."""
import json

from src.orchestrator.semantic_simulation import _append_run_log


def test_append_run_log(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    payload = {"status": "completed", "scenario_id": "scenario_1"}
    _append_run_log(log_path, payload)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["status"] == "completed"
    assert stored["scenario_id"] == "scenario_1"
