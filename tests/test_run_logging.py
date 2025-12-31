"""Tests for semantic run logging."""
import json

from src.orchestrator.semantic_simulation import _append_run_log, get_recent_runs


def test_append_run_log(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    payload = {"status": "completed", "scenario_id": "scenario_1"}
    _append_run_log(log_path, payload)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["status"] == "completed"
    assert stored["scenario_id"] == "scenario_1"


def test_get_recent_runs_filters(tmp_path):
    log_path = tmp_path / "runs.jsonl"
    _append_run_log(log_path, {"status": "completed", "scenario_id": "s1", "motor_backend": "synthetic"})
    _append_run_log(log_path, {"status": "failed", "scenario_id": "s2", "motor_backend": "holosoma"})

    filtered = get_recent_runs(log_path, status="completed", backend="synthetic")
    assert len(filtered) == 1
    assert filtered[0]["scenario_id"] == "s1"
