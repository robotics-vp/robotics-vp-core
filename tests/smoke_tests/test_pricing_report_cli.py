"""Smoke test for pricing report CLI with dummy backend."""
import subprocess
import sys
from pathlib import Path


def test_pricing_report_cli_dummy():
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo_root / "scripts/report_task_pricing_and_performance.py"),
        "--ontology-root",
        str(repo_root / "data/ontology/test_econ_reports"),
        "--task-id",
        "task_econ",
        "--motor-backend",
        "dummy",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "[pricing_report]" in result.stdout
    assert "[datapack_mix]" in result.stdout
    assert "[pricing_snapshot]" in result.stdout
