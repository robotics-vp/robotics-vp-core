"""Smoke test for the golden path contract demo."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.golden_path
def test_run_golden_path_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "golden_path"

    cmd = [
        sys.executable,
        str(repo_root / "scripts/run_golden_path.py"),
        "--env",
        "workcell",
        "--episodes",
        "4",
        "--seed",
        "7",
        "--emit",
        str(output_dir),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Golden Path Complete" in result.stdout

    required_files = [
        output_dir / "objective_tensors.jsonl",
        output_dir / "scalar_rewards.json",
        output_dir / "econ_tensors.json",
        output_dir / "econ_deltas.json",
        output_dir / "governance_report.json",
        output_dir / "episodes.json",
        output_dir / "summary.json",
        output_dir / "artifact_bundle.json",
        output_dir / "plots" / "objective_scalar.png",
        output_dir / "plots" / "econ_governance.png",
    ]
    for path in required_files:
        assert path.exists(), f"missing artifact: {path}"

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["env"] == "workcell"
    assert summary["episodes"] == 4
    assert "mean_scalar_reward" in summary

    governance = json.loads((output_dir / "governance_report.json").read_text())
    assert len(governance) == 4
    assert all("reports" in episode for episode in governance)


@pytest.mark.golden_path
def test_run_golden_path_cli_fail_on_governance_flag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "golden_path_gate"

    cmd = [
        sys.executable,
        str(repo_root / "scripts/run_golden_path.py"),
        "--env",
        "workcell",
        "--episodes",
        "4",
        "--seed",
        "7",
        "--emit",
        str(output_dir),
        "--fail-on-governance-failure",
        "--regal-anomaly-episode",
        "1",
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Governance failure gate enabled" in result.stderr
