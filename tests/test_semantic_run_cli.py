"""Smoke test for semantic_run CLI."""
import subprocess
import sys
from pathlib import Path

from src.ontology.models import Task
from src.ontology.store import OntologyStore


def test_semantic_run_cli_synthetic(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    store = OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_logging",
            name="Logging Task",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=20.0,
            default_energy_cost_per_wh=0.1,
        )
    )

    cmd = [
        sys.executable,
        str(repo_root / "scripts/semantic_run.py"),
        "--ontology-root",
        str(store.root),
        "--intent",
        "test synthetic semantic run",
        "--tags",
        "warehouse",
        "logging",
        "--robot-family",
        "G1",
        "--task-id",
        "task_logging",
        "--motor-backend",
        "synthetic",
        "--num-envs",
        "1",
        "--max-steps",
        "5",
        "--eval-episodes",
        "0",
        "--run-log",
        str(tmp_path / "runs.jsonl"),
        "--datapack-output-dir",
        str(tmp_path / "datapacks"),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "[semantic_run] status=completed" in result.stdout
    assert "[semantic_run] scenario_id=" in result.stdout
    assert "mpl_units_per_hour" in result.stdout
