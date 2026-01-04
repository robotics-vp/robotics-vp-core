"""Smoke tests for pricing report CLI with dummy or mocked backends."""
import subprocess
import sys
from pathlib import Path


def test_pricing_report_cli_dummy_with_configs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    objective_path = tmp_path / "objective.yaml"
    objective_path.write_text(
        "mpl_weight: 1.0\n"
        "energy_weight: 0.2\n"
        "error_weight: 0.5\n"
        "novelty_weight: 0.0\n"
        "risk_weight: 0.1\n"
    )
    datapack_path = tmp_path / "datapack.yaml"
    datapack_path.write_text(
        "id: dp_dummy\n"
        "description: test datapack\n"
        "motion_clips:\n"
        "  - path: data/mocap/dummy_clip.npz\n"
        "    weight: 1.0\n"
        "domain_randomization:\n"
        "  terrain: flat\n"
        "curriculum:\n"
        "  initial_difficulty: 0.1\n"
    )

    cmd = [
        sys.executable,
        str(repo_root / "scripts/report_task_pricing_and_performance.py"),
        "--ontology-root",
        str(repo_root / "data/ontology/test_econ_reports"),
        "--task-id",
        "task_econ",
        "--motor-backend",
        "dummy",
        "--objective-config",
        str(objective_path),
        "--datapacks",
        str(datapack_path),
    ]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "[pricing_report]" in result.stdout
    assert "[datapack_mix]" in result.stdout
    assert "[pricing_snapshot]" in result.stdout


def test_pricing_report_cli_holosoma_mock(monkeypatch, tmp_path, capsys):
    from scripts import report_task_pricing_and_performance as report
    from src.motor_backend.base import MotorEvalResult, MotorTrainingResult
    from src.ontology.models import Task

    objective_path = tmp_path / "objective.yaml"
    objective_path.write_text("mpl_weight: 1.0\n")
    datapack_path = tmp_path / "datapack.yaml"
    datapack_path.write_text("id: dp_mock\n")

    store = report.OntologyStore(root_dir=tmp_path / "ontology")
    store.upsert_task(
        Task(
            task_id="task_mock",
            name="MockTask",
            human_mpl_units_per_hour=60.0,
            human_wage_per_hour=18.0,
            default_energy_cost_per_wh=0.12,
        )
    )

    class DummyBackend:
        def train_policy(self, **kwargs):
            return MotorTrainingResult(
                policy_id="policy.onnx",
                raw_metrics={"mean_reward": 5.0, "mean_episode_length_s": 10.0},
                econ_metrics={"mpl_units_per_hour": 360.0, "wage_parity": 1.0, "reward_scalar_sum": 5.0},
            )

        def evaluate_policy(self, **kwargs):
            return MotorEvalResult(
                policy_id="policy.onnx",
                raw_metrics={"mean_reward": 6.0, "mean_episode_length_s": 9.0},
                econ_metrics={"mpl_units_per_hour": 400.0, "wage_parity": 1.1, "reward_scalar_sum": 6.0},
            )

    monkeypatch.setattr(report, "make_motor_backend", lambda *_args, **_kwargs: DummyBackend())

    argv = [
        "report_task_pricing_and_performance.py",
        "--ontology-root",
        str(store.root),
        "--task-id",
        "task_mock",
        "--motor-backend",
        "holosoma",
        "--objective-config",
        str(objective_path),
        "--datapacks",
        str(datapack_path),
        "--eval-episodes",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    report.main()

    captured = capsys.readouterr()
    assert "[pricing_report]" in captured.out
    assert "[datapack_mix]" in captured.out
    assert "[pricing_snapshot]" in captured.out
