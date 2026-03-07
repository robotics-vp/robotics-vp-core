import json
import subprocess


def test_shadow_advisory_pass_cli_smoke(tmp_path):
    output_dir = tmp_path / "advisory_pass"
    subprocess.run(
        [
            "python3",
            "scripts/run_shadow_advisory_pass.py",
            "--output-dir",
            str(output_dir),
            "--generate-shadow-run",
            "--seed",
            "42",
            "--episodes",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    advisory = json.loads((output_dir / "shadow_advisory.json").read_text())
    assert advisory["summary"]["episodes"] == 3
    assert (output_dir / "shadow_advisory.md").exists()


def test_shadow_learning_ablation_cli_smoke(tmp_path):
    output_dir = tmp_path / "shadow_learning_ablations"
    subprocess.run(
        [
            "python3",
            "scripts/run_shadow_learning_ablations.py",
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            "--episodes",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    comparison = json.loads((output_dir / "shadow_learning_ablation_comparison.json").read_text())
    assert comparison["comparison"]["replay_ingestion_runs"] is True
    assert comparison["comparison"]["bc_policy_runs"] is True
    assert (output_dir / "shadow_learning_ablation_comparison.md").exists()
