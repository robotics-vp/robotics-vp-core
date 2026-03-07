import json
import subprocess

from src.ontology.store import OntologyStore
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_shadow_control_plane_runner_smoke(tmp_path):
    output_dir = tmp_path / "shadow_runner"
    result = run_shadow_control_plane(
        output_dir=output_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert result.run_id == summary["run_id"]
    assert (output_dir / "objective_tensor.json").exists()
    assert (output_dir / "pricing_ticks.jsonl").exists()
    assert (output_dir / "value_ledger.jsonl").exists()
    assert len((output_dir / "value_ledger.jsonl").read_text().strip().splitlines()) == 2

    store = OntologyStore(root_dir=str(output_dir / "ontology"))
    assert len(store.list_episodes()) == 2
    assert len(store.list_datapacks()) == 2


def test_shadow_ablation_harness_smoke(tmp_path):
    output_dir = tmp_path / "shadow_ablations"
    subprocess.run(
        [
            "python3",
            "scripts/run_shadow_econ_ablations.py",
            "--output-dir",
            str(output_dir),
            "--episodes",
            "2",
            "--timestamp-base",
            "2026-01-01T00:00:00+00:00",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    comparison = json.loads((output_dir / "ablation_comparison.json").read_text())
    assert comparison["comparison"]["shadow_loop_runs"] is True
    assert comparison["comparison"]["pricing_traceable"] is True
    assert (output_dir / "ablation_comparison.md").exists()
