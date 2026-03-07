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
    runtime_packets = json.loads((output_dir / "runtime_packets.json").read_text())
    event_spine = json.loads((output_dir / "event_spine.json").read_text())
    decision_ledger = json.loads((output_dir / "decision_ledger.json").read_text())
    assert result.run_id == summary["run_id"]
    assert (output_dir / "objective_tensor.json").exists()
    assert (output_dir / "runtime_packets.json").exists()
    assert (output_dir / "event_spine.json").exists()
    assert (output_dir / "decision_ledger.json").exists()
    assert (output_dir / "pricing_ticks.jsonl").exists()
    assert (output_dir / "value_ledger.jsonl").exists()
    assert len((output_dir / "value_ledger.jsonl").read_text().strip().splitlines()) == 2
    assert summary["artifact_paths"]["runtime_packets"].endswith("runtime_packets.json")
    assert summary["artifact_paths"]["event_spine"].endswith("event_spine.json")
    assert summary["artifact_paths"]["decision_ledger"].endswith("decision_ledger.json")
    assert runtime_packets["packet_count"] == 2
    assert event_spine["event_count"] >= 10
    assert decision_ledger["decision_count"] >= 8
    assert "pricing_tick_published" in {row["event_kind"] for row in event_spine["events"]}
    assert "datapack_credit_assigned" in {row["decision_kind"] for row in decision_ledger["decisions"]}
    assert runtime_packets["episodes"][0]["runtime_packet"]["contract"]["task_id"] == "shadow_kitting"
    assert result.episode_artifacts[0]["runtime_packet"]["contract"]["embodiment_id"] == "shadow_sim_arm_v1"
    assert result.episode_artifacts[0]["event_refs"]
    assert result.episode_artifacts[0]["decision_refs"]

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
