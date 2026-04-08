from __future__ import annotations

import json
from pathlib import Path


def test_bootstrap_semantic_workcell_loop_runs(tmp_path: Path) -> None:
    from scripts.bootstrap_semantic_workcell_loop import _mujoco_available, run_semantic_workcell_bootstrap

    summary = run_semantic_workcell_bootstrap(
        output_root=tmp_path / "semantic_bootstrap",
        episodes=1,
        steps=3,
        max_frames=3,
        seed=13,
        camera="front",
        grounding_mode="real_rgb" if _mujoco_available() else "vector_proxy",
        backend_policy="auto",
        sim_limit=3,
        diffusion_limit=3,
    )

    assert summary["episodes"]
    assert summary["episodes"][0]["scene_tracks_quality"] >= 0.0
    assert Path(summary["episodes"][0]["semantic_world_model_path"]).exists()
    assert summary["episodes"][0]["runtime_packet_ref"].endswith("_runtime_packet_v1.json")
    assert summary["episodes"][0]["event_spine_ref"].endswith("_event_spine_v1.json")
    assert summary["episodes"][0]["decision_ledger_ref"].endswith("_decision_ledger_v1.json")
    assert Path(summary["episodes"][0]["control_plane_context_path"]).exists()
    assert summary["runtime_corpus_summary"]["row_count"] >= 1
    assert summary["runtime_corpus_summary"]["bounded_ready_count"] >= 1
    assert Path(summary["runtime_corpus_paths"]["rows_path"]).exists()
    assert Path(summary["coverage_artifact_paths"]["coverage_graph"]).exists()
    assert summary["coverage_summary"]["covered_edges"] > 0
    assert summary["trace_artifact_summary"]["ready_episode_count"] >= 1
    metadata_path = Path(summary["episodes"][0]["episode_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert Path(metadata["control_plane_context_path"]).exists()
    if summary["episodes"][0]["backend_selected"] == "passthrough":
        assert metadata["scene_tracks_non_stub"] is False
        assert summary["episodes"][0]["grounded_data_ready"] is False
