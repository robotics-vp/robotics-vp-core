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
    assert summary["runtime_corpus_summary"]["row_count"] >= 1
    assert Path(summary["runtime_corpus_paths"]["rows_path"]).exists()
    assert Path(summary["coverage_artifact_paths"]["coverage_graph"]).exists()
    metadata_path = Path(summary["episodes"][0]["episode_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if summary["episodes"][0]["backend_selected"] == "passthrough":
        assert metadata["scene_tracks_non_stub"] is False
