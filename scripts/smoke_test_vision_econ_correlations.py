#!/usr/bin/env python3
"""
Smoke test for vision/econ correlation reporting.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore  # noqa: E402
from src.ontology.models import Episode, EconVector  # noqa: E402
from src.analytics.vision_econ import summarize_vision_econ_correlations  # noqa: E402


def _populate(root: Path):
    store = OntologyStore(root_dir=str(root))
    ep1 = Episode(
        episode_id="ep_vis_1",
        task_id="task_vis",
        robot_id="robot",
        status="success",
        started_at=datetime.utcnow(),
        vision_config={"backend": "pybullet", "camera_intrinsics": {"fov_deg": 80.0}},
        vision_conditions={"lighting_tag": "bright", "occlusion_tag": "none"},
    )
    ep2 = Episode(
        episode_id="ep_vis_2",
        task_id="task_vis",
        robot_id="robot",
        status="success",
        started_at=datetime.utcnow(),
        vision_config={"backend": "isaac_stub", "camera_intrinsics": {"fov_deg": 120.0}},
        vision_conditions={"lighting_tag": "dim", "occlusion_tag": "partial"},
    )
    store.upsert_episode(ep1)
    store.upsert_episode(ep2)
    store.upsert_econ_vector(EconVector(episode_id="ep_vis_1", mpl_units_per_hour=80, wage_parity=1.0, energy_cost=0.5, damage_cost=0.1, novelty_delta=0.0, reward_scalar_sum=5.0))
    store.upsert_econ_vector(EconVector(episode_id="ep_vis_2", mpl_units_per_hour=60, wage_parity=0.9, energy_cost=0.7, damage_cost=0.1, novelty_delta=0.0, reward_scalar_sum=3.0))
    return store


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = _populate(root)
        summary = summarize_vision_econ_correlations(store.list_episodes(), store.list_econ_vectors())
        assert "pybullet" in summary["by_backend"]
        assert "isaac_stub" in summary["by_backend"]
        assert summary["by_fov_bucket"].get("wide")
        out_json = root / "correlations.json"
        out_csv = root / "correlations.csv"
        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "report_vision_econ_correlations.py"),
                "--ontology-root",
                str(root),
                "--task-id",
                "task_vis",
                "--output-json",
                str(out_json),
                "--output-csv",
                str(out_csv),
            ],
            check=True,
        )
        assert out_json.exists() and out_csv.exists()
        data = json.loads(out_json.read_text())
        assert data.get("by_backend", {}).get("pybullet")
        print("[smoke_test_vision_econ_correlations] PASS")


if __name__ == "__main__":
    main()
