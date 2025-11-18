#!/usr/bin/env python3
"""
Smoke test for reward-model + segmentation integration in econ reports.
"""
import json
import subprocess
import tempfile
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.models import Task, Episode, EconVector, Robot
from src.ontology.store import OntologyStore


def _build_fixture(root: str, rewards_path: Path, segments_path: Path, output_path: Path):
    store = OntologyStore(root_dir=root)
    task = Task(task_id="task_econ", name="Test Task", human_mpl_units_per_hour=1.0, human_wage_per_hour=20.0)
    robot = Robot(robot_id="r1", name="bot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    ep = Episode(episode_id="ep_rm", task_id=task.task_id, robot_id=robot.robot_id, status="success")
    store.upsert_episode(ep)
    econ = EconVector(
        episode_id=ep.episode_id,
        mpl_units_per_hour=1.5,
        wage_parity=1.2,
        energy_cost=0.2,
        damage_cost=0.05,
        novelty_delta=0.0,
        reward_scalar_sum=10.0,
        components={"mpl_component": 1.5},
        metadata={"target_mpl_units_per_hour": 1.0},
    )
    store.upsert_econ_vector(econ)

    # Reward model score
    with rewards_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "episode_id": ep.episode_id,
                    "progress_estimate": 0.9,
                    "quality_score": 0.8,
                    "error_probability": 0.1,
                    "subtask_labels": ["drawer", "pull"],
                },
                sort_keys=True,
            )
        )
        f.write("\n")

    # Segmentation tags
    boundaries = [
        {"episode_id": ep.episode_id, "segment_id": "seg0", "timestep": 0, "reason": "start"},
        {"episode_id": ep.episode_id, "segment_id": "seg0", "timestep": 4, "reason": "end"},
        {"episode_id": ep.episode_id, "segment_id": "seg1", "timestep": 5, "reason": "start"},
        {"episode_id": ep.episode_id, "segment_id": "seg1", "timestep": 8, "reason": "recovery"},
        {"episode_id": ep.episode_id, "segment_id": "seg1", "timestep": 9, "reason": "end"},
    ]
    with segments_path.open("w") as f:
        f.write(json.dumps({"episode_id": ep.episode_id, "segment_boundary_tags": boundaries}, sort_keys=True))
        f.write("\n")

    cmd = [
        "python3",
        str(repo_root / "scripts" / "report_task_pricing_and_performance.py"),
        "--ontology-root",
        root,
        "--task-id",
        task.task_id,
        "--reward-model-scores",
        str(rewards_path),
        "--segmentation-tags",
        str(segments_path),
        "--output-json",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        rewards_path = Path(tmpdir) / "rm.jsonl"
        segments_path = Path(tmpdir) / "seg.jsonl"
        output_path = Path(tmpdir) / "report.json"
        _build_fixture(tmpdir, rewards_path, segments_path, output_path)
        data = json.loads(output_path.read_text())
        task_summary = data["task_summary"]
        assert "quality_adjusted_mpl" in task_summary
        assert "quality_grades" in task_summary
        recovery = task_summary.get("recovery_segments", {})
        assert recovery.get("fraction_with_recovery", 0) > 0
        assert data["reward_model_scores"]
        print("[smoke_test_reward_model_econ_integration] PASS")


if __name__ == "__main__":
    main()
