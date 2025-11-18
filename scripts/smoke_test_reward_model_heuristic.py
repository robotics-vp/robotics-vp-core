#!/usr/bin/env python3
"""
Smoke test for heuristic RewardModel policy.
"""
import json
import tempfile
from pathlib import Path

import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.models import Episode, EconVector, Robot, Task
from src.ontology.store import OntologyStore
from src.vla.recap_inference import RecapEpisodeScores
from scripts.score_episodes_with_reward_model import _load_jsonl_map, _load_recap_map, _score


def _build_fixture(tmpdir: str):
    store = OntologyStore(root_dir=tmpdir)
    task = Task(task_id="task_drawer", name="Drawer Task", human_mpl_units_per_hour=1.0, human_wage_per_hour=20.0)
    robot = Robot(robot_id="robot1", name="TestBot")
    store.upsert_task(task)
    store.upsert_robot(robot)

    ep1 = Episode(episode_id="ep_drawer_success", task_id=task.task_id, robot_id=robot.robot_id, status="success")
    ep2 = Episode(episode_id="ep_drawer_fail", task_id=task.task_id, robot_id=robot.robot_id, status="failure")
    store.upsert_episode(ep1)
    store.upsert_episode(ep2)

    econ1 = EconVector(
        episode_id=ep1.episode_id,
        mpl_units_per_hour=1.2,
        wage_parity=1.1,
        energy_cost=0.1,
        damage_cost=0.0,
        novelty_delta=0.0,
        reward_scalar_sum=10.0,
        components={"mpl_component": 1.2, "collision_penalty": 0.0},
        metadata={"target_mpl_units_per_hour": 1.0},
    )
    econ2 = EconVector(
        episode_id=ep2.episode_id,
        mpl_units_per_hour=0.5,
        wage_parity=0.6,
        energy_cost=0.3,
        damage_cost=0.2,
        novelty_delta=0.0,
        reward_scalar_sum=5.0,
        components={"mpl_component": 0.5, "collision_penalty": 0.1},
        metadata={"target_mpl_units_per_hour": 1.0},
    )
    store.upsert_econ_vector(econ1)
    store.upsert_econ_vector(econ2)

    tags_path = Path(tmpdir) / "tags.jsonl"
    tags = [
        {"episode_id": ep1.episode_id, "objects_present": ["drawer", "vase_inside"], "semantic_tags": ["grasp_handle"]},
        {"episode_id": ep2.episode_id, "objects_present": ["drawer"], "semantic_tags": ["recover"], "risk_tags": ["recover"]},
    ]
    with tags_path.open("w") as f:
        for t in tags:
            f.write(json.dumps(t, sort_keys=True))
            f.write("\n")

    recap_path = Path(tmpdir) / "recap.jsonl"
    recap_records = [
        RecapEpisodeScores(
            episode_id=ep1.episode_id,
            advantage_bin_probs_mean=[0.1, 0.2],
            advantage_bin_probs_max=[0.2, 0.3],
            metric_distributions={"error_rate": [0.05, 0.1]},
            recap_goodness_score=0.8,
            num_events=5,
        ).to_dict(),
        RecapEpisodeScores(
            episode_id=ep2.episode_id,
            advantage_bin_probs_mean=[0.2, 0.3],
            advantage_bin_probs_max=[0.3, 0.4],
            metric_distributions={"error_rate": [0.2, 0.3]},
            recap_goodness_score=0.2,
            num_events=5,
        ).to_dict(),
    ]
    with recap_path.open("w") as f:
        for rec in recap_records:
            f.write(json.dumps(rec, sort_keys=True))
            f.write("\n")

    return store, str(tags_path), str(recap_path)


def _assert_score_bounds(score_dict):
    assert 0.0 <= score_dict["progress_estimate"] <= 1.0
    assert 0.0 <= score_dict["quality_score"] <= 1.0
    assert 0.0 <= score_dict["error_probability"] <= 1.0


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        store, tags_path, recap_path = _build_fixture(tmpdir)
        tags_map = _load_jsonl_map(tags_path)
        recap_map = _load_recap_map(recap_path)
        scores1 = _score(store, tags_map, recap_map)
        scores2 = _score(store, tags_map, recap_map)

        assert set(scores1.keys()) == set(scores2.keys())
        for ep_id, score in scores1.items():
            sdict = score.to_dict()
            _assert_score_bounds(sdict)
            assert sdict["subtask_labels"], "Expected subtask labels for drawer task"
            assert sdict == scores2[ep_id].to_dict(), "Scores should be deterministic across runs"
        print("[smoke_test_reward_model_heuristic] PASS")


if __name__ == "__main__":
    main()
