#!/usr/bin/env python3
"""
Smoke test for RECAP dataset builder.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.vla.recap_dataset_builder import build_recap_dataset
from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot, Episode, EconVector, EpisodeEvent


def main():
    root = Path("data/ontology/test_recap")
    if root.exists():
        shutil.rmtree(root)
    store = OntologyStore(root_dir=str(root))
    task_id = "task_recap"
    store.upsert_task(Task(task_id=task_id, name="RecapTask", environment_id="env", human_mpl_units_per_hour=60.0, human_wage_per_hour=18.0, default_energy_cost_per_wh=0.12))
    store.upsert_robot(Robot(robot_id="robot_recap", name="RecapBot"))
    now = datetime.utcnow()
    ep = Episode(episode_id="ep_recap", task_id=task_id, robot_id="robot_recap", started_at=now, status="success", metadata={"sampling_metadata": {"strategy": "balanced"}, "objective_preset": "balanced"})
    store.upsert_episode(ep)
    store.upsert_econ_vector(EconVector(episode_id="ep_recap", mpl_units_per_hour=80, wage_parity=1.0, energy_cost=0.5, damage_cost=0.1, novelty_delta=0.1, reward_scalar_sum=5.0))
    events = [
        EpisodeEvent(episode_id="ep_recap", timestep=0, event_type="step", timestamp=now, reward_scalar=1.0, reward_components={"mpl": 1.0}),
        EpisodeEvent(episode_id="ep_recap", timestep=1, event_type="step", timestamp=now, reward_scalar=2.0, reward_components={"mpl": 2.0}),
    ]
    store.append_events(events)

    out = Path("results/recap/test_recap.jsonl")
    if out.exists():
        out.unlink()
    build_recap_dataset(store, task_id=task_id, output_path=str(out), max_episodes=10)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    import json as _json
    entry = _json.loads(lines[0])
    assert "advantage" in entry and "metrics" in entry
    # Determinism check
    out2 = Path("results/recap/test_recap_2.jsonl")
    build_recap_dataset(store, task_id=task_id, output_path=str(out2), max_episodes=10)
    assert out.read_text() == out2.read_text()
    print("[smoke_test_vla_recap_dataset] All tests passed.")


if __name__ == "__main__":
    main()
