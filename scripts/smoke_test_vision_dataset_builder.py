#!/usr/bin/env python3
"""
Smoke test for vision dataset builder.
"""
import tempfile
from pathlib import Path
import sys
from datetime import datetime

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.ontology.store import OntologyStore
from src.ontology.models import Task, Robot, Episode, EpisodeEvent
from src.vision.dataset_builder import build_frame_dataset_from_ontology


def _populate_store(root: Path):
    store = OntologyStore(root_dir=str(root))
    task = Task(task_id="task_vis", name="Vision Task")
    robot = Robot(robot_id="r1", name="robo")
    ep = Episode(episode_id="ep_vis", task_id=task.task_id, robot_id=robot.robot_id, status="success")
    store.upsert_task(task)
    store.upsert_robot(robot)
    store.upsert_episode(ep)
    events = []
    for t in range(3):
        events.append(
            EpisodeEvent(
                episode_id=ep.episode_id,
                timestep=t,
                event_type="step",
                timestamp=datetime.utcnow(),
                reward_scalar=1.0,
                reward_components={"mpl_component": 0.1 * (t + 1)},
                state_summary={"step": t},
                metadata={},
            )
        )
    store.append_events(events)
    return store


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _populate_store(root)
        out_dir = root / "dataset"
        stats = build_frame_dataset_from_ontology(
            ontology_root=str(root),
            task_id="task_vis",
            output_dir=str(out_dir),
            max_frames=5,
            stride=1,
        )
        meta_path = out_dir / "metadata.jsonl"
        assert meta_path.exists()
        lines = meta_path.read_text().splitlines()
        assert len(lines) == stats.get("frames", 0) == 3
        first = lines[0]
        assert "frame_path" in first and "latent" in first
        print("[smoke_test_vision_dataset_builder] PASS")


if __name__ == "__main__":
    main()
