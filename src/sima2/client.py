"""
SIMA-2 client stub for deterministic semantic rollouts.

Advisory-only: produces canned trajectories that can feed the Stage 2 pipeline
without requiring a live agent or physics backend.
"""
from dataclasses import dataclass
from typing import Dict, List, Iterable, Iterator
import itertools


def _canned_trajectory(task_id: str, idx: int) -> Dict:
    # Deterministic canned primitives keyed by task_id
    base = {
        "episode_id": f"{task_id}_sim_ep{idx}",
        "task": task_id,
        "primitives": [],
        "metadata": {
            "objects_present": ["drawer", "vase_inside"] if "drawer" in task_id else ["dish", "cup"],
            "semantic_primitives": [],
        },
    }
    if "drawer" in task_id:
        prims = [
            {"timestep": 0, "object": "drawer", "action": "approach", "risk": "low"},
            {"timestep": 3, "object": "drawer", "action": "pull", "risk": "medium"},
            {"timestep": 6, "object": "vase_inside", "action": "avoid", "risk": "high"},
        ]
    else:
        prims = [
            {"timestep": 0, "object": "dish", "action": "reach", "risk": "low"},
            {"timestep": 4, "object": "dish", "action": "grasp", "risk": "medium"},
            {"timestep": 8, "object": "cup", "action": "place", "risk": "low"},
        ]
    base["primitives"] = prims
    base["metadata"]["semantic_primitives"] = prims
    base["actions"] = [{"timestep": p["timestep"], "action": p["action"]} for p in prims]
    base["observations"] = [{"timestep": p["timestep"], "state": {"object": p["object"]}} for p in prims]
    base["events"] = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    return base


@dataclass
class Sima2Client:
    """Deterministic SIMA-2 rollout generator (stub)."""

    task_id: str
    seed: int = 0

    def run_episode(self, task_spec: Dict) -> Dict:
        task_id = task_spec.get("task_id", self.task_id)
        idx = int(task_spec.get("episode_index", 0))
        return _canned_trajectory(task_id, idx)

    def stream_rollout(self, task_spec: Dict, count: int = 1) -> Iterator[Dict]:
        task_id = task_spec.get("task_id", self.task_id)
        for idx in itertools.islice(itertools.count(0), count):
            yield _canned_trajectory(task_id, idx)
