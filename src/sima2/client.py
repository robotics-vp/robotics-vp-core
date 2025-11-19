"""
SIMA-2 client stub for deterministic semantic rollouts.

Advisory-only: produces canned trajectories that can feed the Stage 2 pipeline
without requiring a live agent or physics backend.
"""
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterable, Iterator, Optional
import itertools
import json

from src.sima2.config import build_provenance, load_sima2_config


def _canned_trajectory(task_id: str, idx: int, provenance: Dict[str, Any], task_spec: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic canned primitives keyed by task_id
    base = {
        "episode_id": f"{task_id}_sim_ep{idx}",
        "task": task_id,
        "task_spec": json.loads(json.dumps(task_spec or {})),
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
    base["metadata"].update(provenance)
    base.update(provenance)
    base["source"] = provenance.get("sima2_backend_id", "sima2")
    return base


@dataclass
class Sima2Client:
    """Deterministic SIMA-2 rollout generator (stub)."""

    task_id: str
    seed: int = 0
    config_path: Optional[str] = None
    config: Dict[str, Any] = field(init=False)
    backend_mode: str = field(init=False)
    backend_id: str = field(init=False)
    model_version: str = field(init=False)

    def __post_init__(self):
        self.config = load_sima2_config(self.config_path)
        backend_cfg = self.config.get("backend", {}) or {}
        self.backend_mode = str(self.config.get("backend_mode", backend_cfg.get("mode", "stub")))
        self.backend_id = str(backend_cfg.get("id", f"{self.backend_mode}_backend"))
        self.model_version = str(backend_cfg.get("model_version", "sima2_unknown"))

    def _provenance(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        spec = dict(task_spec or {})
        spec.setdefault("task_id", self.task_id)
        prov = build_provenance(spec, self.config)
        prov.setdefault("sima2_backend_id", self.backend_id)
        prov.setdefault("sima2_model_version", self.model_version)
        prov.setdefault("sima2_backend_mode", self.backend_mode)
        return prov

    def run_episode(self, task_spec: Dict) -> Dict:
        task_spec = dict(task_spec or {})
        task_id = task_spec.get("task_id", self.task_id)
        idx = int(task_spec.get("episode_index", 0))
        task_spec["task_id"] = task_id
        task_spec["episode_index"] = idx
        provenance = self._provenance(task_spec)
        return _canned_trajectory(task_id, idx, provenance, task_spec)

    def stream_rollout(self, task_spec: Dict, count: int = 1) -> Iterator[Dict]:
        task_id = task_spec.get("task_id", self.task_id)
        for idx in itertools.islice(itertools.count(0), count):
            spec = dict(task_spec or {})
            spec["task_id"] = task_id
            spec["episode_index"] = idx
            provenance = self._provenance(spec)
            yield _canned_trajectory(task_id, idx, provenance, spec)
