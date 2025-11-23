"""
SIMA-2 client stub for deterministic semantic rollouts.

Advisory-only: produces canned trajectories that can feed the Stage 2 pipeline
without requiring a live agent or physics backend.

Supports task library templates: success, failure, recovery, mixed
"""
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterable, Iterator, Optional
import itertools
import json

from src.sima2.config import build_provenance, load_sima2_config


# Task-specific rollout generators following SIMA2_HARDENING_TASK_LIBRARY.md
def _gen_drawer_open_success(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Normal happy path: approach -> grasp -> pull -> success"""
    prims = [
        {"timestep": 0, "object": "drawer_handle", "action": "approach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.2, "contact": False},
        {"timestep": 3, "object": "drawer_handle", "action": "grasp", "risk": "low", "gripper_width": 0.02, "ee_velocity": 0.0, "contact": True},
        {"timestep": 6, "object": "drawer", "action": "pull", "risk": "medium", "gripper_width": 0.02, "ee_velocity": 0.15, "contact": True},
        {"timestep": 10, "object": "drawer", "action": "release", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.5, 0.0, 0.3, 0, 0, 0]}
    return prims, events, {"outcome": "success", "template": "success", "objects_present": ["drawer", "drawer_handle"]}


def _gen_drawer_open_failure(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Pure failure: grasp -> pull -> slip -> abort"""
    prims = [
        {"timestep": 0, "object": "drawer_handle", "action": "approach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.2, "contact": False},
        {"timestep": 3, "object": "drawer_handle", "action": "grasp", "risk": "medium", "gripper_width": 0.02, "ee_velocity": 0.0, "contact": True},
        {"timestep": 6, "object": "drawer", "action": "pull", "risk": "high", "gripper_width": 0.02, "ee_velocity": 0.15, "contact": True},
        {"timestep": 8, "object": "drawer_handle", "action": "slip", "risk": "critical", "gripper_width": 0.05, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    events.append({"timestep": 8, "event_type": "failure", "payload": {"reason": "grasp_slip", "severity": "high"}})
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.5, 0.0, 0.3, 0, 0, 0]}
    return prims, events, {"outcome": "failure", "template": "failure", "failure_mode": "grasp_slip", "objects_present": ["drawer", "drawer_handle"]}


def _gen_drawer_open_recovery(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Recovery: grasp -> pull -> slip -> regrasp -> pull -> success"""
    prims = [
        {"timestep": 0, "object": "drawer_handle", "action": "approach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.2, "contact": False},
        {"timestep": 3, "object": "drawer_handle", "action": "grasp", "risk": "medium", "gripper_width": 0.02, "ee_velocity": 0.0, "contact": True},
        {"timestep": 6, "object": "drawer", "action": "pull", "risk": "high", "gripper_width": 0.02, "ee_velocity": 0.15, "contact": True},
        {"timestep": 8, "object": "drawer_handle", "action": "slip", "risk": "critical", "gripper_width": 0.05, "ee_velocity": 0.0, "contact": False},
        {"timestep": 10, "object": "drawer_handle", "action": "regrasp", "risk": "medium", "gripper_width": 0.02, "ee_velocity": 0.0, "contact": True, "recovery": True},
        {"timestep": 13, "object": "drawer", "action": "pull", "risk": "medium", "gripper_width": 0.02, "ee_velocity": 0.12, "contact": True},
        {"timestep": 17, "object": "drawer", "action": "release", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    events.append({"timestep": 8, "event_type": "failure", "payload": {"reason": "grasp_slip", "severity": "medium"}})
    events.append({"timestep": 10, "event_type": "recovery_start", "payload": {"strategy": "regrasp"}})
    events.append({"timestep": 17, "event_type": "recovery_complete", "payload": {"success": True}})
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.5, 0.0, 0.3, 0, 0, 0]}
    return prims, events, {"outcome": "recovered", "template": "recovery", "failure_mode": "grasp_slip", "recovery_strategy": "regrasp", "objects_present": ["drawer", "drawer_handle"]}


def _gen_dish_place_success(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Dish placement success: reach -> grasp -> transport -> place"""
    prims = [
        {"timestep": 0, "object": "dish", "action": "reach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.25, "contact": False},
        {"timestep": 4, "object": "dish", "action": "grasp", "risk": "low", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True},
        {"timestep": 7, "object": "dish", "action": "lift", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.1, "contact": True},
        {"timestep": 11, "object": "rack", "action": "approach", "risk": "low", "gripper_width": 0.03, "ee_velocity": 0.2, "contact": True},
        {"timestep": 15, "object": "rack", "action": "place", "risk": "medium", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.4, 0.2, 0.5, 0, 0, 0]}
    return prims, events, {"outcome": "success", "template": "success", "objects_present": ["dish", "rack"]}


def _gen_dish_place_failure(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Dish placement failure: grasp -> lift -> drop"""
    prims = [
        {"timestep": 0, "object": "dish", "action": "reach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.25, "contact": False},
        {"timestep": 4, "object": "dish", "action": "grasp", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True},
        {"timestep": 7, "object": "dish", "action": "lift", "risk": "high", "gripper_width": 0.03, "ee_velocity": 0.1, "contact": True},
        {"timestep": 9, "object": "dish", "action": "drop", "risk": "critical", "gripper_width": 0.05, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    events.append({"timestep": 9, "event_type": "failure", "payload": {"reason": "drop", "severity": "high", "damage_cost": 2.5}})
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.4, 0.2, 0.5, 0, 0, 0]}
    return prims, events, {"outcome": "failure", "template": "failure", "failure_mode": "drop", "objects_present": ["dish", "rack"]}


def _gen_dish_place_recovery(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Dish placement recovery: grasp -> drop -> pick_from_drop -> place"""
    prims = [
        {"timestep": 0, "object": "dish", "action": "reach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.25, "contact": False},
        {"timestep": 4, "object": "dish", "action": "grasp", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True},
        {"timestep": 7, "object": "dish", "action": "lift", "risk": "high", "gripper_width": 0.03, "ee_velocity": 0.1, "contact": True},
        {"timestep": 9, "object": "dish", "action": "drop", "risk": "critical", "gripper_width": 0.05, "ee_velocity": 0.0, "contact": False},
        {"timestep": 12, "object": "dish", "action": "pick_from_drop", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True, "recovery": True},
        {"timestep": 16, "object": "rack", "action": "approach", "risk": "low", "gripper_width": 0.03, "ee_velocity": 0.2, "contact": True},
        {"timestep": 20, "object": "rack", "action": "place", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    events.append({"timestep": 9, "event_type": "failure", "payload": {"reason": "drop", "severity": "medium"}})
    events.append({"timestep": 12, "event_type": "recovery_start", "payload": {"strategy": "pick_from_drop"}})
    events.append({"timestep": 20, "event_type": "recovery_complete", "payload": {"success": True}})
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.4, 0.2, 0.5, 0, 0, 0]}
    return prims, events, {"outcome": "recovered", "template": "recovery", "failure_mode": "drop", "recovery_strategy": "pick_from_drop", "objects_present": ["dish", "rack"]}


def _gen_wipe_surface_success(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Wipe surface success: grasp_tool -> wipe_pattern"""
    prims = [
        {"timestep": 0, "object": "sponge", "action": "reach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.2, "contact": False},
        {"timestep": 3, "object": "sponge", "action": "grasp", "risk": "low", "gripper_width": 0.04, "ee_velocity": 0.0, "contact": True},
        {"timestep": 6, "object": "surface", "action": "wipe", "risk": "low", "gripper_width": 0.04, "ee_velocity": 0.15, "contact": True, "force": 5.0},
        {"timestep": 15, "object": "surface", "action": "wipe", "risk": "low", "gripper_width": 0.04, "ee_velocity": 0.15, "contact": True, "force": 5.0},
        {"timestep": 20, "object": "sponge", "action": "release", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    events = [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in prims]
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.3, 0.0, 0.2, 0, 0, 0]}
    return prims, events, {"outcome": "success", "template": "success", "objects_present": ["sponge", "surface"]}


def _gen_mixed_rollout(task_id: str, idx: int, seed: int) -> tuple[list, list, dict]:
    """Mixed long-horizon: drawer_success -> dish_fail -> dish_recovery"""
    # Drawer task (success)
    drawer_prims, drawer_events, _ = _gen_drawer_open_success("drawer_open", idx, seed)
    # Offset dish task
    offset = 20
    dish_fail_prims = [
        {"timestep": offset + 0, "object": "dish", "action": "reach", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.25, "contact": False},
        {"timestep": offset + 4, "object": "dish", "action": "grasp", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True},
        {"timestep": offset + 7, "object": "dish", "action": "lift", "risk": "high", "gripper_width": 0.03, "ee_velocity": 0.1, "contact": True},
        {"timestep": offset + 9, "object": "dish", "action": "drop", "risk": "critical", "gripper_width": 0.05, "ee_velocity": 0.0, "contact": False},
        {"timestep": offset + 12, "object": "dish", "action": "pick_from_drop", "risk": "medium", "gripper_width": 0.03, "ee_velocity": 0.0, "contact": True, "recovery": True},
        {"timestep": offset + 16, "object": "rack", "action": "place", "risk": "low", "gripper_width": 0.08, "ee_velocity": 0.0, "contact": False},
    ]
    all_prims = drawer_prims + dish_fail_prims
    all_events = drawer_events + [{"timestep": p["timestep"], "event_type": "primitive", "payload": p} for p in dish_fail_prims]
    all_events.append({"timestep": offset + 9, "event_type": "failure", "payload": {"reason": "drop"}})
    all_events.append({"timestep": offset + 12, "event_type": "recovery_start", "payload": {"strategy": "pick_from_drop"}})
    all_events.append({"timestep": offset + 16, "event_type": "recovery_complete", "payload": {"success": True}})
    robot_state = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.3, 0.0, 0.3, 0, 0, 0]}
    return all_prims, all_events, {"outcome": "mixed_success", "template": "mixed", "subtasks": ["drawer_open_success", "dish_place_recovery"], "objects_present": ["drawer", "drawer_handle", "dish", "rack"]}


# Template dispatcher
_TEMPLATE_GENERATORS = {
    "drawer_open": {
        "success": _gen_drawer_open_success,
        "failure": _gen_drawer_open_failure,
        "recovery": _gen_drawer_open_recovery,
    },
    "dish_place": {
        "success": _gen_dish_place_success,
        "failure": _gen_dish_place_failure,
        "recovery": _gen_dish_place_recovery,
    },
    "wipe_surface": {
        "success": _gen_wipe_surface_success,
    },
    "mixed": {
        "mixed": _gen_mixed_rollout,
    },
}


def _canned_trajectory(task_id: str, idx: int, provenance: Dict[str, Any], task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic rollouts based on task_id and template."""
    template = task_spec.get("template", "success")
    seed = task_spec.get("seed", 0)

    # Dispatch to template generator
    task_family = task_id.split("_")[0] + "_" + task_id.split("_")[1] if "_" in task_id else task_id
    if task_family not in _TEMPLATE_GENERATORS:
        task_family = "drawer_open"  # default fallback

    generators = _TEMPLATE_GENERATORS.get(task_family, {})
    generator = generators.get(template, generators.get("success", _gen_drawer_open_success))

    prims, events, metadata_extra = generator(task_id, idx, seed)

    base = {
        "episode_id": f"{task_id}_sim_ep{idx}_{template}",
        "task": task_id,
        "task_spec": json.loads(json.dumps(task_spec or {})),
        "primitives": prims,
        "metadata": {
            "semantic_primitives": prims,
        },
    }
    base["metadata"].update(metadata_extra)
    base["actions"] = [{"timestep": p["timestep"], "action": p["action"]} for p in prims]
    base["observations"] = [{"timestep": p["timestep"], "state": {"object": p["object"]}} for p in prims]
    base["events"] = events
    base["metadata"].update(provenance)
    base.update(provenance)
    base["source"] = provenance.get("sima2_backend_id", "sima2")
    base["robot_state"] = {"gripper_open": True, "joint_positions": [0.0] * 7, "ee_pose": [0.5, 0.0, 0.3, 0, 0, 0]}
    base["object_states"] = {obj: {"pose": [0.0] * 6} for obj in metadata_extra.get("objects_present", [])}
    return base


@dataclass
class Sima2Client:
    """Deterministic SIMA-2 rollout generator (stub) with template support."""

    task_id: str
    seed: int = 0
    config_path: Optional[str] = None
    template: str = "success"  # success, failure, recovery, mixed
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

    def run_episode(self, task_spec: Optional[Dict] = None) -> Dict:
        task_spec = dict(task_spec or {})
        task_id = task_spec.get("task_id", self.task_id)
        idx = int(task_spec.get("episode_index", 0))
        template = task_spec.get("template", self.template)
        seed = task_spec.get("seed", self.seed)
        task_spec["task_id"] = task_id
        task_spec["episode_index"] = idx
        task_spec["template"] = template
        task_spec["seed"] = seed
        provenance = self._provenance(task_spec)
        return _canned_trajectory(task_id, idx, provenance, task_spec)

    def stream_rollout(self, task_spec: Optional[Dict] = None, count: int = 1) -> Iterator[Dict]:
        task_spec = task_spec or {}
        task_id = task_spec.get("task_id", self.task_id)
        template = task_spec.get("template", self.template)
        seed = task_spec.get("seed", self.seed)
        for idx in itertools.islice(itertools.count(0), count):
            spec = dict(task_spec or {})
            spec["task_id"] = task_id
            spec["episode_index"] = idx
            spec["template"] = template
            spec["seed"] = seed + idx  # Deterministic but varying
            provenance = self._provenance(spec)
            yield _canned_trajectory(task_id, idx, provenance, spec)
