"""
Simple JSONL-backed ontology store (Phase A).

Idempotent upserts, deterministic ordering, and JSON-safe persistence.
Performance is not a concern; files are rewritten on each upsert.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Iterable, Mapping, Any, TYPE_CHECKING
from datetime import datetime
from dataclasses import asdict, is_dataclass

from src.ontology.models import Task, Robot, Datapack, Episode, EconVector, EpisodeEvent
if TYPE_CHECKING:
    from src.scenarios.metadata import ScenarioMetadata

T = TypeVar("T")


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default))
            f.write("\n")


def _deserialize_list(records: List[Dict], model: Type[T]) -> List[T]:
    return [model(**rec) for rec in records]


def _serialize(model_obj) -> Dict:
    if is_dataclass(model_obj):
        return asdict(model_obj)
    if hasattr(model_obj, "dict"):
        return model_obj.dict()
    return dict(model_obj)


class OntologyStore:
    def __init__(self, root_dir: str = "data/ontology") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "tasks": self.root / "tasks.jsonl",
            "robots": self.root / "robots.jsonl",
            "datapacks": self.root / "datapacks.jsonl",
            "episodes": self.root / "episodes.jsonl",
            "econ_vectors": self.root / "econ_vectors.jsonl",
            "events": self.root / "events.jsonl",
            "scenarios": self.root / "scenarios.jsonl",
        }

    # Tasks
    def upsert_task(self, task: Task) -> None:
        tasks = {t.task_id: t for t in self.list_tasks()}
        tasks[task.task_id] = task
        ordered = [tasks[k] for k in sorted(tasks.keys())]
        _write_jsonl(self.paths["tasks"], (_serialize(t) for t in ordered))

    def get_task(self, task_id: str) -> Optional[Task]:
        return next((t for t in self.list_tasks() if t.task_id == task_id), None)

    def list_tasks(self) -> List[Task]:
        records = _load_jsonl(self.paths["tasks"])
        tasks = _deserialize_list(records, Task)
        return sorted(tasks, key=lambda t: t.task_id)

    # Robots
    def upsert_robot(self, robot: Robot) -> None:
        robots = {r.robot_id: r for r in self.list_robots()}
        robots[robot.robot_id] = robot
        ordered = [robots[k] for k in sorted(robots.keys())]
        _write_jsonl(self.paths["robots"], (_serialize(r) for r in ordered))

    def get_robot(self, robot_id: str) -> Optional[Robot]:
        return next((r for r in self.list_robots() if r.robot_id == robot_id), None)

    def list_robots(self) -> List[Robot]:
        records = _load_jsonl(self.paths["robots"])
        robots = _deserialize_list(records, Robot)
        return sorted(robots, key=lambda r: r.robot_id)

    # Datapacks
    def append_datapacks(self, datapacks: List[Datapack]) -> None:
        existing = {d.datapack_id: d for d in self.list_datapacks()}
        for d in datapacks:
            existing[d.datapack_id] = d
        ordered = [existing[k] for k in sorted(existing.keys())]
        _write_jsonl(self.paths["datapacks"], (_serialize(d) for d in ordered))

    def list_datapacks(self, task_id: Optional[str] = None, filters: Dict = None) -> List[Datapack]:
        records = _load_jsonl(self.paths["datapacks"])
        datapacks = _deserialize_list(records, Datapack)
        filtered = []
        for d in datapacks:
            if task_id and d.task_id != task_id:
                continue
            if filters and not self._matches_filters(d, filters):
                continue
            filtered.append(d)
        return sorted(filtered, key=lambda d: d.datapack_id)

    # Episodes
    def upsert_episode(self, episode: Episode) -> None:
        episodes = {e.episode_id: e for e in self.list_episodes()}
        episodes[episode.episode_id] = episode
        ordered = [episodes[k] for k in sorted(episodes.keys())]
        _write_jsonl(self.paths["episodes"], (_serialize(e) for e in ordered))

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        return next((e for e in self.list_episodes() if e.episode_id == episode_id), None)

    def list_episodes(self, task_id: Optional[str] = None, filters: Dict = None) -> List[Episode]:
        records = _load_jsonl(self.paths["episodes"])
        episodes = _deserialize_list(records, Episode)
        filtered = []
        for e in episodes:
            if task_id and e.task_id != task_id:
                continue
            if filters and not self._matches_filters(e, filters):
                continue
            filtered.append(e)
        return sorted(filtered, key=lambda e: e.episode_id)

    # Econ vectors
    def upsert_econ_vector(self, econ: EconVector) -> None:
        vectors = {v.episode_id: v for v in self.list_econ_vectors()}
        vectors[econ.episode_id] = econ
        ordered = [vectors[k] for k in sorted(vectors.keys())]
        _write_jsonl(self.paths["econ_vectors"], (_serialize(v) for v in ordered))

    def get_econ_vector(self, episode_id: str) -> Optional[EconVector]:
        return next((v for v in self.list_econ_vectors() if v.episode_id == episode_id), None)

    def list_econ_vectors(self) -> List[EconVector]:
        records = _load_jsonl(self.paths["econ_vectors"])
        vectors = _deserialize_list(records, EconVector)
        return sorted(vectors, key=lambda v: v.episode_id)

    # Events
    def append_events(self, events: List[EpisodeEvent]) -> None:
        existing = _deserialize_list(_load_jsonl(self.paths["events"]), EpisodeEvent)
        existing.extend(events)
        # Preserve append order deterministically by writing in current list order
        _write_jsonl(self.paths["events"], (_serialize(e) for e in existing))

    def get_events(self, episode_id: str) -> List[EpisodeEvent]:
        records = _load_jsonl(self.paths["events"])
        events = [EpisodeEvent(**r) for r in records if r.get("episode_id") == episode_id]
        return events

    # Scenarios
    def record_scenario(
        self,
        scenario: "ScenarioMetadata",
        train_metrics: Mapping[str, Any],
        eval_metrics: Mapping[str, Any],
    ) -> None:
        records = _load_jsonl(self.paths["scenarios"])
        existing = {r.get("scenario_id"): r for r in records if r.get("scenario_id")}

        record: Dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "task_id": scenario.task_id,
            "motor_backend": scenario.motor_backend,
            "objective_name": scenario.objective_name,
            "objective_weights": dict(scenario.objective_weights),
            "datapack_ids": list(scenario.datapack_ids),
            "datapack_tags": list(scenario.datapack_tags),
            "task_tags": list(scenario.task_tags),
            "robot_families": list(scenario.robot_families),
            "notes": scenario.notes,
            "train_metrics": dict(train_metrics),
            "eval_metrics": dict(eval_metrics),
        }
        record.update(_flatten_metrics("train_metrics", train_metrics))
        record.update(_flatten_metrics("eval_metrics", eval_metrics))

        existing[scenario.scenario_id] = record
        ordered = [existing[k] for k in sorted(existing.keys())]
        _write_jsonl(self.paths["scenarios"], ordered)

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return _load_jsonl(self.paths["scenarios"])

    # Helpers
    def _matches_filters(self, obj, filters: Dict) -> bool:
        for key, value in filters.items():
            if getattr(obj, key, None) != value:
                return False
        return True


def _flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key is None:
            continue
        clean_key = str(key).strip().replace(" ", "_").replace("/", "_")
        if not clean_key:
            continue
        flattened[f"{prefix}_{clean_key}"] = value
    return flattened
