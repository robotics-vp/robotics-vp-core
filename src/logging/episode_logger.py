"""
EpisodeLogger: advisory-only episode/event logging into OntologyStore.

No reward math changes; this is a thin wrapper to persist episodes, events, and
optional econ vectors.
"""
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ontology.models import Task, Robot, Datapack, Episode, EpisodeEvent, EconVector
from src.ontology.store import OntologyStore


def _deterministic_episode_id(task_id: str, robot_id: str, datapack_id: Optional[str]) -> str:
    payload = f"{task_id}|{robot_id}|{datapack_id or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"ep_{digest}"


class EpisodeLogger:
    def __init__(self, store: OntologyStore, task: Task, robot: Robot):
        self.store = store
        self.task = task
        self.robot = robot
        self._current_episode: Optional[Episode] = None
        self._events: List[EpisodeEvent] = []

    def start_episode(self, datapack: Optional[Datapack] = None, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        episode_id = _deterministic_episode_id(self.task.task_id, self.robot.robot_id, datapack.datapack_id if datapack else None)
        ep = Episode(
            episode_id=episode_id,
            task_id=self.task.task_id,
            robot_id=self.robot.robot_id,
            datapack_id=datapack.datapack_id if datapack else None,
            started_at=datetime.utcnow(),
            status="running",
            metadata=metadata or {},
        )
        self.store.upsert_episode(ep)
        self._current_episode = ep
        self._events = []
        return ep

    def log_step(
        self,
        timestep: int,
        reward_scalar: float,
        reward_components: Dict[str, float],
        state_summary: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if self._current_episode is None:
            raise RuntimeError("No active episode; call start_episode first.")
        event = EpisodeEvent(
            episode_id=self._current_episode.episode_id,
            timestep=timestep,
            event_type="step",
            timestamp=datetime.utcnow(),
            reward_scalar=float(reward_scalar),
            reward_components=reward_components or {},
            state_summary=state_summary or {},
            metadata=metadata or {},
        )
        self._events.append(event)

    def mark_outcome(self, status: str, metadata: Optional[Dict[str, Any]] = None):
        if self._current_episode is None:
            raise RuntimeError("No active episode; call start_episode first.")
        self._current_episode.status = status
        if metadata:
            self._current_episode.metadata.update(metadata)

    def finalize(self, econ_vector: Optional[EconVector] = None):
        if self._current_episode is None:
            raise RuntimeError("No active episode; call start_episode first.")
        self._current_episode.ended_at = datetime.utcnow()
        # Persist episode and events
        self.store.upsert_episode(self._current_episode)
        if self._events:
            self.store.append_events(self._events)
        if econ_vector:
            self.store.upsert_econ_vector(econ_vector)
        # Reset
        self._current_episode = None
        self._events = []
