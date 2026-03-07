"""Canonical replay schema for deterministic shadow/offline learning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.utils.config_digest import sha256_json


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def _rows(payload: Optional[List[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in (payload or [])]


def _floats(payload: Optional[List[float]]) -> List[float]:
    return [float(value) for value in (payload or [])]


@dataclass(frozen=True)
class ReplayStepRecord:
    """Single canonical replay step."""

    run_id: str
    episode_id: str
    step_idx: int
    obs: Dict[str, Any]
    obs_vector: List[float]
    action: Dict[str, Any]
    action_vector: List[float]
    reward: float
    reward_decomposition: Dict[str, Any]
    done: bool
    task_id: str
    env_id: str
    condition_vector: Dict[str, Any]
    condition_vector_values: List[float]
    skill_mode: str
    objective_tensor_summary: Dict[str, Any]
    objective_tensor_ref: Optional[str]
    econ_tensor_summary: Dict[str, Any]
    econ_tensor_ref: Optional[str]
    constraint_flags: List[Dict[str, Any]]
    pricing_tick_ref: Optional[str]
    ledger_event_ref: Optional[str]
    source_domain: str
    seed: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return f"{self.episode_id}:{self.step_idx:04d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_idx": int(self.step_idx),
            "obs": dict(self.obs),
            "obs_vector": list(self.obs_vector),
            "action": dict(self.action),
            "action_vector": list(self.action_vector),
            "reward": float(self.reward),
            "reward_decomposition": dict(self.reward_decomposition),
            "done": bool(self.done),
            "task_id": self.task_id,
            "env_id": self.env_id,
            "condition_vector": dict(self.condition_vector),
            "condition_vector_values": list(self.condition_vector_values),
            "skill_mode": self.skill_mode,
            "objective_tensor_summary": dict(self.objective_tensor_summary),
            "objective_tensor_ref": self.objective_tensor_ref,
            "econ_tensor_summary": dict(self.econ_tensor_summary),
            "econ_tensor_ref": self.econ_tensor_ref,
            "constraint_flags": [dict(flag) for flag in self.constraint_flags],
            "pricing_tick_ref": self.pricing_tick_ref,
            "ledger_event_ref": self.ledger_event_ref,
            "source_domain": self.source_domain,
            "seed": int(self.seed),
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayStepRecord":
        return cls(
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            step_idx=int(payload.get("step_idx", 0)),
            obs=_mapping(payload.get("obs")),
            obs_vector=_floats(payload.get("obs_vector")),
            action=_mapping(payload.get("action")),
            action_vector=_floats(payload.get("action_vector")),
            reward=float(payload.get("reward", 0.0)),
            reward_decomposition=_mapping(payload.get("reward_decomposition")),
            done=bool(payload.get("done", False)),
            task_id=str(payload.get("task_id", "")),
            env_id=str(payload.get("env_id", "")),
            condition_vector=_mapping(payload.get("condition_vector")),
            condition_vector_values=_floats(payload.get("condition_vector_values")),
            skill_mode=str(payload.get("skill_mode", "efficiency_throughput")),
            objective_tensor_summary=_mapping(payload.get("objective_tensor_summary")),
            objective_tensor_ref=payload.get("objective_tensor_ref"),
            econ_tensor_summary=_mapping(payload.get("econ_tensor_summary")),
            econ_tensor_ref=payload.get("econ_tensor_ref"),
            constraint_flags=_rows(payload.get("constraint_flags")),
            pricing_tick_ref=payload.get("pricing_tick_ref"),
            ledger_event_ref=payload.get("ledger_event_ref"),
            source_domain=str(payload.get("source_domain", "")),
            seed=int(payload.get("seed", 0)),
            timestamp=str(payload.get("timestamp", "")),
            metadata=_mapping(payload.get("metadata")),
            provenance=_mapping(payload.get("provenance")),
        )


@dataclass(frozen=True)
class ReplayEpisodeRecord:
    """Episode-level view over canonical replay data."""

    run_id: str
    episode_id: str
    task_id: str
    env_id: str
    source_domain: str
    seed: int
    status: str
    started_at: str
    ended_at: str
    total_steps: int
    total_reward: float
    skill_mode: str
    condition_vector: Dict[str, Any]
    condition_vector_values: List[float]
    objective_tensor_summary: Dict[str, Any]
    objective_tensor_ref: Optional[str]
    econ_tensor_summary: Dict[str, Any]
    econ_tensor_ref: Optional[str]
    pricing_summary: Dict[str, Any]
    pricing_tick_refs: List[str]
    constraint_flags: List[Dict[str, Any]]
    regal_summary: Dict[str, Any]
    datapack_summary: Dict[str, Any]
    ledger_event_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "source_domain": self.source_domain,
            "seed": int(self.seed),
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_steps": int(self.total_steps),
            "total_reward": float(self.total_reward),
            "skill_mode": self.skill_mode,
            "condition_vector": dict(self.condition_vector),
            "condition_vector_values": list(self.condition_vector_values),
            "objective_tensor_summary": dict(self.objective_tensor_summary),
            "objective_tensor_ref": self.objective_tensor_ref,
            "econ_tensor_summary": dict(self.econ_tensor_summary),
            "econ_tensor_ref": self.econ_tensor_ref,
            "pricing_summary": dict(self.pricing_summary),
            "pricing_tick_refs": list(self.pricing_tick_refs),
            "constraint_flags": [dict(flag) for flag in self.constraint_flags],
            "regal_summary": dict(self.regal_summary),
            "datapack_summary": dict(self.datapack_summary),
            "ledger_event_ids": list(self.ledger_event_ids),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayEpisodeRecord":
        return cls(
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            task_id=str(payload.get("task_id", "")),
            env_id=str(payload.get("env_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            seed=int(payload.get("seed", 0)),
            status=str(payload.get("status", "unknown")),
            started_at=str(payload.get("started_at", "")),
            ended_at=str(payload.get("ended_at", "")),
            total_steps=int(payload.get("total_steps", 0)),
            total_reward=float(payload.get("total_reward", 0.0)),
            skill_mode=str(payload.get("skill_mode", "efficiency_throughput")),
            condition_vector=_mapping(payload.get("condition_vector")),
            condition_vector_values=_floats(payload.get("condition_vector_values")),
            objective_tensor_summary=_mapping(payload.get("objective_tensor_summary")),
            objective_tensor_ref=payload.get("objective_tensor_ref"),
            econ_tensor_summary=_mapping(payload.get("econ_tensor_summary")),
            econ_tensor_ref=payload.get("econ_tensor_ref"),
            pricing_summary=_mapping(payload.get("pricing_summary")),
            pricing_tick_refs=[str(value) for value in payload.get("pricing_tick_refs", []) or []],
            constraint_flags=_rows(payload.get("constraint_flags")),
            regal_summary=_mapping(payload.get("regal_summary")),
            datapack_summary=_mapping(payload.get("datapack_summary")),
            ledger_event_ids=[str(value) for value in payload.get("ledger_event_ids", []) or []],
            metadata=_mapping(payload.get("metadata")),
            provenance=_mapping(payload.get("provenance")),
        )


@dataclass(frozen=True)
class ReplayWindowRecord:
    """Trajectory-window view used for pseudo-real-time replay learning."""

    run_id: str
    episode_id: str
    window_id: str
    start_step: int
    end_step: int
    task_id: str
    env_id: str
    source_domain: str
    seed: int
    timestamp: str
    reward_sum: float
    obs_vector_mean: List[float]
    action_vector_mean: List[float]
    condition_vector: Dict[str, Any]
    condition_vector_values: List[float]
    skill_mode: str
    objective_tensor_summary: Dict[str, Any]
    econ_tensor_summary: Dict[str, Any]
    pricing_summary: Dict[str, Any]
    constraint_flags: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "window_id": self.window_id,
            "start_step": int(self.start_step),
            "end_step": int(self.end_step),
            "task_id": self.task_id,
            "env_id": self.env_id,
            "source_domain": self.source_domain,
            "seed": int(self.seed),
            "timestamp": self.timestamp,
            "reward_sum": float(self.reward_sum),
            "obs_vector_mean": list(self.obs_vector_mean),
            "action_vector_mean": list(self.action_vector_mean),
            "condition_vector": dict(self.condition_vector),
            "condition_vector_values": list(self.condition_vector_values),
            "skill_mode": self.skill_mode,
            "objective_tensor_summary": dict(self.objective_tensor_summary),
            "econ_tensor_summary": dict(self.econ_tensor_summary),
            "pricing_summary": dict(self.pricing_summary),
            "constraint_flags": [dict(flag) for flag in self.constraint_flags],
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayWindowRecord":
        return cls(
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            window_id=str(payload.get("window_id", "")),
            start_step=int(payload.get("start_step", 0)),
            end_step=int(payload.get("end_step", 0)),
            task_id=str(payload.get("task_id", "")),
            env_id=str(payload.get("env_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            seed=int(payload.get("seed", 0)),
            timestamp=str(payload.get("timestamp", "")),
            reward_sum=float(payload.get("reward_sum", 0.0)),
            obs_vector_mean=_floats(payload.get("obs_vector_mean")),
            action_vector_mean=_floats(payload.get("action_vector_mean")),
            condition_vector=_mapping(payload.get("condition_vector")),
            condition_vector_values=_floats(payload.get("condition_vector_values")),
            skill_mode=str(payload.get("skill_mode", "efficiency_throughput")),
            objective_tensor_summary=_mapping(payload.get("objective_tensor_summary")),
            econ_tensor_summary=_mapping(payload.get("econ_tensor_summary")),
            pricing_summary=_mapping(payload.get("pricing_summary")),
            constraint_flags=_rows(payload.get("constraint_flags")),
            metadata=_mapping(payload.get("metadata")),
            provenance=_mapping(payload.get("provenance")),
        )


@dataclass(frozen=True)
class ReplayDatasetManifest:
    """Deterministic manifest for replay datasets and training artifacts."""

    schema_version: str
    run_ids: List[str]
    source_adapters: List[str]
    files: Dict[str, str]
    num_episodes: int
    num_steps: int
    num_windows: int
    obs_dim: int
    action_dim: int
    condition_dim: int
    skill_modes: List[str]
    config_digest: str
    dataset_digest: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact_schema_fingerprint: Dict[str, Any] = field(default_factory=dict)
    provenance_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_ids": list(self.run_ids),
            "source_adapters": list(self.source_adapters),
            "files": dict(self.files),
            "num_episodes": int(self.num_episodes),
            "num_steps": int(self.num_steps),
            "num_windows": int(self.num_windows),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "condition_dim": int(self.condition_dim),
            "skill_modes": list(self.skill_modes),
            "config_digest": self.config_digest,
            "dataset_digest": self.dataset_digest,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "artifact_schema_fingerprint": dict(self.artifact_schema_fingerprint),
            "provenance_summary": dict(self.provenance_summary),
        }

    @property
    def manifest_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayDatasetManifest":
        return cls(
            schema_version=str(payload.get("schema_version", "shadow_replay_dataset_v1")),
            run_ids=[str(value) for value in payload.get("run_ids", []) or []],
            source_adapters=[str(value) for value in payload.get("source_adapters", []) or []],
            files=_mapping(payload.get("files")),
            num_episodes=int(payload.get("num_episodes", 0)),
            num_steps=int(payload.get("num_steps", 0)),
            num_windows=int(payload.get("num_windows", 0)),
            obs_dim=int(payload.get("obs_dim", 0)),
            action_dim=int(payload.get("action_dim", 0)),
            condition_dim=int(payload.get("condition_dim", 0)),
            skill_modes=[str(value) for value in payload.get("skill_modes", []) or []],
            config_digest=str(payload.get("config_digest", "")),
            dataset_digest=str(payload.get("dataset_digest", "")),
            created_at=str(payload.get("created_at", "")),
            metadata=_mapping(payload.get("metadata")),
            artifact_schema_fingerprint=_mapping(payload.get("artifact_schema_fingerprint")),
            provenance_summary=_mapping(payload.get("provenance_summary")),
        )
