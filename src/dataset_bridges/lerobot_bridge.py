"""Lossy LeRobot bridge for canonical replay records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from src.dataset_bridges.sidecar_refs import extract_sidecar_refs
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord


_PROVENANCE_REF_KEYS = {
    "runtime_packet_ref",
    "event_spine_ref",
    "decision_ledger_ref",
    "objective_tensor_ref",
    "econ_tensor_ref",
}


def _row_sidecars(step: ReplayStepRecord) -> Dict[str, Any]:
    return extract_sidecar_refs(step)


def lerobot_rows_from_replay(
    episode: ReplayEpisodeRecord,
    steps: Iterable[ReplayStepRecord],
) -> List[Dict[str, Any]]:
    """Convert replay steps into a tabular LeRobot-like row set."""
    ordered_steps = sorted(steps, key=lambda row: row.step_idx)
    rows: List[Dict[str, Any]] = []
    for frame_index, step in enumerate(ordered_steps):
        rows.append(
            {
                "episode_id": episode.episode_id,
                "frame_index": frame_index,
                "timestamp": step.timestamp,
                "observation": dict(step.obs),
                "action": dict(step.action),
                "reward": float(step.reward),
                "done": bool(step.done),
                "task": step.task_id,
                "environment": step.env_id,
                "source_domain": step.source_domain,
                "metadata": {
                    "record_id": step.record_id,
                    "seed": step.seed,
                    "skill_mode": step.skill_mode,
                    "benchmark_gate": dict(
                        step.metadata.get("benchmark_gate")
                        or step.metadata.get("source_benchmark_gate")
                        or {}
                    ),
                    "future_training_signals": dict(
                        step.metadata.get("future_training_signals", {}) or {}
                    ),
                    "internal_sidecars": _row_sidecars(step),
                },
            }
        )
    return rows


def _split_sidecars(
    sidecars: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    for key, value in dict(sidecars or {}).items():
        if str(key) in _PROVENANCE_REF_KEYS:
            provenance[str(key)] = value
        else:
            metadata[str(key)] = value
    return metadata, provenance


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def replay_episode_from_lerobot(
    rows: Iterable[Mapping[str, Any]],
    *,
    default_run_id: str = "lerobot_rehydrated",
    default_source_domain: str = "lerobot_bridge",
) -> tuple[ReplayEpisodeRecord, list[ReplayStepRecord]]:
    """Rehydrate LeRobot-like rows back into canonical replay rows."""

    ordered_rows = sorted(list(rows), key=lambda row: int(row.get("frame_index", 0)))
    if not ordered_rows:
        raise ValueError("Cannot rehydrate replay episode from empty LeRobot row set")
    first = ordered_rows[0]
    metadata = dict(first.get("metadata", {}) or {})
    sidecars = dict(metadata.get("internal_sidecars", {}) or {})
    episode_id = str(first.get("episode_id", ""))
    task_id = str(first.get("task", "unknown_task") or "unknown_task")
    env_id = str(first.get("environment", "unknown_env") or "unknown_env")
    source_domain = str(
        first.get("source_domain", default_source_domain) or default_source_domain
    )
    replay_steps: list[ReplayStepRecord] = []
    all_event_refs: list[str] = []
    all_decision_refs: list[str] = []
    for row in ordered_rows:
        row_metadata = dict(row.get("metadata", {}) or {})
        row_sidecars = dict(row_metadata.get("internal_sidecars", {}) or {})
        restored_metadata, restored_provenance = _split_sidecars(row_sidecars)
        for key, value in row_metadata.items():
            if key not in {
                "internal_sidecars",
                "benchmark_gate",
                "future_training_signals",
            }:
                restored_metadata[str(key)] = value
        if isinstance(row_metadata.get("benchmark_gate"), Mapping):
            restored_metadata["benchmark_gate"] = dict(row_metadata["benchmark_gate"])
        if isinstance(row_metadata.get("future_training_signals"), Mapping):
            restored_metadata["future_training_signals"] = dict(
                row_metadata["future_training_signals"]
            )
        all_event_refs.extend(
            [str(value) for value in restored_metadata.get("event_refs", []) or []]
        )
        all_decision_refs.extend(
            [str(value) for value in restored_metadata.get("decision_refs", []) or []]
        )
        replay_steps.append(
            ReplayStepRecord(
                run_id=str(
                    row_metadata.get("run_id", default_run_id) or default_run_id
                ),
                episode_id=str(row.get("episode_id", episode_id) or episode_id),
                step_idx=int(row.get("frame_index", 0)),
                obs=dict(row.get("observation", {}) or {}),
                obs_vector=[],
                action=dict(row.get("action", {}) or {}),
                action_vector=[],
                reward=float(row.get("reward", 0.0)),
                reward_decomposition={},
                done=bool(row.get("done", False)),
                task_id=str(row.get("task", task_id) or task_id),
                env_id=str(row.get("environment", env_id) or env_id),
                condition_vector={},
                condition_vector_values=[],
                skill_mode=str(row_metadata.get("skill_mode", "rehydrated")),
                objective_tensor_summary={},
                objective_tensor_ref=_optional_str(
                    restored_provenance.get("objective_tensor_ref")
                    or restored_metadata.get("objective_tensor_ref")
                ),
                econ_tensor_summary={},
                econ_tensor_ref=_optional_str(
                    restored_provenance.get("econ_tensor_ref")
                    or restored_metadata.get("econ_tensor_ref")
                ),
                constraint_flags=[],
                pricing_tick_ref=restored_metadata.get("pricing_tick_ref"),
                ledger_event_ref=restored_metadata.get("ledger_event_ref"),
                source_domain=str(
                    row.get("source_domain", source_domain) or source_domain
                ),
                seed=int(row_metadata.get("seed", 0) or 0),
                timestamp=str(row.get("timestamp", "")),
                metadata=restored_metadata,
                provenance=restored_provenance,
            )
        )
    episode_metadata, episode_provenance = _split_sidecars(sidecars)
    if isinstance(metadata.get("benchmark_gate"), Mapping):
        episode_metadata["benchmark_gate"] = dict(metadata["benchmark_gate"])
    if isinstance(metadata.get("future_training_signals"), Mapping):
        episode_metadata["future_training_signals"] = dict(
            metadata["future_training_signals"]
        )
    episode = ReplayEpisodeRecord(
        run_id=str(metadata.get("run_id", default_run_id) or default_run_id),
        episode_id=episode_id,
        task_id=task_id,
        env_id=env_id,
        source_domain=source_domain,
        seed=int(metadata.get("seed", 0) or 0),
        status="done" if replay_steps[-1].done else "unknown",
        started_at=str(replay_steps[0].timestamp),
        ended_at=str(replay_steps[-1].timestamp),
        total_steps=len(replay_steps),
        total_reward=sum(step.reward for step in replay_steps),
        skill_mode=str(replay_steps[0].skill_mode),
        condition_vector={},
        condition_vector_values=[],
        objective_tensor_summary={},
        objective_tensor_ref=_optional_str(
            episode_provenance.get("objective_tensor_ref")
            or episode_metadata.get("objective_tensor_ref")
        ),
        econ_tensor_summary={},
        econ_tensor_ref=_optional_str(
            episode_provenance.get("econ_tensor_ref")
            or episode_metadata.get("econ_tensor_ref")
        ),
        pricing_summary={},
        pricing_tick_refs=[],
        constraint_flags=[],
        regal_summary={},
        datapack_summary={},
        ledger_event_ids=[],
        metadata={
            **episode_metadata,
            "event_refs": sorted(set(all_event_refs)),
            "decision_refs": sorted(set(all_decision_refs)),
        },
        provenance=episode_provenance,
    )
    return episode, replay_steps


__all__ = ["lerobot_rows_from_replay", "replay_episode_from_lerobot"]
