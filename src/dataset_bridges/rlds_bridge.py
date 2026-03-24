"""Lossy RLDS bridge for canonical replay records.

This adapter intentionally flattens internal replay records into RLDS-friendly
dicts while preserving references to internal sidecars in metadata.
"""

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


def _split_sidecars(sidecars: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}
    for key, value in dict(sidecars or {}).items():
        if key == "provenance" and isinstance(value, Mapping):
            provenance.update(dict(value))
            continue
        if str(key) in _PROVENANCE_REF_KEYS:
            provenance[str(key)] = value
        else:
            metadata[str(key)] = value
    return metadata, provenance


def _sidecar_refs(step: ReplayStepRecord) -> Dict[str, Any]:
    return extract_sidecar_refs(step)


def rlds_episode_from_replay(
    episode: ReplayEpisodeRecord,
    steps: Iterable[ReplayStepRecord],
) -> Dict[str, Any]:
    """Convert one replay episode into a lossy RLDS-shaped dictionary."""
    ordered_steps = sorted(steps, key=lambda row: row.step_idx)
    rlds_steps: List[Dict[str, Any]] = []

    for index, step in enumerate(ordered_steps):
        is_first = index == 0
        is_last = bool(step.done)
        discount = 0.0 if is_last else 1.0
        rlds_steps.append(
            {
                "observation": dict(step.obs),
                "action": dict(step.action),
                "reward": float(step.reward),
                "discount": discount,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_last,
                "metadata": {
                    "record_id": step.record_id,
                    "timestamp": step.timestamp,
                    "task_id": step.task_id,
                    "env_id": step.env_id,
                    "source_domain": step.source_domain,
                    "internal_sidecars": _sidecar_refs(step),
                },
            }
        )

    return {
        "episode_id": episode.episode_id,
        "steps": rlds_steps,
        "metadata": {
            "run_id": episode.run_id,
            "task_id": episode.task_id,
            "env_id": episode.env_id,
            "source_domain": episode.source_domain,
            "internal_sidecars": {
                **extract_sidecar_refs(episode),
                "provenance": dict(episode.provenance),
            },
        },
    }


def replay_episode_from_rlds(
    payload: Mapping[str, Any],
    *,
    default_run_id: str = "rlds_rehydrated",
    default_source_domain: str = "rlds_bridge",
) -> tuple[ReplayEpisodeRecord, list[ReplayStepRecord]]:
    """Rehydrate one RLDS-shaped episode back into canonical replay rows."""

    metadata = dict(payload.get("metadata", {}) or {})
    sidecars = dict(metadata.get("internal_sidecars", {}) or {})
    episode_metadata, episode_provenance = _split_sidecars(sidecars)
    episode_id = str(payload.get("episode_id", metadata.get("episode_id", "")) or "")
    run_id = str(metadata.get("run_id", default_run_id) or default_run_id)
    task_id = str(metadata.get("task_id", "unknown_task") or "unknown_task")
    env_id = str(metadata.get("env_id", "unknown_env") or "unknown_env")
    source_domain = str(metadata.get("source_domain", default_source_domain) or default_source_domain)
    ordered_steps = list(payload.get("steps", []) or [])
    replay_steps: list[ReplayStepRecord] = []
    event_refs: list[str] = []
    decision_refs: list[str] = []
    for index, step_payload in enumerate(ordered_steps):
        step_metadata = dict(step_payload.get("metadata", {}) or {})
        step_sidecars = dict(step_metadata.get("internal_sidecars", {}) or {})
        restored_metadata, restored_provenance = _split_sidecars(step_sidecars)
        step_event_refs = [str(value) for value in restored_metadata.get("event_refs", []) or []]
        step_decision_refs = [str(value) for value in restored_metadata.get("decision_refs", []) or []]
        event_refs.extend(step_event_refs)
        decision_refs.extend(step_decision_refs)
        replay_steps.append(
            ReplayStepRecord(
                run_id=run_id,
                episode_id=episode_id,
                step_idx=index,
                obs=dict(step_payload.get("observation", {}) or {}),
                obs_vector=[],
                action=dict(step_payload.get("action", {}) or {}),
                action_vector=[],
                reward=float(step_payload.get("reward", 0.0)),
                reward_decomposition={},
                done=bool(step_payload.get("is_last", step_payload.get("is_terminal", False))),
                task_id=str(step_metadata.get("task_id", task_id) or task_id),
                env_id=str(step_metadata.get("env_id", env_id) or env_id),
                condition_vector={},
                condition_vector_values=[],
                skill_mode=str(step_metadata.get("skill_mode", "rehydrated")),
                objective_tensor_summary={},
                objective_tensor_ref=None,
                econ_tensor_summary={},
                econ_tensor_ref=None,
                constraint_flags=[],
                pricing_tick_ref=restored_metadata.get("pricing_tick_ref"),
                ledger_event_ref=restored_metadata.get("ledger_event_ref"),
                source_domain=str(step_metadata.get("source_domain", source_domain) or source_domain),
                seed=int(step_metadata.get("seed", metadata.get("seed", 0)) or 0),
                timestamp=str(step_metadata.get("timestamp", "")),
                metadata=restored_metadata,
                provenance=restored_provenance,
            )
        )

    episode = ReplayEpisodeRecord(
        run_id=run_id,
        episode_id=episode_id,
        task_id=task_id,
        env_id=env_id,
        source_domain=source_domain,
        seed=int(metadata.get("seed", 0) or 0),
        status="done" if replay_steps and replay_steps[-1].done else "unknown",
        started_at=str(replay_steps[0].timestamp if replay_steps else ""),
        ended_at=str(replay_steps[-1].timestamp if replay_steps else ""),
        total_steps=len(replay_steps),
        total_reward=sum(step.reward for step in replay_steps),
        skill_mode=str(replay_steps[0].skill_mode if replay_steps else "rehydrated"),
        condition_vector={},
        condition_vector_values=[],
        objective_tensor_summary={},
        objective_tensor_ref=episode_metadata.get("objective_tensor_ref"),
        econ_tensor_summary={},
        econ_tensor_ref=episode_metadata.get("econ_tensor_ref"),
        pricing_summary={},
        pricing_tick_refs=[],
        constraint_flags=[],
        regal_summary={},
        datapack_summary={},
        ledger_event_ids=[],
        metadata={
            **episode_metadata,
            "event_refs": sorted(set(event_refs)),
            "decision_refs": sorted(set(decision_refs)),
        },
        provenance=episode_provenance,
    )
    return episode, replay_steps


__all__ = ["rlds_episode_from_replay", "replay_episode_from_rlds"]
