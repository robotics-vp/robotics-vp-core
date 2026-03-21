"""Lossy RLDS bridge for canonical replay records.

This adapter intentionally flattens internal replay records into RLDS-friendly
dicts while preserving references to internal sidecars in metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord


def _sidecar_refs(step: ReplayStepRecord) -> Dict[str, Any]:
    return {
        "objective_tensor_ref": step.objective_tensor_ref,
        "econ_tensor_ref": step.econ_tensor_ref,
        "pricing_tick_ref": step.pricing_tick_ref,
        "ledger_event_ref": step.ledger_event_ref,
        "event_refs": step.metadata.get("event_refs", []),
        "decision_refs": step.metadata.get("decision_refs", []),
        "governance_trace_ref": step.provenance.get("governance_trace_ref"),
        "runtime_packet_ref": step.provenance.get("runtime_packet_ref"),
    }


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
                "objective_tensor_ref": episode.objective_tensor_ref,
                "econ_tensor_ref": episode.econ_tensor_ref,
                "pricing_tick_refs": list(episode.pricing_tick_refs),
                "ledger_event_ids": list(episode.ledger_event_ids),
                "provenance": dict(episode.provenance),
            },
        },
    }


__all__ = ["rlds_episode_from_replay"]
