"""Lossy LeRobot bridge for canonical replay records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from src.dataset_bridges.sidecar_refs import extract_sidecar_refs
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord


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
                    "internal_sidecars": _row_sidecars(step),
                },
            }
        )
    return rows


__all__ = ["lerobot_rows_from_replay"]
