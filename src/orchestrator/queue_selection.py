"""Additive live queue-selection shim that consumes advisory outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class QueueSelectionEntry:
    """Queue entry emitted from advisory-only signals."""

    episode_id: str
    queue_name: str
    priority_score: float
    tags: list[str]
    replay_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": sha256_json(
                {
                    "episode_id": self.episode_id,
                    "queue_name": self.queue_name,
                    "priority_score": self.priority_score,
                    "tags": self.tags,
                }
            )[:18],
            "episode_id": self.episode_id,
            "queue_name": self.queue_name,
            "priority_score": float(self.priority_score),
            "tags": list(self.tags),
            "replay_action": self.replay_action,
            "metadata": dict(self.metadata),
        }


def build_live_queue_selection(
    advisory_output: Mapping[str, Any],
    *,
    queue_name: str = "shadow_advisory_queue",
) -> Dict[str, Any]:
    episodes = list(advisory_output.get("episodes", []) or [])
    entries: list[QueueSelectionEntry] = []
    for episode in episodes:
        entries.append(
            QueueSelectionEntry(
                episode_id=str(episode.get("episode_id", "")),
                queue_name=queue_name,
                priority_score=float(episode.get("sampling_priority_score", 0.0)),
                tags=[str(value) for value in episode.get("replay_queue_tags", []) or []],
                replay_action=str(episode.get("replay_action", "holdout")),
                metadata={
                    "deploy_recommendation": episode.get("deploy_recommendation"),
                    "pricing_recommendation": episode.get("pricing_recommendation"),
                    "datapack_recommendation": episode.get("datapack_recommendation"),
                },
            )
        )
    entries = sorted(entries, key=lambda entry: (-entry.priority_score, entry.episode_id))
    payload = {
        "queue_name": queue_name,
        "entries": [entry.to_dict() for entry in entries],
        "summary": {
            "num_entries": len(entries),
            "top_episode_id": entries[0].episode_id if entries else None,
            "queue_digest": sha256_json([entry.to_dict() for entry in entries]),
        },
    }
    return payload
