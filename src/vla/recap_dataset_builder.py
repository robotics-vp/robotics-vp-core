"""
Build RECAP-style dataset from ontology episodes/events/econ vectors.
"""
import json
from typing import Dict
from pathlib import Path

from src.ontology.store import OntologyStore


def build_recap_dataset(
    store: OntologyStore,
    task_id: str,
    output_path: str,
    max_episodes: int = 1000,
) -> None:
    episodes = store.list_episodes(task_id=task_id)[:max_episodes]
    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    events_by_ep: Dict[str, list] = {}
    for ep in episodes:
        events_by_ep[ep.episode_id] = store.get_events(ep.episode_id)

    rewards = [ev.reward_scalar_sum for ev in econ_map.values() if hasattr(ev, "reward_scalar_sum")]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ep in episodes:
            econ = econ_map.get(ep.episode_id)
            events = events_by_ep.get(ep.episode_id, [])
            for evt in events:
                advantage = float(evt.reward_scalar - reward_mean)
                entry = {
                    "task_id": task_id,
                    "episode_id": ep.episode_id,
                    "timestep": evt.timestep,
                    "advantage": advantage,
                    "metrics": evt.reward_components,
                    "sampler_strategy": ep.metadata.get("sampling_metadata", {}).get("strategy") if ep.metadata else None,
                    "curriculum_phase": ep.metadata.get("metadata", {}).get("curriculum_phase") if hasattr(ep, "metadata") else None,
                    "objective_preset": ep.metadata.get("objective_preset") if ep.metadata else None,
                }
                f.write(json.dumps(entry, sort_keys=True))
                f.write("\n")
