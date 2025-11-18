"""
Build RECAP-style dataset from ontology episodes/events/econ vectors.
"""
import json
from typing import Dict, Optional, Callable, Any
from pathlib import Path

from src.ontology.store import OntologyStore
from src.policies.registry import build_all_policies
from src.vision.policy_observation_builder import PolicyObservationBuilder
from src.vision.interfaces import VisionFrame, compute_state_digest
from src.vision.config import load_vision_config
from src.vla.recap_features import summarize_vision_features


def build_recap_dataset(
    store: OntologyStore,
    task_id: str,
    output_path: str,
    max_episodes: int = 1000,
    use_vision_features: bool = False,
    vision_frame_provider: Optional[Callable[[Any, Any], VisionFrame]] = None,
    vision_builder: Optional[PolicyObservationBuilder] = None,
) -> None:
    episodes = store.list_episodes(task_id=task_id)[:max_episodes]
    econ_map = {e.episode_id: e for e in store.list_econ_vectors()}
    events_by_ep: Dict[str, list] = {}
    for ep in episodes:
        events_by_ep[ep.episode_id] = store.get_events(ep.episode_id)

    rewards = [ev.reward_scalar_sum for ev in econ_map.values() if hasattr(ev, "reward_scalar_sum")]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    builder = vision_builder
    if use_vision_features and builder is None:
        builder = PolicyObservationBuilder(build_all_policies().vision_encoder)
    cfg = load_vision_config() if use_vision_features else None

    def _default_frame_provider(ep, evt):
        width, height = (cfg.get("input_resolution", [224, 224]) if cfg else [224, 224])
        channels = int(cfg.get("channels", 3)) if cfg else 3
        dtype = str(cfg.get("dtype", "uint8")) if cfg else "uint8"
        return VisionFrame(
            backend="recap_stub",
            backend_id="recap_stub",
            task_id=task_id,
            episode_id=ep.episode_id,
            timestep=evt.timestep,
            width=int(width),
            height=int(height),
            channels=channels,
            dtype=dtype,
            camera_intrinsics={
                "resolution": [int(width), int(height)],
                "fov_deg": float(cfg.get("fov_deg", 90.0)) if cfg else 90.0,
            },
            camera_extrinsics={"frame": "world", "translation": [0.0, 0.0, 1.0], "rotation_rpy": [0.0, 0.0, 0.0]},
            state_digest=compute_state_digest(evt.state_summary or {"timestep": evt.timestep}),
            metadata={"source": "recap_dataset_builder"},
        )

    provider = vision_frame_provider or _default_frame_provider
    with open(output_path, "w") as f:
        for ep in episodes:
            econ = econ_map.get(ep.episode_id)
            events = events_by_ep.get(ep.episode_id, [])
            for evt in events:
                advantage = float(evt.reward_scalar - reward_mean)
                metrics = dict(evt.reward_components)
                metrics["mobility_penalty"] = getattr(econ, "mobility_penalty", 0.0) if econ else 0.0
                metrics["precision_bonus"] = getattr(econ, "precision_bonus", 0.0) if econ else 0.0
                metrics["stability_risk_score"] = getattr(econ, "stability_risk_score", 0.0) if econ else 0.0
                entry = {
                    "task_id": task_id,
                    "episode_id": ep.episode_id,
                    "timestep": evt.timestep,
                    "advantage": advantage,
                    "metrics": metrics,
                    "sampler_strategy": ep.metadata.get("sampling_metadata", {}).get("strategy") if ep.metadata else None,
                    "curriculum_phase": ep.metadata.get("metadata", {}).get("curriculum_phase") if hasattr(ep, "metadata") else None,
                    "objective_preset": ep.metadata.get("objective_preset") if ep.metadata else None,
                    "mobility_penalty": getattr(econ, "mobility_penalty", 0.0) if econ else 0.0,
                    "precision_bonus": getattr(econ, "precision_bonus", 0.0) if econ else 0.0,
                    "stability_risk_score": getattr(econ, "stability_risk_score", 0.0) if econ else 0.0,
                }
                mob_meta = evt.metadata.get("mobility_adjustment", {}) if hasattr(evt, "metadata") else {}
                if mob_meta:
                    entry["mobility_adjustment"] = mob_meta
                if use_vision_features and builder is not None:
                    frame = provider(ep, evt)
                    policy_feats = builder.build_policy_features(frame, evt.state_summary or {})
                    entry["vision_features"] = summarize_vision_features(policy_feats)
                f.write(json.dumps(entry, sort_keys=True))
                f.write("\n")
