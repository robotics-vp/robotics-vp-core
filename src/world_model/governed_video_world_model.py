"""Governed video-state scaffolding layered above the stable latent backbone."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.evidence.belief_state import BeliefState
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


def _hash_to_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _objective_one_hot(objective_preset: str) -> list[float]:
    order = ["balanced", "throughput", "safety", "energy_saver"]
    return [1.0 if objective_preset == item else 0.0 for item in order]


@dataclass(frozen=True)
class VideoStateConfig:
    """Configuration for the governed video-state service."""

    token_dim: int = 128
    hypothesis_budget: int = 4
    novelty_bias: float = 0.15
    min_render_priority: float = 0.15


@dataclass(frozen=True)
class VideoStateSnapshot:
    """Geometry- and evidence-first video state for downstream planning."""

    state_id: str
    episode_id: str
    timestamp: str
    objective_preset: str
    token_vector: list[float]
    state_features: Dict[str, float]
    semantic_tags: list[str] = field(default_factory=list)
    media_refs: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "video_state_snapshot_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "objective_preset": self.objective_preset,
            "token_vector": list(self.token_vector),
            "state_features": _float_mapping(self.state_features),
            "semantic_tags": list(self.semantic_tags),
            "media_refs": list(self.media_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class GovernedVideoHypothesis:
    """Candidate future proposed before any rendering/generation step."""

    hypothesis_id: str
    episode_id: str
    mode: str
    action_conditioning: Dict[str, float]
    scores: Dict[str, float]
    semantic_tags: list[str] = field(default_factory=list)
    rationale: str = ""
    render_intent: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "governed_video_hypothesis_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "episode_id": self.episode_id,
            "mode": self.mode,
            "action_conditioning": _float_mapping(self.action_conditioning),
            "scores": _float_mapping(self.scores),
            "semantic_tags": list(self.semantic_tags),
            "rationale": self.rationale,
            "render_intent": _mapping(self.render_intent),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


class GovernedVideoWorldModel:
    """Model-neutral service for evidence-first video-state rollouts."""

    def __init__(self, config: Optional[VideoStateConfig] = None) -> None:
        self.config = config or VideoStateConfig()

    def build_state_snapshot(
        self,
        *,
        episode_id: str,
        timestamp: str,
        belief_state: BeliefState,
        objective_preset: str,
        semantic_tags: Optional[Sequence[Any]] = None,
        media_refs: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        extra_metrics: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VideoStateSnapshot:
        tags = sorted(set(list(belief_state.semantic_tags) + _strings(semantic_tags)))
        state_features = {
            **_float_mapping(belief_state.state_vector),
            **_float_mapping(extra_metrics),
        }
        token_vector = self._build_token_vector(
            tags=tags,
            objective_preset=objective_preset,
            state_features=state_features,
        )
        payload = {
            "episode_id": str(episode_id),
            "timestamp": str(timestamp),
            "objective_preset": str(objective_preset),
            "token_vector": token_vector,
            "state_features": state_features,
            "semantic_tags": tags,
            "media_refs": _strings(media_refs),
            "artifact_refs": {
                **_mapping(belief_state.artifact_refs),
                **_mapping(artifact_refs),
            },
            "provenance": {
                **_mapping(belief_state.provenance),
                **_mapping(provenance),
            },
            "metadata": {
                "belief_id": belief_state.belief_id,
                **_mapping(belief_state.metadata),
                **_mapping(metadata),
            },
            "version": "video_state_snapshot_v1",
        }
        state_id = f"video_state_{sha256_json(payload)[:16]}"
        return VideoStateSnapshot(
            state_id=state_id,
            episode_id=str(episode_id),
            timestamp=str(timestamp),
            objective_preset=str(objective_preset),
            token_vector=token_vector,
            state_features=state_features,
            semantic_tags=tags,
            media_refs=_strings(media_refs),
            artifact_refs={
                **_mapping(belief_state.artifact_refs),
                **_mapping(artifact_refs),
            },
            provenance={
                **_mapping(belief_state.provenance),
                **_mapping(provenance),
            },
            metadata={
                "belief_id": belief_state.belief_id,
                **_mapping(belief_state.metadata),
                **_mapping(metadata),
            },
        )

    def propose_hypotheses(
        self,
        *,
        snapshot: VideoStateSnapshot,
        constraint_set: Optional[Mapping[str, Any]] = None,
        candidate_actions: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> list[GovernedVideoHypothesis]:
        modes = self._candidate_modes(snapshot)
        constraint_pressure = self._constraint_pressure(constraint_set)
        hypotheses: list[GovernedVideoHypothesis] = []
        for idx, mode in enumerate(modes):
            action_conditioning = self._action_conditioning_for_mode(mode, candidate_actions, idx)
            scores = self._score_hypothesis(
                snapshot=snapshot,
                mode=mode,
                constraint_pressure=constraint_pressure,
                action_conditioning=action_conditioning,
            )
            render_intent = {
                "should_render": scores["render_priority"] >= self.config.min_render_priority,
                "geometry_first": True,
                "render_priority": scores["render_priority"],
                "constraint_pressure": constraint_pressure,
            }
            payload = {
                "episode_id": snapshot.episode_id,
                "mode": mode,
                "action_conditioning": action_conditioning,
                "scores": scores,
                "render_intent": render_intent,
            }
            hypothesis_id = f"hyp_{sha256_json(payload)[:16]}"
            hypotheses.append(
                GovernedVideoHypothesis(
                    hypothesis_id=hypothesis_id,
                    episode_id=snapshot.episode_id,
                    mode=mode,
                    action_conditioning=action_conditioning,
                    scores=scores,
                    semantic_tags=list(snapshot.semantic_tags),
                    rationale=self._rationale_for_mode(mode, snapshot),
                    render_intent=render_intent,
                    artifact_refs=_mapping(snapshot.artifact_refs),
                    metadata={"state_id": snapshot.state_id},
                )
            )
        return sorted(
            hypotheses,
            key=lambda hypothesis: (
                hypothesis.scores.get("render_priority", 0.0),
                hypothesis.scores.get("plausibility", 0.0),
            ),
            reverse=True,
        )[: self.config.hypothesis_budget]

    def _build_token_vector(
        self,
        *,
        tags: Sequence[str],
        objective_preset: str,
        state_features: Mapping[str, Any],
    ) -> list[float]:
        token = np.zeros((self.config.token_dim,), dtype=np.float32)
        feature_values = list(_float_mapping(state_features).values())
        base = feature_values[: min(len(feature_values), 16)]
        token[: len(base)] = np.asarray(base, dtype=np.float32)

        offset = min(16, self.config.token_dim)
        objective_vec = _objective_one_hot(objective_preset)
        token[offset: offset + len(objective_vec)] = np.asarray(objective_vec, dtype=np.float32)

        for tag in tags:
            slot = int(_hash_to_unit(tag) * float(max(self.config.token_dim - 1, 1)))
            token[slot] = max(token[slot], 0.25 + 0.75 * _hash_to_unit(f"{tag}:weight"))
        return token.tolist()

    def _candidate_modes(self, snapshot: VideoStateSnapshot) -> list[str]:
        tags = set(snapshot.semantic_tags)
        disagreement = float(snapshot.state_features.get("evidence_disagreement_mean", 0.0))
        modes = ["geometry_guarded_continuation"]
        if disagreement >= 0.2:
            modes.append("semantic_disambiguation")
        if {"fragile", "avoid_collision", "safety"} & tags or snapshot.objective_preset == "safety":
            modes.append("fragile_object_preservation")
        if snapshot.objective_preset == "throughput":
            modes.append("throughput_push")
        if snapshot.objective_preset == "energy_saver":
            modes.append("energy_saver_retiming")
        if "error_recovery" in tags or disagreement >= 0.35:
            modes.append("recovery_branch")
        return modes

    def _constraint_pressure(self, constraint_set: Optional[Mapping[str, Any]]) -> float:
        if not isinstance(constraint_set, Mapping):
            return 0.0
        hard_bounds = constraint_set.get("hard_bounds")
        if not isinstance(hard_bounds, Mapping):
            return 0.0
        return _clip01(float(len(hard_bounds)) / 6.0)

    def _action_conditioning_for_mode(
        self,
        mode: str,
        candidate_actions: Optional[Sequence[Mapping[str, Any]]],
        idx: int,
    ) -> Dict[str, float]:
        if candidate_actions and idx < len(candidate_actions):
            return _float_mapping(candidate_actions[idx])
        defaults = {
            "geometry_guarded_continuation": {"speed_scale": 0.45, "clearance_bias": 0.85},
            "semantic_disambiguation": {"camera_reframe": 1.0, "speed_scale": 0.2},
            "fragile_object_preservation": {"speed_scale": 0.25, "clearance_bias": 1.0},
            "throughput_push": {"speed_scale": 0.9, "clearance_bias": 0.4},
            "energy_saver_retiming": {"speed_scale": 0.3, "smoothness_bias": 0.95},
            "recovery_branch": {"speed_scale": 0.35, "regrasp_bias": 0.9},
        }
        return defaults.get(mode, {"speed_scale": 0.5})

    def _score_hypothesis(
        self,
        *,
        snapshot: VideoStateSnapshot,
        mode: str,
        constraint_pressure: float,
        action_conditioning: Mapping[str, Any],
    ) -> Dict[str, float]:
        features = snapshot.state_features
        confidence = _clip01(features.get("evidence_confidence_mean", 0.0))
        disagreement = _clip01(features.get("evidence_disagreement_mean", 0.0))
        geometry_quality = _clip01(features.get("geometry_quality", confidence))
        semantic_quality = _clip01(features.get("semantic_quality", confidence))
        teacher_alignment = _clip01(features.get("teacher_alignment", 0.0))
        coverage = _clip01(features.get("evidence_coverage", 0.0))

        mode_bias = {
            "geometry_guarded_continuation": 0.05,
            "semantic_disambiguation": 0.18,
            "fragile_object_preservation": 0.14,
            "throughput_push": 0.12,
            "energy_saver_retiming": 0.1,
            "recovery_branch": 0.16,
        }.get(mode, 0.0)
        novelty = _clip01(
            disagreement * 0.55
            + (1.0 - coverage) * 0.2
            + self.config.novelty_bias
            + mode_bias
        )
        plausibility = _clip01(
            geometry_quality * 0.4
            + semantic_quality * 0.2
            + teacher_alignment * 0.15
            + confidence * 0.15
            + (1.0 - constraint_pressure) * 0.1
        )
        objective_fit = _clip01(
            {
                "throughput_push": 0.9 if snapshot.objective_preset == "throughput" else 0.45,
                "energy_saver_retiming": 0.9 if snapshot.objective_preset == "energy_saver" else 0.5,
                "fragile_object_preservation": 0.9 if snapshot.objective_preset == "safety" else 0.65,
                "semantic_disambiguation": 0.7,
                "recovery_branch": 0.75,
            }.get(mode, 0.6)
        )
        action_norm = float(sum(abs(float(value)) for value in _float_mapping(action_conditioning).values()))
        action_pressure = _clip01(action_norm / 4.0)
        render_priority = _clip01(
            novelty * 0.45 + plausibility * 0.35 + objective_fit * 0.2 - action_pressure * 0.1
        )
        return {
            "novelty": novelty,
            "plausibility": plausibility,
            "objective_fit": objective_fit,
            "render_priority": render_priority,
        }

    def _rationale_for_mode(self, mode: str, snapshot: VideoStateSnapshot) -> str:
        disagreement = float(snapshot.state_features.get("evidence_disagreement_mean", 0.0))
        geometry_quality = float(snapshot.state_features.get("geometry_quality", 0.0))
        return (
            f"{mode} selected for objective={snapshot.objective_preset} "
            f"with disagreement={disagreement:.3f} and geometry_quality={geometry_quality:.3f}"
        )


__all__ = [
    "GovernedVideoHypothesis",
    "GovernedVideoWorldModel",
    "VideoStateConfig",
    "VideoStateSnapshot",
]
