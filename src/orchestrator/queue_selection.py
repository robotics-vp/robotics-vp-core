"""Additive live queue-selection shim that consumes advisory outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

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


class QueueDispatchMode(str, Enum):
    """Bounded influence stages for queue-driven dispatch."""

    DISABLED = "disabled"
    LOG_ONLY = "log_only"
    COMPARE_ONLY = "compare_only"
    ADVISORY_REORDER = "advisory_reorder"
    BOUNDED_REWEIGHT = "bounded_reweight"
    PROMOTED_GATE_ELIGIBLE = "promoted_gate_eligible"


@dataclass(frozen=True)
class QueueDispatchConfig:
    """Configurable bounded dispatch behavior for training-time selection."""

    mode: QueueDispatchMode | str = QueueDispatchMode.LOG_ONLY
    max_upweight: float = 2.0
    max_downweight: float = 0.5
    allow_slice_removal_on_integrity_failure: bool = False
    severe_integrity_actions: tuple[str, ...] = ("deny_shadow", "suppress")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": _normalize_mode(self.mode).value,
            "max_upweight": float(self.max_upweight),
            "max_downweight": float(self.max_downweight),
            "allow_slice_removal_on_integrity_failure": bool(self.allow_slice_removal_on_integrity_failure),
            "severe_integrity_actions": list(self.severe_integrity_actions),
        }


@dataclass(frozen=True)
class QueueDispatchDecision:
    """Dispatch decision for one queue candidate."""

    episode_id: str
    queue_name: str
    original_rank: int
    adjusted_rank: int
    priority_score: float
    replay_action: str
    tags: list[str]
    base_weight: float
    adjusted_weight: float
    reweight_factor: float
    dropped: bool
    promotion_stage: str
    influence_source: str
    reasons: list[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "queue_name": self.queue_name,
            "original_rank": int(self.original_rank),
            "adjusted_rank": int(self.adjusted_rank),
            "priority_score": float(self.priority_score),
            "replay_action": self.replay_action,
            "tags": list(self.tags),
            "base_weight": float(self.base_weight),
            "adjusted_weight": float(self.adjusted_weight),
            "reweight_factor": float(self.reweight_factor),
            "dropped": bool(self.dropped),
            "promotion_stage": self.promotion_stage,
            "influence_source": self.influence_source,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
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
                    "signal_yield_score": episode.get("signal_yield_score"),
                    "inferential_signal_yield": episode.get("inferential_signal_yield"),
                    "inferential_budget_decision": episode.get("inferential_budget_decision"),
                    "promotion_stage": _episode_promotion_stage(episode),
                    "promotion_stages": _episode_promotion_stages(episode),
                    "influence_source": _episode_influence_source(episode),
                    "evidence": {
                        "sampling_recommendation": dict(episode.get("sampling_recommendation", {}) or {}),
                        "receipt_feedback": dict(episode.get("receipt_feedback", {}) or {}),
                        "inferential_signal_yield": episode.get("inferential_signal_yield"),
                        "inferential_reward": (
                            dict(
                                (episode.get("inferential_budget_decision", {}) or {})
                                .get("artifact_summary", {})
                                .get("inferential_reward", {})
                                or {}
                            )
                        ),
                        "advisor_modes": _episode_advisor_modes(episode),
                    },
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


def apply_live_queue_selection(
    episodes: Sequence[Mapping[str, Any]],
    *,
    live_queue_selection: Optional[Mapping[str, Any] | str | Path] = None,
    base_weights: Optional[Mapping[str, float]] = None,
    config: QueueDispatchConfig | None = None,
) -> Dict[str, Any]:
    """Apply bounded queue influence to an episode pool without mutating reward math."""
    dispatch_config = config or QueueDispatchConfig()
    mode = _normalize_mode(dispatch_config.mode)
    if mode == QueueDispatchMode.DISABLED:
        entries = [
            QueueDispatchDecision(
                episode_id=_episode_id(row, index),
                queue_name="disabled",
                original_rank=index,
                adjusted_rank=index,
                priority_score=0.0,
                replay_action="holdout",
                tags=[],
                base_weight=float(_base_weight(row, base_weights)),
                adjusted_weight=float(_base_weight(row, base_weights)),
                reweight_factor=1.0,
                dropped=False,
                promotion_stage="compare_only",
                influence_source="heuristic",
                reasons=["queue_dispatch_disabled"],
                evidence={},
                metadata={},
            ).to_dict()
            for index, row in enumerate(episodes)
        ]
        return {
            "mode": mode.value,
            "entries": entries,
            "ordered_episode_ids": [row["episode_id"] for row in entries],
            "original_queue_order": [row["episode_id"] for row in entries],
            "adjusted_queue_order": [row["episode_id"] for row in entries],
            "reweight_factors": {row["episode_id"]: 1.0 for row in entries},
            "summary": {
                "num_entries": len(entries),
                "num_reweighted": 0,
                "num_dropped": 0,
                "queue_digest": sha256_json(entries),
            },
        }

    queue_payload = _load_queue_payload(live_queue_selection)
    queue_entries = {
        str(entry.get("episode_id", "")): dict(entry)
        for entry in list(queue_payload.get("entries", []) or [])
    }
    decisions: list[QueueDispatchDecision] = []
    for original_rank, episode in enumerate(episodes):
        episode_id = _episode_id(episode, original_rank)
        queue_entry = queue_entries.get(episode_id, {})
        base_weight = float(_base_weight(episode, base_weights))
        priority_score = float(queue_entry.get("priority_score", 0.0) or 0.0)
        replay_action = str(queue_entry.get("replay_action", "holdout") or "holdout")
        tags = [str(value) for value in queue_entry.get("tags", []) or []]
        metadata = dict(queue_entry.get("metadata", {}) or {})
        evidence = dict(metadata.get("evidence", {}) or {})
        adjusted_weight = base_weight
        reasons: list[str] = []
        dropped = False
        promotion_stage = str(metadata.get("promotion_stage", "compare_only") or "compare_only")
        influence_source = str(metadata.get("influence_source", "heuristic") or "heuristic")

        if mode in {QueueDispatchMode.BOUNDED_REWEIGHT, QueueDispatchMode.PROMOTED_GATE_ELIGIBLE}:
            multiplier = _bounded_multiplier(
                priority_score=priority_score,
                replay_action=replay_action,
                tags=tags,
                max_upweight=dispatch_config.max_upweight,
                max_downweight=dispatch_config.max_downweight,
            )
            adjusted_weight = base_weight * multiplier
            if abs(multiplier - 1.0) > 1e-6:
                reasons.append("bounded_reweight_applied")
        else:
            multiplier = 1.0

        if (
            mode == QueueDispatchMode.PROMOTED_GATE_ELIGIBLE
            and dispatch_config.allow_slice_removal_on_integrity_failure
            and _gate_stage_allows_slice_removal(promotion_stage)
            and _severe_integrity_failure(metadata, dispatch_config.severe_integrity_actions)
        ):
            dropped = True
            adjusted_weight = 0.0
            reasons.append("severe_integrity_drop")

        if mode == QueueDispatchMode.LOG_ONLY and not reasons:
            reasons.append("log_only_no_dispatch_change")
        elif mode == QueueDispatchMode.COMPARE_ONLY and not reasons:
            reasons.append("compare_only_legacy_alias")
        elif mode == QueueDispatchMode.ADVISORY_REORDER:
            reasons.append("advisory_reorder")
        elif mode == QueueDispatchMode.PROMOTED_GATE_ELIGIBLE and not reasons:
            reasons.append("promoted_gate_eligible_no_drop")

        decisions.append(
            QueueDispatchDecision(
                episode_id=episode_id,
                queue_name=str(queue_payload.get("queue_name", "shadow_advisory_queue")),
                original_rank=original_rank,
                adjusted_rank=original_rank,
                priority_score=priority_score,
                replay_action=replay_action,
                tags=tags,
                base_weight=base_weight,
                adjusted_weight=adjusted_weight,
                reweight_factor=(adjusted_weight / base_weight) if abs(base_weight) > 1e-9 else 1.0,
                dropped=dropped,
                promotion_stage=promotion_stage,
                influence_source=influence_source,
                reasons=reasons,
                evidence=evidence,
                metadata={
                    **metadata,
                    "mode": mode.value,
                    "base_weight_source": "explicit" if base_weights else "descriptor",
                },
            )
        )

    ordered = sorted(
        decisions,
        key=lambda decision: _dispatch_sort_key(decision, mode),
    )
    finalized: list[QueueDispatchDecision] = []
    for adjusted_rank, decision in enumerate(ordered):
        finalized.append(
            QueueDispatchDecision(
                episode_id=decision.episode_id,
                queue_name=decision.queue_name,
                original_rank=decision.original_rank,
                adjusted_rank=adjusted_rank,
                priority_score=decision.priority_score,
                replay_action=decision.replay_action,
                tags=list(decision.tags),
                base_weight=decision.base_weight,
                adjusted_weight=decision.adjusted_weight,
                reweight_factor=decision.reweight_factor,
                dropped=decision.dropped,
                promotion_stage=decision.promotion_stage,
                influence_source=decision.influence_source,
                reasons=list(decision.reasons),
                evidence=dict(decision.evidence),
                metadata=dict(decision.metadata),
            )
        )
    entries = [decision.to_dict() for decision in finalized]
    original_queue_order = [
        decision.episode_id
        for decision in sorted(finalized, key=lambda row: (row.original_rank, row.episode_id))
        if not decision.dropped
    ]
    adjusted_queue_order = [
        decision.episode_id
        for decision in sorted(finalized, key=lambda row: (row.adjusted_rank, row.episode_id))
        if not decision.dropped
    ]
    payload = {
        "mode": mode.value,
        "entries": entries,
        "ordered_episode_ids": adjusted_queue_order,
        "original_queue_order": original_queue_order,
        "adjusted_queue_order": adjusted_queue_order,
        "reweight_factors": {
            decision.episode_id: float(decision.reweight_factor)
            for decision in finalized
        },
        "summary": {
            "num_entries": len(finalized),
            "num_reweighted": sum(
                1 for decision in finalized
                if abs(decision.adjusted_weight - decision.base_weight) > 1e-6
            ),
            "num_dropped": sum(1 for decision in finalized if decision.dropped),
            "queue_digest": sha256_json(entries),
        },
    }
    return payload


def _load_queue_payload(
    live_queue_selection: Optional[Mapping[str, Any] | str | Path],
) -> Dict[str, Any]:
    if live_queue_selection is None:
        return {"queue_name": "shadow_advisory_queue", "entries": []}
    if isinstance(live_queue_selection, (str, Path)):
        payload = json.loads(Path(live_queue_selection).read_text(encoding="utf-8"))
        return dict(payload or {})
    return dict(live_queue_selection or {})


def _episode_id(row: Mapping[str, Any], index: int) -> str:
    descriptor = dict(row.get("descriptor", {}) or {})
    return str(
        descriptor.get("pack_id")
        or descriptor.get("episode_id")
        or row.get("episode_id")
        or row.get("pack_id")
        or f"episode_{index:04d}"
    )


def _base_weight(row: Mapping[str, Any], base_weights: Optional[Mapping[str, float]]) -> float:
    episode_id = _episode_id(row, 0)
    if base_weights is not None and episode_id in base_weights:
        return float(base_weights[episode_id])
    descriptor = dict(row.get("descriptor", {}) or {})
    return float(descriptor.get("sampling_weight", row.get("sampling_weight", 1.0)) or 1.0)


def _bounded_multiplier(
    *,
    priority_score: float,
    replay_action: str,
    tags: Sequence[str],
    max_upweight: float,
    max_downweight: float,
) -> float:
    multiplier = 1.0
    normalized_score = max(0.0, min(1.0, float(priority_score)))
    positive_tags = {
        "high_value_uncertain",
        "frontier_candidate",
        "collect_more_like_this",
        "pricing_truth_review",
        "pricing_review",
        "reward_safety_review",
    }
    negative_tags = {
        "low_provenance_review",
        "low_provenance",
        "downweight_candidate",
        "holdout_candidate",
    }
    if replay_action in {"upweight", "collect_more_like_this"}:
        multiplier += 0.5 * normalized_score
    elif replay_action == "downweight":
        multiplier -= 0.5 * normalized_score
    elif replay_action == "holdout":
        multiplier -= 0.15 * normalized_score

    multiplier += 0.05 * sum(1 for tag in tags if tag in positive_tags)
    multiplier -= 0.08 * sum(1 for tag in tags if tag in negative_tags)
    return max(float(max_downweight), min(float(max_upweight), multiplier))


def _severe_integrity_failure(metadata: Mapping[str, Any], severe_integrity_actions: Sequence[str]) -> bool:
    deploy_recommendation = str(metadata.get("deploy_recommendation", "") or "")
    pricing_recommendation = str(metadata.get("pricing_recommendation", "") or "")
    datapack_recommendation = str(metadata.get("datapack_recommendation", "") or "")
    return (
        deploy_recommendation in severe_integrity_actions
        or pricing_recommendation in severe_integrity_actions
        or datapack_recommendation in {"review", "deny", "suppress"}
    )


def _dispatch_sort_key(
    decision: QueueDispatchDecision,
    mode: QueueDispatchMode,
) -> tuple[Any, ...]:
    if mode == QueueDispatchMode.ADVISORY_REORDER:
        return (
            decision.dropped,
            -decision.priority_score,
            -decision.base_weight,
            decision.episode_id,
        )
    if mode in {QueueDispatchMode.BOUNDED_REWEIGHT, QueueDispatchMode.PROMOTED_GATE_ELIGIBLE}:
        return (
            decision.dropped,
            -decision.adjusted_weight,
            -decision.priority_score,
            decision.episode_id,
        )
    return (decision.dropped, decision.original_rank, decision.episode_id)


def _normalize_mode(value: QueueDispatchMode | str) -> QueueDispatchMode:
    if isinstance(value, QueueDispatchMode):
        if value == QueueDispatchMode.COMPARE_ONLY:
            return QueueDispatchMode.LOG_ONLY
        return value
    raw = str(value or QueueDispatchMode.LOG_ONLY.value).strip().lower()
    if raw == QueueDispatchMode.COMPARE_ONLY.value:
        return QueueDispatchMode.LOG_ONLY
    return QueueDispatchMode(raw)


def _episode_advisor_modes(episode: Mapping[str, Any]) -> Dict[str, str]:
    advisor_modes: Dict[str, str] = {}
    for key in (
        "policy_advisor",
        "pricing_advisor",
        "data_value_advisor",
        "regal_support_advisor",
    ):
        payload = dict(episode.get(key, {}) or {})
        if payload.get("mode"):
            advisor_modes[key] = str(payload["mode"])
    return advisor_modes


def _episode_promotion_stages(episode: Mapping[str, Any]) -> Dict[str, str]:
    stages: Dict[str, str] = {}
    for key in (
        "policy_advisor",
        "pricing_advisor",
        "data_value_advisor",
        "regal_support_advisor",
    ):
        payload = dict(episode.get(key, {}) or {})
        guard = dict(payload.get("promotion_guard", {}) or {})
        stage = guard.get("stage")
        if stage:
            stages[key] = str(stage)
    return stages


def _episode_promotion_stage(episode: Mapping[str, Any]) -> str:
    stages = list(_episode_promotion_stages(episode).values())
    if not stages:
        return "compare_only"
    order = {
        "compare_only": 0,
        "advisory": 1,
        "budget_gate": 2,
        "narrow_hard_gate": 3,
    }
    return max(stages, key=lambda stage: order.get(str(stage), -1))


def _episode_influence_source(episode: Mapping[str, Any]) -> str:
    learned_modes = {
        "learned_only",
        "heuristic_learned_residual",
        "heuristic_plus_learned_residual",
        "learned_compare_only",
        "learned_shadow_only",
        "promoted_gate_eligible",
    }
    hybrid_modes = {
        "heuristic_learned_residual",
        "heuristic_plus_learned_residual",
        "heuristic_learned_compare_only",
    }
    modes = [value for value in _episode_advisor_modes(episode).values() if value]
    if any(mode in hybrid_modes for mode in modes):
        return "hybrid"
    if any(mode in learned_modes for mode in modes):
        return "learned"
    return "heuristic"


def _gate_stage_allows_slice_removal(stage: str) -> bool:
    return str(stage or "").lower() in {"budget_gate", "narrow_hard_gate"}
