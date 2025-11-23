"""
OODTag and RecoveryTag for SIMA-2 hardening.

Implements deterministic firing semantics from
specs/sima2_hardening/SIMA2_INVARIANTS_AND_PHASE_H_HOOKS.md.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class OODTag:
    """Out-Of-Distribution tag for anomaly detection."""
    severity: float  # 0.0-1.0
    source: str  # "visual", "kinematic", "temporal"
    details: Dict[str, Any] = field(default_factory=dict)
    segment_id: Optional[str] = None
    timestep: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_type": "OODTag",
            "severity": float(self.severity),
            "source": self.source,
            "details": dict(self.details),
            "segment_id": self.segment_id,
            "timestep": self.timestep,
        }


@dataclass
class RecoveryTag:
    """Recovery action tag marking failure -> correction -> success patterns."""
    value_add: str  # "low", "medium", "high"
    correction_type: str  # "re-grasp", "pick_from_drop", etc.
    cost_wh: float  # Energy cost of recovery
    segment_id: Optional[str] = None
    timestep_start: Optional[int] = None
    timestep_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_type": "RecoveryTag",
            "value_add": self.value_add,
            "correction_type": self.correction_type,
            "cost_wh": float(self.cost_wh),
            "segment_id": self.segment_id,
            "timestep_start": self.timestep_start,
            "timestep_end": self.timestep_end,
        }


def detect_ood_from_segment(segment: Any, config: Optional[Dict[str, Any]] = None) -> Optional[OODTag]:
    """
    Detect OOD conditions from a segment.

    Branches:
    - Visual OOD (CLIP embedding distance)
    - Kinematic OOD (vel/force beyond p99)
    - Temporal OOD (duration > 3x mean)

    Suppression:
    - Recovery segments
    - Debug mode
    - TrustMatrix[\"OODTag\"].trust_score <= 0.5
    """
    cfg = config or {}
    trust_matrix = cfg.get("trust_matrix", {}) or {}
    trust_score = float(trust_matrix.get("OODTag", {}).get("trust_score", 1.0))
    if trust_score <= 0.5:
        return None

    metadata = getattr(segment, "metadata", {}) if not isinstance(segment, dict) else segment.get("metadata", {}) or {}
    if metadata.get("debug_mode"):
        return None
    if metadata.get("recovery_observed"):
        return None

    # Visual branch
    embedding_distance = float(metadata.get("embedding_distance", metadata.get("clip_embedding_distance", 0.0)))
    visual_threshold = float(cfg.get("visual_threshold", 0.7))
    visual_fired = should_fire_visual_ood(embedding_distance, visual_threshold)
    visual_severity = _severity_visual(embedding_distance) if visual_fired else 0.0

    # Kinematic branch
    joint_velocity = np.asarray(metadata.get("joint_velocity", []), dtype=float) if metadata.get("joint_velocity") is not None else np.asarray([])
    force_val = float(metadata.get("force", metadata.get("contact_force", 0.0)))
    training_stats = cfg.get("training_stats", {"vel_p99": 1.0, "force_p99": 10.0})
    kin_fired = should_fire_kinematic_ood(joint_velocity, force_val, training_stats)
    kin_severity = _severity_kinematic(joint_velocity, force_val, training_stats) if kin_fired else 0.0

    # Temporal branch
    duration = int(metadata.get("duration", max(0, getattr(segment, "end_t", 0) - getattr(segment, "start_t", 0))))
    primitive_type = str(getattr(segment, "label", metadata.get("label", metadata.get("primitive_type", "unknown"))))
    mean_durations = cfg.get("mean_durations", {})
    temporal_fired = should_fire_temporal_ood(duration, primitive_type, mean_durations)
    temporal_severity = _severity_temporal(duration, primitive_type, mean_durations) if temporal_fired else 0.0

    # Choose the highest severity source deterministically
    severities = [
        ("visual", visual_severity, {"embedding_distance": embedding_distance, "threshold": visual_threshold}),
        ("kinematic", kin_severity, {"force": force_val, "training_stats": training_stats}),
        ("temporal", temporal_severity, {"duration": duration, "mean_durations": mean_durations}),
    ]
    severities.sort(key=lambda s: s[1], reverse=True)
    top_source, top_sev, details = severities[0]

    if top_sev <= 0.0:
        return None

    seg_id = getattr(segment, "segment_id", None) if not isinstance(segment, dict) else segment.get("segment_id")
    timestep = getattr(segment, "start_t", None) if not isinstance(segment, dict) else segment.get("start_t")
    return OODTag(
        severity=float(min(1.0, top_sev)),
        source=top_source,
        details=details,
        segment_id=seg_id,
        timestep=timestep,
    )


def detect_recovery_from_segments(segments: Sequence[Any], config: Optional[Dict[str, Any]] = None) -> List[RecoveryTag]:
    """
    Detect recovery patterns from segment sequence.

    Pattern: failure → recovery → success/recovered.
    Suppression:
    - No prior failure
    - Recovery fails
    - TrustMatrix[\"RecoveryTag\"].trust_score < 0.5
    """
    cfg = config or {}
    trust_matrix = cfg.get("trust_matrix", {}) or {}
    trust_score = float(trust_matrix.get("RecoveryTag", {}).get("trust_score", 1.0))
    if trust_score < 0.5:
        return []

    recovery_tags: List[RecoveryTag] = []

    def _meta(seg: Any) -> Dict[str, Any]:
        if isinstance(seg, dict):
            return seg.get("metadata", {}) or {}
        return getattr(seg, "metadata", {}) or {}

    def _outcome(seg: Any) -> str:
        if isinstance(seg, dict):
            return str(seg.get("outcome", seg.get("status", "")))
        return str(getattr(seg, "outcome", ""))

    for i in range(len(segments)):
        if not should_fire_recovery_tag(segments, i):
            continue
        curr = segments[i]
        prev = segments[i - 1]
        nxt = segments[i + 1] if i + 1 < len(segments) else None

        prev_meta = _meta(prev)
        curr_meta = _meta(curr)
        nxt_outcome = _outcome(nxt) if nxt else "success"
        final_success = nxt_outcome in {"success", "recovered"}
        failure_sev = str(prev_meta.get("failure_severity", prev_meta.get("severity", "medium")))
        recovery_duration = int(curr_meta.get("duration", max(0, getattr(curr, "end_t", 0) - getattr(curr, "start_t", 0))))
        if isinstance(curr, dict):
            correction_type = curr.get("label", curr_meta.get("label", curr_meta.get("primitive_type", "unknown")))
        else:
            correction_type = getattr(curr, "label", curr_meta.get("label", curr_meta.get("primitive_type", "unknown")))
        value_add = classify_recovery_value(failure_sev, recovery_duration, final_success)
        cost_wh = estimate_recovery_cost_wh(
            recovery_duration,
            primitive_type=correction_type,
            base_power_per_timestep=float(cfg.get("base_power_per_timestep", 0.5)),
        )

        seg_id = getattr(curr, "segment_id", None) if not isinstance(curr, dict) else curr.get("segment_id")
        start_t = getattr(curr, "start_t", None) if not isinstance(curr, dict) else curr.get("start_t")
        end_t = getattr(curr, "end_t", None) if not isinstance(curr, dict) else curr.get("end_t")
        recovery_tags.append(
            RecoveryTag(
                value_add=value_add,
                correction_type=correction_type,
                cost_wh=float(cost_wh),
                segment_id=seg_id,
                timestep_start=start_t,
                timestep_end=end_t,
            )
        )

    return recovery_tags


# ==== Spec helpers ========================================================

def should_fire_visual_ood(embedding_distance: float, threshold: float = 0.7) -> bool:
    """Fire if CLIP embedding distance from training centroid exceeds threshold."""
    return float(embedding_distance) > float(threshold)


def should_fire_kinematic_ood(joint_velocity: np.ndarray, force: float, training_stats: Dict[str, float]) -> bool:
    """Fire if joint velocities or contact forces exceed 99th percentile of training data."""
    vel_ood = False
    if joint_velocity.size > 0:
        vel_ood = float(np.max(np.abs(joint_velocity))) > float(training_stats.get("vel_p99", 1.0))
    force_ood = float(force) > float(training_stats.get("force_p99", 10.0))
    return bool(vel_ood or force_ood)


def should_fire_temporal_ood(segment_duration: int, primitive_type: str, mean_durations: Dict[str, float]) -> bool:
    """Fire if segment duration exceeds 3x mean duration for the primitive type."""
    expected_mean = float(mean_durations.get(primitive_type, 10.0))
    return float(segment_duration) > 3.0 * expected_mean


def _severity_visual(distance: float) -> float:
    if distance >= 0.95:
        return 1.0
    if distance > 0.85:
        return 0.8
    if distance > 0.7:
        return 0.5
    return 0.0


def _severity_kinematic(joint_velocity: np.ndarray, force: float, training_stats: Dict[str, float]) -> float:
    vel_p99 = float(training_stats.get("vel_p99", 1.0))
    force_p99 = float(training_stats.get("force_p99", 10.0))
    severity_candidates = []

    if joint_velocity.size > 0:
        max_vel = float(np.max(np.abs(joint_velocity)))
        if max_vel > vel_p99 * 1.5:
            severity_candidates.append(1.0)
        elif max_vel > vel_p99 * 1.25:
            severity_candidates.append(0.8)
        elif max_vel > vel_p99:
            severity_candidates.append(0.6)

    if force > force_p99 * 1.5:
        severity_candidates.append(1.0)
    elif force > force_p99 * 1.25:
        severity_candidates.append(0.8)
    elif force > force_p99:
        severity_candidates.append(0.6)

    return max(severity_candidates) if severity_candidates else 0.0


def _severity_temporal(duration: int, primitive_type: str, mean_durations: Dict[str, float]) -> float:
    expected_mean = float(mean_durations.get(primitive_type, 10.0))
    ratio = float(duration) / max(expected_mean, 1e-6)
    if ratio > 6.0:
        return 1.0
    if ratio > 4.0:
        return 0.7
    if ratio > 3.0:
        return 0.5
    return 0.0


def classify_recovery_value(failure_severity: str, recovery_duration: int, final_success: bool) -> str:
    """Classify recovery value-add per spec."""
    sev = str(failure_severity).lower()
    if not final_success:
        return "low"
    if sev == "critical" and recovery_duration < 10:
        return "high"
    if sev in {"high", "critical"}:
        return "medium"
    return "low"


def estimate_recovery_cost_wh(recovery_duration: int, primitive_type: str, base_power_per_timestep: float = 0.5) -> float:
    """Estimate Wh consumed during recovery."""
    multipliers = {
        "regrasp": 1.0,
        "pick_from_drop": 1.5,
        "wiggle_pull": 0.8,
    }
    mult = float(multipliers.get(str(primitive_type), 1.0))
    return float(recovery_duration) * float(base_power_per_timestep) * mult


def should_fire_recovery_tag(segments: Sequence[Any], i: int) -> bool:
    """Check failure -> correction -> success/recovered pattern."""
    if i == 0 or i >= len(segments):
        return False

    def _outcome(seg: Any) -> str:
        if isinstance(seg, dict):
            return str(seg.get("outcome", seg.get("status", "")))
        return str(getattr(seg, "outcome", ""))

    def _meta(seg: Any) -> Dict[str, Any]:
        if isinstance(seg, dict):
            return seg.get("metadata", {}) or {}
        return getattr(seg, "metadata", {}) or {}

    prev_failed = _outcome(segments[i - 1]) == "failure"
    curr_is_recovery = bool(_meta(segments[i]).get("recovery_observed", False))
    if i + 1 < len(segments):
        next_succeeds = _outcome(segments[i + 1]) in {"success", "recovered"}
    else:
        next_succeeds = True
    return prev_failed and curr_is_recovery and next_succeeds
