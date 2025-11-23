"""
OODTag and RecoveryTag for SIMA-2 hardening.

Per SIMA2_OOD_AND_RECOVERY_SPEC.md:
- OODTag: Marks segments deviating from training distribution
- RecoveryTag: Marks successful recovery from failure states
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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

    Heuristics:
    - Kinematic: Unusual velocity/force patterns
    - Temporal: Segment duration >> expected
    """
    cfg = config or {}
    temporal_threshold = float(cfg.get("ood_temporal_threshold", 3.0))  # 3x mean duration

    metadata = getattr(segment, 'metadata', {}) if not isinstance(segment, dict) else segment.get('metadata', {})
    duration = metadata.get('duration', 0)

    # Temporal OOD: segment too long
    if duration > temporal_threshold * 10:  # Assuming 10 is baseline mean
        return OODTag(
            severity=min(1.0, duration / (temporal_threshold * 10)),
            source="temporal",
            details={"duration": duration, "threshold": temporal_threshold * 10},
            segment_id=segment.segment_id if hasattr(segment, 'segment_id') else segment.get('segment_id'),
        )

    return None


def detect_recovery_from_segments(segments: list, config: Optional[Dict[str, Any]] = None) -> list:
    """
    Detect recovery patterns from segment sequence.

    Pattern: failure segment → recovery segment → success segment
    """
    cfg = config or {}
    recovery_tags = []

    for i in range(len(segments) - 1):
        curr = segments[i]
        next_seg = segments[i + 1]

        # Check for failure → recovery pattern
        curr_meta = getattr(curr, 'metadata', {}) if not isinstance(curr, dict) else curr.get('metadata', {})
        next_meta = getattr(next_seg, 'metadata', {}) if not isinstance(next_seg, dict) else next_seg.get('metadata', {})

        curr_failure = curr_meta.get('failure_observed', False)
        next_recovery = next_meta.get('recovery_observed', False)

        if curr_failure and next_recovery:
            # Estimate value-add based on whether recovery succeeded
            value_add = "high" if next_seg.outcome == "recovered" else "medium"

            # Extract correction type from label
            correction_type = next_seg.label if hasattr(next_seg, 'label') else next_seg.get('label', 'unknown')

            # Estimate cost (mock for now)
            duration = next_meta.get('duration', 5)
            cost_wh = duration * 0.5  # Mock: 0.5 Wh per timestep

            recovery_tags.append(RecoveryTag(
                value_add=value_add,
                correction_type=correction_type,
                cost_wh=cost_wh,
                segment_id=next_seg.segment_id if hasattr(next_seg, 'segment_id') else next_seg.get('segment_id'),
                timestep_start=next_seg.start_t if hasattr(next_seg, 'start_t') else next_seg.get('start_t'),
                timestep_end=next_seg.end_t if hasattr(next_seg, 'end_t') else next_seg.get('end_t'),
            ))

    return recovery_tags
