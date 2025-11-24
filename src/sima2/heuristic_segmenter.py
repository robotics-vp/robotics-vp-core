"""
Heuristic-based segmentation from raw robot state.

Detects semantic primitives from physical signals with deterministic rules:
- Velocity thresholds:
  * low velocity + gripper open  -> approach
  * high velocity + contact      -> pull
- Contact state changes:
  * start_contact/end_contact    -> hard segment boundary
- Gripper width transitions:
  * width decreasing             -> grasp
  * width increasing             -> release
- Temporal decay window prevents micro-segments
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PhysicsSignal:
    """Extracted physics signals from a timestep."""
    timestep: int
    gripper_width: float
    ee_velocity: float
    contact: bool
    force: Optional[float] = None
    object_name: Optional[str] = None


@dataclass
class DetectedPrimitive:
    """A detected semantic primitive from raw state."""
    start_t: int
    end_t: int
    label: str
    object_name: str
    confidence: float
    failure: bool = False
    recovery: bool = False
    metadata: Optional[Dict[str, Any]] = None


class HeuristicSegmenter:
    """
    Detects segment boundaries from robot_state and events.

    Heuristics:
    1. Gripper close (width < threshold) -> grasp
    2. Gripper open (width > threshold) -> release
    3. Contact True -> contact/manipulation phase
    4. Velocity > threshold -> approach/move
    5. Velocity ~ 0 -> static
    6. Event markers -> failure/recovery boundaries
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.gripper_close_threshold = float(cfg.get("gripper_close_threshold", 0.04))
        self.gripper_open_threshold = float(cfg.get("gripper_open_threshold", 0.06))
        self.velocity_low_threshold = float(cfg.get("velocity_low_threshold", cfg.get("velocity_threshold", 0.02)))
        self.velocity_high_threshold = float(cfg.get("velocity_high_threshold", cfg.get("velocity_threshold", 0.08)))
        self.width_delta_threshold = float(cfg.get("width_delta_threshold", 0.005))
        self.temporal_decay_window = int(cfg.get("temporal_decay_window", 2))
        self.min_segment_duration = int(cfg.get("min_segment_duration", 2))

    def segment(self, rollout: Dict[str, Any]) -> List[DetectedPrimitive]:
        """
        Extract primitives from rollout's primitives + raw signals.

        Returns:
            List of DetectedPrimitive objects
        """
        events = rollout.get("events", []) or []
        # Extract physics signals from any available source; we intentionally
        # ignore client-provided primitive labels.
        signals = self._extract_signals(rollout)

        # Detect boundaries using heuristics
        detected = self._detect_primitives_from_signals(signals, events)

        return detected

    def _extract_signals(self, rollout: Dict[str, Any]) -> List[PhysicsSignal]:
        """Extract physics signals from rollout primitives or metadata."""
        primitive_sources = (
            rollout.get("primitives")
            or rollout.get("primitive_events")
            or rollout.get("metadata", {}).get("semantic_primitives")
            or []
        )
        signals: List[PhysicsSignal] = []
        for p in primitive_sources:
            if not isinstance(p, dict):
                continue
            sig = PhysicsSignal(
                timestep=int(p.get("timestep", len(signals))),
                gripper_width=float(p.get("gripper_width", rollout.get("gripper_width", 0.08) or 0.08)),
                ee_velocity=float(p.get("ee_velocity", p.get("velocity", rollout.get("ee_velocity", 0.0) or 0.0))),
                contact=bool(p.get("contact", False)),
                force=p.get("force"),
                object_name=str(p.get("object") or p.get("object_name") or "unknown"),
            )
            signals.append(sig)
        return sorted(signals, key=lambda s: s.timestep)

    def _detect_primitives_from_signals(
        self, signals: List[PhysicsSignal], events: List[Dict[str, Any]]
    ) -> List[DetectedPrimitive]:
        """
        Main detection logic.

        State machine:
        1. Track gripper width deltas for grasp/release
        2. Track contact transitions for boundaries
        3. Velocity-based labeling (approach/pull)
        4. Temporal decay window suppresses micro-segments
        """
        if not signals:
            return []

        primitives: List[DetectedPrimitive] = []
        event_lookup = {int(e.get("timestep", -1)): e for e in events if isinstance(e, dict)}

        prev_width = signals[0].gripper_width
        prev_contact = signals[0].contact
        current_segment_start = signals[0].timestep
        current_object = signals[0].object_name
        current_label = self._label_for_signal(signals[0], None, prev_contact, None)
        last_switch_t = current_segment_start

        for sig in signals[1:]:
            candidate_label = self._label_for_signal(sig, prev_width, prev_contact, current_label)
            contact_changed = sig.contact != prev_contact
            label_changed = candidate_label != current_label
            should_split = contact_changed or label_changed
            since_switch = sig.timestep - last_switch_t

            # Temporal decay window suppresses rapid oscillations unless contact toggles
            if should_split and not contact_changed and since_switch < self.temporal_decay_window:
                candidate_label = current_label
                label_changed = False
                should_split = False

            if should_split:
                end_t = sig.timestep
                duration = end_t - current_segment_start
                if duration >= 1 and (duration >= self.min_segment_duration or contact_changed):
                    primitives.append(
                        self._build_detected_primitive(
                            start_t=current_segment_start,
                            end_t=end_t,
                            label=current_label,
                            object_name=current_object,
                            contact_state=prev_contact,
                            event_lookup=event_lookup,
                        )
                    )
                    current_segment_start = sig.timestep
                    last_switch_t = sig.timestep
                current_label = candidate_label

            if sig.object_name and sig.object_name != "unknown":
                current_object = sig.object_name
            prev_width = sig.gripper_width
            prev_contact = sig.contact

        # Finalize last segment (inclusive of final timestep)
        last_sig = signals[-1]
        end_t = last_sig.timestep + 1
        duration = end_t - current_segment_start
        if duration >= 1:
            primitives.append(
                self._build_detected_primitive(
                    start_t=current_segment_start,
                    end_t=end_t,
                    label=current_label,
                    object_name=current_object,
                    contact_state=prev_contact,
                    event_lookup=event_lookup,
                )
            )

        return primitives

    def _label_for_signal(
        self,
        sig: PhysicsSignal,
        prev_width: Optional[float],
        prev_contact: bool,
        current_label: Optional[str],
    ) -> str:
        width_delta = 0.0 if prev_width is None else float(prev_width - sig.gripper_width)
        if prev_width is not None and width_delta > self.width_delta_threshold:
            return "grasp"
        if prev_width is not None and width_delta < -self.width_delta_threshold:
            return "release"
        if sig.contact and sig.ee_velocity >= self.velocity_high_threshold:
            return "pull"
        if not sig.contact and sig.ee_velocity <= self.velocity_low_threshold and sig.gripper_width >= self.gripper_open_threshold:
            return "approach"
        if sig.contact and not prev_contact:
            return "contact"
        if not sig.contact and prev_contact:
            return "depart"
        if sig.contact:
            return current_label or "manipulate"
        return current_label or "move"

    def _build_detected_primitive(
        self,
        start_t: int,
        end_t: int,
        label: str,
        object_name: str,
        contact_state: bool,
        event_lookup: Dict[int, Any],
    ) -> DetectedPrimitive:
        failure = self._check_failure(start_t, end_t, event_lookup)
        recovery = self._check_recovery(start_t, end_t, event_lookup)
        metadata = {
            "duration": end_t - start_t,
            "contact": contact_state,
        }
        confidence = 0.95 if contact_state else 0.85
        return DetectedPrimitive(
            start_t=start_t,
            end_t=end_t,
            label=label,
            object_name=object_name,
            confidence=confidence,
            failure=failure,
            recovery=recovery,
            metadata=metadata,
        )

    def _check_failure(self, start_t: int, end_t: int, event_lookup: Dict[int, Any]) -> bool:
        """Check if this segment contains a failure event."""
        for t in range(start_t, end_t):
            event = event_lookup.get(t)
            if event and event.get("event_type") == "failure":
                return True
        return False

    def _check_recovery(self, start_t: int, end_t: int, event_lookup: Dict[int, Any]) -> bool:
        """Check if this segment contains a recovery event."""
        for t in range(start_t, end_t):
            event = event_lookup.get(t)
            if event and event.get("event_type") in ["recovery_start", "recovery_complete"]:
                return True
        return False
