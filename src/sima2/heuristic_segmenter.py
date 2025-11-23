"""
Heuristic-based segmentation from raw robot state.

Detects semantic primitives from physical signals:
- Gripper width changes -> grasp/release
- Contact changes -> approach/contact/lift
- Velocity patterns -> move/static
- Event markers -> failure/recovery
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
        self.velocity_threshold = float(cfg.get("velocity_threshold", 0.05))
        self.min_segment_duration = int(cfg.get("min_segment_duration", 2))

    def segment(self, rollout: Dict[str, Any]) -> List[DetectedPrimitive]:
        """
        Extract primitives from rollout's primitives + raw signals.

        Returns:
            List of DetectedPrimitive objects
        """
        primitives_raw = rollout.get("primitives", [])
        events = rollout.get("events", [])

        # Extract physics signals from primitives (which now contain raw state)
        signals = self._extract_signals(primitives_raw)

        # Detect boundaries using heuristics
        detected = self._detect_primitives_from_signals(signals, events)

        return detected

    def _extract_signals(self, primitives: List[Dict[str, Any]]) -> List[PhysicsSignal]:
        """Extract physics signals from primitive dicts."""
        signals = []
        for p in primitives:
            sig = PhysicsSignal(
                timestep=p.get("timestep", 0),
                gripper_width=float(p.get("gripper_width", 0.08)),
                ee_velocity=float(p.get("ee_velocity", 0.0)),
                contact=bool(p.get("contact", False)),
                force=p.get("force"),
                object_name=p.get("object", "unknown"),
            )
            signals.append(sig)
        return sorted(signals, key=lambda s: s.timestep)

    def _detect_primitives_from_signals(
        self, signals: List[PhysicsSignal], events: List[Dict[str, Any]]
    ) -> List[DetectedPrimitive]:
        """
        Main detection logic.

        State machine:
        1. Track gripper state (open/closed)
        2. Track contact state
        3. Detect transitions
        4. Label segments based on transition patterns
        """
        if not signals:
            return []

        primitives: List[DetectedPrimitive] = []

        # Build event lookup
        event_lookup = {e.get("timestep", -1): e for e in events}

        # State tracking
        gripper_closed = False
        in_contact = False
        current_segment_start = signals[0].timestep
        current_label = "approach"  # Default initial state
        current_object = signals[0].object_name

        for i, sig in enumerate(signals):
            # Check for events at this timestep
            event = event_lookup.get(sig.timestep)

            # Detect gripper transitions
            new_gripper_closed = sig.gripper_width < self.gripper_close_threshold
            new_in_contact = sig.contact

            # Detect state change
            changed = False
            new_label = current_label

            # Gripper close -> grasp
            if not gripper_closed and new_gripper_closed and new_in_contact:
                new_label = "grasp"
                changed = True

            # Contact lost (not a release) -> slip/failure
            elif in_contact and not new_in_contact and gripper_closed:
                new_label = "slip"
                changed = True

            # Gripper open (after slip or normal) -> release
            elif gripper_closed and not new_gripper_closed:
                # Only mark as release if not slipping
                if current_label != "slip":
                    new_label = "release"
                    changed = True

            # Contact established -> contact action
            elif not in_contact and new_in_contact and gripper_closed:
                new_label = "pull" if "drawer" in current_object else "lift"
                changed = True

            # Movement with object -> transport
            elif gripper_closed and new_in_contact and sig.ee_velocity > self.velocity_threshold:
                if current_label not in ["pull", "lift", "transport"]:
                    new_label = "transport"
                    changed = True

            # Approaching without contact
            elif not new_in_contact and sig.ee_velocity > self.velocity_threshold:
                if current_label != "approach":
                    new_label = "approach"
                    changed = True

            # Recovery action detection
            if event and event.get("event_type") == "recovery_start":
                new_label = event.get("payload", {}).get("strategy", "regrasp")
                changed = True

            # If state changed, finalize current segment
            if changed and i > 0:
                # Finalize previous segment
                duration = sig.timestep - current_segment_start
                if duration >= self.min_segment_duration or current_label in ["slip", "regrasp", "pick_from_drop"]:
                    failure = self._check_failure(current_segment_start, sig.timestep, event_lookup) or current_label == "slip"
                    recovery = self._check_recovery(current_segment_start, sig.timestep, event_lookup) or current_label in ["regrasp", "pick_from_drop"]

                    primitives.append(DetectedPrimitive(
                        start_t=current_segment_start,
                        end_t=sig.timestep,
                        label=current_label,
                        object_name=current_object,
                        confidence=0.9,
                        failure=failure,
                        recovery=recovery,
                        metadata={"duration": duration}
                    ))

                # Start new segment
                current_segment_start = sig.timestep
                current_label = new_label

            # Update state
            gripper_closed = new_gripper_closed
            in_contact = new_in_contact
            if sig.object_name and sig.object_name != "unknown":
                current_object = sig.object_name

        # Finalize last segment
        if signals:
            last_sig = signals[-1]
            duration = last_sig.timestep - current_segment_start
            if duration >= 1 or current_label in ["slip", "regrasp", "pick_from_drop"]:
                failure = self._check_failure(current_segment_start, last_sig.timestep + 1, event_lookup) or current_label == "slip"
                recovery = self._check_recovery(current_segment_start, last_sig.timestep + 1, event_lookup) or current_label in ["regrasp", "pick_from_drop"]

                primitives.append(DetectedPrimitive(
                    start_t=current_segment_start,
                    end_t=last_sig.timestep + 1,
                    label=current_label,
                    object_name=current_object,
                    confidence=0.9,
                    failure=failure,
                    recovery=recovery,
                    metadata={"duration": duration}
                ))

        return primitives

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
