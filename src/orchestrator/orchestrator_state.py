"""Orchestrator state persistence for provenance closure.

Phase 1: Tracks orchestrator decision state for deterministic replay.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import OrchestratorStateV1, KnobDeltaV1


class OrchestratorStateTracker:
    """Tracks orchestrator state for provenance closure.

    Captures all decision factors for replay: failure counts, patience counters,
    clamp/noop decisions, cooldowns, and applied knob deltas.
    """

    def __init__(self, step: int = 0):
        self._step = step
        self._failure_counts: Dict[str, int] = {}
        self._patience_counters: Dict[str, int] = {}
        self._clamp_decisions: List[Dict[str, Any]] = []
        self._noop_decisions: List[Dict[str, Any]] = []
        self._cooldown_remaining: Dict[str, int] = {}
        self._backoff_multipliers: Dict[str, float] = {}
        self._applied_knob_deltas: List[KnobDeltaV1] = []

    def record_failure(self, gate_id: str) -> None:
        """Record a gate failure.

        Args:
            gate_id: ID of the gate that failed
        """
        self._failure_counts[gate_id] = self._failure_counts.get(gate_id, 0) + 1

    def set_patience(self, gate_id: str, remaining: int) -> None:
        """Set patience counter for a gate.

        Args:
            gate_id: ID of the gate
            remaining: Remaining patience count
        """
        self._patience_counters[gate_id] = remaining

    def record_clamp(
        self,
        gate_id: str,
        trigger: str,
        clamped_value: Any,
    ) -> None:
        """Record a clamp decision.

        Args:
            gate_id: ID of the gate
            trigger: Trigger condition for the clamp
            clamped_value: Value after clamping
        """
        self._clamp_decisions.append({
            "gate": gate_id,
            "trigger": trigger,
            "clamped_value": clamped_value,
            "step": self._step,
        })

    def record_noop(
        self,
        gate_id: str,
        trigger: str,
        reason: str,
    ) -> None:
        """Record a noop decision.

        Args:
            gate_id: ID of the gate
            trigger: Trigger condition
            reason: Reason for noop
        """
        self._noop_decisions.append({
            "gate": gate_id,
            "trigger": trigger,
            "reason": reason,
            "step": self._step,
        })

    def set_cooldown(self, gate_id: str, steps_remaining: int) -> None:
        """Set cooldown for a gate.

        Args:
            gate_id: ID of the gate
            steps_remaining: Steps remaining in cooldown
        """
        self._cooldown_remaining[gate_id] = steps_remaining

    def set_backoff(self, gate_id: str, multiplier: float) -> None:
        """Set backoff multiplier for a gate.

        Args:
            gate_id: ID of the gate
            multiplier: Current backoff multiplier
        """
        self._backoff_multipliers[gate_id] = multiplier

    def record_applied_knob_delta(self, delta: KnobDeltaV1) -> None:
        """Record an applied knob delta.

        Args:
            delta: The KnobDeltaV1 that was applied
        """
        self._applied_knob_deltas.append(delta)

    def update_step(self, step: int) -> None:
        """Update current step.

        Args:
            step: Current step number
        """
        self._step = step

    def build_state(self) -> OrchestratorStateV1:
        """Build the orchestrator state snapshot.

        Returns:
            OrchestratorStateV1 with current state
        """
        return OrchestratorStateV1(
            failure_counts=dict(self._failure_counts),
            patience_counters=dict(self._patience_counters),
            clamp_decisions=list(self._clamp_decisions),
            noop_decisions=list(self._noop_decisions),
            cooldown_remaining=dict(self._cooldown_remaining),
            backoff_multipliers=dict(self._backoff_multipliers),
            applied_knob_deltas=list(self._applied_knob_deltas),
            step=self._step,
        )


def write_orchestrator_state(path: str, state: OrchestratorStateV1) -> str:
    """Write orchestrator state to JSON file.

    Args:
        path: Output path
        state: Orchestrator state to write

    Returns:
        SHA-256 of written file content
    """
    from src.utils.config_digest import sha256_file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2)
    return sha256_file(str(output_path))


def load_orchestrator_state(path: str) -> OrchestratorStateV1:
    """Load orchestrator state from JSON file.

    Args:
        path: Path to state file

    Returns:
        OrchestratorStateV1 loaded from file
    """
    with open(path, "r") as f:
        data = json.load(f)
    return OrchestratorStateV1.model_validate(data)


__all__ = [
    "OrchestratorStateTracker",
    "write_orchestrator_state",
    "load_orchestrator_state",
]
