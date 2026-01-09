"""Plan applier for hot-reload actuation.

Single choke point for applying semantic update plans to the sampler.
Supports polling for plan file changes, hysteresis, and safe boundaries.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import (
    SemanticUpdatePlanV1,
    TaskSamplerOverrides,
    DatapackSelectionOverrides,
    PlanOpType,
)


@dataclass
class PlanApplyResult:
    """Result of applying a plan."""

    applied: bool
    plan_id: str
    plan_sha: str
    prev_plan_sha: Optional[str] = None
    overrides: Optional[TaskSamplerOverrides] = None
    datapack_overrides: Optional[DatapackSelectionOverrides] = None
    error: Optional[str] = None
    step: int = 0
    ts: str = field(default_factory=lambda: datetime.now().isoformat())


class PlanApplier:
    """Hot-reload plan applier with hysteresis.

    Loads and validates SemanticUpdatePlanV1 from a file path,
    polls for changes, and applies updates atomically at safe boundaries.
    """

    def __init__(
        self,
        plan_path: Optional[str] = None,
        poll_steps: int = 100,
        enabled: bool = True,
        min_apply_interval_steps: int = 0,
        apply_only_on_boundary: bool = False,
        boundary_interval: int = 100,
        max_plan_changes_per_window: int = 0,
        events_path: Optional[str] = None,
    ):
        """Initialize plan applier.

        Args:
            plan_path: Path to plan JSON file
            poll_steps: Steps between file polls (0 = every step)
            enabled: Whether plan application is enabled
            min_apply_interval_steps: Minimum steps between applies (hysteresis)
            apply_only_on_boundary: Only apply at boundary steps
            boundary_interval: Interval for boundary steps
            max_plan_changes_per_window: Max changes per window (0 = unlimited)
            events_path: Path to write plan_applied_events.jsonl
        """
        self.plan_path = plan_path
        self.poll_steps = max(0, poll_steps)
        self.enabled = enabled

        # Hysteresis controls
        self.min_apply_interval_steps = max(0, min_apply_interval_steps)
        self.apply_only_on_boundary = apply_only_on_boundary
        self.boundary_interval = max(1, boundary_interval)
        self.max_plan_changes_per_window = max_plan_changes_per_window
        self.events_path = events_path

        # State
        self._current_plan: Optional[SemanticUpdatePlanV1] = None
        self._current_sha: Optional[str] = None
        self._last_poll_step: int = -1
        self._last_apply_step: int = -1
        self._file_mtime: float = 0.0
        self._window_change_count: int = 0

        # Computed overrides
        self._task_overrides: TaskSamplerOverrides = TaskSamplerOverrides()
        self._datapack_overrides: DatapackSelectionOverrides = DatapackSelectionOverrides()

        # History for debugging
        self._apply_history: List[PlanApplyResult] = []

    def load(self, path: Optional[str] = None, step: int = 0) -> PlanApplyResult:
        """Load and validate plan from file.

        Args:
            path: Optional path override
            step: Current step for logging

        Returns:
            PlanApplyResult with status
        """
        target_path = path or self.plan_path
        if not target_path:
            return PlanApplyResult(
                applied=False,
                plan_id="",
                plan_sha="",
                error="No plan path specified",
                step=step,
            )

        try:
            with open(target_path, "r") as f:
                data = json.load(f)

            plan = SemanticUpdatePlanV1.model_validate(data)
            new_sha = plan.sha256()

            if new_sha == self._current_sha:
                return PlanApplyResult(
                    applied=False,
                    plan_id=plan.plan_id,
                    plan_sha=new_sha,
                    error="Plan unchanged",
                    step=step,
                )

            prev_sha = self._current_sha
            self._current_plan = plan
            self._current_sha = new_sha
            self._file_mtime = os.path.getmtime(target_path)
            self._last_apply_step = step
            self._window_change_count += 1

            # Compute overrides
            self._compute_overrides(plan)

            result = PlanApplyResult(
                applied=True,
                plan_id=plan.plan_id,
                plan_sha=new_sha,
                prev_plan_sha=prev_sha,
                overrides=self._task_overrides,
                datapack_overrides=self._datapack_overrides,
                step=step,
            )
            self._apply_history.append(result)

            # Write event
            if self.events_path:
                self._write_event(result)

            return result

        except Exception as e:
            return PlanApplyResult(
                applied=False,
                plan_id="",
                plan_sha="",
                error=str(e),
                step=step,
            )

    def poll_and_apply(self, step: int) -> Optional[PlanApplyResult]:
        """Poll for plan file changes and apply if changed.

        Respects hysteresis and boundary constraints.

        Args:
            step: Current training step

        Returns:
            PlanApplyResult if plan was applied, None otherwise
        """
        if not self.enabled or not self.plan_path:
            return None

        # Check if we should poll
        if self.poll_steps > 0:
            if step - self._last_poll_step < self.poll_steps:
                return None

        self._last_poll_step = step

        # Hysteresis: check minimum interval since last apply
        if self.min_apply_interval_steps > 0:
            if self._last_apply_step >= 0:
                if step - self._last_apply_step < self.min_apply_interval_steps:
                    return None

        # Boundary check
        if self.apply_only_on_boundary:
            if step % self.boundary_interval != 0:
                return None

        # Max changes per window
        if self.max_plan_changes_per_window > 0:
            if self._window_change_count >= self.max_plan_changes_per_window:
                return None

        # Check file modification time
        try:
            current_mtime = os.path.getmtime(self.plan_path)
            if current_mtime <= self._file_mtime:
                return None
        except OSError:
            return None

        # Load and apply
        return self.load(step=step)

    def _compute_overrides(self, plan: SemanticUpdatePlanV1) -> None:
        """Compute overrides from plan."""
        weights: Dict[str, float] = {}
        disabled: List[str] = []

        for op in plan.task_graph_changes:
            if op.op == PlanOpType.SET_WEIGHT and op.weight is not None:
                weights[op.task_family] = op.weight
            elif op.op == PlanOpType.DISABLE:
                disabled.append(op.task_family)
            elif op.op == PlanOpType.ENABLE:
                if op.task_family in disabled:
                    disabled.remove(op.task_family)

        self._task_overrides = TaskSamplerOverrides(
            weights=weights,
            disabled=disabled,
        )

        if plan.datapack_selection:
            self._datapack_overrides = DatapackSelectionOverrides(
                allowlist=plan.datapack_selection.allowlist,
                denylist=plan.datapack_selection.denylist,
                quotas=plan.datapack_selection.quotas,
            )
        else:
            self._datapack_overrides = DatapackSelectionOverrides()

    def _write_event(self, result: PlanApplyResult) -> None:
        """Write plan apply event to JSONL."""
        if not self.events_path:
            return

        event = {
            "ts": result.ts,
            "step": result.step,
            "plan_id": result.plan_id,
            "plan_sha": result.plan_sha,
            "prev_plan_sha": result.prev_plan_sha,
            "applied": result.applied,
        }

        path = Path(self.events_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def reset_window(self) -> None:
        """Reset window change counter (call at window boundaries)."""
        self._window_change_count = 0

    @property
    def current_plan(self) -> Optional[SemanticUpdatePlanV1]:
        """Current loaded plan."""
        return self._current_plan

    @property
    def current_sha(self) -> Optional[str]:
        """SHA-256 of current plan."""
        return self._current_sha

    @property
    def task_overrides(self) -> TaskSamplerOverrides:
        """Current task sampler overrides."""
        return self._task_overrides

    @property
    def datapack_overrides(self) -> DatapackSelectionOverrides:
        """Current datapack selection overrides."""
        return self._datapack_overrides

    @property
    def apply_history(self) -> List[PlanApplyResult]:
        """History of plan applications."""
        return self._apply_history

    def reset(self) -> None:
        """Reset applier state."""
        self._current_plan = None
        self._current_sha = None
        self._file_mtime = 0.0
        self._last_apply_step = -1
        self._window_change_count = 0
        self._task_overrides = TaskSamplerOverrides()
        self._datapack_overrides = DatapackSelectionOverrides()


__all__ = [
    "PlanApplier",
    "PlanApplyResult",
]

