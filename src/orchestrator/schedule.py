from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class BudgetConfig:
    max_concurrent_runs: int = 1
    daily_step_budget: int = 1_000_000
    daily_run_budget: int = 100


@dataclass
class BudgetState:
    current_concurrent_runs: int = 0
    steps_today: int = 0
    runs_today: int = 0


class BudgetExceeded(Exception):
    pass


_budget_config = BudgetConfig()
_budget_state = BudgetState()
_inflight_steps: list[int] = []
_last_reset_date = date.today()


def set_budget_config(cfg: BudgetConfig) -> None:
    global _budget_config
    _budget_config = cfg


def get_budget_state() -> BudgetState:
    _maybe_reset()
    return BudgetState(
        current_concurrent_runs=_budget_state.current_concurrent_runs,
        steps_today=_budget_state.steps_today,
        runs_today=_budget_state.runs_today,
    )


def reset_budget_state() -> None:
    global _budget_state, _inflight_steps
    _budget_state = BudgetState()
    _inflight_steps = []


def acquire_run_budget(estimated_steps: int) -> None:
    _maybe_reset()
    if _budget_state.current_concurrent_runs >= _budget_config.max_concurrent_runs:
        raise BudgetExceeded("Max concurrent runs exceeded.")
    if _budget_state.steps_today + estimated_steps > _budget_config.daily_step_budget:
        raise BudgetExceeded("Daily step budget exceeded.")
    if _budget_state.runs_today + 1 > _budget_config.daily_run_budget:
        raise BudgetExceeded("Daily run budget exceeded.")

    _budget_state.current_concurrent_runs += 1
    _budget_state.steps_today += max(0, int(estimated_steps))
    _budget_state.runs_today += 1
    _inflight_steps.append(max(0, int(estimated_steps)))


def release_run_budget(steps_used: int) -> None:
    _maybe_reset()
    _budget_state.current_concurrent_runs = max(0, _budget_state.current_concurrent_runs - 1)
    estimated = _inflight_steps.pop() if _inflight_steps else 0
    try:
        delta = int(steps_used) - int(estimated)
    except Exception:
        delta = 0
    _budget_state.steps_today = max(0, _budget_state.steps_today + delta)


def _maybe_reset() -> None:
    global _last_reset_date, _budget_state, _inflight_steps
    today = date.today()
    if today != _last_reset_date:
        _budget_state = BudgetState()
        _inflight_steps = []
        _last_reset_date = today
