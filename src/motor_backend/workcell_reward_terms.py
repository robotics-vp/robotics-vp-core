from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.envs.workcell_env.rewards.reward_terms import WorkcellRewardTerms, compute_reward


@dataclass(frozen=True)
class WorkcellRewardResult:
    """Breakdown of reward terms for a single workcell step."""

    reward: float
    base_reward: float
    process_reward: float
    progress: float
    success: bool
    error_count: int
    tolerance_met: bool


def compute_workcell_reward(
    terms: WorkcellRewardTerms,
    *,
    task_reward: float | None,
    task_info: Mapping[str, Any] | None,
    step_time_s: float,
    process_reward: Any | None = None,
    process_reward_scale: float = 1.0,
    success: bool | None = None,
    progress: float | None = None,
    error_count: int | None = None,
    tolerance_met: bool | None = None,
) -> WorkcellRewardResult:
    """Compute a wrapped reward that can incorporate process-reward shaping."""
    task_info = task_info or {}
    progress_val = progress if progress is not None else extract_task_progress(task_info)
    error_val = error_count if error_count is not None else extract_error_count(task_info)
    success_val = bool(success if success is not None else _infer_success(task_info))
    tolerance_val = bool(tolerance_met if tolerance_met is not None else extract_tolerance_met(task_info))

    base = task_reward if task_reward is not None else compute_reward(
        terms,
        success=success_val,
        progress=progress_val,
        time_cost=step_time_s,
        error_count=error_val,
        tolerance_met=tolerance_val,
    )
    shaped = process_reward_scale * extract_process_reward_shaping(process_reward)
    return WorkcellRewardResult(
        reward=float(base + shaped),
        base_reward=float(base),
        process_reward=float(shaped),
        progress=float(progress_val),
        success=success_val,
        error_count=int(error_val),
        tolerance_met=tolerance_val,
    )


def extract_task_progress(task_info: Mapping[str, Any]) -> float:
    """Infer task-graph progress from task info payloads."""
    if not task_info:
        return 0.0
    if "progress" in task_info:
        return _clamp01(task_info.get("progress"))
    if "correct_count" in task_info:
        remaining = task_info.get("remaining")
        total = _safe_float(task_info.get("correct_count"), 0.0) + _safe_float(remaining, 0.0)
        if total > 0.0:
            return _clamp01(task_info.get("correct_count") / total)
    if "correct_sorted" in task_info:
        remaining = task_info.get("remaining")
        total = _safe_float(task_info.get("correct_sorted"), 0.0) + _safe_float(remaining, 0.0)
        if total > 0.0:
            return _clamp01(task_info.get("correct_sorted") / total)
    if "correct_pick" in task_info:
        return 1.0 if bool(task_info.get("correct_pick")) else 0.0
    if "completed" in task_info:
        return 1.0 if bool(task_info.get("completed")) else 0.0
    return 0.0


def extract_error_count(task_info: Mapping[str, Any]) -> int:
    """Infer error count from task info payloads."""
    if not task_info:
        return 0
    if "collision_count" in task_info:
        return max(int(task_info.get("collision_count", 0)), 0)
    if "incorrect_sorted" in task_info:
        return max(int(task_info.get("incorrect_sorted", 0)), 0)
    if task_info.get("force_violation"):
        return 1
    if task_info.get("missing_distance"):
        return 1
    return 0


def extract_tolerance_met(task_info: Mapping[str, Any]) -> bool:
    """Infer tolerance compliance from task info payloads."""
    if not task_info:
        return False
    if "tolerance_met" in task_info:
        return bool(task_info.get("tolerance_met"))
    if "inserted" in task_info:
        return bool(task_info.get("inserted"))
    return bool(task_info.get("completed")) if "completed" in task_info else False


def extract_process_reward_shaping(process_reward: Any | None) -> float:
    """Extract shaped reward value from process_reward outputs or dicts."""
    if process_reward is None:
        return 0.0
    if hasattr(process_reward, "r_shape"):
        return _extract_r_shape(getattr(process_reward, "r_shape"))
    if isinstance(process_reward, Mapping):
        for key in ("r_shape", "r_shape_step", "r_shape_t", "process_reward"):
            if key in process_reward:
                return _extract_r_shape(process_reward.get(key))
    return 0.0


def analyze_anti_reward_hacking(log_dict: Mapping[str, float]) -> tuple[bool, list[str], Mapping[str, float]]:
    """Heuristic checks for suspicious reward patterns."""
    reasons: list[str] = []
    suspicious = False

    reward = _first_metric(log_dict, ("mean_reward", "episode_reward_mean", "reward_mean"))
    success_rate = _first_metric(log_dict, ("success_rate",))
    energy = _first_metric(log_dict, ("energy_kwh_mean", "energy_kwh"))
    duration = _first_metric(log_dict, ("mean_episode_length", "episode_length_mean", "mean_episode_length_s"))
    expected_duration = _first_metric(log_dict, ("expected_duration",))

    if reward is not None and success_rate is not None and duration is not None:
        duration_threshold = expected_duration * 0.1 if expected_duration else duration * 0.1
        if reward > 10 * max(success_rate, 1e-3) and duration <= duration_threshold:
            suspicious = True
            reasons.append("Reward disproportionately high relative to success and duration")

    if reward is not None and energy is not None and energy <= 0.0 and reward > 0.0:
        suspicious = True
        reasons.append("Positive reward with near-zero energy usage")

    summary = {
        "episode_reward_mean": reward or 0.0,
        "success_rate": success_rate or 0.0,
        "energy_kwh_mean": energy or 0.0,
        "episode_length_mean": duration or 0.0,
    }
    return suspicious, reasons, summary


def _extract_r_shape(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _safe_float(value[-1], 0.0)
    try:
        if hasattr(value, "shape"):
            flattened = list(value.flatten())
            return _safe_float(flattened[-1], 0.0) if flattened else 0.0
    except Exception:
        pass
    return _safe_float(value, 0.0)


def _infer_success(task_info: Mapping[str, Any]) -> bool:
    for key in ("success", "completed", "inserted"):
        if key in task_info:
            return bool(task_info.get(key))
    return False


def _first_metric(log_dict: Mapping[str, float], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in log_dict and log_dict[key] is not None:
            try:
                return float(log_dict[key])
            except (TypeError, ValueError):
                continue
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    val = _safe_float(value, 0.0)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val
