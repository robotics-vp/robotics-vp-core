"""
Safety and rejection rules for Text-Front-Door.
"""
import re
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tfd.condition_mapper import TFDConditionVector  # pragma: no cover

# Rejection patterns from spec
REJECTED_PATTERNS = [
    r"ignore safety",
    r"break (the )?rule",
    r"override (safety|econ)",
    r"don't care (about|if)",
    r"just do it",
]


def should_reject_instruction(text: str) -> Tuple[bool, str]:
    text_lower = text.lower()
    for pattern in REJECTED_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"Instruction contains unsafe pattern: '{pattern}'"
    return False, ""


def apply_safety_constraints(cv: Any, system_state: Dict) -> Any:
    """
    Clamp ConditionVector fields to safe/economically-viable ranges.
    """
    if cv.risk_tolerance is not None:
        cv.risk_tolerance = max(0.1, min(0.9, float(cv.risk_tolerance)))
    if cv.energy_budget_wh is not None:
        cv.energy_budget_wh = max(10.0, float(cv.energy_budget_wh))
    if cv.time_budget_sec is not None:
        cv.time_budget_sec = max(5.0, float(cv.time_budget_sec))
    if cv.target_mpl_uplift is not None:
        max_feasible = float(system_state.get("max_mpl_uplift", 20.0)) if system_state else 20.0
        cv.target_mpl_uplift = min(float(cv.target_mpl_uplift), max_feasible)
    return cv
