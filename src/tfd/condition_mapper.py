"""
Condition mapping for Text-Front-Door.

Implements deterministic Intent -> ConditionVector mapping with safety
clamps and advisory fields (novelty_bias, safety_emphasis, etc.).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.tfd.intents import InstructionType, ParsedIntent
from src.tfd.safety_rules import apply_safety_constraints
from src.utils.json_safe import to_json_safe


@dataclass
class TFDConditionVector:
    # Core fields
    risk_tolerance: Optional[float] = None
    safety_language_modulation: Optional[float] = None
    energy_budget_wh: Optional[float] = None
    time_budget_sec: Optional[float] = None
    skill_mode: Optional[str] = None
    skill_id: Optional[str] = None
    curriculum_phase: Optional[str] = None
    novelty_tier: Optional[int] = None
    instruction_priority: Optional[float] = None
    target_mpl_uplift: Optional[float] = None
    objective_vector: Optional[Dict[str, float]] = None

    # Advisory fields from spec
    novelty_bias: Optional[float] = None
    safety_emphasis: Optional[float] = None
    exploration_priority: Optional[float] = None
    efficiency_preference: Optional[float] = None
    fragility_avoidance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(
            {
                "risk_tolerance": self.risk_tolerance,
                "safety_language_modulation": self.safety_language_modulation,
                "energy_budget_wh": self.energy_budget_wh,
                "time_budget_sec": self.time_budget_sec,
                "skill_mode": self.skill_mode,
                "skill_id": self.skill_id,
                "curriculum_phase": self.curriculum_phase,
                "novelty_tier": self.novelty_tier,
                "instruction_priority": self.instruction_priority,
                "target_mpl_uplift": self.target_mpl_uplift,
                "objective_vector": self.objective_vector,
                "novelty_bias": self.novelty_bias,
                "safety_emphasis": self.safety_emphasis,
                "exploration_priority": self.exploration_priority,
                "efficiency_preference": self.efficiency_preference,
                "fragility_avoidance": self.fragility_avoidance,
            }
        )


def _get_current_mpl(system_state: Optional[Dict[str, Any]]) -> float:
    if system_state is None:
        return 60.0
    try:
        return float(system_state.get("current_mpl", 60.0))
    except Exception:
        return 60.0


def _infer_objective_vector(raw_text: str) -> Optional[Dict[str, float]]:
    txt = raw_text.lower()
    if "blue" in txt and "red" in txt:
        return {"blue": 1.0, "red": 0.0}
    if "left" in txt:
        return {"left_region": 1.0, "right_region": 0.0}
    if "center" in txt:
        return {"center": 1.0}
    if "weird object" in txt or "weird" in txt:
        return {"weird_object": 1.0}
    return None


def compile_to_condition_vector(
    intent: ParsedIntent, system_state: Optional[Dict[str, Any]] = None
) -> TFDConditionVector:
    """
    Compile ParsedIntent into a TFDConditionVector.
    Deterministic mapping, no randomness.
    """
    cv = TFDConditionVector()
    raw_text = intent.raw_text or ""
    current_mpl = _get_current_mpl(system_state)
    max_mpl_uplift = float(system_state.get("max_mpl_uplift", 20.0)) if system_state else 20.0
    obj_vec = _infer_objective_vector(raw_text)

    itype = intent.intent_type
    if itype == InstructionType.MODIFY_RISK:
        cv.risk_tolerance = 0.2
        cv.safety_language_modulation = 0.9
        cv.skill_mode = "safety_critical"
        cv.safety_emphasis = 0.9
        cv.fragility_avoidance = 0.8
    elif itype == InstructionType.SPEED_MODE:
        cv.risk_tolerance = 0.8
        cv.energy_budget_wh = None
        cv.skill_mode = "speed"
        cv.instruction_priority = 0.9
        cv.efficiency_preference = 0.2
    elif itype == InstructionType.ENERGY_CONSTRAINT:
        energy_cap = intent.parameters.get("energy_cap_wh", 50.0)
        if "don't sacrifice quality" in raw_text:
            energy_cap = 75.0
            cv.risk_tolerance = 0.4
            cv.safety_language_modulation = 0.7
        cv.energy_budget_wh = energy_cap
        cv.skill_mode = "energy_efficient"
        cv.risk_tolerance = 0.5 if cv.risk_tolerance is None else cv.risk_tolerance
        cv.efficiency_preference = 0.8
        cv.safety_emphasis = 0.6
    elif itype == InstructionType.PRIORITIZE_NOVELTY:
        cv.curriculum_phase = "frontier"
        cv.novelty_tier = 2
        cv.skill_mode = "exploration"
        cv.novelty_bias = 1.0
        cv.exploration_priority = 0.9
    elif itype == InstructionType.TARGET_MPL:
        uplift_pct = float(intent.parameters.get("target_uplift_pct", 0.0))
        cv.target_mpl_uplift = min(max_mpl_uplift, current_mpl * (uplift_pct / 100.0))
        cv.instruction_priority = 0.8
    elif itype == InstructionType.DEPLOY_SKILL:
        cv.skill_id = intent.parameters.get("skill_id")
        cv.skill_mode = "precision"
        cv.objective_vector = obj_vec
    elif itype == InstructionType.PRECISION_MODE:
        cv.skill_mode = "precision"
        cv.risk_tolerance = 0.3
        cv.safety_language_modulation = 0.8
        cv.safety_emphasis = 0.8
        cv.fragility_avoidance = 0.7
        cv.objective_vector = obj_vec
    elif itype == InstructionType.TIME_CONSTRAINT:
        budget = intent.parameters.get("time_budget_sec", 30.0)
        cv.time_budget_sec = float(budget)
        cv.skill_mode = "speed"
        cv.risk_tolerance = 0.6
        cv.instruction_priority = 0.8
    elif itype == InstructionType.EXPLORATION_MODE:
        cv.skill_mode = "exploration"
        cv.curriculum_phase = "frontier"
        cv.novelty_tier = 2
        cv.exploration_priority = 1.0
        cv.novelty_bias = 1.0
        cv.objective_vector = obj_vec
    elif itype == InstructionType.SAFETY_OVERRIDE:
        # Parsed but should be rejected upstream; keep defaults
        cv.risk_tolerance = 0.5

    # Attach inferred objective vector if not set
    if cv.objective_vector is None and obj_vec is not None:
        cv.objective_vector = obj_vec

    return apply_safety_constraints(cv, system_state or {})
