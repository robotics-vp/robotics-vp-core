"""
Intent parsing for Text-Front-Door (TFD).

Deterministic regex-only parser for 10 InstructionTypes as specified in
TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple


class InstructionType(Enum):
    DEPLOY_SKILL = "deploy_skill"
    MODIFY_RISK = "modify_risk_tolerance"
    PRIORITIZE_NOVELTY = "prioritize_novelty"
    TARGET_MPL = "target_mpl_uplift"
    ENERGY_CONSTRAINT = "energy_constraint"
    TIME_CONSTRAINT = "time_constraint"
    SAFETY_OVERRIDE = "safety_override"
    EXPLORATION_MODE = "exploration_mode"
    PRECISION_MODE = "precision_mode"
    SPEED_MODE = "speed_mode"


@dataclass
class ParsedIntent:
    intent_type: Optional[InstructionType]
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type.value if self.intent_type else None,
            "parameters": dict(self.parameters),
            "confidence": float(self.confidence),
            "raw_text": self.raw_text,
        }


def _compile_patterns() -> List[Tuple[InstructionType, List[Pattern[str]]]]:
    """Compile INTENT_PATTERNS deterministically (ordered)."""
    raw_patterns: List[Tuple[InstructionType, List[str]]] = [
        (
            InstructionType.MODIFY_RISK,
            [
                r"be (more|very) careful",
                r"don't break",
                r"avoid damage",
                r"high precision",
                r"be cautious",
            ],
        ),
        (
            InstructionType.TARGET_MPL,
            [
                r"increase productivity by (\d+)%",  # group 1
                r"work (\d+)% faster",
                r"improve output by (\d+)%",
            ],
        ),
        (
            InstructionType.SPEED_MODE,
            [
                r"(go |move )?fast(er)?",
                r"quick(ly)?",
                r"hurry",
                r"speed up",
                r"don't care about energy",
            ],
        ),
        (
            InstructionType.ENERGY_CONSTRAINT,
            [
                r"save (power|energy)",
                r"minimize (power|energy|consumption)",
                r"be efficient",
                r"low power mode",
                r"stay(ing)? under (\d+)\s*wh",
            ],
        ),
        (
            InstructionType.EXPLORATION_MODE,
            [
                r"activate exploration mode",
                r"figure out",
                r"try to discover",
                r"explore new (ways|methods)",
            ],
        ),
        (
            InstructionType.PRIORITIZE_NOVELTY,
            [
                r"\bexplore\b",
                r"try (a )?(new|different) approach",
                r"be creative",
                r"find alternatives",
            ],
        ),
        (
            InstructionType.DEPLOY_SKILL,
            [
                r"use (the )?(\w+) skill",  # group 2 = skill
                r"activate (\w+) mode",
                r"switch to (\w+)",
                r"sort the .*blocks",
                r"focus on clearing the left side",
            ],
        ),
        (
            InstructionType.TIME_CONSTRAINT,
            [
                r"finish (this )?(task )?(in|within) (\d+)\s*(seconds|sec|s)",
                r"done in (\d+)\s*(seconds|sec|s)",
            ],
        ),
        (
            InstructionType.PRECISION_MODE,
            [
                r"gentle movements",
                r"exactly",
                r"precision mode",
                r"be precise",
                r"take your time",
                r"no rush",
                r"double[- ]check",
            ],
        ),
        (
            InstructionType.SAFETY_OVERRIDE,
            [
                r"ignore safety",
                r"override (safety|econ)",
            ],
        ),
    ]
    compiled: List[Tuple[InstructionType, List[Pattern[str]]]] = []
    for intent, patterns in raw_patterns:
        compiled.append((intent, [re.compile(p, re.IGNORECASE) for p in patterns]))
    return compiled


INTENT_PATTERNS: List[Tuple[InstructionType, List[Pattern[str]]]] = _compile_patterns()


def _extract_parameters(match: re.Match, intent_type: InstructionType) -> Dict[str, Any]:
    if intent_type == InstructionType.TARGET_MPL:
        # Prioritize first captured group
        for i in range(1, (match.lastindex or 0) + 1):
            try:
                return {"target_uplift_pct": float(match.group(i))}
            except Exception:
                continue
        return {"target_uplift_pct": 0.0}
    if intent_type == InstructionType.DEPLOY_SKILL:
        skill_name = ""
        if match.lastindex:
            skill_name = match.group(match.lastindex)
        else:
            text = match.group(0).lower()
            if "sort" in text:
                skill_name = "sort_objects"
            elif "left side" in text:
                skill_name = "left_focus"
        return {"skill_id": skill_name.strip()}
    if intent_type == InstructionType.ENERGY_CONSTRAINT:
        if match.lastindex:
            for i in range(1, match.lastindex + 1):
                try:
                    return {"energy_cap_wh": float(match.group(i))}
                except Exception:
                    continue
    if intent_type == InstructionType.TIME_CONSTRAINT:
        if match.lastindex:
            for i in range(1, match.lastindex + 1):
                try:
                    return {"time_budget_sec": float(match.group(i))}
                except Exception:
                    continue
    return {}


def parse_instruction(text: str) -> ParsedIntent:
    """
    Parse natural language instruction into structured intent.
    Deterministic, regex-only (no ML).
    """
    text_lower = text.lower().strip()
    for intent_type, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            match = pattern.search(text_lower)
            if match:
                params = _extract_parameters(match, intent_type)
                confidence = 0.9 if intent_type != InstructionType.SAFETY_OVERRIDE else 0.8
                return ParsedIntent(intent_type=intent_type, parameters=params, confidence=confidence, raw_text=text)

    return ParsedIntent(intent_type=None, parameters={}, confidence=0.0, raw_text=text)
