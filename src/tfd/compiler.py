"""
TFD compiler: orchestrates rejection checks, intent parsing, condition mapping, and safety clamping.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.tfd.intents import ParsedIntent, parse_instruction
from src.tfd.safety_rules import should_reject_instruction
from src.tfd.condition_mapper import TFDConditionVector, compile_to_condition_vector, apply_safety_constraints
from src.utils.json_safe import to_json_safe


@dataclass
class TFDInstruction:
    status: str
    parsed_intent: Optional[ParsedIntent]
    condition_vector: Optional[TFDConditionVector]
    reason: str = ""
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe(
            {
                "status": self.status,
                "reason": self.reason,
                "raw_text": self.raw_text,
                "parsed_intent": self.parsed_intent.to_dict() if self.parsed_intent else None,
                "condition_vector": self.condition_vector.to_dict() if self.condition_vector else None,
            }
        )


class TextFrontDoor:
    """Entry point for Text-Front-Door module."""

    def __init__(self, system_state: Optional[Dict[str, Any]] = None):
        self.system_state = system_state or {"current_mpl": 60.0, "max_mpl_uplift": 20.0}

    def process_instruction(self, text: str) -> TFDInstruction:
        should_reject, reason = should_reject_instruction(text)
        if should_reject:
            return TFDInstruction(status="rejected", parsed_intent=None, condition_vector=None, reason=reason, raw_text=text)

        conflict_cv = self._resolve_conflict(text)
        if conflict_cv:
            conflict_cv = apply_safety_constraints(conflict_cv, self.system_state)
            return TFDInstruction(
                status="accepted",
                parsed_intent=None,
                condition_vector=conflict_cv,
                reason="Resolved conflicting instructions",
                raw_text=text,
            )

        intent = parse_instruction(text)
        if intent.confidence < 0.5 or intent.intent_type is None:
            return TFDInstruction(
                status="low_confidence",
                parsed_intent=intent,
                condition_vector=None,
                reason="Could not parse instruction",
                raw_text=text,
            )

        cv = compile_to_condition_vector(intent, system_state=self.system_state)

        return TFDInstruction(status="accepted", parsed_intent=intent, condition_vector=cv, raw_text=text, reason="")

    def _resolve_conflict(self, text: str) -> Optional[TFDConditionVector]:
        txt = text.lower()
        if ("fast" in txt or "quick" in txt) and "careful" in txt:
            return TFDConditionVector(
                risk_tolerance=0.4,
                skill_mode="precision",
                time_budget_sec=20.0,
                safety_language_modulation=0.7,
                safety_emphasis=0.8,
            )
        return None
