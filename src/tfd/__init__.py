"""
Text-Front-Door (TFD) minimal working implementation.

Deterministic, regex-only parsing and condition mapping per
specs/phase_h_design/TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md.
"""
from src.tfd.compiler import TextFrontDoor, TFDInstruction
from src.tfd.intents import InstructionType, ParsedIntent, parse_instruction
from src.tfd.safety_rules import should_reject_instruction
from src.tfd.condition_mapper import compile_to_condition_vector, apply_safety_constraints

__all__ = [
    "TextFrontDoor",
    "TFDInstruction",
    "InstructionType",
    "ParsedIntent",
    "parse_instruction",
    "should_reject_instruction",
    "compile_to_condition_vector",
    "apply_safety_constraints",
]
