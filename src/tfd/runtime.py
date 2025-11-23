"""
TFD Runtime Session Layer.

Maintains active set of instructions over time with deterministic conflict resolution.
Per TEXT_FRONT_DOOR_COMPLETE_SEMANTICS.md.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tfd.compiler import TFDInstruction, TextFrontDoor
from src.tfd.condition_mapper import TFDConditionVector
from src.tfd.intents import InstructionType
from src.utils.json_safe import to_json_safe


@dataclass
class TimestampedInstruction:
    """Single instruction with timestamp."""
    instruction: TFDInstruction
    timestamp: float
    index: int  # Sequence index for determinism

    def to_dict(self) -> Dict[str, Any]:
        return to_json_safe({
            "instruction": self.instruction.to_dict(),
            "timestamp": self.timestamp,
            "index": self.index,
        })


class TFDSession:
    """
    Runtime TFD session that manages active instructions over time.

    Maintains:
    - History of all instructions (ordered by timestamp/index)
    - Current canonical instruction (after conflict resolution)
    - Deterministic conflict resolution rules
    """

    # Safety-first instruction types always override others
    SAFETY_FIRST_TYPES = {
        InstructionType.MODIFY_RISK,
        InstructionType.PRECISION_MODE,
    }

    # Speed/efficiency types (lower priority than safety)
    SPEED_TYPES = {
        InstructionType.SPEED_MODE,
        InstructionType.TIME_CONSTRAINT,
    }

    # Exploration types
    EXPLORATION_TYPES = {
        InstructionType.EXPLORATION_MODE,
        InstructionType.PRIORITIZE_NOVELTY,
    }

    def __init__(self, system_state: Optional[Dict[str, Any]] = None):
        """Initialize TFD session."""
        self.tfd_compiler = TextFrontDoor(system_state=system_state)
        self.instructions: List[TimestampedInstruction] = []
        self._current_canonical: Optional[TFDInstruction] = None
        self._next_index = 0

    def add_instruction(self, text: str, timestamp: Optional[float] = None) -> TFDInstruction:
        """
        Add a new instruction to the session.

        Args:
            text: Natural language instruction
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Processed TFDInstruction
        """
        if timestamp is None:
            timestamp = time.time()

        # Process instruction through TFD compiler
        instruction = self.tfd_compiler.process_instruction(text)

        # Add to history if accepted
        if instruction.status == "accepted":
            timestamped = TimestampedInstruction(
                instruction=instruction,
                timestamp=timestamp,
                index=self._next_index,
            )
            self.instructions.append(timestamped)
            self._next_index += 1

            # Recompute canonical instruction
            self._update_canonical()

        return instruction

    def get_canonical_instruction(self) -> Optional[TFDInstruction]:
        """
        Get the current canonical instruction after conflict resolution.

        Returns:
            Current canonical TFDInstruction, or None if no active instructions
        """
        return self._current_canonical

    def get_canonical_condition_vector(self) -> Optional[TFDConditionVector]:
        """
        Get the condition vector from the canonical instruction.

        Returns:
            TFDConditionVector from canonical instruction, or None
        """
        if self._current_canonical is None:
            return None
        return self._current_canonical.condition_vector

    def clear(self):
        """Clear all instructions and reset session."""
        self.instructions.clear()
        self._current_canonical = None
        self._next_index = 0

    def _update_canonical(self):
        """
        Update the canonical instruction using conflict resolution rules.

        Conflict Resolution Rules (from spec):
        1. Safety-first instructions always win
        2. Most recent instruction wins within same priority tier
        3. If conflicting (e.g., "fast" + "careful"), apply mixed resolution

        Safety priority order:
        - Tier 1 (Highest): MODIFY_RISK, PRECISION_MODE
        - Tier 2: ENERGY_CONSTRAINT, TARGET_MPL
        - Tier 3: DEPLOY_SKILL, EXPLORATION_MODE, PRIORITIZE_NOVELTY
        - Tier 4 (Lowest): SPEED_MODE, TIME_CONSTRAINT
        """
        if not self.instructions:
            self._current_canonical = None
            return

        # Group instructions by type
        by_type: Dict[InstructionType, List[TimestampedInstruction]] = {}
        for ts_inst in self.instructions:
            inst = ts_inst.instruction
            if inst.parsed_intent and inst.parsed_intent.intent_type:
                itype = inst.parsed_intent.intent_type
                if itype not in by_type:
                    by_type[itype] = []
                by_type[itype].append(ts_inst)

        # Check for safety-first instructions
        safety_instructions = []
        for itype in self.SAFETY_FIRST_TYPES:
            if itype in by_type:
                safety_instructions.extend(by_type[itype])

        if safety_instructions:
            # Safety wins: use most recent safety instruction
            safety_instructions.sort(key=lambda x: (x.timestamp, x.index))
            latest_safety = safety_instructions[-1]

            # Check if there are also speed instructions (conflict case)
            speed_instructions = []
            for itype in self.SPEED_TYPES:
                if itype in by_type:
                    speed_instructions.extend(by_type[itype])

            if speed_instructions:
                # Mixed case: "be quick but careful"
                # Apply blended resolution favoring safety
                self._current_canonical = self._blend_safety_and_speed(
                    latest_safety, speed_instructions[-1]
                )
            else:
                # Pure safety mode
                self._current_canonical = latest_safety.instruction
        else:
            # No safety instructions; use most recent instruction overall
            all_instructions = list(self.instructions)
            all_instructions.sort(key=lambda x: (x.timestamp, x.index))
            self._current_canonical = all_instructions[-1].instruction

    def _blend_safety_and_speed(
        self,
        safety_inst: TimestampedInstruction,
        speed_inst: TimestampedInstruction,
    ) -> TFDInstruction:
        """
        Blend safety and speed instructions into a balanced instruction.

        Safety takes precedence but acknowledges speed request with moderate time budget.
        """
        # Start with safety instruction's condition vector
        safety_cv = safety_inst.instruction.condition_vector
        if safety_cv is None:
            return safety_inst.instruction

        # Create blended condition vector
        blended_cv = TFDConditionVector(
            risk_tolerance=0.4,  # Moderate (between safety 0.2 and speed 0.8)
            safety_language_modulation=0.7,  # Still high
            skill_mode="precision",  # Favor safety
            time_budget_sec=20.0,  # Moderate deadline (acknowledge speed)
            safety_emphasis=0.8,  # High
            efficiency_preference=0.5,  # Balanced
            # Inherit other fields from safety
            energy_budget_wh=safety_cv.energy_budget_wh,
            skill_id=safety_cv.skill_id,
            curriculum_phase=safety_cv.curriculum_phase,
            novelty_tier=safety_cv.novelty_tier,
            instruction_priority=0.8,
            target_mpl_uplift=safety_cv.target_mpl_uplift,
            objective_vector=safety_cv.objective_vector,
            novelty_bias=safety_cv.novelty_bias,
            exploration_priority=safety_cv.exploration_priority,
            fragility_avoidance=safety_cv.fragility_avoidance,
        )

        # Create blended instruction
        return TFDInstruction(
            status="accepted",
            parsed_intent=safety_inst.instruction.parsed_intent,
            condition_vector=blended_cv,
            reason="Blended safety-first with speed request",
            raw_text=f"{safety_inst.instruction.raw_text} + {speed_inst.instruction.raw_text}",
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Export session state as JSON-safe dict.

        Returns:
            Dict with instruction history and current canonical
        """
        return to_json_safe({
            "instruction_count": len(self.instructions),
            "instructions": [inst.to_dict() for inst in self.instructions],
            "canonical_instruction": (
                self._current_canonical.to_dict() if self._current_canonical else None
            ),
        })

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get compact session summary for logging.

        Returns:
            Dict with key session metrics
        """
        canonical_cv = self.get_canonical_condition_vector()
        canonical_inst = self.get_canonical_instruction()

        summary = {
            "active": canonical_inst is not None,
            "instruction_count": len(self.instructions),
        }

        if canonical_inst:
            summary["canonical_text"] = canonical_inst.raw_text
            summary["canonical_status"] = canonical_inst.status

            if canonical_inst.parsed_intent:
                summary["canonical_type"] = (
                    canonical_inst.parsed_intent.intent_type.value
                    if canonical_inst.parsed_intent.intent_type
                    else None
                )

        if canonical_cv:
            # Add key condition vector fields
            summary["skill_mode"] = canonical_cv.skill_mode
            summary["risk_tolerance"] = canonical_cv.risk_tolerance
            summary["safety_emphasis"] = canonical_cv.safety_emphasis
            summary["exploration_priority"] = canonical_cv.exploration_priority

        return to_json_safe(summary)
