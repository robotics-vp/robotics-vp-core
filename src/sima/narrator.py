"""
SIMA-2 Narrator.

Generates natural language narrations for robot actions.
"""

import numpy as np
from src.hrl.skills import SkillID


class Narrator:
    """
    Generates step-level and skill-level narrations.

    Converts robot state/action pairs into natural language descriptions.
    """

    def __init__(self):
        # Vocabulary for narrations
        self.skill_templates = {
            SkillID.LOCATE_DRAWER: [
                "Looking for the drawer handle",
                "Scanning the cabinet for drawer",
                "Identifying drawer location",
                "Searching for the drawer"
            ],
            SkillID.LOCATE_VASE: [
                "Detecting fragile vase position",
                "Identifying the vase location",
                "Scanning for fragile objects",
                "Locating the vase"
            ],
            SkillID.PLAN_SAFE_APPROACH: [
                "Computing safe trajectory",
                "Planning approach path around vase",
                "Calculating collision-free path",
                "Planning safe route"
            ],
            SkillID.GRASP_HANDLE: [
                "Moving to grasp drawer handle",
                "Approaching the handle",
                "Reaching for drawer handle",
                "Moving arm towards handle"
            ],
            SkillID.OPEN_WITH_CLEARANCE: [
                "Pulling drawer open while avoiding vase",
                "Opening drawer carefully",
                "Maintaining clearance while opening",
                "Pulling drawer with safe distance"
            ],
            SkillID.RETRACT_SAFE: [
                "Retracting to safe position",
                "Moving back to home position",
                "Returning to start position",
                "Pulling back safely"
            ]
        }

        # Action descriptors
        self.direction_words = {
            'up': ['lifting', 'raising', 'moving upward'],
            'down': ['lowering', 'moving downward', 'descending'],
            'forward': ['pushing forward', 'advancing', 'moving forward'],
            'backward': ['pulling back', 'retracting', 'moving backward'],
            'left': ['moving left', 'shifting left'],
            'right': ['moving right', 'shifting right']
        }

        # Safety phrases
        self.safety_phrases = [
            "maintaining safe distance from vase",
            "keeping clearance from fragile object",
            "avoiding collision with vase",
            "staying clear of vase"
        ]

    def narrate_skill_start(self, skill_id):
        """
        Generate narration for skill initiation.

        Args:
            skill_id: SkillID

        Returns:
            narration: str
        """
        templates = self.skill_templates.get(skill_id, ["Executing action"])
        return np.random.choice(templates)

    def narrate_step(self, obs, action, info, skill_id=None):
        """
        Generate narration for a single step.

        Args:
            obs: (13,) observation
            action: (3,) action vector
            info: Info dictionary
            skill_id: Optional current skill

        Returns:
            narration: str
        """
        ee_pos = obs[0:3]
        ee_vel = obs[3:6]
        drawer_frac = obs[6]
        vase_pos = obs[7:10]
        min_clearance = obs[11]

        # Analyze action direction
        dx, dy, dz = action

        parts = []

        # Movement description
        if abs(dz) > 0.3:
            if dz > 0:
                parts.append("moving upward")
            else:
                parts.append("moving downward")

        if abs(dy) > 0.3:
            if dy > 0:
                parts.append("pushing forward")
            else:
                parts.append("pulling back")

        if abs(dx) > 0.3:
            if dx > 0:
                parts.append("shifting right")
            else:
                parts.append("shifting left")

        # If no significant movement
        if len(parts) == 0:
            parts.append("making small adjustment")

        movement_desc = " and ".join(parts)

        # Add context
        context_parts = []

        # Drawer progress
        if skill_id == SkillID.OPEN_WITH_CLEARANCE:
            context_parts.append(f"drawer {drawer_frac*100:.0f}% open")

        # Safety context
        if min_clearance < 0.15:
            context_parts.append(f"clearance {min_clearance:.2f}m")

        if min_clearance < 0.1:
            context_parts.append("close to vase")

        # Build narration
        narration = movement_desc.capitalize()

        if context_parts:
            narration += f" ({', '.join(context_parts)})"

        return narration

    def narrate_skill_completion(self, skill_id, success, reason=""):
        """
        Generate narration for skill completion.

        Args:
            skill_id: SkillID that completed
            success: Whether skill succeeded
            reason: Optional reason string

        Returns:
            narration: str
        """
        skill_name = SkillID.name(skill_id)

        if success:
            completions = {
                SkillID.LOCATE_DRAWER: "Found the drawer handle",
                SkillID.LOCATE_VASE: "Identified vase position",
                SkillID.PLAN_SAFE_APPROACH: "Safe path computed",
                SkillID.GRASP_HANDLE: "Handle grasped successfully",
                SkillID.OPEN_WITH_CLEARANCE: "Drawer opened successfully",
                SkillID.RETRACT_SAFE: "Safely retracted to home position"
            }
            return completions.get(skill_id, f"Completed {skill_name}")
        else:
            failures = {
                SkillID.GRASP_HANDLE: "Failed to grasp handle",
                SkillID.OPEN_WITH_CLEARANCE: f"Drawer opening failed: {reason}",
            }
            return failures.get(skill_id, f"Skill {skill_name} failed: {reason}")

    def narrate_task_start(self, instruction):
        """
        Generate narration for task start.

        Args:
            instruction: Original instruction text

        Returns:
            narration: str
        """
        return f"Starting task: {instruction}"

    def narrate_task_completion(self, success, stats=None):
        """
        Generate narration for task completion.

        Args:
            success: Whether task succeeded
            stats: Optional statistics dict

        Returns:
            narration: str
        """
        if success:
            base = "Task completed successfully"
            if stats:
                details = []
                if 'drawer_frac' in stats:
                    details.append(f"drawer {stats['drawer_frac']*100:.0f}% open")
                if 'vase_intact' in stats:
                    details.append("vase intact" if stats['vase_intact'] else "vase damaged")
                if 'total_steps' in stats:
                    details.append(f"{stats['total_steps']} steps")
                if details:
                    base += f" ({', '.join(details)})"
            return base
        else:
            base = "Task failed"
            if stats and 'reason' in stats:
                base += f": {stats['reason']}"
            return base

    def narrate_safety_warning(self, min_clearance, threshold=0.1):
        """
        Generate safety warning narration.

        Args:
            min_clearance: Current minimum clearance
            threshold: Warning threshold

        Returns:
            narration: str or None
        """
        if min_clearance < threshold:
            return f"Warning: Clearance from vase only {min_clearance:.3f}m"
        return None

    def summarize_trajectory(self, skill_sequence, narrations):
        """
        Generate summary of entire trajectory.

        Args:
            skill_sequence: List of skill IDs executed
            narrations: List of step narrations

        Returns:
            summary: str
        """
        skill_names = [SkillID.name(sid) for sid in skill_sequence]
        sequence_str = " → ".join(skill_names)

        summary = f"Executed skill sequence: {sequence_str}. "
        summary += f"Total {len(narrations)} actions performed."

        return summary

    def generate_instruction_variations(self, base_instruction):
        """
        Generate variations of an instruction for data augmentation.

        Args:
            base_instruction: Original instruction

        Returns:
            variations: List of varied instructions
        """
        variations = [base_instruction]

        # Synonym replacements
        synonyms = {
            'open': ['pull open', 'slide open', 'extract'],
            'drawer': ['top drawer', 'cabinet drawer', 'storage drawer'],
            'vase': ['fragile vase', 'ceramic vase', 'delicate object'],
            'avoid': ['not hit', 'stay away from', 'keep clear of'],
            'carefully': ['safely', 'gently', 'cautiously']
        }

        # Generate variations
        for original, alternatives in synonyms.items():
            if original in base_instruction.lower():
                for alt in alternatives[:2]:  # Limit variations
                    var = base_instruction.lower().replace(original, alt)
                    variations.append(var)

        # Add modifiers
        modifiers = ['please ', 'robot, ', 'i need you to ', '']
        for mod in modifiers:
            variations.append(mod + base_instruction)

        return list(set(variations))[:10]  # Remove duplicates, limit to 10
