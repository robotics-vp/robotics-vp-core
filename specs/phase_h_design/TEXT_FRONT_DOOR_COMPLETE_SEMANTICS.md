# Text-Front-Door: Complete Semantics Specification

**Status**: Canonical Specification
**Owner**: Claude (Semantic Architect)
**Context**: Unambiguous mapping from natural language → economic/semantic control

---

## 1. Instruction Taxonomy

### 1.1. Core Instruction Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class InstructionType(Enum):
    """Canonical instruction categories."""
    DEPLOY_SKILL = "deploy_skill"              # "Use the dish-washing skill"
    MODIFY_RISK = "modify_risk_tolerance"      # "Be more careful"
    PRIORITIZE_NOVELTY = "prioritize_novelty"  # "Explore new approaches"
    TARGET_MPL = "target_mpl_uplift"           # "Increase productivity by 15%"
    ENERGY_CONSTRAINT = "energy_constraint"    # "Minimize power consumption"
    TIME_CONSTRAINT = "time_constraint"        # "Finish within 30 seconds"
    SAFETY_OVERRIDE = "safety_override"        # "Ignore fragile objects" (REJECTED)
    EXPLORATION_MODE = "exploration_mode"      # "Figure out how to open this latch"
    PRECISION_MODE = "precision_mode"          # "Place the cup exactly here"
    SPEED_MODE = "speed_mode"                  # "Go as fast as possible"
```

---

## 2. Text → Intent Mapping

### 2.1. Pattern Recognition Rules

```python
INTENT_PATTERNS = {
    InstructionType.MODIFY_RISK: [
        r"be (more|very) careful",
        r"gentle(ly)?",
        r"don't break",
        r"avoid damage",
        r"high precision",
        r"be cautious"
    ],
    InstructionType.SPEED_MODE: [
        r"(go |move )?fast(er)?",
        r"quick(ly)?",
        r"hurry",
        r"speed up",
        r"don't care about energy"
    ],
    InstructionType.ENERGY_CONSTRAINT: [
        r"save (power|energy)",
        r"minimize (power|energy|consumption)",
        r"be efficient",
        r"low power mode"
    ],
    InstructionType.PRIORITIZE_NOVELTY: [
        r"explore",
        r"try (new|different) approach",
        r"be creative",
        r"find alternatives"
    ],
    InstructionType.TARGET_MPL: [
        r"increase productivity by (\d+)%",
        r"work (\d+)% faster",
        r"improve output by (\d+)%"
    ],
    InstructionType.DEPLOY_SKILL: [
        r"use (the )?(\w+) skill",
        r"activate (\w+) mode",
        r"switch to (\w+)"
    ]
}
```

---

### 2.2. Semantic Compiler Implementation

```python
@dataclass
class ParsedIntent:
    """Parsed instruction intent."""
    intent_type: InstructionType
    parameters: Dict[str, Any]
    confidence: float  # 0.0-1.0
    raw_text: str

def parse_instruction(text: str) -> ParsedIntent:
    """
    Parse natural language instruction into structured intent.

    Returns:
        ParsedIntent with detected type, extracted parameters, and confidence
    """
    text_lower = text.lower().strip()

    # Check each pattern
    for intent_type, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                params = _extract_parameters(match, intent_type)
                return ParsedIntent(
                    intent_type=intent_type,
                    parameters=params,
                    confidence=0.9,
                    raw_text=text
                )

    # No match found
    return ParsedIntent(
        intent_type=None,
        parameters={},
        confidence=0.0,
        raw_text=text
    )

def _extract_parameters(match: re.Match, intent_type: InstructionType) -> Dict[str, Any]:
    """Extract numeric/categorical parameters from regex match."""
    if intent_type == InstructionType.TARGET_MPL:
        # Extract percentage: "increase productivity by 15%"
        pct = float(match.group(1))
        return {"target_uplift_pct": pct}

    elif intent_type == InstructionType.DEPLOY_SKILL:
        # Extract skill name: "use the dishwashing skill"
        skill_name = match.group(2) if match.lastindex >= 2 else match.group(1)
        return {"skill_id": skill_name.strip()}

    return {}
```

---

## 3. Intent → ConditionVector Mapping

### 3.1. ConditionVector Field Definitions

```python
@dataclass
class ConditionVector:
    """
    Semantic control vector for VLA stack.

    All fields are optional; unset fields use system defaults.
    """
    # Risk & Safety
    risk_tolerance: Optional[float] = None  # [0, 1]: 0=max_safety, 1=max_speed
    safety_language_modulation: Optional[float] = None  # [0, 1]: text-derived safety bias

    # Energy & Time
    energy_budget_wh: Optional[float] = None  # Max Watt-hours (None = unlimited)
    time_budget_sec: Optional[float] = None   # Max seconds (None = unlimited)

    # Skill & Mode
    skill_mode: Optional[str] = None  # "precision", "speed", "exploration", "safety_critical"
    skill_id: Optional[str] = None    # Specific skill to activate

    # Curriculum & Exploration
    curriculum_phase: Optional[str] = None  # "early", "mid", "advanced", "frontier"
    novelty_tier: Optional[int] = None      # 0=redundant, 1=context_novel, 2=frontier

    # Economic
    instruction_priority: Optional[float] = None  # [0, 1]: urgency weight
    target_mpl_uplift: Optional[float] = None     # Target MPL increase (absolute units/hr)

    # Vision Modulation
    objective_vector: Optional[Dict[str, float]] = None  # {"blue": 1.0, "red": 0.0}
```

---

### 3.2. Mapping Rules (Deterministic)

```python
def compile_to_condition_vector(intent: ParsedIntent) -> ConditionVector:
    """
    Compile ParsedIntent into ConditionVector.

    Rules are deterministic and priority-ordered.
    """
    cv = ConditionVector()

    if intent.intent_type == InstructionType.MODIFY_RISK:
        # "Be careful" → low risk tolerance
        cv.risk_tolerance = 0.2  # Conservative
        cv.safety_language_modulation = 0.9  # High safety bias
        cv.skill_mode = "safety_critical"

    elif intent.intent_type == InstructionType.SPEED_MODE:
        # "Go fast" → high risk tolerance
        cv.risk_tolerance = 0.8  # Aggressive
        cv.energy_budget_wh = None  # Unlimited energy
        cv.skill_mode = "speed"
        cv.instruction_priority = 0.9  # Urgent

    elif intent.intent_type == InstructionType.ENERGY_CONSTRAINT:
        # "Save energy" → efficient mode
        cv.energy_budget_wh = 50.0  # Conservative budget
        cv.skill_mode = "energy_efficient"
        cv.risk_tolerance = 0.5  # Balanced (avoid failures that waste energy)

    elif intent.intent_type == InstructionType.PRIORITIZE_NOVELTY:
        # "Explore" → frontier mode
        cv.curriculum_phase = "frontier"
        cv.novelty_tier = 2  # Prioritize novel data
        cv.skill_mode = "exploration"

    elif intent.intent_type == InstructionType.TARGET_MPL:
        # "Increase productivity by X%" → set target
        uplift_pct = intent.parameters.get("target_uplift_pct", 0)
        current_mpl = _get_current_mpl()  # Retrieve from system
        cv.target_mpl_uplift = current_mpl * (uplift_pct / 100.0)

    elif intent.intent_type == InstructionType.DEPLOY_SKILL:
        # "Use dishwashing skill" → activate skill
        cv.skill_id = intent.parameters.get("skill_id")
        cv.skill_mode = "precision"  # Default for explicit skill requests

    elif intent.intent_type == InstructionType.PRECISION_MODE:
        # "Place exactly here" → precision
        cv.skill_mode = "precision"
        cv.risk_tolerance = 0.3  # Low tolerance (precision requires caution)

    elif intent.intent_type == InstructionType.TIME_CONSTRAINT:
        # "Finish in 30 seconds" → time budget
        cv.time_budget_sec = 30.0
        cv.skill_mode = "speed"
        cv.risk_tolerance = 0.6  # Moderate (balance speed vs safety)

    elif intent.intent_type == InstructionType.EXPLORATION_MODE:
        # "Figure out how to open this latch" → exploration + novelty
        cv.skill_mode = "exploration"
        cv.curriculum_phase = "frontier"
        cv.novelty_tier = 2

    return cv
```

---

## 4. Safety & Economic Constraint Rules

### 4.1. Clamping & Overrides

```python
def apply_safety_constraints(cv: ConditionVector, system_state: Dict) -> ConditionVector:
    """
    Clamp ConditionVector fields to safe/economically-viable ranges.

    Overrides textual instructions that violate hard constraints.
    """
    # Hard minimum risk tolerance (never go below 0.1, even if user says "be very careful")
    if cv.risk_tolerance is not None:
        cv.risk_tolerance = max(0.1, min(0.9, cv.risk_tolerance))

    # Energy budget floor (never below 10 Wh, even in "save energy" mode)
    if cv.energy_budget_wh is not None:
        cv.energy_budget_wh = max(10.0, cv.energy_budget_wh)

    # Time budget floor (never below 5 sec, prevents unsafe rushing)
    if cv.time_budget_sec is not None:
        cv.time_budget_sec = max(5.0, cv.time_budget_sec)

    # Economic viability check: If target_mpl_uplift exceeds max feasible, clamp
    if cv.target_mpl_uplift is not None:
        max_feasible_uplift = system_state.get("max_mpl_uplift", 10.0)
        if cv.target_mpl_uplift > max_feasible_uplift:
            cv.target_mpl_uplift = max_feasible_uplift
            # Emit warning: "Requested MPL uplift exceeds system capacity; clamped to X"

    return cv
```

---

### 4.2. Rejection Rules

**Reject instruction if:**

```python
REJECTED_PATTERNS = [
    r"ignore safety",
    r"break (the )?rule",
    r"override (safety|econ)",
    r"don't care (about|if)",
    r"just do it"  # Ambiguous; requires clarification
]

def should_reject_instruction(text: str) -> tuple[bool, str]:
    """
    Check if instruction violates safety policy.

    Returns:
        (should_reject: bool, reason: str)
    """
    text_lower = text.lower()

    for pattern in REJECTED_PATTERNS:
        if re.search(pattern, text_lower):
            return (True, f"Instruction contains unsafe pattern: '{pattern}'")

    # Check for conflicting constraints
    if "fast" in text_lower and "careful" in text_lower:
        return (True, "Conflicting instructions: 'fast' and 'careful' cannot both be satisfied")

    return (False, "")
```

---

## 5. TFD Interaction with Other Modules

### 5.1. SIMA-2 Integration

```python
def modulate_sima2_with_tfd(
    sima2_tags: List[Tag],
    condition_vector: ConditionVector
) -> List[Tag]:
    """
    Modulate SIMA-2 tag weights based on TFD instruction.

    Example:
    - If cv.risk_tolerance = 0.2 (careful), amplify RiskTag sensitivity
    - If cv.novelty_tier = 2 (explore), amplify OODTag importance
    """
    modulated_tags = []

    for tag in sima2_tags:
        if tag.type == "RiskTag" and condition_vector.risk_tolerance is not None:
            # Lower risk tolerance → higher risk sensitivity
            tag.severity *= (1.0 - condition_vector.risk_tolerance + 0.5)

        elif tag.type == "OODTag" and condition_vector.novelty_tier is not None:
            # Higher novelty tier → amplify OOD importance
            tag.severity *= (condition_vector.novelty_tier + 1.0)

        modulated_tags.append(tag)

    return modulated_tags
```

---

### 5.2. SemanticOrchestrator Integration

```python
def tfd_propose_task_refinement(
    condition_vector: ConditionVector,
    current_task_graph: TaskGraph
) -> Optional[TaskGraphRefinement]:
    """
    Propose task graph refinements based on TFD instruction.

    Example:
    - If cv.skill_mode = "precision", propose adding verification nodes
    - If cv.time_budget_sec is tight, propose removing optional subtasks
    """
    if condition_vector.skill_mode == "precision":
        return TaskGraphRefinement(
            type="add_verification_node",
            reason="Precision mode requested; adding post-task verification"
        )

    if condition_vector.time_budget_sec is not None and condition_vector.time_budget_sec < 20:
        return TaskGraphRefinement(
            type="remove_optional_subtasks",
            reason="Time budget is tight; removing non-critical subtasks"
        )

    return None
```

---

### 5.3. EconController Interaction (Advisory Only)

```python
def tfd_advise_econ_controller(
    condition_vector: ConditionVector,
    current_econ_state: Dict
) -> Dict[str, Any]:
    """
    Provide advisory inputs to EconController.

    TFD MUST NOT directly modify pricing or rewards.
    Instead, it provides "soft suggestions" that EconController can consider.
    """
    advice = {}

    if condition_vector.target_mpl_uplift is not None:
        advice["suggested_target_mpl"] = current_econ_state["current_mpl"] + condition_vector.target_mpl_uplift
        advice["urgency"] = condition_vector.instruction_priority or 0.5

    if condition_vector.energy_budget_wh is not None:
        advice["suggested_energy_cap"] = condition_vector.energy_budget_wh

    # EconController decides whether to accept advice based on viability
    return advice
```

---

## 6. Canonical Examples (20 Instructions)

### 6.1. Safety & Risk

**1. "Be very careful with the vase"**
```json
{
  "intent_type": "modify_risk_tolerance",
  "parameters": {},
  "condition_vector": {
    "risk_tolerance": 0.2,
    "safety_language_modulation": 0.9,
    "skill_mode": "safety_critical"
  }
}
```

**2. "Don't worry about damage, just go fast"**
```json
{
  "intent_type": "speed_mode",
  "parameters": {},
  "condition_vector": {
    "risk_tolerance": 0.8,
    "energy_budget_wh": null,
    "skill_mode": "speed",
    "instruction_priority": 0.9
  }
}
```

**3. "Gentle movements only"**
```json
{
  "intent_type": "precision_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "precision",
    "risk_tolerance": 0.3,
    "safety_language_modulation": 0.8
  }
}
```

---

### 6.2. Energy & Efficiency

**4. "Minimize power consumption"**
```json
{
  "intent_type": "energy_constraint",
  "parameters": {},
  "condition_vector": {
    "energy_budget_wh": 50.0,
    "skill_mode": "energy_efficient",
    "risk_tolerance": 0.5
  }
}
```

**5. "Save energy but don't sacrifice quality"**
```json
{
  "intent_type": "energy_constraint",
  "parameters": {},
  "condition_vector": {
    "energy_budget_wh": 75.0,
    "skill_mode": "energy_efficient",
    "risk_tolerance": 0.4,
    "safety_language_modulation": 0.7
  }
}
```

---

### 6.3. Skill Deployment

**6. "Use the dishwashing skill"**
```json
{
  "intent_type": "deploy_skill",
  "parameters": {"skill_id": "dishwashing"},
  "condition_vector": {
    "skill_id": "dishwashing",
    "skill_mode": "precision"
  }
}
```

**7. "Activate exploration mode"**
```json
{
  "intent_type": "exploration_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "exploration",
    "curriculum_phase": "frontier",
    "novelty_tier": 2
  }
}
```

---

### 6.4. Productivity & Targets

**8. "Increase productivity by 15%"**
```json
{
  "intent_type": "target_mpl_uplift",
  "parameters": {"target_uplift_pct": 15.0},
  "condition_vector": {
    "target_mpl_uplift": 9.0
  }
}
```
*(Assumes current MPL = 60 units/hr → 15% = 9 units/hr uplift)*

**9. "Work 20% faster"**
```json
{
  "intent_type": "target_mpl_uplift",
  "parameters": {"target_uplift_pct": 20.0},
  "condition_vector": {
    "target_mpl_uplift": 12.0
  }
}
```

---

### 6.5. Time Constraints

**10. "Finish this task in 30 seconds"**
```json
{
  "intent_type": "time_constraint",
  "parameters": {},
  "condition_vector": {
    "time_budget_sec": 30.0,
    "skill_mode": "speed",
    "risk_tolerance": 0.6
  }
}
```

**11. "Take your time, no rush"**
```json
{
  "intent_type": "precision_mode",
  "parameters": {},
  "condition_vector": {
    "time_budget_sec": null,
    "skill_mode": "precision",
    "risk_tolerance": 0.3
  }
}
```

---

### 6.6. Exploration & Novelty

**12. "Explore new ways to open this drawer"**
```json
{
  "intent_type": "exploration_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "exploration",
    "curriculum_phase": "frontier",
    "novelty_tier": 2
  }
}
```

**13. "Try a different approach"**
```json
{
  "intent_type": "prioritize_novelty",
  "parameters": {},
  "condition_vector": {
    "curriculum_phase": "frontier",
    "novelty_tier": 2,
    "skill_mode": "exploration"
  }
}
```

**14. "Figure out how to grip this weird object"**
```json
{
  "intent_type": "exploration_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "exploration",
    "curriculum_phase": "frontier",
    "novelty_tier": 2,
    "objective_vector": {"weird_object": 1.0}
  }
}
```

---

### 6.7. Mixed Constraints

**15. "Be quick but careful"**
```json
{
  "intent_type": "time_constraint",
  "parameters": {},
  "condition_vector": {
    "time_budget_sec": 20.0,
    "skill_mode": "precision",
    "risk_tolerance": 0.4,
    "safety_language_modulation": 0.7
  }
}
```

**16. "Maximize output while staying under 100 Wh"**
```json
{
  "intent_type": "energy_constraint",
  "parameters": {},
  "condition_vector": {
    "energy_budget_wh": 100.0,
    "skill_mode": "speed",
    "risk_tolerance": 0.6,
    "target_mpl_uplift": 5.0
  }
}
```

---

### 6.8. Precision & Verification

**17. "Place the cup exactly in the center"**
```json
{
  "intent_type": "precision_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "precision",
    "risk_tolerance": 0.2,
    "objective_vector": {"center": 1.0}
  }
}
```

**18. "Double-check before placing"**
```json
{
  "intent_type": "precision_mode",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "safety_critical",
    "risk_tolerance": 0.3,
    "instruction_priority": 0.8
  }
}
```

---

### 6.9. Contextual / Object-Specific

**19. "Sort the blue blocks, ignore the red ones"**
```json
{
  "intent_type": "deploy_skill",
  "parameters": {"skill_id": "sort_objects"},
  "condition_vector": {
    "skill_id": "sort_objects",
    "skill_mode": "precision",
    "objective_vector": {"blue": 1.0, "red": 0.0}
  }
}
```

**20. "Focus on clearing the left side first"**
```json
{
  "intent_type": "deploy_skill",
  "parameters": {},
  "condition_vector": {
    "skill_mode": "precision",
    "objective_vector": {"left_region": 1.0, "right_region": 0.0},
    "instruction_priority": 0.7
  }
}
```

---

## 7. Conflicts & Ambiguity Resolution

### 7.1. Conflicting Instructions

**Example**: "Go fast but be very careful"

**Resolution**:
```python
def resolve_conflict(intents: List[ParsedIntent]) -> ConditionVector:
    """
    Resolve conflicting intents using priority rules.

    Priority (highest first):
    1. Safety (MODIFY_RISK, PRECISION_MODE)
    2. Economic constraints (ENERGY_CONSTRAINT, TARGET_MPL)
    3. Speed/Time (SPEED_MODE, TIME_CONSTRAINT)
    """
    cv = ConditionVector()

    # Apply in priority order
    safety_intents = [i for i in intents if i.intent_type in [InstructionType.MODIFY_RISK, InstructionType.PRECISION_MODE]]
    speed_intents = [i for i in intents if i.intent_type in [InstructionType.SPEED_MODE, InstructionType.TIME_CONSTRAINT]]

    if safety_intents:
        # Safety wins: apply conservative settings
        cv.risk_tolerance = 0.3  # Careful
        cv.skill_mode = "precision"

        # But acknowledge speed request: reduce time budget slightly
        if speed_intents:
            cv.time_budget_sec = 15.0  # Moderate deadline

    return cv
```

---

### 7.2. Ambiguous Instructions

**Example**: "Just do it"

**Resolution**: Reject and request clarification
```json
{
  "status": "rejected",
  "reason": "Ambiguous instruction; please specify risk tolerance, time constraints, or skill mode"
}
```

---

## 8. Implementation Contracts

### 8.1. Module Structure

```
src/tfd/
├── __init__.py
├── text_op.py              # TextOp entry point
├── semantic_compiler.py    # parse_instruction(), compile_to_condition_vector()
├── safety_filter.py        # apply_safety_constraints(), should_reject_instruction()
├── patterns.py             # INTENT_PATTERNS, REJECTED_PATTERNS
└── examples.py             # Canonical examples for testing
```

---

### 8.2. API Contract

```python
class TextOp:
    """Entry point for Text-Front-Door module."""

    def process_instruction(self, text: str) -> TextOpResult:
        """
        Process natural language instruction.

        Returns:
            TextOpResult with condition_vector, status, and metadata
        """
        # 1. Check for rejection
        should_reject, reject_reason = should_reject_instruction(text)
        if should_reject:
            return TextOpResult(status="rejected", reason=reject_reason)

        # 2. Parse intent
        intent = parse_instruction(text)
        if intent.confidence < 0.5:
            return TextOpResult(status="low_confidence", reason="Could not parse instruction")

        # 3. Compile to ConditionVector
        cv = compile_to_condition_vector(intent)

        # 4. Apply safety constraints
        cv = apply_safety_constraints(cv, system_state=self.get_system_state())

        return TextOpResult(
            status="accepted",
            condition_vector=cv,
            parsed_intent=intent
        )
```

---

### 8.3. Determinism & Testing

**All TFD operations MUST be deterministic**:
- Same text → same ConditionVector (no randomness)
- Regex patterns are fixed
- Clamping rules are fixed

**Smoke Test**:
```python
def test_tfd_determinism():
    """Test that TFD produces identical output for same input."""
    text = "Be very careful with the vase"

    tfd = TextOp()
    result1 = tfd.process_instruction(text)
    result2 = tfd.process_instruction(text)

    assert result1.condition_vector == result2.condition_vector
```

---

**End of Text-Front-Door Complete Semantics**
