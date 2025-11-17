# Codex Implementation Instructions: Stage 2.2 — OntologyUpdateEngine

**Date**: 2025-11-17
**Implementer**: Codex
**Prerequisites**: Stage 2.1 complete (smoke test green)
**Estimated Time**: 2-3 hours

---

## Quick Start

### What You're Building
A **proposal generation system** that:
1. Reads `SemanticPrimitive` objects from Stage 2.1
2. Generates `OntologyUpdateProposal` objects (advisory-only)
3. Validates proposals against econ/datapack/task-graph constraints
4. Outputs JSON-safe proposals for downstream consumption

### What You're NOT Building
- ❌ Ontology mutation logic (advisory only)
- ❌ Reward math changes
- ❌ Data valuation logic
- ❌ Task graph mutation

---

## Implementation Checklist

### File 1: `src/sima2/ontology_proposals.py`

**Purpose**: Proposal schema dataclasses

#### Step 1.1: Create ProposalType Enum
```python
from enum import Enum

class ProposalType(Enum):
    """Types of ontology update proposals."""
    ADD_AFFORDANCE = "add_affordance"
    ADJUST_RISK = "adjust_risk"
    INFER_FRAGILITY = "infer_fragility"
    ADD_OBJECT_CATEGORY = "add_object_category"
    ADD_SEMANTIC_TAG = "add_semantic_tag"
    ADD_SKILL_GATE = "add_skill_gate"
    ADD_SAFETY_CONSTRAINT = "add_safety_constraint"
    ADD_ENERGY_HEURISTIC = "add_energy_heuristic"
    UPDATE_OBJECT_RELATIONSHIP = "update_object_relationship"
```

#### Step 1.2: Create ProposalPriority Enum
```python
class ProposalPriority(Enum):
    """Priority levels for proposal application."""
    CRITICAL = "critical"  # Safety-critical (fragility, collision risk)
    HIGH = "high"  # Economic urgency-driven
    MEDIUM = "medium"  # Quality-of-life improvements
    LOW = "low"  # Nice-to-have, no urgency
```

#### Step 1.3: Create OntologyUpdateProposal Dataclass
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class OntologyUpdateProposal:
    """
    Schema for a single ontology update proposal.

    Advisory-only; does not mutate ontology directly.
    SemanticOrchestrator decides whether/how to apply.
    """
    # Identification
    proposal_id: str  # Unique ID (e.g., "prop_123abc")
    proposal_type: ProposalType
    priority: ProposalPriority = ProposalPriority.MEDIUM

    # Source tracking
    source_primitive_id: str = ""  # Which SemanticPrimitive triggered this
    source: str = "sima2"  # "sima2", "rule_based", "heuristic"

    # Target specification
    target_object_id: Optional[str] = None  # If affecting specific object
    target_skill_id: Optional[int] = None  # If affecting specific skill
    target_affordance_type: Optional[str] = None  # If affecting affordance

    # Proposal content (type-specific)
    proposed_changes: Dict[str, Any] = field(default_factory=dict)

    # Justification
    rationale: str = ""  # Human-readable explanation
    confidence: float = 1.0  # Confidence in proposal (0-1)

    # Constraint compliance
    respects_econ_constraints: bool = True
    respects_datapack_constraints: bool = True
    respects_task_graph: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "priority": self.priority.value,
            "source_primitive_id": self.source_primitive_id,
            "source": self.source,
            "target_object_id": self.target_object_id,
            "target_skill_id": self.target_skill_id,
            "target_affordance_type": self.target_affordance_type,
            "proposed_changes": self.proposed_changes,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "respects_econ_constraints": self.respects_econ_constraints,
            "respects_datapack_constraints": self.respects_datapack_constraints,
            "respects_task_graph": self.respects_task_graph,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OntologyUpdateProposal":
        """Create from dictionary."""
        return cls(
            proposal_id=d["proposal_id"],
            proposal_type=ProposalType(d["proposal_type"]),
            priority=ProposalPriority(d.get("priority", "medium")),
            source_primitive_id=d.get("source_primitive_id", ""),
            source=d.get("source", "sima2"),
            target_object_id=d.get("target_object_id"),
            target_skill_id=d.get("target_skill_id"),
            target_affordance_type=d.get("target_affordance_type"),
            proposed_changes=d.get("proposed_changes", {}),
            rationale=d.get("rationale", ""),
            confidence=d.get("confidence", 1.0),
            respects_econ_constraints=d.get("respects_econ_constraints", True),
            respects_datapack_constraints=d.get("respects_datapack_constraints", True),
            respects_task_graph=d.get("respects_task_graph", True),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )
```

**Validation**:
- ✅ Run: `python3 -c "from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority; print('OK')"`

---

### File 2: `src/sima2/ontology_update_engine.py`

**Purpose**: Proposal generation engine

#### Step 2.1: Imports
```python
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalType,
    ProposalPriority,
)
from src.orchestrator.ontology import (
    EnvironmentOntology,
    AffordanceType,
    ObjectCategory,
)
from src.orchestrator.task_graph import TaskGraph
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals
```

#### Step 2.2: OntologyUpdateEngine Class Skeleton
```python
class OntologyUpdateEngine:
    """
    Generates ontology update proposals from semantic primitives.

    Advisory-only; does not mutate ontology state.
    """

    def __init__(
        self,
        ontology: EnvironmentOntology,
        task_graph: Optional[TaskGraph] = None,
        econ_signals: Optional[EconSignals] = None,
        datapack_signals: Optional[DatapackSignals] = None,
    ):
        """
        Initialize OntologyUpdateEngine.

        Args:
            ontology: Current ontology (read-only)
            task_graph: Current task graph (read-only)
            econ_signals: Economic constraints (optional)
            datapack_signals: Data constraints (optional)
        """
        self.ontology = ontology
        self.task_graph = task_graph
        self.econ_signals = econ_signals or EconSignals()
        self.datapack_signals = datapack_signals or DatapackSignals()

        # Proposal generation state (for determinism)
        self._proposal_counter = 0
```

#### Step 2.3: Core Methods

##### generate_proposals()
```python
def generate_proposals(
    self, primitives: List[SemanticPrimitive]
) -> List[OntologyUpdateProposal]:
    """
    Generate ontology update proposals from semantic primitives.

    Args:
        primitives: List of SemanticPrimitives from Stage 2.1

    Returns:
        List of OntologyUpdateProposals (advisory-only)
    """
    proposals = []

    for prim in primitives:
        # 1. Affordance proposals
        proposals.extend(self._propose_affordances(prim))

        # 2. Risk adjustment proposals
        proposals.extend(self._propose_risk_adjustments(prim))

        # 3. Fragility inference proposals
        proposals.extend(self._propose_fragility_inference(prim))

        # 4. Skill gating proposals
        proposals.extend(self._propose_skill_gates(prim))

        # 5. Energy heuristic proposals
        proposals.extend(self._propose_energy_heuristics(prim))

        # 6. Semantic tag proposals
        proposals.extend(self._propose_semantic_tags(prim))

    return proposals
```

##### _make_proposal_id()
```python
def _make_proposal_id(self) -> str:
    """Generate unique proposal ID."""
    self._proposal_counter += 1
    return f"prop_{self._proposal_counter:06d}_{uuid.uuid4().hex[:6]}"
```

##### _risk_level_to_float()
```python
@staticmethod
def _risk_level_to_float(risk_level: str) -> float:
    """Convert categorical risk to float."""
    mapping = {"low": 0.1, "medium": 0.5, "high": 0.9}
    return mapping.get(risk_level, 0.5)
```

#### Step 2.4: Proposal Generation Methods

See full implementation in `STAGE_2_2_ONTOLOGY_UPDATE_ENGINE_SPEC.md` Section 4.2.

**Required Methods** (copy from spec):
1. `_propose_affordances(prim)` → List[OntologyUpdateProposal]
2. `_propose_risk_adjustments(prim)` → List[OntologyUpdateProposal]
3. `_propose_fragility_inference(prim)` → List[OntologyUpdateProposal]
4. `_propose_skill_gates(prim)` → List[OntologyUpdateProposal]
5. `_propose_energy_heuristics(prim)` → List[OntologyUpdateProposal]
6. `_propose_semantic_tags(prim)` → List[OntologyUpdateProposal]

**Copy-paste from spec** — all logic is pre-written.

#### Step 2.5: Validation Methods

```python
def validate_proposals(
    self, proposals: List[OntologyUpdateProposal]
) -> List[OntologyUpdateProposal]:
    """
    Validate proposals against constraints.

    Filters out proposals that violate econ/datapack/task-graph constraints.

    Args:
        proposals: List of proposals to validate

    Returns:
        List of valid proposals
    """
    valid = []

    for prop in proposals:
        # Check econ constraints
        if not self._check_econ_constraints(prop):
            prop.respects_econ_constraints = False
            continue

        # Check datapack constraints
        if not self._check_datapack_constraints(prop):
            prop.respects_datapack_constraints = False
            continue

        # Check task graph constraints
        if not self._check_task_graph_constraints(prop):
            prop.respects_task_graph = False
            continue

        valid.append(prop)

    return valid

def _check_econ_constraints(self, prop: OntologyUpdateProposal) -> bool:
    """
    Check if proposal respects economic constraints.

    Forbidden:
    - Modifying price_per_unit
    - Modifying damage_cost
    - Modifying reward weights
    """
    # Proposals cannot set economic parameters
    forbidden_keys = {
        "price_per_unit",
        "damage_cost",
        "wage_parity",
        "alpha",
        "beta",
        "gamma",
    }

    if any(k in prop.proposed_changes for k in forbidden_keys):
        return False

    return True

def _check_datapack_constraints(self, prop: OntologyUpdateProposal) -> bool:
    """
    Check if proposal respects datapack constraints.

    Forbidden:
    - Modifying tier classification
    - Modifying novelty scoring
    """
    forbidden_keys = {"tier", "novelty_score", "data_premium"}

    if any(k in prop.proposed_changes for k in forbidden_keys):
        return False

    return True

def _check_task_graph_constraints(self, prop: OntologyUpdateProposal) -> bool:
    """
    Check if proposal respects task graph constraints.

    Forbidden:
    - Deleting tasks
    - Modifying task dependencies directly
    """
    if prop.proposal_type == ProposalType.ADD_SKILL_GATE:
        # Skill gates are allowed (they add preconditions, not modify)
        return True

    # Other proposals should not reference task deletion
    if "delete_task" in prop.proposed_changes:
        return False

    return True
```

**Validation**:
- ✅ Run: `python3 -c "from src.sima2.ontology_update_engine import OntologyUpdateEngine; print('OK')"`

---

### File 3: `scripts/smoke_test_ontology_update_engine.py`

**Purpose**: Smoke test validation

**Copy full implementation from** `STAGE_2_2_ONTOLOGY_UPDATE_ENGINE_SPEC.md` Section 7.2.

**Key Test Cases**:
1. Proposal generation
2. JSON-safety
3. Required fields
4. Constraint compliance
5. Proposal type coverage
6. Fragility inference
7. Risk adjustment
8. Skill gating
9. Priority assignment
10. Determinism

**Validation**:
- ✅ Run: `python3 scripts/smoke_test_ontology_update_engine.py`
- ✅ Expected output: `[smoke_test_ontology_update_engine] All tests passed!`

---

### File 4: Update `scripts/run_all_smokes.py`

#### Step 4.1: Add to SMOKES List
```python
SMOKES = [
    ["python3", "scripts/smoke_test_dependency_hierarchy.py"],
    ["python3", "scripts/smoke_test_pareto_frontier.py"],
    ["python3", "scripts/smoke_test_semantic_feedback_loop.py"],
    ["python3", "scripts/smoke_test_reward_builder.py"],
    ["python3", "scripts/smoke_test_reward_heads.py"],
    ["python3", "scripts/smoke_test_datapack_rl_ingestion.py"],
    ["python3", "scripts/smoke_test_stage1_pipeline.py"],
    ["python3", "scripts/smoke_test_stage1_to_rl_sampling.py"],
    ["python3", "scripts/smoke_test_sima2_semantic_extraction.py"],
    ["python3", "scripts/smoke_test_ontology_update_engine.py"],  # NEW
]
```

**Validation**:
- ✅ Run: `python3 scripts/run_all_smokes.py`
- ✅ Expected: All smokes pass (including new test)

---

## Implementation Order

Execute in this order:

1. **File 1**: `src/sima2/ontology_proposals.py`
   - Create enums + dataclass
   - Validate imports work

2. **File 2**: `src/sima2/ontology_update_engine.py`
   - Implement class skeleton
   - Copy-paste proposal methods from spec
   - Implement validation methods

3. **File 3**: `scripts/smoke_test_ontology_update_engine.py`
   - Copy full test from spec
   - Run test, fix any errors

4. **File 4**: Update `scripts/run_all_smokes.py`
   - Add new smoke test to list
   - Run full smoke suite

---

## Common Pitfalls

### Pitfall 1: Ontology Mutation
**Wrong**:
```python
def generate_proposals(self, primitives):
    for prim in primitives:
        # ❌ FORBIDDEN: Direct mutation
        self.ontology.add_object(ObjectSpec(...))
```

**Right**:
```python
def generate_proposals(self, primitives):
    proposals = []
    for prim in primitives:
        # ✅ Advisory proposal only
        proposal = OntologyUpdateProposal(
            proposal_type=ProposalType.ADD_OBJECT_CATEGORY,
            proposed_changes={"category": "fragile", ...}
        )
        proposals.append(proposal)
    return proposals
```

### Pitfall 2: Non-JSON-Safe Data
**Wrong**:
```python
proposed_changes = {
    "confidence": np.float32(0.8),  # ❌ Not JSON-safe
}
```

**Right**:
```python
proposed_changes = {
    "confidence": float(0.8),  # ✅ Native Python float
}
```

### Pitfall 3: Violating Constraints
**Wrong**:
```python
proposed_changes = {
    "price_per_unit": 10.0,  # ❌ Econ parameter (forbidden)
}
```

**Right**:
```python
proposed_changes = {
    "risk_level": 0.8,  # ✅ Ontology parameter (allowed)
}
```

---

## Testing Strategy

### Unit Testing (Smoke Test)
Run after implementing each file:
```bash
# Test File 1
python3 -c "from src.sima2.ontology_proposals import *; print('File 1 OK')"

# Test File 2
python3 -c "from src.sima2.ontology_update_engine import *; print('File 2 OK')"

# Test File 3
python3 scripts/smoke_test_ontology_update_engine.py
```

### Integration Testing
Run after all files complete:
```bash
python3 scripts/run_all_smokes.py
```

### JSON Serialization Test
```python
import json
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.orchestrator.ontology import build_drawer_vase_ontology

engine = OntologyUpdateEngine(ontology=build_drawer_vase_ontology())
prim = SemanticPrimitive(
    primitive_id="test",
    task_type="test",
    tags=["fragile"],
    risk_level="high",
    energy_intensity=0.1,
    success_rate=0.9,
    avg_steps=5.0,
)
proposals = engine.generate_proposals([prim])

# Validate JSON-safety
for prop in proposals:
    json_str = json.dumps(prop.to_dict())
    print(f"✅ Proposal {prop.proposal_id} is JSON-safe")
```

---

## Debugging Checklist

If smoke test fails:

### Test 1 Fails (Proposal Generation)
- [ ] Check `generate_proposals()` returns non-empty list
- [ ] Check all `_propose_*()` methods are implemented
- [ ] Verify primitives have required fields

### Test 2 Fails (JSON-Safety)
- [ ] Check `proposed_changes` contains only JSON-safe types
- [ ] No numpy types (use `float()`, not `np.float32()`)
- [ ] No torch tensors

### Test 3 Fails (Required Fields)
- [ ] Check `proposal_id` is generated
- [ ] Check `source_primitive_id` is set
- [ ] Check `rationale` is non-empty

### Test 4 Fails (Constraint Compliance)
- [ ] Check validation methods reject forbidden keys
- [ ] Verify `respects_*_constraints` flags are set correctly

### Test 7 Fails (Risk Adjustment)
- [ ] Check `econ_signals.error_urgency` is > 0.5 in test
- [ ] Verify risk multiplier logic in `_propose_risk_adjustments()`

### Test 10 Fails (Determinism)
- [ ] Proposal IDs will differ (UUID), but types/counts should match
- [ ] Check `_proposal_counter` is incremented consistently

---

## Completion Checklist

Before marking Stage 2.2 complete:

- [ ] File 1 created: `src/sima2/ontology_proposals.py`
- [ ] File 2 created: `src/sima2/ontology_update_engine.py`
- [ ] File 3 created: `scripts/smoke_test_ontology_update_engine.py`
- [ ] File 4 updated: `scripts/run_all_smokes.py`
- [ ] Smoke test passes: `python3 scripts/smoke_test_ontology_update_engine.py`
- [ ] All smokes pass: `python3 scripts/run_all_smokes.py`
- [ ] JSON serialization validated (manual test above)
- [ ] No ontology mutations (code review)
- [ ] Constraint validation tested (forbidden keys rejected)
- [ ] Spec committed: `STAGE_2_2_ONTOLOGY_UPDATE_ENGINE_SPEC.md`
- [ ] Codex instructions committed: `CODEX_NEXT_STEPS_STAGE_2_2.md`

---

## Questions / Blockers

If you encounter issues:

1. **Import errors**: Check all dependencies exist in `src/` directories
2. **Attribute errors**: Verify `EconSignals`/`DatapackSignals` have expected fields
3. **Type errors**: Ensure all enums use `.value` for serialization
4. **Test failures**: Check test expectations match implementation

**Slack me if**:
- Imports fail (missing dependencies)
- Constraint logic is unclear
- Proposal format is ambiguous

---

## Success Criteria

Stage 2.2 is **COMPLETE** when:

```bash
$ python3 scripts/smoke_test_ontology_update_engine.py
[smoke_test_ontology_update_engine] Starting tests...
[TEST 1 PASS] Generated 15 proposals
[TEST 2 PASS] All 15 proposals are JSON-safe
[TEST 3 PASS] All proposals have required fields
[TEST 4 PASS] 15/15 proposals valid
[TEST 5 PASS] Proposal types: ['add_affordance', 'adjust_risk', 'infer_fragility', 'add_skill_gate', ...]
[TEST 6 PASS] Fragility inference working (3 proposals)
[TEST 7 PASS] Risk adjustment working (2 proposals)
[TEST 8 PASS] Skill gating working (6 proposals)
[TEST 9 PASS] Priority assignment working (3 CRITICAL)
[TEST 10 PASS] Determinism validated
[smoke_test_ontology_update_engine] All tests passed!

$ python3 scripts/run_all_smokes.py
[run_all_smokes] Running python3 scripts/smoke_test_dependency_hierarchy.py
[run_all_smokes] Running python3 scripts/smoke_test_pareto_frontier.py
...
[run_all_smokes] Running python3 scripts/smoke_test_ontology_update_engine.py
[run_all_smokes] All smokes passed.
```

---

**Go build! 🚀**
