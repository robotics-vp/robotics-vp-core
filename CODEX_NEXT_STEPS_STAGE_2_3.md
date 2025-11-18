# Codex Implementation Instructions: Stage 2.3 — TaskGraphRefiner

**Date**: 2025-11-17
**Implementer**: Codex
**Prerequisites**: Stage 2.1 + Stage 2.2 complete (smoke tests green)
**Estimated Time**: 2-3 hours

---

## Quick Start

### What You're Building
A **task graph refinement system** that:
1. Reads `OntologyUpdateProposal[]` from Stage 2.2
2. Reads `SemanticPrimitive[]` from Stage 2.1 (optional)
3. Generates `TaskGraphRefinementProposal[]` (advisory-only)
4. Validates refinements against econ/datapack/DAG constraints
5. Outputs JSON-safe refinements for downstream consumption

### What You're NOT Building
- ❌ Task graph mutation logic (advisory only)
- ❌ DAG edge modification
- ❌ Node deletion (replace only)
- ❌ Reward math changes
- ❌ Economic/data parameter setting

---

## Implementation Checklist

### File 1: `src/sima2/task_graph_proposals.py`

**Purpose**: Refinement schema dataclasses

#### Step 1.1: Create RefinementType Enum
```python
from enum import Enum

class RefinementType(Enum):
    """Types of task graph refinement proposals."""
    SPLIT_TASK = "split_task"
    INSERT_CHECKPOINT = "insert_checkpoint"
    REORDER_TASKS = "reorder_tasks"
    MERGE_TASKS = "merge_tasks"
    ADD_PRECONDITION = "add_precondition"
    PARALLELIZE_TASKS = "parallelize_tasks"
    INSERT_RECOVERY = "insert_recovery"
    ADJUST_PRIORITY = "adjust_priority"
```

#### Step 1.2: Create RefinementPriority Enum
```python
class RefinementPriority(Enum):
    """Priority levels for refinement application."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

#### Step 1.3: Create TaskGraphRefinementProposal Dataclass
See full implementation in `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md` Section 3.1.

**Copy-paste from spec** — all fields are pre-defined.

**Validation**:
- ✅ Run: `python3 -c "from src.sima2.task_graph_proposals import *; print('OK')"`

---

### File 2: `src/sima2/task_graph_refiner.py`

**Purpose**: Refinement generation engine

#### Step 2.1: Imports
```python
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.sima2.ontology_proposals import (
    OntologyUpdateProposal,
    ProposalType as OntologyProposalType,
)
from src.sima2.task_graph_proposals import (
    TaskGraphRefinementProposal,
    RefinementType,
    RefinementPriority,
)
from src.orchestrator.task_graph import TaskGraph, TaskNode, TaskType, TaskStatus
from src.orchestrator.ontology import EnvironmentOntology
from src.orchestrator.economic_controller import EconSignals
from src.orchestrator.datapack_engine import DatapackSignals
```

#### Step 2.2: TaskGraphRefiner Class
See full implementation in `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md` Section 4.2.

**Required Methods** (copy from spec):
1. `generate_refinements(ontology_proposals, primitives)` → List[TaskGraphRefinementProposal]
2. `validate_refinements(refinements)` → List[TaskGraphRefinementProposal]
3. `_insert_checkpoint_from_gate(ont_prop)` → List[TaskGraphRefinementProposal]
4. `_split_task_for_fragility(ont_prop)` → List[TaskGraphRefinementProposal]
5. `_reorder_for_safety(ont_prop)` → List[TaskGraphRefinementProposal]
6. `_insert_recovery_for_risk(ont_prop)` → List[TaskGraphRefinementProposal]
7. `_reorder_for_energy(prim)` → List[TaskGraphRefinementProposal]
8. `_merge_redundant_tasks(prim)` → List[TaskGraphRefinementProposal]
9. `_adjust_priority_for_safety()` → List[TaskGraphRefinementProposal]
10. `_parallelize_for_throughput()` → List[TaskGraphRefinementProposal]
11. `_check_econ_constraints(ref)` → bool
12. `_check_datapack_constraints(ref)` → bool
13. `_check_dag_topology(ref)` → bool
14. `_check_preserves_nodes(ref)` → bool
15. `_make_proposal_id()` → str

**Copy-paste from spec** — all logic is pre-written.

**Validation**:
- ✅ Run: `python3 -c "from src.sima2.task_graph_refiner import *; print('OK')"`

---

### File 3: `scripts/smoke_test_task_graph_refiner.py`

**Purpose**: Smoke test validation

**Copy full implementation from** `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md` Section 7.2.

**Key Test Cases**:
1. Refinement generation
2. JSON-safety
3. Required fields
4. Constraint compliance
5. Refinement type coverage
6. Mandatory checkpoint insertion
7. Task splitting for fragility
8. Safety reordering
9. Priority assignment
10. Determinism
11. DAG topology preservation
12. Node preservation

**Validation**:
- ✅ Run: `python3 scripts/smoke_test_task_graph_refiner.py`
- ✅ Expected output: `[smoke_test_task_graph_refiner] All tests passed!`

---

### File 4: Update `scripts/run_all_smokes.py`

#### Step 4.1: Add to SMOKES List
```python
SMOKES = [
    # ... existing smokes ...
    ["python3", "scripts/smoke_test_ontology_update_engine.py"],
    ["python3", "scripts/smoke_test_task_graph_refiner.py"],  # NEW
]
```

**Validation**:
- ✅ Run: `python3 scripts/run_all_smokes.py`
- ✅ Expected: All smokes pass (including new test)

---

## Implementation Order

Execute in this order:

1. **File 1**: `src/sima2/task_graph_proposals.py`
   - Create enums + dataclass
   - Validate imports work

2. **File 2**: `src/sima2/task_graph_refiner.py`
   - Implement class skeleton
   - Copy-paste refinement methods from spec
   - Implement validation methods

3. **File 3**: `scripts/smoke_test_task_graph_refiner.py`
   - Copy full test from spec
   - Run test, fix any errors

4. **File 4**: Update `scripts/run_all_smokes.py`
   - Add new smoke test to list
   - Run full smoke suite

---

## Common Pitfalls

### Pitfall 1: Task Graph Mutation
**Wrong**:
```python
def generate_refinements(self, ontology_proposals):
    for prop in ontology_proposals:
        # ❌ FORBIDDEN: Direct mutation
        self.task_graph.add_child(TaskNode(...))
```

**Right**:
```python
def generate_refinements(self, ontology_proposals):
    refinements = []
    for prop in ontology_proposals:
        # ✅ Advisory refinement only
        refinement = TaskGraphRefinementProposal(
            refinement_type=RefinementType.INSERT_CHECKPOINT,
            proposed_changes={"checkpoint_task": {...}}
        )
        refinements.append(refinement)
    return refinements
```

### Pitfall 2: Node Deletion
**Wrong**:
```python
proposed_changes = {
    "delete_task": "pull_drawer",  # ❌ Forbidden
}
```

**Right**:
```python
proposed_changes = {
    "task_ids_to_merge": ["approach", "grasp"],  # ✅ Replace, not delete
    "merged_task": {...}
}
```

### Pitfall 3: DAG Cycles
**Wrong**:
```python
# Creating cycle: A → B → C → A
proposed_changes = {
    "new_preconditions": ["task_c"],  # ❌ Creates cycle if task_a → task_b → task_c
}
```

**Right**:
```python
# Validation should detect and reject cycles
# (Simplified in smoke test; production needs full topological sort)
```

---

## Testing Strategy

### Unit Testing (Smoke Test)
Run after implementing each file:
```bash
# Test File 1
python3 -c "from src.sima2.task_graph_proposals import *; print('File 1 OK')"

# Test File 2
python3 -c "from src.sima2.task_graph_refiner import *; print('File 2 OK')"

# Test File 3
python3 scripts/smoke_test_task_graph_refiner.py
```

### Integration Testing
Run after all files complete:
```bash
python3 scripts/run_all_smokes.py
```

### JSON Serialization Test
```python
import json
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.orchestrator.ontology import build_drawer_vase_ontology

refiner = TaskGraphRefiner(
    task_graph=build_drawer_vase_task_graph(),
    ontology=build_drawer_vase_ontology(),
)

ont_prop = OntologyUpdateProposal(
    proposal_id="test",
    proposal_type=ProposalType.ADD_SKILL_GATE,
    priority=ProposalPriority.HIGH,
    target_skill_id=2,
    proposed_changes={"gated_skill_id": 2, "preconditions": ["check_passed"]},
)

refinements = refiner.generate_refinements([ont_prop])

# Validate JSON-safety
for ref in refinements:
    json_str = json.dumps(ref.to_dict())
    print(f"✅ Refinement {ref.proposal_id} is JSON-safe")
```

---

## Debugging Checklist

If smoke test fails:

### Test 1 Fails (Refinement Generation)
- [ ] Check `generate_refinements()` returns non-empty list
- [ ] Check all `_insert/split/reorder_*()` methods are implemented
- [ ] Verify ontology proposals have required fields

### Test 6 Fails (Checkpoint Insertion)
- [ ] Check `_insert_checkpoint_from_gate()` is called
- [ ] Verify `ADD_SKILL_GATE` proposals trigger checkpoint insertion
- [ ] Check `mandatory=True` is set in proposed_changes

### Test 7 Fails (Task Splitting)
- [ ] Check `_split_task_for_fragility()` is called
- [ ] Verify `INFER_FRAGILITY` with fragility > 0.7 triggers split
- [ ] Check `new_sub_tasks` contains at least 2 sub-tasks

### Test 11 Fails (DAG Topology)
- [ ] Check `_check_dag_topology()` returns True (simplified check)
- [ ] In production, implement full cycle detection

### Test 12 Fails (Node Preservation)
- [ ] Check no `delete_task` keys in proposed_changes
- [ ] Verify only SPLIT_TASK/MERGE_TASKS can replace nodes

---

## Completion Checklist

Before marking Stage 2.3 complete:

- [ ] File 1 created: `src/sima2/task_graph_proposals.py`
- [ ] File 2 created: `src/sima2/task_graph_refiner.py`
- [ ] File 3 created: `scripts/smoke_test_task_graph_refiner.py`
- [ ] File 4 updated: `scripts/run_all_smokes.py`
- [ ] Smoke test passes: `python3 scripts/smoke_test_task_graph_refiner.py`
- [ ] All smokes pass: `python3 scripts/run_all_smokes.py`
- [ ] JSON serialization validated (manual test above)
- [ ] No task graph mutations (code review)
- [ ] Constraint validation tested (forbidden keys rejected)
- [ ] Spec committed: `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md`
- [ ] Codex instructions committed: `CODEX_NEXT_STEPS_STAGE_2_3.md`

---

## Success Criteria

Stage 2.3 is **COMPLETE** when:

```bash
$ python3 scripts/smoke_test_task_graph_refiner.py
[smoke_test_task_graph_refiner] Starting tests...
[TEST 1 PASS] Generated X refinements
[TEST 2 PASS] All X refinements are JSON-safe
[TEST 3 PASS] All refinements have required fields
[TEST 4 PASS] X/X refinements valid
[TEST 5 PASS] Refinement types: [...]
[TEST 6 PASS] Mandatory checkpoint insertion working (X checkpoints)
[TEST 7 PASS] Task splitting for fragility working (X splits)
[TEST 8 PASS/SKIP] Safety reordering working (X reorders)
[TEST 9 PASS] Priority assignment working (X CRITICAL)
[TEST 10 PASS] Determinism validated
[TEST 11 PASS] DAG topology preserved (no cycles)
[TEST 12 PASS] Node preservation validated
[smoke_test_task_graph_refiner] All tests passed!

$ python3 scripts/run_all_smokes.py
[run_all_smokes] All smokes passed.
```

---

**Go build! 🚀**
