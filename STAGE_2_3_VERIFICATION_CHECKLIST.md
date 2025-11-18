# Stage 2.3 Verification Checklist

**Date**: 2025-11-17
**Stage**: TaskGraphRefiner Design → Codex Implementation
**Status**: Design Complete — Awaiting Codex

---

## Pre-Implementation Checklist (User Review)

Before handing off to Codex, verify:

### Documentation Deliverables
- [ ] `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md` exists
  - [ ] Section 1: Overview complete
  - [ ] Section 2: Constraint sources mapped (6 upstream sources)
  - [ ] Section 3: TaskGraphRefinementProposal schema defined (8 types)
  - [ ] Section 4: TaskGraphRefiner module design complete
  - [ ] Section 5: Causality constraints documented
  - [ ] Section 6: Pipeline contract specified
  - [ ] Section 7: Smoke test spec complete (12 test cases)
  - [ ] Section 8: Acceptance criteria defined

- [ ] `CODEX_NEXT_STEPS_STAGE_2_3.md` exists
  - [ ] File 1 instructions (task_graph_proposals.py)
  - [ ] File 2 instructions (task_graph_refiner.py)
  - [ ] File 3 instructions (smoke_test_task_graph_refiner.py)
  - [ ] File 4 instructions (update run_all_smokes.py)
  - [ ] Common pitfalls documented (mutation, deletion, cycles)
  - [ ] Validation commands provided
  - [ ] Success criteria defined

- [ ] `STAGE_2_3_PIPELINE_DIAGRAM.md` exists
  - [ ] Stage 2.3 architecture diagram
  - [ ] Refinement generation flow
  - [ ] Validation pipeline
  - [ ] Refinement type mappings
  - [ ] End-to-end examples

- [ ] `STAGE_2_3_SUMMARY.md` exists
  - [ ] Deliverables listed
  - [ ] Key design decisions explained
  - [ ] Contract guarantees documented
  - [ ] Comparison with Stage 2.2

### Design Validation
- [ ] All 8 RefinementTypes defined:
  - [ ] SPLIT_TASK
  - [ ] INSERT_CHECKPOINT
  - [ ] REORDER_TASKS
  - [ ] MERGE_TASKS
  - [ ] ADD_PRECONDITION
  - [ ] PARALLELIZE_TASKS
  - [ ] INSERT_RECOVERY
  - [ ] ADJUST_PRIORITY

- [ ] TaskGraphRefinementProposal schema complete:
  - [ ] proposal_id: str
  - [ ] refinement_type: RefinementType
  - [ ] priority: RefinementPriority
  - [ ] source_primitive_ids: List[str]
  - [ ] source_ontology_proposal_ids: List[str]
  - [ ] target_task_ids: List[str]
  - [ ] parent_task_id: Optional[str]
  - [ ] proposed_changes: Dict[str, Any]
  - [ ] rationale: str
  - [ ] confidence: float
  - [ ] respects_*_constraints: bool (4 flags)
  - [ ] to_dict() / from_dict() methods

- [ ] TaskGraphRefiner methods specified:
  - [ ] generate_refinements()
  - [ ] validate_refinements()
  - [ ] _insert_checkpoint_from_gate()
  - [ ] _split_task_for_fragility()
  - [ ] _reorder_for_safety()
  - [ ] _insert_recovery_for_risk()
  - [ ] _reorder_for_energy()
  - [ ] _merge_redundant_tasks()
  - [ ] _adjust_priority_for_safety()
  - [ ] _parallelize_for_throughput()
  - [ ] _check_econ_constraints()
  - [ ] _check_datapack_constraints()
  - [ ] _check_dag_topology()
  - [ ] _check_preserves_nodes()
  - [ ] _make_proposal_id()

- [ ] Constraint compliance rules defined:
  - [ ] Forbidden econ keys: price_per_unit, damage_cost, alpha/beta/gamma
  - [ ] Forbidden datapack keys: tier, novelty_score, data_premium
  - [ ] Forbidden operations: node deletion (except SPLIT/MERGE), cycle creation

- [ ] Smoke test coverage complete:
  - [ ] Test 1: Refinement generation
  - [ ] Test 2: JSON-safety
  - [ ] Test 3: Required fields
  - [ ] Test 4: Constraint compliance
  - [ ] Test 5: Refinement type coverage
  - [ ] Test 6: Mandatory checkpoint insertion (from ADD_SKILL_GATE)
  - [ ] Test 7: Task splitting for fragility
  - [ ] Test 8: Safety reordering
  - [ ] Test 9: Priority assignment
  - [ ] Test 10: Determinism
  - [ ] Test 11: DAG topology preservation
  - [ ] Test 12: Node preservation

- [ ] Mandatory checkpoint insertion contract:
  - [ ] ADD_SKILL_GATE → INSERT_CHECKPOINT (MUST trigger)
  - [ ] Priority: CRITICAL
  - [ ] mandatory: True

---

## Post-Implementation Checklist (Codex Validation)

After Codex completes implementation:

### Files Created
- [ ] `src/sima2/task_graph_proposals.py` exists
  - [ ] RefinementType enum (8 values)
  - [ ] RefinementPriority enum (4 values)
  - [ ] TaskGraphRefinementProposal dataclass
  - [ ] to_dict() method
  - [ ] from_dict() method
  - [ ] No syntax errors
  - [ ] Imports work: `python3 -c "from src.sima2.task_graph_proposals import *"`

- [ ] `src/sima2/task_graph_refiner.py` exists
  - [ ] TaskGraphRefiner class
  - [ ] __init__() method
  - [ ] generate_refinements() method
  - [ ] validate_refinements() method
  - [ ] All 10 refinement generation methods
  - [ ] All 4 constraint check methods
  - [ ] _make_proposal_id() helper
  - [ ] No syntax errors
  - [ ] Imports work: `python3 -c "from src.sima2.task_graph_refiner import *"`

- [ ] `scripts/smoke_test_task_graph_refiner.py` exists
  - [ ] _make_test_ontology_proposals() helper
  - [ ] _make_test_primitives() helper
  - [ ] main() function
  - [ ] All 12 test cases implemented
  - [ ] No syntax errors
  - [ ] Executable: `chmod +x scripts/smoke_test_task_graph_refiner.py`

- [ ] `scripts/run_all_smokes.py` updated
  - [ ] New smoke test added to SMOKES list
  - [ ] Line: `["python3", "scripts/smoke_test_task_graph_refiner.py"]`

### Smoke Test Results
Run: `python3 scripts/smoke_test_task_graph_refiner.py`

- [ ] Test 1 PASS: Refinement generation
  - [ ] Output: `Generated X refinements` (X > 0)

- [ ] Test 2 PASS: JSON-safety
  - [ ] All refinements serialize to JSON without errors

- [ ] Test 3 PASS: Required fields
  - [ ] All refinements have proposal_id, refinement_type, priority, rationale

- [ ] Test 4 PASS: Constraint compliance
  - [ ] Output: `X/X refinements valid`
  - [ ] All valid refinements have respects_*_constraints = True

- [ ] Test 5 PASS: Refinement type coverage
  - [ ] At least 3 refinement types generated (INSERT_CHECKPOINT, SPLIT_TASK, REORDER_TASKS expected)

- [ ] Test 6 PASS: Mandatory checkpoint insertion
  - [ ] ADD_SKILL_GATE proposal triggers INSERT_CHECKPOINT refinement
  - [ ] Checkpoint has mandatory=True
  - [ ] Priority is CRITICAL

- [ ] Test 7 PASS: Task splitting
  - [ ] INFER_FRAGILITY with high fragility triggers SPLIT_TASK
  - [ ] new_sub_tasks contains >= 2 sub-tasks

- [ ] Test 8 PASS/SKIP: Safety reordering
  - [ ] ADD_SAFETY_CONSTRAINT triggers REORDER_TASKS (or SKIP if graph optimal)

- [ ] Test 9 PASS: Priority assignment
  - [ ] At least 1 CRITICAL priority refinement

- [ ] Test 10 PASS: Determinism
  - [ ] Same inputs → same refinement types/counts (UUIDs may differ)

- [ ] Test 11 PASS: DAG topology preservation
  - [ ] respects_dag_topology = True for all valid refinements

- [ ] Test 12 PASS: Node preservation
  - [ ] No "delete_task" in proposed_changes (except SPLIT/MERGE)
  - [ ] preserves_existing_nodes = True for all valid refinements

- [ ] Final output: `[smoke_test_task_graph_refiner] All tests passed!`

### Full Smoke Suite
Run: `python3 scripts/run_all_smokes.py`

- [ ] All previous smokes pass
- [ ] New smoke test passes
- [ ] Final output: `[run_all_smokes] All smokes passed.`

### Code Quality
- [ ] No task graph mutations in TaskGraphRefiner
  - [ ] Search for: `task_graph.add_child(` → should NOT appear
  - [ ] Search for: `task_graph.mark_` → should NOT appear
  - [ ] Search for: `task_node.children.append(` → should NOT appear

- [ ] No node deletions (except SPLIT/MERGE)
  - [ ] Search for: `delete_task` in proposed_changes → only in SPLIT/MERGE

- [ ] All proposed_changes are JSON-safe
  - [ ] No numpy types in proposed_changes
  - [ ] No torch tensors in proposed_changes

- [ ] Constraint validation works
  - [ ] Forbidden econ keys trigger validation failure
  - [ ] Forbidden datapack keys trigger validation failure
  - [ ] Cycle creation triggers validation failure (if detected)

- [ ] Mandatory checkpoint insertion
  - [ ] Every ADD_SKILL_GATE proposal produces INSERT_CHECKPOINT refinement
  - [ ] Checkpoint task has all required fields
  - [ ] insert_before_task_id references correct task

### Manual Validation Tests

#### Test A: JSON Serialization
```bash
python3 -c "
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
    proposal_id='test',
    proposal_type=ProposalType.ADD_SKILL_GATE,
    priority=ProposalPriority.HIGH,
    target_skill_id=2,
    proposed_changes={'gated_skill_id': 2, 'preconditions': ['check']},
)

refinements = refiner.generate_refinements([ont_prop])
for ref in refinements:
    json_str = json.dumps(ref.to_dict())
    assert json_str, 'JSON serialization failed'
print('✅ Manual JSON test passed')
"
```
- [ ] Output: `✅ Manual JSON test passed`

#### Test B: Constraint Violation Detection
```bash
python3 -c "
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.task_graph_proposals import TaskGraphRefinementProposal, RefinementType, RefinementPriority
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.orchestrator.ontology import build_drawer_vase_ontology

refiner = TaskGraphRefiner(
    task_graph=build_drawer_vase_task_graph(),
    ontology=build_drawer_vase_ontology(),
)

# Create invalid refinement (contains forbidden key)
invalid_ref = TaskGraphRefinementProposal(
    proposal_id='test_invalid',
    refinement_type=RefinementType.REORDER_TASKS,
    priority=RefinementPriority.HIGH,
    proposed_changes={'price_per_unit': 10.0},  # FORBIDDEN
)

valid = refiner.validate_refinements([invalid_ref])
assert len(valid) == 0, 'Validation should reject forbidden keys'
print('✅ Constraint violation detection works')
"
```
- [ ] Output: `✅ Constraint violation detection works`

#### Test C: Mandatory Checkpoint Insertion
```bash
python3 -c "
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.sima2.task_graph_proposals import RefinementType, RefinementPriority
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.orchestrator.ontology import build_drawer_vase_ontology

refiner = TaskGraphRefiner(
    task_graph=build_drawer_vase_task_graph(),
    ontology=build_drawer_vase_ontology(),
)

# ADD_SKILL_GATE must trigger INSERT_CHECKPOINT
gate_prop = OntologyUpdateProposal(
    proposal_id='gate_test',
    proposal_type=ProposalType.ADD_SKILL_GATE,
    priority=ProposalPriority.HIGH,
    target_skill_id=2,
    proposed_changes={'gated_skill_id': 2, 'preconditions': ['check']},
)

refinements = refiner.generate_refinements([gate_prop])
checkpoint_refs = [r for r in refinements if r.refinement_type == RefinementType.INSERT_CHECKPOINT]
assert checkpoint_refs, 'ADD_SKILL_GATE must produce INSERT_CHECKPOINT'
assert checkpoint_refs[0].priority == RefinementPriority.CRITICAL, 'Checkpoint must be CRITICAL'
assert checkpoint_refs[0].proposed_changes.get('mandatory') is True, 'Checkpoint must be mandatory'
print('✅ Mandatory checkpoint insertion works')
"
```
- [ ] Output: `✅ Mandatory checkpoint insertion works`

#### Test D: Determinism
```bash
python3 -c "
from src.sima2.task_graph_refiner import TaskGraphRefiner
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.orchestrator.task_graph import build_drawer_vase_task_graph
from src.orchestrator.ontology import build_drawer_vase_ontology

refiner = TaskGraphRefiner(
    task_graph=build_drawer_vase_task_graph(),
    ontology=build_drawer_vase_ontology(),
)

ont_prop = OntologyUpdateProposal(
    proposal_id='test',
    proposal_type=ProposalType.ADD_SKILL_GATE,
    priority=ProposalPriority.HIGH,
    target_skill_id=2,
    proposed_changes={'gated_skill_id': 2},
)

# Generate twice
refinements_1 = refiner.generate_refinements([ont_prop])
refinements_2 = refiner.generate_refinements([ont_prop])

# Check counts match
assert len(refinements_1) == len(refinements_2), 'Refinement count mismatch'

# Check types match (UUIDs will differ)
types_1 = sorted([r.refinement_type.value for r in refinements_1])
types_2 = sorted([r.refinement_type.value for r in refinements_2])
assert types_1 == types_2, 'Refinement types mismatch'

print('✅ Determinism validated')
"
```
- [ ] Output: `✅ Determinism validated`

---

## Final Approval Checklist

Before proceeding to Stage 2.4:

- [ ] All documentation files exist and are complete
- [ ] All implementation files created by Codex
- [ ] Smoke test passes (12/12 tests)
- [ ] Full smoke suite passes
- [ ] Manual validation tests pass (A, B, C, D)
- [ ] Code quality checks pass (no mutations, JSON-safe)
- [ ] Constraint validation works correctly
- [ ] Mandatory checkpoint insertion works
- [ ] Determinism validated
- [ ] DAG topology preserved
- [ ] No blockers or errors

---

## Approval Sign-Off

**Reviewed by**: _______________________
**Date**: _______________________
**Status**: [ ] APPROVED — Ready for Stage 2.4

**Notes**:
```
[Any additional observations or concerns]
```

---

## If Issues Found

### Common Issues & Fixes

**Issue**: Smoke test fails on Test 1 (no refinements generated)
- **Fix**: Check `generate_refinements()` processes all ontology proposals
- **Verify**: Ontology proposals have correct proposal_types

**Issue**: Smoke test fails on Test 6 (no checkpoint insertion)
- **Fix**: Check `_insert_checkpoint_from_gate()` is called for ADD_SKILL_GATE
- **Verify**: Task graph contains nodes with target skill_id

**Issue**: Smoke test fails on Test 7 (no task splitting)
- **Fix**: Check `_split_task_for_fragility()` fragility threshold (>= 0.7)
- **Verify**: INFER_FRAGILITY proposals have high fragility values

**Issue**: Smoke test fails on Test 11 (DAG topology)
- **Fix**: Check `_check_dag_topology()` returns True (simplified check)
- **Note**: Production should implement full topological sort

**Issue**: Smoke test fails on Test 12 (node deletion detected)
- **Fix**: Check no `delete_task` keys except in SPLIT_TASK/MERGE_TASKS
- **Verify**: `_check_preserves_nodes()` logic

---

## Next Steps After Approval

1. **User**: Mark Stage 2.3 as COMPLETE
2. **User**: Ask Claude: "Ready for Stage 2.4?"
3. **Claude**: Design Stage 2.4 spec (SemanticTagPropagator or orchestrator integration)
4. **Codex**: Implement Stage 2.4
5. **Repeat**: Until all Stage 2 modules complete

---

**End of Verification Checklist**
