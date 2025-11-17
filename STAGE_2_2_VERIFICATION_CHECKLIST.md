# Stage 2.2 Verification Checklist

**Date**: 2025-11-17
**Stage**: OntologyUpdateEngine Design → Codex Implementation
**Status**: Design Complete — Awaiting Codex

---

## Pre-Implementation Checklist (User Review)

Before handing off to Codex, verify:

### Documentation Deliverables
- [ ] `STAGE_2_2_ONTOLOGY_UPDATE_ENGINE_SPEC.md` exists
  - [ ] Section 1: Overview complete
  - [ ] Section 2: Constraint sources mapped
  - [ ] Section 3: OntologyUpdateProposal schema defined (9 types)
  - [ ] Section 4: OntologyUpdateEngine module design complete
  - [ ] Section 5: Causality constraints documented
  - [ ] Section 6: Pipeline contract specified
  - [ ] Section 7: Smoke test spec complete (10 test cases)
  - [ ] Section 8: Codex instructions included
  - [ ] Section 9: Pipeline diagrams present
  - [ ] Section 10: SemanticOrchestratorV2 preview included

- [ ] `CODEX_NEXT_STEPS_STAGE_2_2.md` exists
  - [ ] File 1 instructions (ontology_proposals.py)
  - [ ] File 2 instructions (ontology_update_engine.py)
  - [ ] File 3 instructions (smoke_test_ontology_update_engine.py)
  - [ ] File 4 instructions (update run_all_smokes.py)
  - [ ] Step-by-step implementation order
  - [ ] Common pitfalls documented
  - [ ] Validation commands provided
  - [ ] Success criteria defined

- [ ] `STAGE_2_2_SUMMARY.md` exists
  - [ ] Deliverables listed
  - [ ] Key design decisions explained
  - [ ] Contract guarantees documented
  - [ ] Next steps outlined

- [ ] `STAGE_2_PIPELINE_DIAGRAM.md` exists
  - [ ] Full Stage 2 architecture diagram
  - [ ] Constraint flow diagram
  - [ ] Proposal generation flow
  - [ ] Validation pipeline
  - [ ] Storage format example
  - [ ] End-to-end example

### Design Validation
- [ ] All 9 ProposalTypes defined:
  - [ ] ADD_AFFORDANCE
  - [ ] ADJUST_RISK
  - [ ] INFER_FRAGILITY
  - [ ] ADD_OBJECT_CATEGORY
  - [ ] ADD_SEMANTIC_TAG
  - [ ] ADD_SKILL_GATE
  - [ ] ADD_SAFETY_CONSTRAINT
  - [ ] ADD_ENERGY_HEURISTIC
  - [ ] UPDATE_OBJECT_RELATIONSHIP

- [ ] OntologyUpdateProposal schema complete:
  - [ ] proposal_id: str
  - [ ] proposal_type: ProposalType
  - [ ] priority: ProposalPriority
  - [ ] source_primitive_id: str
  - [ ] proposed_changes: Dict[str, Any]
  - [ ] rationale: str
  - [ ] confidence: float
  - [ ] respects_*_constraints: bool (3 flags)
  - [ ] to_dict() / from_dict() methods

- [ ] OntologyUpdateEngine methods specified:
  - [ ] generate_proposals()
  - [ ] validate_proposals()
  - [ ] _propose_affordances()
  - [ ] _propose_risk_adjustments()
  - [ ] _propose_fragility_inference()
  - [ ] _propose_skill_gates()
  - [ ] _propose_energy_heuristics()
  - [ ] _propose_semantic_tags()
  - [ ] _check_econ_constraints()
  - [ ] _check_datapack_constraints()
  - [ ] _check_task_graph_constraints()

- [ ] Constraint compliance rules defined:
  - [ ] Forbidden econ keys: price_per_unit, damage_cost, alpha/beta/gamma
  - [ ] Forbidden datapack keys: tier, novelty_score, data_premium
  - [ ] Forbidden task graph keys: delete_task, modify_dependencies

- [ ] Smoke test coverage complete:
  - [ ] Test 1: Proposal generation
  - [ ] Test 2: JSON-safety
  - [ ] Test 3: Required fields
  - [ ] Test 4: Constraint compliance
  - [ ] Test 5: Proposal type coverage
  - [ ] Test 6: Fragility inference
  - [ ] Test 7: Risk adjustment
  - [ ] Test 8: Skill gating
  - [ ] Test 9: Priority assignment
  - [ ] Test 10: Determinism

---

## Post-Implementation Checklist (Codex Validation)

After Codex completes implementation:

### Files Created
- [ ] `src/sima2/ontology_proposals.py` exists
  - [ ] ProposalType enum (9 values)
  - [ ] ProposalPriority enum (4 values)
  - [ ] OntologyUpdateProposal dataclass
  - [ ] to_dict() method
  - [ ] from_dict() method
  - [ ] No syntax errors
  - [ ] Imports work: `python3 -c "from src.sima2.ontology_proposals import *"`

- [ ] `src/sima2/ontology_update_engine.py` exists
  - [ ] OntologyUpdateEngine class
  - [ ] __init__() method
  - [ ] generate_proposals() method
  - [ ] validate_proposals() method
  - [ ] All 6 _propose_*() methods
  - [ ] All 3 _check_*_constraints() methods
  - [ ] _make_proposal_id() helper
  - [ ] _risk_level_to_float() helper
  - [ ] No syntax errors
  - [ ] Imports work: `python3 -c "from src.sima2.ontology_update_engine import *"`

- [ ] `scripts/smoke_test_ontology_update_engine.py` exists
  - [ ] _make_test_primitives() helper
  - [ ] main() function
  - [ ] All 10 test cases implemented
  - [ ] No syntax errors
  - [ ] Executable: `chmod +x scripts/smoke_test_ontology_update_engine.py`

- [ ] `scripts/run_all_smokes.py` updated
  - [ ] New smoke test added to SMOKES list
  - [ ] Line: `["python3", "scripts/smoke_test_ontology_update_engine.py"]`

### Smoke Test Results
Run: `python3 scripts/smoke_test_ontology_update_engine.py`

- [ ] Test 1 PASS: Proposal generation
  - [ ] Output: `Generated X proposals` (X > 0)

- [ ] Test 2 PASS: JSON-safety
  - [ ] All proposals serialize to JSON without errors

- [ ] Test 3 PASS: Required fields
  - [ ] All proposals have proposal_id, proposal_type, priority, rationale

- [ ] Test 4 PASS: Constraint compliance
  - [ ] Output: `X/X proposals valid`
  - [ ] All valid proposals have respects_*_constraints = True

- [ ] Test 5 PASS: Proposal type coverage
  - [ ] At least 4 proposal types generated (ADD_AFFORDANCE, ADJUST_RISK, INFER_FRAGILITY, ADD_SKILL_GATE expected)

- [ ] Test 6 PASS: Fragility inference
  - [ ] Fragility proposals contain inferred_fragility in [0.0, 1.0]

- [ ] Test 7 PASS: Risk adjustment
  - [ ] Risk adjustment proposals have new_risk_level > old_risk_level

- [ ] Test 8 PASS: Skill gating
  - [ ] Skill gate proposals contain gated_skill_id, preconditions, safety_threshold

- [ ] Test 9 PASS: Priority assignment
  - [ ] At least 1 CRITICAL priority proposal (fragility/high-urgency)

- [ ] Test 10 PASS: Determinism
  - [ ] Same primitives → same proposal types/counts (UUIDs may differ)

- [ ] Final output: `[smoke_test_ontology_update_engine] All tests passed!`

### Full Smoke Suite
Run: `python3 scripts/run_all_smokes.py`

- [ ] All previous smokes pass
- [ ] New smoke test passes
- [ ] Final output: `[run_all_smokes] All smokes passed.`

### Code Quality
- [ ] No ontology mutations in OntologyUpdateEngine
  - [ ] Search for: `ontology.add_object(` → should NOT appear
  - [ ] Search for: `ontology.objects[` = → should NOT appear (read-only OK)

- [ ] No task graph mutations
  - [ ] Search for: `task_graph.add_child(` → should NOT appear
  - [ ] Search for: `task_graph.mark_` → should NOT appear

- [ ] All proposed_changes are JSON-safe
  - [ ] No numpy types: search for `np.float`, `np.int` → should NOT appear in proposed_changes
  - [ ] No torch tensors: search for `torch.` → should NOT appear in proposed_changes

- [ ] Constraint validation works
  - [ ] Forbidden econ keys trigger validation failure
  - [ ] Forbidden datapack keys trigger validation failure
  - [ ] Forbidden task graph keys trigger validation failure

### Manual Validation Tests

#### Test A: JSON Serialization
```bash
python3 -c "
import json
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.orchestrator.ontology import build_drawer_vase_ontology

engine = OntologyUpdateEngine(ontology=build_drawer_vase_ontology())
prim = SemanticPrimitive(
    primitive_id='test',
    task_type='test',
    tags=['fragile'],
    risk_level='high',
    energy_intensity=0.1,
    success_rate=0.9,
    avg_steps=5.0,
)
proposals = engine.generate_proposals([prim])
for prop in proposals:
    json_str = json.dumps(prop.to_dict())
    assert json_str, 'JSON serialization failed'
print('✅ Manual JSON test passed')
"
```
- [ ] Output: `✅ Manual JSON test passed`

#### Test B: Constraint Violation Detection
```bash
python3 -c "
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.ontology_proposals import OntologyUpdateProposal, ProposalType, ProposalPriority
from src.orchestrator.ontology import build_drawer_vase_ontology

engine = OntologyUpdateEngine(ontology=build_drawer_vase_ontology())

# Create invalid proposal (contains forbidden key)
invalid_prop = OntologyUpdateProposal(
    proposal_id='test_invalid',
    proposal_type=ProposalType.ADJUST_RISK,
    priority=ProposalPriority.HIGH,
    proposed_changes={'price_per_unit': 10.0},  # FORBIDDEN
)

valid = engine.validate_proposals([invalid_prop])
assert len(valid) == 0, 'Validation should reject forbidden keys'
print('✅ Constraint violation detection works')
"
```
- [ ] Output: `✅ Constraint violation detection works`

#### Test C: Determinism
```bash
python3 -c "
from src.sima2.ontology_update_engine import OntologyUpdateEngine
from src.sima2.semantic_primitive_extractor import SemanticPrimitive
from src.orchestrator.ontology import build_drawer_vase_ontology

engine = OntologyUpdateEngine(ontology=build_drawer_vase_ontology())
prim = SemanticPrimitive(
    primitive_id='test',
    task_type='test',
    tags=['fragile', 'lift'],
    risk_level='high',
    energy_intensity=0.1,
    success_rate=0.9,
    avg_steps=5.0,
)

# Generate twice
proposals_1 = engine.generate_proposals([prim])
proposals_2 = engine.generate_proposals([prim])

# Check counts match
assert len(proposals_1) == len(proposals_2), 'Proposal count mismatch'

# Check types match (UUIDs will differ)
types_1 = sorted([p.proposal_type.value for p in proposals_1])
types_2 = sorted([p.proposal_type.value for p in proposals_2])
assert types_1 == types_2, 'Proposal types mismatch'

print('✅ Determinism validated')
"
```
- [ ] Output: `✅ Determinism validated`

---

## Final Approval Checklist

Before proceeding to Stage 2.3:

- [ ] All documentation files exist and are complete
- [ ] All implementation files created by Codex
- [ ] Smoke test passes (10/10 tests)
- [ ] Full smoke suite passes
- [ ] Manual validation tests pass (A, B, C)
- [ ] Code quality checks pass (no mutations, JSON-safe)
- [ ] Constraint validation works correctly
- [ ] Determinism validated
- [ ] No blockers or errors

---

## Approval Sign-Off

**Reviewed by**: _______________________
**Date**: _______________________
**Status**: [ ] APPROVED — Ready for Stage 2.3

**Notes**:
```
[Any additional observations or concerns]
```

---

## If Issues Found

### Common Issues & Fixes

**Issue**: Smoke test fails on Test 1 (no proposals generated)
- **Fix**: Check `generate_proposals()` calls all 6 `_propose_*()` methods
- **Verify**: Primitives have required fields (tags, risk_level, etc.)

**Issue**: Smoke test fails on Test 2 (JSON error)
- **Fix**: Check `proposed_changes` contains only Python native types
- **Verify**: No numpy types (use `float()`, not `np.float32()`)

**Issue**: Smoke test fails on Test 4 (constraint violations)
- **Fix**: Check `_check_*_constraints()` methods reject forbidden keys
- **Verify**: Validation logic matches spec (Section 4.2)

**Issue**: Smoke test fails on Test 7 (risk adjustment)
- **Fix**: Check `econ_signals.error_urgency` is set in test
- **Verify**: Risk multiplier logic in `_propose_risk_adjustments()`

**Issue**: Smoke test fails on Test 10 (determinism)
- **Fix**: Proposal IDs will differ (UUID), but types/counts should match
- **Verify**: `_proposal_counter` is incremented consistently

**Issue**: Full smoke suite fails (previous tests)
- **Fix**: Likely import error or circular dependency
- **Verify**: No changes to existing files (Stage 2.1)

---

## Next Steps After Approval

1. **User**: Mark Stage 2.2 as COMPLETE
2. **User**: Ask Claude: "Ready for Stage 2.3 (TaskGraphRefiner)?"
3. **Claude**: Design Stage 2.3 spec (similar deliverable set)
4. **Codex**: Implement Stage 2.3
5. **Repeat**: Until all Stage 2 modules complete

---

**End of Verification Checklist**
