# Stage 2.2 Deliverable Summary

**Date**: 2025-11-17
**Status**: ✅ Design Complete — Ready for Codex Implementation
**Estimated Implementation Time**: 2-3 hours

---

## What Was Delivered

### 1. Full Technical Specification
**File**: `STAGE_2_2_ONTOLOGY_UPDATE_ENGINE_SPEC.md`

**Contents**:
- ✅ OntologyUpdateProposal schema (9 proposal types)
- ✅ OntologyUpdateEngine module design (advisory-only, no mutation)
- ✅ Constraint mapping (Econ/Datapack/TaskGraph/SemanticOrchestrator)
- ✅ Causality & dependency constraints
- ✅ Stage 2 pipeline contract (Input→Output→Storage→Downstream)
- ✅ Smoke test specification (10 test cases)
- ✅ SemanticOrchestratorV2 proposal consumption interface (preview)
- ✅ Pipeline diagrams

### 2. Codex Implementation Guide
**File**: `CODEX_NEXT_STEPS_STAGE_2_2.md`

**Contents**:
- ✅ Step-by-step implementation checklist
- ✅ Exact file paths and class signatures
- ✅ Copy-paste-ready code snippets
- ✅ Validation commands for each step
- ✅ Common pitfalls and debugging checklist
- ✅ Success criteria (smoke test expectations)

---

## Key Design Decisions

### Advisory-Only Architecture
- **OntologyUpdateEngine does NOT mutate ontology**
- All outputs are `OntologyUpdateProposal` objects
- SemanticOrchestrator decides whether/how to apply proposals
- Clean separation: proposal generation ≠ proposal application

### Constraint Hierarchy Respected
```
UPSTREAM (Older Siblings)
├── EconomicController  → Cannot propose econ params
├── DatapackEngine      → Cannot propose data valuation
└── TaskGraph           → Cannot delete tasks

STAGE 2.2 (This Stage)
└── OntologyUpdateEngine → Proposes ontology changes only

DOWNSTREAM (Younger Siblings)
├── SemanticOrchestratorV2 → Consumes proposals
├── TaskGraphRefiner       → Refines tasks based on proposals
└── VLA/SIMA/Diffusion/RL  → Receive constraints via orchestrator
```

### 9 Proposal Types
1. `ADD_AFFORDANCE`: New affordances from primitive actions
2. `ADJUST_RISK`: Risk elevation from econ urgency
3. `INFER_FRAGILITY`: Object fragility from tags
4. `ADD_OBJECT_CATEGORY`: New object categories
5. `ADD_SEMANTIC_TAG`: Unified semantic tags
6. `ADD_SKILL_GATE`: Safety preconditions for skills
7. `ADD_SAFETY_CONSTRAINT`: Collision avoidance, clearance
8. `ADD_ENERGY_HEURISTIC`: Prefer efficient paths
9. `UPDATE_OBJECT_RELATIONSHIP`: Spatial relationships

### Validation & Safety
- **Econ constraints**: Cannot set `price_per_unit`, `damage_cost`, `alpha/beta/gamma`
- **Datapack constraints**: Cannot set `tier`, `novelty_score`, `data_premium`
- **Task graph constraints**: Cannot delete tasks, modify dependencies
- **JSON-safety**: All proposals must serialize to JSON
- **Determinism**: Same inputs → same proposal types/counts

---

## Files to Be Created (by Codex)

### File 1: `src/sima2/ontology_proposals.py`
- **Lines**: ~150
- **Classes**: `ProposalType`, `ProposalPriority`, `OntologyUpdateProposal`
- **Methods**: `to_dict()`, `from_dict()`

### File 2: `src/sima2/ontology_update_engine.py`
- **Lines**: ~400
- **Classes**: `OntologyUpdateEngine`
- **Methods**: 13 total (see Codex guide Section 2.2)

### File 3: `scripts/smoke_test_ontology_update_engine.py`
- **Lines**: ~200
- **Test Cases**: 10 (generation, JSON-safety, constraints, determinism, etc.)

### File 4: `scripts/run_all_smokes.py` (update)
- **Change**: Add 1 line to `SMOKES` list

---

## Smoke Test Expectations

When Codex implementation is complete, the smoke test should produce:

```
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
```

**Full smoke suite**:
```
$ python3 scripts/run_all_smokes.py
[run_all_smokes] All smokes passed.
```

---

## Contract Guarantees

### What OntologyUpdateEngine WILL Do
1. ✅ Consume `SemanticPrimitive[]` from Stage 2.1
2. ✅ Generate `OntologyUpdateProposal[]` (advisory-only)
3. ✅ Validate proposals against econ/datapack/task-graph constraints
4. ✅ Output JSON-safe proposals for storage/logging
5. ✅ Provide deterministic proposal generation

### What OntologyUpdateEngine WILL NOT Do
1. ❌ Mutate ontology directly
2. ❌ Set economic parameters (`price_per_unit`, `damage_cost`, etc.)
3. ❌ Set data valuation logic (`tier`, `novelty_score`, etc.)
4. ❌ Delete task graph nodes
5. ❌ Modify reward math or RL training loops

---

## Downstream Integration (Stage 2.3 Preview)

### SemanticOrchestratorV2 Interface
```python
def apply_ontology_proposals(
    self,
    proposals: List[OntologyUpdateProposal],
    apply_mode: Literal["advisory", "immediate"] = "advisory",
) -> Dict[str, Any]:
    """
    Apply ontology update proposals (advisory-only by default).

    Returns:
        Application report with accepted/rejected/deferred proposals
    """
    # To be implemented in Stage 2.3
    pass
```

### TaskGraphRefiner Interface (Stage 2.3)
```python
def refine_task_graph(
    self,
    proposals: List[OntologyUpdateProposal],
) -> List[TaskGraphUpdate]:
    """
    Generate task graph updates from ontology proposals.

    E.g., ADD_SKILL_GATE → insert checkpoint task before gated skill
    """
    # To be implemented in Stage 2.3
    pass
```

---

## Stage 2 Roadmap

```
Stage 2.1: SemanticPrimitiveExtractor  ✅ COMPLETE
Stage 2.2: OntologyUpdateEngine        🔄 DESIGN COMPLETE → READY FOR CODEX
Stage 2.3: TaskGraphRefiner            ⏸️  NEXT
Stage 2.4: SemanticTagPropagator       ⏸️  NEXT
```

---

## Next Steps

### For Codex (Immediate)
1. Read `CODEX_NEXT_STEPS_STAGE_2_2.md`
2. Implement File 1 → validate imports
3. Implement File 2 → validate imports
4. Implement File 3 → run smoke test
5. Update File 4 → run full smoke suite
6. Commit all files if tests pass

### For User (After Codex)
1. Review smoke test results
2. Manually validate JSON serialization
3. Approve Stage 2.2 completion
4. Proceed to Stage 2.3 (TaskGraphRefiner) design

---

## Success Metrics

Stage 2.2 is **COMPLETE** when:

- ✅ All 4 files created/updated
- ✅ Smoke test passes (10/10 tests)
- ✅ Full smoke suite passes (all previous + new test)
- ✅ JSON serialization validated
- ✅ No ontology mutations (code review)
- ✅ Constraint validation working (forbidden keys rejected)
- ✅ Determinism validated (same inputs → same types/counts)

---

## Appendix: Proposal Schema Quick Reference

```python
@dataclass
class OntologyUpdateProposal:
    proposal_id: str                  # "prop_000001_abc123"
    proposal_type: ProposalType       # ADD_AFFORDANCE, ADJUST_RISK, etc.
    priority: ProposalPriority        # CRITICAL, HIGH, MEDIUM, LOW
    source_primitive_id: str          # "prim_001"
    source: str                       # "sima2"
    target_object_id: Optional[str]   # "vase_01" (if applicable)
    target_skill_id: Optional[int]    # 2 (PULL skill, if applicable)
    target_affordance_type: Optional[str]  # "graspable" (if applicable)
    proposed_changes: Dict[str, Any]  # Type-specific changes
    rationale: str                    # Human-readable explanation
    confidence: float                 # 0.0 - 1.0
    respects_econ_constraints: bool   # True
    respects_datapack_constraints: bool  # True
    respects_task_graph: bool         # True
    tags: List[str]                   # ["fragile", "safety"]
    metadata: Dict[str, Any]          # Additional context
```

---

## Questions Before Implementation?

**Contact**: Slack user (before Codex starts)

**Common Questions**:
- Q: Can proposals modify reward weights?
  - A: ❌ No — econ constraint violation
- Q: Can proposals delete task nodes?
  - A: ❌ No — task graph constraint violation
- Q: Can proposals suggest new affordances?
  - A: ✅ Yes — ontology parameter (allowed)
- Q: Are proposal IDs deterministic?
  - A: ❌ No (UUID), but proposal types/counts are deterministic

---

**Ready for Stage 2.3 (TaskGraphRefiner)?**

Type: "Ready for Stage 2.3" when:
1. Codex has implemented Stage 2.2
2. All smoke tests pass
3. User has reviewed and approved
