# Stage 2.3 Deliverable Summary

**Date**: 2025-11-17
**Status**: ✅ Design Complete — Ready for Codex Implementation
**Estimated Implementation Time**: 2-3 hours

---

## What Was Delivered

### 1. Full Technical Specification
**File**: `STAGE_2_3_TASK_GRAPH_REFINER_SPEC.md`

**Contents**:
- ✅ TaskGraphRefinementProposal schema (8 refinement types)
- ✅ TaskGraphRefiner module design (advisory-only, no DAG mutations)
- ✅ Constraint mapping (Econ/Datapack/Ontology/TaskGraph/Stage2.1/Stage2.2)
- ✅ Causality & dependency constraints
- ✅ Stage 2.3 pipeline contract (Input→Output→Storage→Downstream)
- ✅ Smoke test specification (12 test cases)
- ✅ Integration with Stage 2.1 + 2.2 outputs

### 2. Codex Implementation Guide
**File**: `CODEX_NEXT_STEPS_STAGE_2_3.md`

**Contents**:
- ✅ Step-by-step implementation checklist (4 files)
- ✅ Exact file paths and class signatures
- ✅ Copy-paste-ready code snippets
- ✅ Validation commands for each step
- ✅ Common pitfalls and debugging guide
- ✅ Success criteria (smoke test expectations)

### 3. Pipeline Diagrams
**File**: `STAGE_2_3_PIPELINE_DIAGRAM.md`

**Contents**:
- ✅ Stage 2.3 architecture diagram
- ✅ Refinement generation flow (detailed)
- ✅ Validation pipeline
- ✅ Refinement type mappings (Ontology→TaskGraph)
- ✅ End-to-end examples (checkpoint insertion, task splitting)

### 4. This Summary
**File**: `STAGE_2_3_SUMMARY.md`

---

## Key Design Decisions

### Advisory-Only Architecture
- **TaskGraphRefiner does NOT mutate task graph**
- All outputs are `TaskGraphRefinementProposal` objects
- SemanticOrchestrator decides whether/how to apply refinements
- Clean separation: proposal generation ≠ graph mutation

### Constraint Hierarchy Respected
```
UPSTREAM (Older Siblings)
├── EconomicController  → Cannot propose econ params
├── DatapackEngine      → Cannot propose data valuation
├── Ontology (read-only)→ Cannot mutate objects/affordances
├── TaskGraph (read-only)→ Cannot modify DAG edges
├── Stage 2.1           → Consumes SemanticPrimitives
└── Stage 2.2           → Consumes OntologyUpdateProposals (PRIMARY)

STAGE 2.3 (This Stage)
└── TaskGraphRefiner → Proposes task graph refinements only

DOWNSTREAM (Younger Siblings)
├── SemanticOrchestratorV2 → Applies refinements
├── HRL Scheduler          → Executes checkpoints/gates
└── VLA/Diffusion/RL       → Receives task decomposition hints
```

### 8 Refinement Types
1. `SPLIT_TASK`: Decompose complex tasks into safer sub-tasks
2. `INSERT_CHECKPOINT`: Add safety/verification checkpoints (mandatory from skill gates)
3. `REORDER_TASKS`: Suggest efficiency/safety-driven reordering
4. `MERGE_TASKS`: Combine redundant tasks
5. `ADD_PRECONDITION`: Add safety preconditions
6. `PARALLELIZE_TASKS`: Suggest parallel execution for throughput
7. `INSERT_RECOVERY`: Add error recovery tasks
8. `ADJUST_PRIORITY`: Change task priority based on urgency

### Validation & Safety
- **Econ constraints**: Cannot set `price_per_unit`, `damage_cost`, `alpha/beta/gamma`
- **Datapack constraints**: Cannot set `tier`, `novelty_score`, `data_premium`
- **DAG topology**: Cannot create cycles
- **Node preservation**: Cannot delete nodes (replace only via SPLIT/MERGE)
- **JSON-safety**: All refinements must serialize to JSON
- **Determinism**: Same inputs → same refinement types/counts

---

## Files to Be Created (by Codex)

### File 1: `src/sima2/task_graph_proposals.py`
- **Lines**: ~180
- **Classes**: `RefinementType`, `RefinementPriority`, `TaskGraphRefinementProposal`
- **Methods**: `to_dict()`, `from_dict()`

### File 2: `src/sima2/task_graph_refiner.py`
- **Lines**: ~500
- **Classes**: `TaskGraphRefiner`
- **Methods**: 15 total (see Codex guide Section 2.2)

### File 3: `scripts/smoke_test_task_graph_refiner.py`
- **Lines**: ~250
- **Test Cases**: 12 (generation, JSON-safety, constraints, checkpoint insertion, DAG preservation, etc.)

### File 4: `scripts/run_all_smokes.py` (update)
- **Change**: Add 1 line to `SMOKES` list

---

## Smoke Test Expectations

When Codex implementation is complete, the smoke test should produce:

```
[smoke_test_task_graph_refiner] Starting tests...
[TEST 1 PASS] Generated X refinements
[TEST 2 PASS] All X refinements are JSON-safe
[TEST 3 PASS] All refinements have required fields
[TEST 4 PASS] X/X refinements valid
[TEST 5 PASS] Refinement types: ['insert_checkpoint', 'split_task', 'reorder_tasks', ...]
[TEST 6 PASS] Mandatory checkpoint insertion working (X checkpoints)
[TEST 7 PASS] Task splitting for fragility working (X splits)
[TEST 8 PASS/SKIP] Safety reordering working (X reorders)
[TEST 9 PASS] Priority assignment working (X CRITICAL)
[TEST 10 PASS] Determinism validated
[TEST 11 PASS] DAG topology preserved (no cycles)
[TEST 12 PASS] Node preservation validated
[smoke_test_task_graph_refiner] All tests passed!
```

**Full smoke suite**:
```
$ python3 scripts/run_all_smokes.py
[run_all_smokes] All smokes passed.
```

---

## Contract Guarantees

### What TaskGraphRefiner WILL Do
1. ✅ Consume `OntologyUpdateProposal[]` from Stage 2.2
2. ✅ Consume `SemanticPrimitive[]` from Stage 2.1 (optional)
3. ✅ Generate `TaskGraphRefinementProposal[]` (advisory-only)
4. ✅ Validate refinements against econ/datapack/DAG constraints
5. ✅ Output JSON-safe refinements for storage/logging
6. ✅ Provide deterministic refinement generation
7. ✅ Insert mandatory checkpoints from skill gates

### What TaskGraphRefiner WILL NOT Do
1. ❌ Mutate task graph directly
2. ❌ Set economic parameters (`price_per_unit`, `damage_cost`, etc.)
3. ❌ Set data valuation logic (`tier`, `novelty_score`, etc.)
4. ❌ Delete task nodes (replace only)
5. ❌ Create DAG cycles
6. ❌ Modify reward math or RL training loops

---

## Downstream Integration

### Mandatory Checkpoint Insertion
**Contract**: `ADD_SKILL_GATE` proposals from Stage 2.2 **MUST** trigger `INSERT_CHECKPOINT` refinements.

```python
# Stage 2.2 Output
OntologyUpdateProposal {
  proposal_type: ADD_SKILL_GATE,
  target_skill_id: 2,  # PULL skill
  proposed_changes: {
    "gated_skill_id": 2,
    "preconditions": ["fragility_check_passed"]
  }
}

# Stage 2.3 Output (MANDATORY)
TaskGraphRefinementProposal {
  refinement_type: INSERT_CHECKPOINT,
  priority: CRITICAL,
  proposed_changes: {
    "checkpoint_task": {...},
    "insert_before_task_id": "pull_drawer",
    "mandatory": True
  }
}
```

### SemanticOrchestratorV2 Interface (Stage 2.4+)
```python
def apply_task_graph_refinements(
    self,
    refinements: List[TaskGraphRefinementProposal],
    apply_mode: Literal["advisory", "immediate"] = "advisory",
) -> Dict[str, Any]:
    """
    Apply task graph refinements (advisory-only by default).

    Returns:
        Application report with accepted/rejected/deferred refinements
    """
    # To be implemented in Stage 2.4+
    pass
```

---

## Stage 2 Roadmap Progress

```
Stage 2.1: SemanticPrimitiveExtractor  ✅ COMPLETE
Stage 2.2: OntologyUpdateEngine        ✅ DESIGN COMPLETE → Codex implementing
Stage 2.3: TaskGraphRefiner            🔄 DESIGN COMPLETE → READY FOR CODEX
Stage 2.4: SemanticTagPropagator       ⏸️  NEXT
```

---

## Success Metrics

Stage 2.3 is **COMPLETE** when:

- ✅ All 4 files created/updated
- ✅ Smoke test passes (12/12 tests)
- ✅ Full smoke suite passes (all previous + new test)
- ✅ JSON serialization validated
- ✅ No task graph mutations (code review)
- ✅ Constraint validation working (forbidden keys rejected)
- ✅ Determinism validated (same inputs → same types/counts)
- ✅ DAG topology preserved (no cycles)
- ✅ Mandatory checkpoint insertion working

---

## Key Differences from Stage 2.2

### Stage 2.2 (OntologyUpdateEngine)
- **Input**: SemanticPrimitives
- **Output**: OntologyUpdateProposals
- **Focus**: Object properties, affordances, risk, fragility
- **Trigger**: Primitive tags/risk levels
- **Priority**: Fragility/safety

### Stage 2.3 (TaskGraphRefiner)
- **Input**: OntologyUpdateProposals + SemanticPrimitives
- **Output**: TaskGraphRefinementProposals
- **Focus**: Task structure, execution order, decomposition
- **Trigger**: Ontology proposals (especially ADD_SKILL_GATE)
- **Priority**: Checkpoint insertion, task splitting, reordering

**Relationship**: Stage 2.3 is **downstream** of Stage 2.2. It consumes ontology proposals and translates them into task graph changes.

---

## Appendix: Refinement Schema Quick Reference

```python
@dataclass
class TaskGraphRefinementProposal:
    proposal_id: str                      # "tgr_000001_abc123"
    refinement_type: RefinementType       # SPLIT_TASK, INSERT_CHECKPOINT, etc.
    priority: RefinementPriority          # CRITICAL, HIGH, MEDIUM, LOW
    source_primitive_ids: List[str]       # From Stage 2.1 (optional)
    source_ontology_proposal_ids: List[str]  # From Stage 2.2 (primary)
    source: str                           # "task_graph_refiner"
    target_task_ids: List[str]            # Affected task nodes
    parent_task_id: Optional[str]         # Parent for INSERT operations
    proposed_changes: Dict[str, Any]      # Type-specific changes
    rationale: str                        # Human-readable explanation
    confidence: float                     # 0.0 - 1.0
    respects_econ_constraints: bool       # True
    respects_datapack_constraints: bool   # True
    respects_dag_topology: bool           # True (no cycles)
    preserves_existing_nodes: bool        # True (no deletions)
    tags: List[str]                       # ["checkpoint", "safety"]
    metadata: Dict[str, Any]              # Additional context
```

---

## Questions Before Implementation?

**Common Questions**:
- Q: Can refinements delete task nodes?
  - A: ❌ No — only SPLIT_TASK/MERGE_TASKS can replace nodes
- Q: Can refinements modify DAG edges directly?
  - A: ❌ No — proposals only, orchestrator applies
- Q: Are checkpoint insertions mandatory?
  - A: ✅ Yes — if triggered by ADD_SKILL_GATE
- Q: Can refinements reorder tasks arbitrarily?
  - A: ⚠️ Only if DAG topology preserved (no cycles)

---

**Ready for Stage 2.4 (SemanticTagPropagator)?**

After Codex implements Stage 2.3 and smoke tests pass.
