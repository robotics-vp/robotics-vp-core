# Codex Implementation Priorities

**Context**: Stage 1 complete, all smokes passing. Ready for Stage 2-5 buildout.

**Status**: User asked for roadmap + Codex recommendations for next implementation steps.

---

## 🎯 Immediate Priority: Stage 2 Foundation

### **Why Stage 2 First?**

Stage 2 establishes the **semantic feedback loop** that all other stages depend on:
- SIMA-2 discovers semantic primitives
- Ontology gets updated with new affordances/constraints
- Task graph refined based on discoveries
- Semantic tags propagate to datapacks

Without Stage 2, Stages 3-5 lack the semantic infrastructure to properly condition RL sampling, diffusion generation, and real deployment analysis.

---

## 🔨 Implementation Queue (Codex should do in this order)

### **1. SemanticPrimitiveExtractor** ⭐ START HERE

**File**: `src/sima2/semantic_primitive_extractor.py`

**Purpose**: Extract semantic primitives from SIMA-2 agent rollouts and map them to ontology concepts.

**Key methods**:
```python
def extract_primitives(
    sima_rollout: Dict[str, Any],
    context: Dict[str, Any]
) -> List[SemanticPrimitive]:
    """Parse SIMA-2 semantic rollout and extract primitives"""

def map_to_ontology(
    primitive: SemanticPrimitive,
    ontology: EnvironmentOntology
) -> Optional[str]:
    """Find closest ontology concept for a primitive"""
```

**Smoke test**: `scripts/smoke_test_sima2_semantic_extraction.py`
- Create stub SIMA-2 rollout with semantic primitives
- Extract primitives
- Map to drawer_vase ontology
- Assert: At least 3 primitives extracted, 2 mapped to ontology

**Dependencies**:
- `src/sima2/sima2_agent_stub.py` (exists)
- `src/orchestrator/ontology.py` (exists)

**Success criteria**:
- [ ] `SemanticPrimitive` dataclass defined
- [ ] `extract_primitives()` returns non-empty list
- [ ] `map_to_ontology()` finds matches for common primitives
- [ ] Smoke test passes

---

### **2. OntologyUpdateEngine**

**File**: `src/orchestrator/ontology_updater.py`

**Purpose**: Propose and apply ontology updates based on semantic discoveries.

**Key methods**:
```python
def propose_update_from_primitive(
    primitive: SemanticPrimitive,
    ontology: EnvironmentOntology
) -> Optional[OntologyUpdate]:
    """Propose ontology update from semantic primitive"""

def apply_update(
    update: OntologyUpdate,
    ontology: EnvironmentOntology
) -> bool:
    """Apply validated update to ontology"""

def reconcile_conflicts(
    updates: List[OntologyUpdate]
) -> List[OntologyUpdate]:
    """Resolve conflicting updates"""
```

**Smoke test**: `scripts/smoke_test_ontology_updater.py`
- Create primitives from smoke test #1
- Propose updates
- Apply to ontology
- Assert: Ontology modified, no conflicts, affordance confidences updated

**Dependencies**:
- `SemanticPrimitiveExtractor` (from step 1)
- `src/orchestrator/ontology.py` (exists)

**Success criteria**:
- [ ] `OntologyUpdate` dataclass defined
- [ ] At least 3 update types supported (add_affordance, update_confidence, add_constraint)
- [ ] Conflict resolution handles duplicates
- [ ] Smoke test passes

---

### **3. TaskGraphRefiner**

**File**: `src/orchestrator/task_graph_refiner.py`

**Purpose**: Refine task graph structure based on semantic discoveries and econ signals.

**Key methods**:
```python
def refine_from_primitives(
    task_graph: TaskGraph,
    primitives: List[SemanticPrimitive],
    ontology: EnvironmentOntology
) -> List[TaskNode]:
    """Propose task graph refinements"""

def insert_safety_checkpoints(
    task_graph: TaskGraph,
    fragile_objects: List[ObjectSpec]
) -> List[str]:
    """Insert checkpoint nodes near fragile interactions"""

def reorder_for_efficiency(
    task_graph: TaskGraph,
    energy_costs: Dict[str, float]
) -> TaskGraph:
    """Reorder tasks to minimize energy"""
```

**Smoke test**: `scripts/smoke_test_task_graph_refiner.py`
- Load drawer_vase task graph
- Refine based on fragility primitives
- Assert: Safety checkpoints inserted before vase interactions
- Assert: Task ordering changed if energy costs warrant

**Dependencies**:
- `SemanticPrimitiveExtractor` (from step 1)
- `src/orchestrator/task_graph.py` (exists)
- `src/orchestrator/ontology.py` (exists)

**Success criteria**:
- [ ] Refinement proposals generated
- [ ] Safety checkpoints inserted correctly
- [ ] Task reordering respects dependencies
- [ ] Smoke test passes

---

### **4. SemanticTagPropagator**

**File**: `src/valuation/semantic_tag_propagator.py`

**Purpose**: Propagate semantic discoveries back to datapack tags.

**Key methods**:
```python
def update_datapack_tags(
    datapack: DataPackMeta,
    ontology: EnvironmentOntology,
    primitives: List[SemanticPrimitive]
) -> List[str]:
    """Add semantic tags to existing datapack"""

def compute_semantic_quality_score(
    datapack: DataPackMeta,
    ontology: EnvironmentOntology
) -> float:
    """Compute semantic_quality based on ontology alignment"""
```

**Smoke test**: `scripts/smoke_test_semantic_tag_propagation.py`
- Load Stage 1 datapacks
- Update with new semantic primitives
- Assert: New tags added to econ_semantic_tags
- Assert: semantic_quality score computed

**Dependencies**:
- `SemanticPrimitiveExtractor` (from step 1)
- `src/valuation/datapack_schema.py` (exists)
- `src/orchestrator/ontology.py` (exists)

**Success criteria**:
- [ ] Tags retroactively added to datapacks
- [ ] Semantic quality score in [0, 1]
- [ ] Higher scores for better ontology alignment
- [ ] Smoke test passes

---

### **5. Stage 2 Integrated Pipeline**

**File**: `scripts/run_stage2_semantic_colearning.py`

**Purpose**: End-to-end Stage 2 pipeline.

**Flow**:
1. Generate SIMA-2 rollouts (stub)
2. Extract semantic primitives
3. Propose ontology updates
4. Refine task graph
5. Propagate tags to datapacks
6. Export updated ontology + task graph + datapacks

**Smoke test**: `scripts/smoke_test_stage2_pipeline.py`
- Run full pipeline with stub data
- Assert: Ontology updated
- Assert: Task graph refined
- Assert: Datapacks enriched
- Assert: Output files created

**Dependencies**: All components 1-4

**Success criteria**:
- [ ] Pipeline runs end-to-end
- [ ] Statistics exported
- [ ] Output compatible with Stage 3
- [ ] Smoke test passes

---

## 📋 Implementation Checklist

**Before starting**:
- [x] Review ROADMAP_STAGES_2_5.md
- [ ] Confirm understanding of ontology/task graph structures
- [ ] Verify SIMA-2 stubs are sufficient

**During implementation**:
- [ ] Write module docstrings
- [ ] Add type hints to all methods
- [ ] Follow existing naming conventions
- [ ] Keep modules < 500 lines
- [ ] No modifications to Phase B/reward/RL training

**After each module**:
- [ ] Write smoke test
- [ ] Run smoke test (must pass)
- [ ] Update ROADMAP_STAGES_2_5.md with ✅
- [ ] Commit with message pattern: "Add [module name] for Stage 2 semantic co-learning"

**After Stage 2 complete**:
- [ ] Run `scripts/run_all_smokes.py` (all must pass)
- [ ] Update ROADMAP with "Stage 2 Complete" section
- [ ] Ask user: "Stage 2 complete. Proceed with Stage 3 or adjust?"

---

## 🚫 Things Codex Should NOT Do

1. **Do NOT modify**:
   - `src/valuation/reward_builder.py`
   - `src/config/econ_params.py` (Phase B parameters)
   - `src/training/sac.py` or `src/training/ppo.py` (RL loops)
   - Any reward calculation logic
   - Episode descriptor normalization (already done)

2. **Do NOT implement**:
   - Actual SIMA-2 training (use stubs only)
   - Real diffusion model training (use stubs)
   - GPU-dependent code (CPU stubs only)
   - Production database connections

3. **Do NOT skip**:
   - Smoke tests for each module
   - Type hints and docstrings
   - Contract validation (schemas)
   - Error handling for edge cases

---

## 🎓 Key Design Patterns

### **Pattern 1: Advisory Outputs**
All semantic orchestration produces *suggestions*, not commands:
```python
# ✅ GOOD: Advisory
plan = semantic_orchestrator.build_update_plan(econ_signals, datapack_signals)
# User/downstream decides whether to apply
semantic_orchestrator.apply_update_plan(plan)

# ❌ BAD: Automatic modification
semantic_orchestrator.auto_update_everything()  # Too aggressive
```

### **Pattern 2: Contract-Validated Schemas**
All inter-module communication uses validated dataclasses:
```python
# ✅ GOOD: Schema-validated
@dataclass
class SemanticPrimitive:
    primitive_id: str
    primitive_type: str  # Validated enum
    confidence: float  # In [0, 1]

# ❌ BAD: Raw dictionaries
primitive = {"id": "foo", "type": "unknown", "conf": 2.5}  # No validation
```

### **Pattern 3: Smoke Test Structure**
Every module has a standalone smoke test:
```python
def test_component():
    # 1. Setup
    component = ComponentUnderTest()
    stub_input = create_stub_input()

    # 2. Execute
    output = component.process(stub_input)

    # 3. Assert
    assert output is not None
    assert len(output.results) > 0
    assert output.validate()

    print("✓ Component smoke test passed")

if __name__ == "__main__":
    test_component()
```

### **Pattern 4: Econ/Semantic Separation**
Economics drives semantics, not vice versa:
```python
# ✅ GOOD: Econ signals drive semantic updates
econ_signals = econ_controller.compute_signals()
semantic_plan = semantic_orch.build_update_plan(econ_signals, ...)

# ❌ BAD: Semantics try to drive economics
semantic_orch.set_reward_weights(...)  # FORBIDDEN
```

---

## 🔍 Verification Steps

After implementing each module, Codex should verify:

1. **Type correctness**:
   ```bash
   python -m mypy src/sima2/semantic_primitive_extractor.py
   ```

2. **Smoke test passes**:
   ```bash
   python scripts/smoke_test_sima2_semantic_extraction.py
   ```

3. **No breaking changes**:
   ```bash
   python scripts/run_all_smokes.py  # All existing smokes must pass
   ```

4. **Schema compatibility**:
   ```bash
   python scripts/smoke_test_episode_descriptor_contract.py  # Descriptors still valid
   ```

---

## 📊 Progress Tracking

**Current Status**:
- ✅ Stage 1: Complete
- 🔨 Stage 2: Ready to implement
- ⏳ Stage 3: Waiting on Stage 2
- ⏳ Stage 4: Waiting on Stage 3
- ⏳ Stage 5: Waiting on Stage 4

**Next Milestone**:
🎯 Stage 2 complete = 5 modules + 5 smoke tests + integrated pipeline

**Estimated Implementation Time**:
- SemanticPrimitiveExtractor: ~30 min
- OntologyUpdateEngine: ~45 min
- TaskGraphRefiner: ~45 min
- SemanticTagPropagator: ~30 min
- Stage 2 Pipeline: ~30 min
- **Total**: ~3 hours for Stage 2

---

## 🎉 Success Criteria for Stage 2

When Codex can answer YES to all:

- [ ] `SemanticPrimitiveExtractor` extracts primitives from SIMA-2 rollouts
- [ ] Primitives mapped to ontology concepts (graspable, fragile, etc.)
- [ ] `OntologyUpdateEngine` proposes and applies updates
- [ ] Conflicts resolved (e.g., two confidence scores for same affordance)
- [ ] `TaskGraphRefiner` inserts safety checkpoints near fragile objects
- [ ] Task reordering respects dependencies and energy costs
- [ ] `SemanticTagPropagator` enriches datapacks with new tags
- [ ] Semantic quality scores computed
- [ ] Stage 2 pipeline runs end-to-end
- [ ] All 5 new smoke tests pass
- [ ] All existing smoke tests still pass
- [ ] No modifications to Phase B, reward, or RL training

**Then**: Report to user and await instructions for Stage 3.

---

## 💡 Tips for Codex

1. **Start small**: Get `SemanticPrimitiveExtractor` working before moving to next module
2. **Test incrementally**: Run smoke test after each method implementation
3. **Use existing patterns**: Look at `src/orchestrator/semantic_orchestrator.py` for style
4. **Stub liberally**: Use synthetic data for SIMA-2 rollouts (no real agent needed)
5. **Document assumptions**: If you make design choices, add comments explaining why
6. **Ask before breaking**: If you need to modify existing code, ASK FIRST

---

## 📞 When to Ask User

**Ask before**:
- Modifying any existing Phase B, reward, or RL code
- Changing schema definitions (DataPackMeta, EpisodeDescriptor, etc.)
- Implementing GPU-dependent features
- Deviating from roadmap structure

**Report after**:
- Each module complete + smoke test passing
- Stage 2 fully complete
- Any blockers or ambiguities discovered

---

**End of Next Steps Guide**

**Codex**: Begin with `SemanticPrimitiveExtractor` (Module 1). Good luck! 🚀
