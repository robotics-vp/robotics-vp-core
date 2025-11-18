# Stage 2.4: SemanticTagPropagator - Design Pack Summary

**Version**: 1.0
**Last Updated**: 2025-11-17
**Status**: Design Complete, Implementation Pending

---

## Design Pack Contents

This is the **complete Stage 2.4 SemanticTagPropagator design pack**, matching the format and depth of Stage 2.2 (OntologyUpdateEngine) and Stage 2.3 (TaskGraphRefiner).

### Documentation Files

1. **[STAGE_2_4_SEMANTIC_TAG_PROPAGATOR_SPEC.md](STAGE_2_4_SEMANTIC_TAG_PROPAGATOR_SPEC.md)**
   - Full technical specification
   - Inputs/outputs and data formats
   - Contract boundaries (what MAY/MAY NOT do)
   - Tag type definitions (Fragility, Risk, Affordance, Efficiency, Novelty, Intervention)
   - Coherence checking and conflict detection
   - Supervision hints for orchestrator
   - Determinism and reproducibility guarantees
   - Schema compliance and validation
   - Cross-consistency requirements
   - Failure modes and resilience

2. **[STAGE_2_4_PIPELINE_DIAGRAMS.md](STAGE_2_4_PIPELINE_DIAGRAMS.md)**
   - Full Stage 2 pipeline flow (2.1/2.2/2.3 → 2.4 → Stage 3)
   - SemanticTagPropagator internal processing flow
   - Example workflow: Drawer + vase scenario (step-by-step)
   - Failure mode examples (missing data, conflicts, etc.)
   - Multi-video aggregation example
   - Tag flow decision tree
   - Stage 2 to Stage 3 handoff

3. **[STAGE_2_4_SMOKE_TESTS.md](STAGE_2_4_SMOKE_TESTS.md)**
   - 22 comprehensive smoke tests
   - Determinism tests (3 tests)
   - Schema compliance tests (3 tests)
   - JSON safety tests (2 tests)
   - Cross-consistency tests (3 tests)
   - Forbidden field tests (2 tests)
   - Stable ordering tests (2 tests)
   - Multi-video tests (2 tests)
   - Resilience tests (5 tests)
   - Test fixtures and expected outputs
   - Acceptance criteria

4. **[STAGE_2_4_CODEX_IMPLEMENTATION_GUIDE.md](STAGE_2_4_CODEX_IMPLEMENTATION_GUIDE.md)**
   - File structure and dependencies
   - Step-by-step implementation guide
   - Tag dataclasses (complete code)
   - Ontology matcher (complete code)
   - Task graph matcher (complete code)
   - Economics matcher (complete code)
   - Coherence checker (complete code)
   - Supervision hints generator (complete code)
   - Main propagator (complete code)
   - Test fixture generation
   - Integration commands

5. **[STAGE_2_4_VERIFICATION_CHECKLIST.md](STAGE_2_4_VERIFICATION_CHECKLIST.md)**
   - Pre-implementation review checklist
   - Post-implementation validation checklist
   - Determinism verification
   - Schema compliance verification
   - JSON safety verification
   - Cross-consistency verification
   - Contract boundary verification
   - Final sign-off conditions
   - Troubleshooting guide

---

## Quick Reference

### What is SemanticTagPropagator?

The **SemanticTagPropagator** is the final Stage 2 component that enriches Stage 1 datapacks with semantic metadata derived from:
- **Stage 2.2** (OntologyUpdateEngine): Affordances, fragilities, object relations
- **Stage 2.3** (TaskGraphRefiner): Task dependencies, risks, optimization opportunities
- **Economics Module**: Novelty scores, expected MPL gains, tier assignments

It produces **advisory-only semantic tags** that help the Stage 3 orchestrator:
- Prioritize high-value training episodes
- Weight safety-critical scenarios appropriately
- Schedule curriculum based on prerequisite knowledge
- Flag semantic conflicts for human review
- Provide supervision hints for training loop

### Key Constraints

✅ **MAY**:
- Read Stage 1 datapacks, Stage 2 proposals, Economics outputs (read-only)
- Generate semantic tags (fragility, risk, affordance, efficiency, novelty, intervention)
- Detect semantic conflicts and coherence issues
- Suggest supervision hints (priority, weight, curriculum stage)
- Extend datapack schema with `enrichment` field (backward-compatible)

❌ **MAY NOT**:
- Modify economic parameters (MPL, wage parity, damage costs)
- Change reward values in datapacks
- Alter sampling weights or training priorities (only suggest)
- Mutate task graph structure or ontology definitions
- Write to global state or trigger side effects

### Tag Types Produced

1. **FragilityTag**: Flags fragile objects, damage costs, contact frames
2. **RiskTag**: Flags safety risks (collision, tip-over, etc.), severity, mitigation hints
3. **AffordanceTag**: Tags demonstrated affordances (graspable, pullable, etc.)
4. **EfficiencyTag**: Scores execution efficiency (time, energy, precision)
5. **NoveltyTag**: Quantifies novelty and expected MPL gain
6. **InterventionTag**: Tags human corrections or failure recoveries
7. **SemanticConflict**: Detects tag contradictions (e.g., high fragility + low risk)
8. **SupervisionHints**: Orchestrator guidance (priority, weight, curriculum stage, prerequisites)

### Smoke Test Coverage

**22 tests** covering:
- ✅ Determinism (identical outputs for identical inputs)
- ✅ Schema compliance (all fields conform to JSON schema)
- ✅ JSON safety (all types serializable)
- ✅ Cross-consistency (tags align with source proposals)
- ✅ Forbidden fields (no contract violations)
- ✅ Stable ordering (sorted tags for reproducibility)
- ✅ Multi-video aggregation (tag merging across videos)
- ✅ Resilience (graceful degradation on missing data)

### Implementation Steps

1. **Implement tag dataclasses** (`tag_types.py`) - 8 dataclasses with validation
2. **Implement matchers** (`ontology_matcher.py`, `task_graph_matcher.py`, `economics_matcher.py`)
3. **Implement coherence checker** (`coherence_checker.py`) - conflict detection + scoring
4. **Implement supervision hints generator** (`supervision_hints_generator.py`)
5. **Implement main propagator** (`semantic_tag_propagator.py`)
6. **Create test fixtures** (sample datapacks, proposals, economics outputs)
7. **Run smoke tests** (verify all 22 tests pass)
8. **Generate sample enrichments** (drawer+vase scenario)
9. **Integrate with Stage 3** (merge enrichments with datapacks)

### Example Output (Drawer + Vase Scenario)

**Input**: Datapack of robot opening drawer with fragile vase inside

**Output Enrichment** (excerpt):
```json
{
  "episode_id": "ep_12345",
  "enrichment": {
    "fragility_tags": [
      {
        "object_name": "vase_inside",
        "fragility_level": "high",
        "damage_cost_usd": 50.0,
        "contact_frames": [45, 46, 47]
      }
    ],
    "risk_tags": [
      {
        "risk_type": "collision",
        "severity": "medium",
        "affected_frames": [44, 50],
        "mitigation_hints": ["Reduce pull velocity"]
      }
    ],
    "novelty_tags": [
      {
        "novelty_type": "edge_case",
        "novelty_score": 0.73,
        "expected_mpl_gain": 4.2
      }
    ],
    "coherence_score": 0.95,
    "supervision_hints": {
      "prioritize_for_training": true,
      "priority_level": "high",
      "suggested_weight_multiplier": 2.0,
      "safety_critical": true,
      "curriculum_stage": "advanced",
      "prerequisite_tags": ["basic_drawer_open", "fragile_object_awareness"]
    },
    "confidence": 0.87,
    "validation_status": "passed"
  }
}
```

**Orchestrator Actions**:
- Sample episode 2x more often (weight=2.0)
- Mark as safety-critical (fragile object present)
- Schedule in advanced curriculum (after prerequisites met)
- Penalize rapid movements near fragile objects

---

## Verification Quick Check

Before implementation:
- [ ] Read all 5 specification documents
- [ ] Understand contract boundaries (MAY/MAY NOT)
- [ ] Review smoke tests (22 tests)
- [ ] Check Stage 2.2/2.3 dependencies available

After implementation:
- [ ] All 22 smoke tests pass
- [ ] No forbidden fields in output
- [ ] No mutations to inputs
- [ ] Determinism verified (10+ identical runs)
- [ ] JSON serialization works
- [ ] Cross-consistency validated
- [ ] Sample enrichments generated and inspected

---

## Integration with Broader Pipeline

```
Stage 1 (Datapacks)
    ↓
Stage 2.1 (Economics: MPL, Novelty, Tiers)
    ↓
Stage 2.2 (Ontology: Affordances, Fragilities) ──┐
    ↓                                            │
Stage 2.3 (TaskGraph: Dependencies, Risks) ──────┤
    ↓                                            │
Stage 2.4 (SemanticTagPropagator) ←──────────────┘
    ↓
Enriched Datapacks (JSONL)
    ↓
Stage 3 (Orchestrator: Training with supervision)
```

**Stage 2.4 Role**:
- Final semantic enrichment layer
- Translates Stage 2 knowledge into actionable tags
- Bridges understanding (2.2/2.3) with execution (Stage 3)
- Provides orchestrator with rich metadata for intelligent training

---

## Next Steps

### Immediate (Design Phase)

✅ Stage 2.4 design pack complete
- All 5 specification documents created
- Full contract defined
- Smoke tests specified
- Implementation guide provided
- Verification checklist complete

### Pending (Implementation Phase)

⏸️ **AWAITING DIRECTION** before proceeding to:
- Stage 2.5 specification (if exists)
- Stage 3 orchestrator design
- Stage 2.4 implementation
- End-to-end pipeline integration

**DO NOT PROCEED** without explicit user approval.

---

## Design Quality Metrics

This design pack meets all requirements from the original prompt:

✅ **Full Technical Specification**:
- What SemanticTagPropagator is ✓
- Inputs/outputs ✓
- Contract boundaries ✓
- Tag types (8 types defined) ✓
- Determinism guarantees ✓
- Zero side effects ✓

✅ **Proposal Types**:
- Fragility, Risk, Affordance, Efficiency, Novelty, Intervention tags ✓
- Semantic conflicts and coherence ✓
- Supervision hints for orchestrator ✓
- Datapack schema integration ✓

✅ **Contract + Constraints**:
- What it may propose (advisory tags) ✓
- What it may not touch (economics, rewards, task graph) ✓
- How it consumes inputs ✓
- How it emits JSONL enrichments ✓

✅ **Smoke Test Suite**:
- 22 comprehensive tests ✓
- Determinism, schema, JSON, consistency, ordering, resilience ✓
- Multi-video aggregation ✓
- Forbidden field enforcement ✓

✅ **Pipeline Diagrams + Examples**:
- Full Stage 2 flow ✓
- Drawer+vase scenario (step-by-step) ✓
- Failure modes ✓
- Multi-video aggregation ✓

✅ **Codex Implementation Guide**:
- File paths ✓
- Method signatures ✓
- Dataclasses (complete code) ✓
- Test scaffolding ✓
- Integration commands ✓

✅ **Verification Checklist**:
- Pre-implementation review ✓
- Post-implementation validation ✓
- Determinism, schema, JSON, consistency checks ✓
- Final sign-off conditions ✓

---

## Document Index

| Document | Purpose | Pages |
|----------|---------|-------|
| [SPEC.md](STAGE_2_4_SEMANTIC_TAG_PROPAGATOR_SPEC.md) | Technical specification | ~50 |
| [PIPELINE_DIAGRAMS.md](STAGE_2_4_PIPELINE_DIAGRAMS.md) | Flow diagrams + examples | ~40 |
| [SMOKE_TESTS.md](STAGE_2_4_SMOKE_TESTS.md) | Test suite (22 tests) | ~35 |
| [CODEX_GUIDE.md](STAGE_2_4_CODEX_IMPLEMENTATION_GUIDE.md) | Implementation guide | ~45 |
| [VERIFICATION.md](STAGE_2_4_VERIFICATION_CHECKLIST.md) | Verification checklist | ~30 |
| **TOTAL** | **Complete design pack** | **~200** |

---

## Format Consistency

This design pack matches the format and depth of:
- **Stage 2.2** (OntologyUpdateEngine): Proposal types, validation, smoke tests, Codex guide
- **Stage 2.3** (TaskGraphRefiner): Advisory refinements, determinism, cross-consistency, verification

All three Stage 2 components (2.2, 2.3, 2.4) follow **identical design methodology**:
1. Full technical spec with contract boundaries
2. Proposal type definitions
3. Comprehensive smoke tests
4. Step-by-step Codex implementation guide
5. Pre/post verification checklists

---

## Contact & Feedback

**Design Author**: Claude (Anthropic)
**Specification Version**: 1.0
**Design Completion Date**: 2025-11-17

For questions or clarifications:
- Review individual specification documents
- Check verification checklist for common issues
- Consult smoke tests for expected behavior
- Reference Codex guide for implementation details

---

**End of Stage 2.4 Design Pack Summary**

**Status**: ✅ Design Complete, ⏸️ Awaiting Implementation Direction
