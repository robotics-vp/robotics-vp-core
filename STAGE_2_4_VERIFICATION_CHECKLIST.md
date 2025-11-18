# Stage 2.4: SemanticTagPropagator - Verification Checklist

**Version**: 1.0
**Last Updated**: 2025-11-17

---

## Overview

This checklist provides a **comprehensive validation workflow** for SemanticTagPropagator implementations. Use this before and after implementation to ensure specification compliance.

---

## Pre-Implementation Review

### Design Review

- [ ] **Specification Read-Through**
  - [ ] Read STAGE_2_4_SEMANTIC_TAG_PROPAGATOR_SPEC.md in full
  - [ ] Understand all tag types (Fragility, Risk, Affordance, Efficiency, Novelty, Intervention)
  - [ ] Understand coherence checking and conflict detection
  - [ ] Understand supervision hints generation
  - [ ] Understand contract boundaries (what MAY/MAY NOT do)

- [ ] **Pipeline Understanding**
  - [ ] Read STAGE_2_4_PIPELINE_DIAGRAMS.md
  - [ ] Trace flow: Stage 2.1/2.2/2.3 → 2.4 → enriched datapacks
  - [ ] Understand drawer+vase example workflow
  - [ ] Understand failure modes and graceful degradation

- [ ] **Test Coverage Planning**
  - [ ] Read STAGE_2_4_SMOKE_TESTS.md
  - [ ] Understand all 22 smoke tests
  - [ ] Plan test fixture generation
  - [ ] Identify edge cases to test beyond smoke tests

- [ ] **Implementation Planning**
  - [ ] Read STAGE_2_4_CODEX_IMPLEMENTATION_GUIDE.md
  - [ ] Review file structure and dependencies
  - [ ] Plan implementation order (tag types → matchers → coherence → propagator)
  - [ ] Identify reusable components from Stage 2.2/2.3

### Dependency Check

- [ ] **Stage 2.2 (OntologyUpdateEngine) Complete**
  - [ ] OntologyUpdateEngine implemented
  - [ ] Smoke tests pass
  - [ ] Sample proposals generated and validated
  - [ ] JSONL output format confirmed

- [ ] **Stage 2.3 (TaskGraphRefiner) Complete**
  - [ ] TaskGraphRefiner implemented
  - [ ] Smoke tests pass
  - [ ] Sample proposals generated and validated
  - [ ] JSONL output format confirmed

- [ ] **Economics Module (Stage 2.1) Operational**
  - [ ] Novelty scores computed for datapacks
  - [ ] Expected MPL gains computed
  - [ ] Tier assignments (0/1/2) working
  - [ ] JSONL output format confirmed

- [ ] **Stage 1 Datapacks Available**
  - [ ] Datapacks in JSONL format
  - [ ] Schema includes: episode_id, task, frames, actions, metadata
  - [ ] Metadata includes: success, objects_present (optional)

### Contract Review

- [ ] **Forbidden Operations Identified**
  - [ ] No mutations to economic parameters
  - [ ] No changes to reward values
  - [ ] No modifications to sampling weights
  - [ ] No alterations to task graph structure
  - [ ] No modifications to ontology definitions
  - [ ] No writes to global state

- [ ] **Read-Only Access Confirmed**
  - [ ] Datapacks: read-only
  - [ ] Ontology proposals: read-only
  - [ ] Task graph proposals: read-only
  - [ ] Economics outputs: read-only

- [ ] **Advisory-Only Outputs Understood**
  - [ ] All tags are suggestions, not enforcement
  - [ ] Orchestrator decides how to use hints
  - [ ] No direct control of training loop

---

## Post-Implementation Validation

### Code Quality Checks

- [ ] **Code Structure**
  - [ ] File structure matches Codex guide
  - [ ] All required files present:
    - [ ] `tag_types.py`
    - [ ] `ontology_matcher.py`
    - [ ] `task_graph_matcher.py`
    - [ ] `economics_matcher.py`
    - [ ] `coherence_checker.py`
    - [ ] `supervision_hints_generator.py`
    - [ ] `semantic_tag_propagator.py`
  - [ ] No missing imports
  - [ ] No circular dependencies

- [ ] **Dataclass Definitions**
  - [ ] All tag types defined as dataclasses
  - [ ] All fields typed correctly
  - [ ] `to_dict()` methods implemented for JSON serialization
  - [ ] `__post_init__` validation for required constraints

- [ ] **Type Hints**
  - [ ] All function signatures have type hints
  - [ ] Return types specified
  - [ ] Complex types (List, Dict, Optional) properly annotated

- [ ] **Docstrings**
  - [ ] All classes have docstrings
  - [ ] All public methods have docstrings
  - [ ] Complex logic has inline comments

### Determinism Tests

- [ ] **test_deterministic_generation**
  - [ ] Run test 10 times → same outputs every time
  - [ ] Proposal IDs identical across runs
  - [ ] Tag lists identical (same tags, same order)
  - [ ] Confidence scores identical
  - [ ] Validation status identical

- [ ] **test_deterministic_ids**
  - [ ] Proposal IDs generated from hash of inputs
  - [ ] Same inputs → same ID every time
  - [ ] Different inputs → different IDs

- [ ] **test_no_random_sources**
  - [ ] Changing random seeds has no effect
  - [ ] No `random.random()` calls in code
  - [ ] No `np.random` calls in code
  - [ ] No timestamp-based IDs (use input timestamps only)

### Schema Compliance Tests

- [ ] **test_enrichment_schema_compliance**
  - [ ] Load enrichment_schema.json
  - [ ] Generate sample proposal
  - [ ] Validate with jsonschema library
  - [ ] No validation errors

- [ ] **test_required_fields_present**
  - [ ] `coherence_score` present (float, 0.0-1.0)
  - [ ] `confidence` present (float, 0.0-1.0)
  - [ ] `validation_status` present (string, "pending"/"passed"/"failed")
  - [ ] All required tag fields present

- [ ] **test_value_ranges**
  - [ ] coherence_score ∈ [0.0, 1.0]
  - [ ] confidence ∈ [0.0, 1.0]
  - [ ] novelty_score ∈ [0.0, 1.0]
  - [ ] efficiency_score ∈ [0.0, 1.0]
  - [ ] No NaN or Inf values

### JSON Safety Tests

- [ ] **test_json_serialization**
  - [ ] `json.dumps()` succeeds for all proposals
  - [ ] `json.loads()` roundtrip preserves data
  - [ ] No serialization errors

- [ ] **test_no_unsafe_types**
  - [ ] No `numpy.ndarray` in output
  - [ ] No `torch.Tensor` in output
  - [ ] No lambda functions in output
  - [ ] No file handles in output
  - [ ] All values are: str, int, float, bool, list, dict, None

### Cross-Consistency Tests

- [ ] **test_ontology_consistency**
  - [ ] All AffordanceTags reference valid ontology proposals
  - [ ] All FragilityTags reference valid ontology proposals or economics
  - [ ] Source proposal IDs match referenced proposals
  - [ ] No orphan tags (tags without sources)

- [ ] **test_task_graph_consistency**
  - [ ] All RiskTags reference valid task graph proposals
  - [ ] All EfficiencyTags reference valid task graph proposals
  - [ ] Task names match between tags and proposals
  - [ ] Source proposal IDs match referenced proposals

- [ ] **test_economics_consistency**
  - [ ] NoveltyTag scores match economics outputs (within 0.01 tolerance)
  - [ ] Expected MPL gains match economics outputs (within 0.01 tolerance)
  - [ ] Tier assignments consistent

### Forbidden Field Tests

- [ ] **test_no_forbidden_fields**
  - [ ] No `rewards` field in enrichment
  - [ ] No `mpl_value` field in enrichment
  - [ ] No `wage_parity` field in enrichment
  - [ ] No `sampling_weight` field in enrichment
  - [ ] No `task_order` field in enrichment
  - [ ] No `affordance_definitions` field in enrichment
  - [ ] Recursive check in nested dicts passes

- [ ] **test_no_mutations_to_inputs**
  - [ ] Datapacks unchanged after processing
  - [ ] Ontology proposals unchanged after processing
  - [ ] Task graph proposals unchanged after processing
  - [ ] Economics outputs unchanged after processing
  - [ ] Deep copy comparison passes

### Stable Ordering Tests

- [ ] **test_tag_ordering**
  - [ ] Fragility tags sorted by: object_name, contact_frames
  - [ ] Risk tags sorted by: affected_frames, risk_type
  - [ ] Affordance tags sorted by: object_name, affordance_name
  - [ ] Efficiency tags sorted by: metric, -score (descending)

- [ ] **test_proposal_ordering**
  - [ ] Proposals sorted by episode_id (lexicographic)
  - [ ] Order stable across multiple runs

### Multi-Video Tests

- [ ] **test_multi_video_aggregation**
  - [ ] Aggregated affordances = union of individual affordances
  - [ ] Aggregated coherence = min of individual coherence scores
  - [ ] No duplicate tags in aggregated output
  - [ ] Source proposals combined correctly

- [ ] **test_cross_video_conflict_detection**
  - [ ] Conflicting tags across videos detected
  - [ ] Semantic conflicts list populated
  - [ ] Coherence score reduced when conflicts present

### Resilience Tests

- [ ] **test_missing_ontology_graceful**
  - [ ] Empty ontology → no crash
  - [ ] AffordanceTags = [] (empty)
  - [ ] FragilityTags may be empty or from economics
  - [ ] Confidence reduced
  - [ ] Validation status = "passed"

- [ ] **test_missing_task_graph_graceful**
  - [ ] Empty task graph → no crash
  - [ ] RiskTags = [] (empty)
  - [ ] EfficiencyTags = [] (empty)
  - [ ] Confidence reduced
  - [ ] Validation status = "passed"

- [ ] **test_missing_economics_hard_fail**
  - [ ] Missing economics → validation fails
  - [ ] validation_status = "failed"
  - [ ] validation_errors mentions "economics"

- [ ] **test_partial_datapack_resilient**
  - [ ] Missing metadata.objects_present → no crash
  - [ ] Fewer tags generated (graceful degradation)
  - [ ] Confidence reduced
  - [ ] Validation status = "passed"

- [ ] **test_rejected_source_proposals_filtered**
  - [ ] Proposals with validation_status="failed" not used
  - [ ] Source_proposals list excludes rejected IDs
  - [ ] No tags derived from rejected proposals

### Coherence Tests

- [ ] **Conflict Detection**
  - [ ] High fragility + low risk → conflict detected
  - [ ] Safety-critical + no risk tags → conflict detected
  - [ ] High efficiency + high novelty → conflict detected (low severity)
  - [ ] Conflict severity assigned correctly (low/medium/high)

- [ ] **Coherence Score Computation**
  - [ ] No conflicts → coherence_score = 1.0
  - [ ] Low-severity conflict → coherence_score ≈ 0.9
  - [ ] Medium-severity conflict → coherence_score ≈ 0.7
  - [ ] High-severity conflict → coherence_score ≈ 0.5
  - [ ] Multiple conflicts → cumulative penalty

### Supervision Hints Tests

- [ ] **Priority Level Assignment**
  - [ ] High novelty (>0.7) → priority="high"
  - [ ] High risk (severity="high"/"critical") → priority="high"
  - [ ] Medium novelty (0.4-0.7) → priority="medium"
  - [ ] Low novelty (<0.4) → priority="low"

- [ ] **Weight Multiplier Computation**
  - [ ] priority="low" → weight=1.0
  - [ ] priority="medium" → weight=1.5
  - [ ] priority="high" → weight=2.0
  - [ ] priority="critical" → weight=3.0

- [ ] **Replay Frequency Assignment**
  - [ ] priority="high"/"critical" → frequency="frequent"
  - [ ] priority="medium" → frequency="standard"
  - [ ] priority="low" → frequency="rare"

- [ ] **Safety Flags**
  - [ ] High fragility → safety_critical=True
  - [ ] High risk → safety_critical=True
  - [ ] High-severity conflict → requires_human_review=True

- [ ] **Curriculum Stage Assignment**
  - [ ] Fragile objects present → curriculum_stage="advanced"
  - [ ] High risks present → curriculum_stage="mid"/"advanced"
  - [ ] No risks → curriculum_stage="early"

- [ ] **Prerequisites Identification**
  - [ ] Advanced stage → includes "basic_{task}"
  - [ ] Fragile objects → includes "fragile_object_awareness"

### Integration Tests

- [ ] **End-to-End Pipeline**
  - [ ] Load real Stage 1 datapacks
  - [ ] Load real Stage 2.2 ontology proposals
  - [ ] Load real Stage 2.3 task graph proposals
  - [ ] Load real economics outputs
  - [ ] Generate enrichments
  - [ ] Write JSONL output
  - [ ] Verify JSONL format
  - [ ] Merge with datapacks (mock Stage 3)

- [ ] **Sample Output Inspection**
  - [ ] Generate enrichment for drawer+vase scenario
  - [ ] Manually inspect tags:
    - [ ] FragilityTag for vase present
    - [ ] RiskTag for collision present
    - [ ] AffordanceTags for handle, drawer present
    - [ ] NoveltyTag with high score present
    - [ ] Coherence score > 0.9
    - [ ] SupervisionHints priority="high"
  - [ ] Verify JSONL format matches spec

---

## Final Sign-Off Conditions

### All Tests Pass

- [ ] **Smoke Test Suite**: 22/22 tests pass
- [ ] **No warnings or errors** in pytest output
- [ ] **No test modifications** required (tests as-is)

### Contract Compliance

- [ ] **No forbidden fields** in any output
- [ ] **No mutations** to any input
- [ ] **Read-only access** verified
- [ ] **Advisory-only** outputs confirmed

### Determinism Verified

- [ ] **10 consecutive runs** produce identical outputs
- [ ] **Proposal IDs stable** across runs
- [ ] **Tag ordering stable** across runs

### Schema Compliance

- [ ] **JSON schema validation** passes for all proposals
- [ ] **Required fields** present in all outputs
- [ ] **Value ranges** correct for all numeric fields

### JSON Safety

- [ ] **All proposals serializable** to JSON
- [ ] **Roundtrip preservation** verified
- [ ] **No unsafe types** in any output

### Cross-Consistency

- [ ] **Ontology alignment** verified
- [ ] **Task graph alignment** verified
- [ ] **Economics alignment** verified
- [ ] **No orphan tags** (all tags have sources)

### Graceful Degradation

- [ ] **Missing ontology** handled gracefully
- [ ] **Missing task graph** handled gracefully
- [ ] **Missing economics** fails with clear error
- [ ] **Partial datapacks** handled gracefully
- [ ] **Rejected proposals** filtered out correctly

### Documentation

- [ ] **All public APIs documented** with docstrings
- [ ] **Complex logic commented** inline
- [ ] **README added** to economics/ directory explaining usage
- [ ] **Example usage** provided in README

### Performance

- [ ] **Processing time** reasonable (<1s per datapack on average)
- [ ] **Memory usage** bounded (no memory leaks)
- [ ] **Batch processing** works (100+ datapacks)

---

## Post-Deployment Monitoring

### First-Run Validation

After deploying to production:

- [ ] **Generate enrichments** for 100 real datapacks
- [ ] **Inspect sample outputs** manually
- [ ] **Check coherence score distribution** (should be mostly >0.8)
- [ ] **Check priority level distribution** (balanced across low/medium/high)
- [ ] **Verify no crashes** on real data

### Quality Metrics

- [ ] **Average coherence score** > 0.85
- [ ] **Validation pass rate** > 95%
- [ ] **Tag coverage** (% datapacks with at least 1 tag) > 80%
- [ ] **Confidence distribution** (median > 0.7)

### Integration Validation

- [ ] **Stage 3 orchestrator** successfully merges enrichments
- [ ] **Supervision hints** applied correctly in training loop
- [ ] **High-priority episodes** weighted higher
- [ ] **Safety-critical episodes** flagged for review

---

## Troubleshooting Common Issues

### Issue: Determinism test fails

**Symptoms**: Proposal IDs or tag ordering differs across runs

**Diagnosis**:
- [ ] Check for timestamp-based IDs → use input timestamps only
- [ ] Check for random number generation → remove
- [ ] Check for unsorted collections → add explicit sorting

**Fix**:
- [ ] Replace `time.time()` with `datapack['metadata']['timestamp']`
- [ ] Remove all `random` and `np.random` calls
- [ ] Sort all tag lists before returning

### Issue: JSON serialization fails

**Symptoms**: `TypeError: Object of type X is not JSON serializable`

**Diagnosis**:
- [ ] Check for numpy arrays → convert to lists
- [ ] Check for torch tensors → convert to lists
- [ ] Check for custom objects → implement `to_dict()`
- [ ] Check for tuples → convert to lists for JSON

**Fix**:
- [ ] Add `to_dict()` methods to all dataclasses
- [ ] Convert numpy/torch types: `array.tolist()`
- [ ] Convert tuples in tag fields: `list(tuple_field)`

### Issue: Cross-consistency test fails

**Symptoms**: Tags reference proposals not in source_proposals list

**Diagnosis**:
- [ ] Check matcher filtering → ensure validated proposals used
- [ ] Check source_proposal_ids collection → ensure matchers return IDs
- [ ] Check tag generation logic → ensure source tracking

**Fix**:
- [ ] Filter proposals: `[p for p in proposals if p['validation_status']=='passed']`
- [ ] Track sources in matchers: `return proposal['proposal_id']`
- [ ] Include sources in final proposal: `source_proposals = matcher.get_source_ids()`

### Issue: Graceful degradation fails (crashes on missing data)

**Symptoms**: Exception raised when ontology/task graph empty

**Diagnosis**:
- [ ] Check for missing null checks → add `if not proposals: return []`
- [ ] Check for missing try/except → wrap tag generation
- [ ] Check for missing defaults → provide fallback values

**Fix**:
- [ ] Add early returns: `if not ontology_proposals: return []`
- [ ] Wrap in try/except: `try: tags = matcher.match() except: tags = []`
- [ ] Reduce confidence when data missing

---

## Specification Compliance Statement

An implementation is **specification-compliant** if and only if:

✅ All 22 smoke tests pass without modification
✅ All contract boundaries respected (no forbidden operations)
✅ Determinism verified (10+ identical runs)
✅ Schema compliance verified (JSON schema validation passes)
✅ JSON safety verified (all outputs serializable)
✅ Cross-consistency verified (tags align with sources)
✅ Graceful degradation verified (missing data handled)
✅ No mutations to inputs
✅ Stable tag ordering
✅ Multi-video aggregation works

**Signature**: ___________________________
**Date**: ___________________________
**Implementation Version**: ___________________________

---

**End of Verification Checklist**
