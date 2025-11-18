# Stage 2.4: SemanticTagPropagator - Smoke Test Suite

**Version**: 1.0
**Last Updated**: 2025-11-17

---

## Overview

This document defines the **comprehensive smoke test suite** for validating SemanticTagPropagator implementations. All tests must pass for specification compliance.

**Test Organization**:
1. **Determinism Tests**: Verify reproducibility
2. **Schema Compliance Tests**: Validate structure and types
3. **JSON Safety Tests**: Ensure serialization works
4. **Cross-Consistency Tests**: Verify alignment with source proposals
5. **Forbidden Field Tests**: Prevent contract violations
6. **Stable Ordering Tests**: Check deterministic sorting
7. **Multi-Video Tests**: Validate aggregation logic
8. **Resilience Tests**: Handle missing/partial data gracefully

---

## Test Suite

### 1. Determinism Tests

#### 1.1 test_deterministic_generation

**Purpose**: Verify identical outputs for identical inputs.

```python
def test_deterministic_generation():
    """Equal inputs must produce equal outputs"""
    # Load test inputs
    datapacks = load_fixture("test_datapacks.jsonl")
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("test_task_graph_proposals.json")
    economics = load_fixture("test_economics_outputs.json")

    # Generate proposals twice
    propagator = SemanticTagPropagator()
    proposals_1 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)
    proposals_2 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Verify identical outputs
    assert len(proposals_1) == len(proposals_2)

    for p1, p2 in zip(proposals_1, proposals_2):
        assert p1.proposal_id == p2.proposal_id, "Proposal IDs must match"
        assert p1.episode_id == p2.episode_id, "Episode IDs must match"
        assert p1.fragility_tags == p2.fragility_tags, "Fragility tags must match"
        assert p1.risk_tags == p2.risk_tags, "Risk tags must match"
        assert p1.affordance_tags == p2.affordance_tags, "Affordance tags must match"
        assert p1.efficiency_tags == p2.efficiency_tags, "Efficiency tags must match"
        assert p1.novelty_tags == p2.novelty_tags, "Novelty tags must match"
        assert p1.intervention_tags == p2.intervention_tags, "Intervention tags must match"
        assert p1.semantic_conflicts == p2.semantic_conflicts, "Conflicts must match"
        assert p1.coherence_score == p2.coherence_score, "Coherence score must match"
        assert p1.confidence == p2.confidence, "Confidence must match"
        assert p1.validation_status == p2.validation_status, "Validation status must match"

    print("✓ Determinism test passed")
```

#### 1.2 test_deterministic_ids

**Purpose**: Verify proposal IDs are deterministically generated.

```python
def test_deterministic_ids():
    """Proposal IDs must be deterministic (hash of inputs)"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()

    # Generate 10 times
    ids = []
    for _ in range(10):
        proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)
        ids.append(proposal.proposal_id)

    # All IDs must be identical
    assert len(set(ids)) == 1, "Proposal IDs must be deterministic"
    assert ids[0].startswith("enrich_"), "Proposal ID must have 'enrich_' prefix"

    print("✓ Deterministic ID test passed")
```

#### 1.3 test_no_random_sources

**Purpose**: Ensure no randomness in generation.

```python
def test_no_random_sources():
    """Implementation must not use random number generation"""
    import random
    import numpy as np

    # Set random seeds (should have no effect if deterministic)
    random.seed(12345)
    np.random.seed(12345)

    datapacks = load_fixture("test_datapacks.jsonl")
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("test_task_graph_proposals.json")
    economics = load_fixture("test_economics_outputs.json")

    propagator = SemanticTagPropagator()
    proposals_1 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Change seeds
    random.seed(99999)
    np.random.seed(99999)

    proposals_2 = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Outputs must still be identical (no randomness)
    assert proposals_1 == proposals_2, "Changing random seeds should have no effect"

    print("✓ No random sources test passed")
```

---

### 2. Schema Compliance Tests

#### 2.1 test_enrichment_schema_compliance

**Purpose**: Validate enrichment structure matches schema.

```python
def test_enrichment_schema_compliance():
    """Enrichments must conform to defined schema"""
    import jsonschema

    # Load schema
    schema = load_fixture("enrichment_schema.json")

    # Generate sample proposal
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    # Convert to JSONL format
    enrichment = proposal.to_jsonl_enrichment()

    # Validate against schema
    try:
        jsonschema.validate(instance=enrichment['enrichment'], schema=schema)
    except jsonschema.ValidationError as e:
        pytest.fail(f"Schema validation failed: {e}")

    print("✓ Schema compliance test passed")
```

#### 2.2 test_required_fields_present

**Purpose**: Ensure all required fields are present.

```python
def test_required_fields_present():
    """Required fields must be present in all enrichments"""
    REQUIRED_FIELDS = {'coherence_score', 'confidence', 'validation_status'}

    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    enrichment = proposal.to_jsonl_enrichment()['enrichment']

    # Check required fields
    missing = REQUIRED_FIELDS - enrichment.keys()
    assert not missing, f"Missing required fields: {missing}"

    # Check types
    assert isinstance(enrichment['coherence_score'], float), "coherence_score must be float"
    assert isinstance(enrichment['confidence'], float), "confidence must be float"
    assert isinstance(enrichment['validation_status'], str), "validation_status must be string"

    # Check valid values
    assert enrichment['validation_status'] in ['pending', 'passed', 'failed'], \
        "validation_status must be pending/passed/failed"

    print("✓ Required fields test passed")
```

#### 2.3 test_value_ranges

**Purpose**: Verify numerical values are in valid ranges.

```python
def test_value_ranges():
    """Numerical values must be in valid ranges"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    # Check coherence_score in [0.0, 1.0]
    assert 0.0 <= proposal.coherence_score <= 1.0, \
        f"coherence_score {proposal.coherence_score} out of range [0.0, 1.0]"

    # Check confidence in [0.0, 1.0]
    assert 0.0 <= proposal.confidence <= 1.0, \
        f"confidence {proposal.confidence} out of range [0.0, 1.0]"

    # Check novelty scores
    for tag in proposal.novelty_tags:
        assert 0.0 <= tag.novelty_score <= 1.0, \
            f"novelty_score {tag.novelty_score} out of range [0.0, 1.0]"

    # Check efficiency scores
    for tag in proposal.efficiency_tags:
        assert 0.0 <= tag.score <= 1.0, \
            f"efficiency score {tag.score} out of range [0.0, 1.0]"

    print("✓ Value range test passed")
```

---

### 3. JSON Safety Tests

#### 3.1 test_json_serialization

**Purpose**: Ensure all proposals are JSON-serializable.

```python
def test_json_serialization():
    """All enrichments must be JSON-serializable"""
    import json

    datapacks = load_fixture("test_datapacks.jsonl")
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("test_task_graph_proposals.json")
    economics = load_fixture("test_economics_outputs.json")

    propagator = SemanticTagPropagator()
    proposals = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    for proposal in proposals:
        enrichment = proposal.to_jsonl_enrichment()

        # Attempt JSON serialization
        try:
            json_str = json.dumps(enrichment)
            reloaded = json.loads(json_str)
            assert enrichment == reloaded, "Serialization roundtrip failed"
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed for {proposal.episode_id}: {e}")

    print("✓ JSON serialization test passed")
```

#### 3.2 test_no_unsafe_types

**Purpose**: Detect unsafe types (numpy arrays, tensors, etc.).

```python
def test_no_unsafe_types():
    """Enrichments must not contain numpy/torch/other unsafe types"""
    import numpy as np
    import torch

    UNSAFE_TYPES = (np.ndarray, torch.Tensor, type(lambda: None), type(open(__file__)))

    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    enrichment = proposal.to_jsonl_enrichment()

    # Recursively check for unsafe types
    def check_value(value, path=""):
        if isinstance(value, UNSAFE_TYPES):
            pytest.fail(f"Unsafe type {type(value)} found at {path}")
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                check_value(v, f"{path}[{i}]")

    check_value(enrichment)

    print("✓ No unsafe types test passed")
```

---

### 4. Cross-Consistency Tests

#### 4.1 test_ontology_consistency

**Purpose**: Verify tags align with ontology proposals.

```python
def test_ontology_consistency():
    """Affordance/fragility tags must reference valid ontology proposals"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology_proposals = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology_proposals, task_graph, economics)

    # Check affordance tags reference ontology
    for tag in proposal.affordance_tags:
        # Find source proposal
        found = False
        for onto_prop in ontology_proposals:
            if onto_prop['proposal_id'] in proposal.source_proposals:
                for affordance in onto_prop['new_affordances']:
                    if (affordance['name'] == tag.affordance_name and
                        affordance['object'] == tag.object_name):
                        found = True
                        break
            if found:
                break

        assert found, f"AffordanceTag {tag.affordance_name}:{tag.object_name} not in ontology"

    # Check fragility tags reference ontology
    for tag in proposal.fragility_tags:
        found = False
        for onto_prop in ontology_proposals:
            if onto_prop['proposal_id'] in proposal.source_proposals:
                for fragility in onto_prop.get('fragility_updates', []):
                    if fragility['object'] == tag.object_name:
                        found = True
                        break
            if found:
                break

        # Note: Fragility can also come from economics damage_cost table
        # So don't fail if not in ontology, just check if source_proposals referenced
        if proposal.source_proposals:  # Only check if sources claimed
            assert found or tag.object_name in economics_damage_costs, \
                f"FragilityTag {tag.object_name} not in ontology or economics"

    print("✓ Ontology consistency test passed")
```

#### 4.2 test_task_graph_consistency

**Purpose**: Verify tags align with task graph proposals.

```python
def test_task_graph_consistency():
    """Risk/efficiency tags must reference valid task graph proposals"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph_proposals = load_fixture("test_task_graph_proposals.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph_proposals, economics)

    # Check risk tags reference task graph
    for tag in proposal.risk_tags:
        found = False
        for tg_prop in task_graph_proposals:
            if tg_prop['proposal_id'] in proposal.source_proposals:
                for risk in tg_prop.get('risk_annotations', []):
                    if (risk['task'] == proposal.task and
                        risk['risk_type'] == tag.risk_type):
                        found = True
                        break
            if found:
                break

        assert found, f"RiskTag {tag.risk_type} for task {proposal.task} not in task graph"

    # Check efficiency tags reference task graph
    for tag in proposal.efficiency_tags:
        found = False
        for tg_prop in task_graph_proposals:
            if tg_prop['proposal_id'] in proposal.source_proposals:
                for eff in tg_prop.get('efficiency_hints', []):
                    if (eff['task'] == proposal.task and
                        eff['metric'] == tag.metric):
                        found = True
                        break
            if found:
                break

        assert found, f"EfficiencyTag {tag.metric} for task {proposal.task} not in task graph"

    print("✓ Task graph consistency test passed")
```

#### 4.3 test_economics_consistency

**Purpose**: Verify novelty tags match economics outputs.

```python
def test_economics_consistency():
    """Novelty tags must match economics data"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    episode_id = datapack['episode_id']
    econ_data = economics[episode_id]

    # Check novelty tags match economics
    for tag in proposal.novelty_tags:
        # Novelty score must match (within tolerance)
        assert abs(tag.novelty_score - econ_data['novelty_score']) < 0.01, \
            f"Novelty score mismatch: {tag.novelty_score} != {econ_data['novelty_score']}"

        # Expected MPL gain must match (within tolerance)
        assert abs(tag.expected_mpl_gain - econ_data['expected_mpl_gain']) < 0.01, \
            f"MPL gain mismatch: {tag.expected_mpl_gain} != {econ_data['expected_mpl_gain']}"

    print("✓ Economics consistency test passed")
```

---

### 5. Forbidden Field Tests

#### 5.1 test_no_forbidden_fields

**Purpose**: Ensure forbidden fields are not present.

```python
def test_no_forbidden_fields():
    """Enrichments must not contain forbidden fields"""
    FORBIDDEN_FIELDS = {
        'rewards', 'mpl_value', 'wage_parity', 'sampling_weight',
        'task_order', 'affordance_definitions'
    }

    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    enrichment = proposal.to_jsonl_enrichment()['enrichment']

    # Check for forbidden fields
    forbidden_found = FORBIDDEN_FIELDS.intersection(enrichment.keys())
    assert not forbidden_found, f"Forbidden fields found: {forbidden_found}"

    # Recursively check nested dicts
    def check_nested(obj, path=""):
        if isinstance(obj, dict):
            forbidden_nested = FORBIDDEN_FIELDS.intersection(obj.keys())
            assert not forbidden_nested, f"Forbidden fields found at {path}: {forbidden_nested}"
            for key, value in obj.items():
                check_nested(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                check_nested(item, f"{path}[{i}]")

    check_nested(enrichment)

    print("✓ Forbidden field test passed")
```

#### 5.2 test_no_mutations_to_inputs

**Purpose**: Verify inputs are not mutated during processing.

```python
def test_no_mutations_to_inputs():
    """Inputs must not be mutated during proposal generation"""
    import copy

    # Load inputs
    datapacks = load_fixture("test_datapacks.jsonl")
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    # Deep copy for comparison
    datapacks_copy = copy.deepcopy(datapacks)
    ontology_copy = copy.deepcopy(ontology)
    task_graph_copy = copy.deepcopy(task_graph)
    economics_copy = copy.deepcopy(economics)

    # Generate proposals
    propagator = SemanticTagPropagator()
    _ = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Verify no mutations
    assert datapacks == datapacks_copy, "Datapacks were mutated"
    assert ontology == ontology_copy, "Ontology proposals were mutated"
    assert task_graph == task_graph_copy, "Task graph proposals were mutated"
    assert economics == economics_copy, "Economics outputs were mutated"

    print("✓ No mutation test passed")
```

---

### 6. Stable Ordering Tests

#### 6.1 test_tag_ordering

**Purpose**: Verify tags are sorted deterministically.

```python
def test_tag_ordering():
    """Tags within each category must be sorted deterministically"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    # Check fragility tags sorted by object_name, then contact_frames
    fragility_tags = proposal.fragility_tags
    for i in range(len(fragility_tags) - 1):
        t1, t2 = fragility_tags[i], fragility_tags[i + 1]
        assert t1.object_name <= t2.object_name, \
            f"Fragility tags not sorted by object_name: {t1.object_name} > {t2.object_name}"
        if t1.object_name == t2.object_name:
            assert t1.contact_frames[0] <= t2.contact_frames[0], \
                f"Fragility tags not sorted by contact_frames"

    # Check risk tags sorted by affected_frames
    risk_tags = proposal.risk_tags
    for i in range(len(risk_tags) - 1):
        t1, t2 = risk_tags[i], risk_tags[i + 1]
        assert t1.affected_frames[0] <= t2.affected_frames[0], \
            f"Risk tags not sorted by affected_frames"

    # Check affordance tags sorted by object_name
    affordance_tags = proposal.affordance_tags
    for i in range(len(affordance_tags) - 1):
        t1, t2 = affordance_tags[i], affordance_tags[i + 1]
        assert t1.object_name <= t2.object_name, \
            f"Affordance tags not sorted by object_name"

    print("✓ Tag ordering test passed")
```

#### 6.2 test_proposal_ordering

**Purpose**: Verify proposals are sorted by episode_id.

```python
def test_proposal_ordering():
    """Proposals must be sorted by episode_id"""
    datapacks = load_fixture("test_datapacks.jsonl")  # Multiple episodes
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("test_economics_outputs.json")

    propagator = SemanticTagPropagator()
    proposals = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Check sorted by episode_id
    episode_ids = [p.episode_id for p in proposals]
    sorted_ids = sorted(episode_ids)
    assert episode_ids == sorted_ids, \
        f"Proposals not sorted by episode_id: {episode_ids} != {sorted_ids}"

    print("✓ Proposal ordering test passed")
```

---

### 7. Multi-Video Tests

#### 7.1 test_multi_video_aggregation

**Purpose**: Verify tag aggregation across multiple videos.

```python
def test_multi_video_aggregation():
    """Aggregating tags across videos must be consistent"""
    # Three videos of same task
    datapacks = load_fixture("test_drawer_multi_video.jsonl")  # 3 episodes
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("test_economics_outputs.json")

    propagator = SemanticTagPropagator()
    proposals = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # All should be for same task
    tasks = [p.task for p in proposals]
    assert len(set(tasks)) == 1, "Multi-video test requires same task"

    # Aggregate
    aggregated = propagator.aggregate_proposals_for_task(proposals)

    # Check union of affordances
    all_affordances = set()
    for p in proposals:
        for tag in p.affordance_tags:
            all_affordances.add((tag.affordance_name, tag.object_name))

    agg_affordances = set()
    for tag in aggregated.affordance_tags:
        agg_affordances.add((tag.affordance_name, tag.object_name))

    assert agg_affordances == all_affordances, \
        "Aggregated affordances should be union of individual affordances"

    # Check coherence score (conservative: min of individuals)
    min_coherence = min(p.coherence_score for p in proposals)
    assert aggregated.coherence_score == min_coherence, \
        "Aggregated coherence should be min of individuals"

    print("✓ Multi-video aggregation test passed")
```

#### 7.2 test_cross_video_conflict_detection

**Purpose**: Detect conflicts between videos of same task.

```python
def test_cross_video_conflict_detection():
    """Conflicting tags across videos must be detected"""
    # Two videos with conflicting tags
    datapacks = load_fixture("test_drawer_conflicting_videos.jsonl")
    ontology = load_fixture("test_ontology_proposals.json")
    task_graph = load_fixture("test_task_graph_proposals.json")
    economics = load_fixture("test_economics_outputs.json")

    propagator = SemanticTagPropagator()
    proposals = propagator.generate_proposals(datapacks, ontology, task_graph, economics)

    # Aggregate
    aggregated = propagator.aggregate_proposals_for_task(proposals)

    # Should detect conflicts
    assert len(aggregated.semantic_conflicts) > 0, \
        "Cross-video conflicts should be detected"

    # Coherence score should be reduced
    avg_coherence = sum(p.coherence_score for p in proposals) / len(proposals)
    assert aggregated.coherence_score < avg_coherence, \
        "Conflicts should reduce coherence score"

    print("✓ Cross-video conflict detection test passed")
```

---

### 8. Resilience Tests

#### 8.1 test_missing_ontology_graceful

**Purpose**: Handle missing ontology proposals gracefully.

```python
def test_missing_ontology_graceful():
    """Missing ontology proposals should not crash"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = []  # Empty ontology
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()

    # Should not crash
    try:
        proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)
    except Exception as e:
        pytest.fail(f"Missing ontology caused crash: {e}")

    # Should have empty affordance/fragility tags
    assert len(proposal.affordance_tags) == 0, "Should have no affordance tags"
    # (fragility_tags may still exist from economics damage_cost table)

    # Confidence should be reduced
    assert proposal.confidence < 0.8, "Confidence should be reduced when ontology missing"

    # Validation should still pass
    assert proposal.validation_status == "passed", "Should still pass validation"

    print("✓ Missing ontology graceful degradation test passed")
```

#### 8.2 test_missing_task_graph_graceful

**Purpose**: Handle missing task graph proposals gracefully.

```python
def test_missing_task_graph_graceful():
    """Missing task graph proposals should not crash"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = []  # Empty task graph
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()

    # Should not crash
    try:
        proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)
    except Exception as e:
        pytest.fail(f"Missing task graph caused crash: {e}")

    # Should have empty risk/efficiency tags
    assert len(proposal.risk_tags) == 0, "Should have no risk tags"
    assert len(proposal.efficiency_tags) == 0, "Should have no efficiency tags"

    # Confidence should be reduced
    assert proposal.confidence < 0.8, "Confidence should be reduced when task graph missing"

    # Validation should still pass
    assert proposal.validation_status == "passed", "Should still pass validation"

    print("✓ Missing task graph graceful degradation test passed")
```

#### 8.3 test_missing_economics_hard_fail

**Purpose**: Verify hard failure when economics data missing.

```python
def test_missing_economics_hard_fail():
    """Missing economics data should cause validation failure"""
    datapack = load_fixture("drawer_vase_episode.json")
    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = {}  # Empty economics (missing episode)

    propagator = SemanticTagPropagator()

    # Should fail validation (economics required)
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    assert proposal.validation_status == "failed", \
        "Missing economics should cause validation failure"

    assert "economics" in str(proposal.validation_errors).lower(), \
        "Validation errors should mention missing economics"

    print("✓ Missing economics hard fail test passed")
```

#### 8.4 test_partial_datapack_resilient

**Purpose**: Handle partial datapacks (missing optional fields).

```python
def test_partial_datapack_resilient():
    """Partial datapacks should be processed gracefully"""
    # Datapack missing optional metadata.objects_present
    datapack = {
        "episode_id": "ep_partial",
        "task": "open_drawer",
        "frames": [...],
        "actions": [...],
        "metadata": {
            "success": True
            # Missing: objects_present
        }
    }

    ontology = load_fixture("ontology_proposal_45.json")
    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = {"ep_partial": {"novelty_score": 0.5, "expected_mpl_gain": 2.0}}

    propagator = SemanticTagPropagator()

    # Should not crash
    try:
        proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)
    except Exception as e:
        pytest.fail(f"Partial datapack caused crash: {e}")

    # May have fewer tags (object-specific tags missing)
    # But should still validate
    assert proposal.validation_status == "passed", "Should pass validation despite partial data"

    # Confidence should be reduced
    assert proposal.confidence < 0.9, "Confidence should be reduced for partial data"

    print("✓ Partial datapack resilience test passed")
```

#### 8.5 test_rejected_source_proposals_filtered

**Purpose**: Verify rejected proposals are filtered out.

```python
def test_rejected_source_proposals_filtered():
    """Rejected source proposals should not be used"""
    datapack = load_fixture("drawer_vase_episode.json")

    # Ontology proposal with failed validation
    ontology = [
        {
            "proposal_id": "onto_prop_failed",
            "validation_status": "failed",  # FAILED
            "new_affordances": [...]
        }
    ]

    task_graph = load_fixture("task_graph_proposal_78.json")
    economics = load_fixture("economics_ep_12345.json")

    propagator = SemanticTagPropagator()
    proposal = propagator.generate_single_proposal(datapack, ontology, task_graph, economics)

    # Failed ontology should not be in source_proposals
    assert "onto_prop_failed" not in proposal.source_proposals, \
        "Rejected proposals should not be referenced"

    # Should have no affordance tags derived from rejected proposal
    assert len(proposal.affordance_tags) == 0, \
        "Should not generate tags from rejected proposals"

    print("✓ Rejected proposals filtered test passed")
```

---

## Test Fixtures

### Required Test Data Files

**test_datapacks.jsonl**:
```jsonl
{"video_id": "drawer_001", "episode_id": "ep_001", "task": "open_drawer", "metadata": {"success": true, "objects_present": ["drawer", "handle"]}}
{"video_id": "drawer_002", "episode_id": "ep_002", "task": "open_drawer", "metadata": {"success": true, "objects_present": ["drawer", "handle", "vase_inside"]}}
{"video_id": "drawer_003", "episode_id": "ep_003", "task": "open_drawer", "metadata": {"success": false, "objects_present": ["drawer", "handle"]}}
```

**test_ontology_proposals.json**:
```json
[
  {
    "proposal_id": "onto_prop_45",
    "validation_status": "passed",
    "new_affordances": [
      {"name": "graspable", "object": "handle", "confidence": 0.92},
      {"name": "pullable", "object": "drawer", "confidence": 0.88}
    ],
    "fragility_updates": [
      {"object": "vase_inside", "level": "high", "damage_cost": 50.0}
    ]
  }
]
```

**test_task_graph_proposals.json**:
```json
[
  {
    "proposal_id": "task_prop_78",
    "validation_status": "passed",
    "risk_annotations": [
      {"task": "open_drawer", "risk_type": "collision", "severity": "medium"}
    ],
    "efficiency_hints": [
      {"task": "open_drawer", "metric": "time", "suggestion": "slow down"}
    ]
  }
]
```

**test_economics_outputs.json**:
```json
{
  "ep_001": {"novelty_score": 0.2, "expected_mpl_gain": 0.5, "tier": 0},
  "ep_002": {"novelty_score": 0.73, "expected_mpl_gain": 4.2, "tier": 2},
  "ep_003": {"novelty_score": 0.3, "expected_mpl_gain": 1.0, "tier": 1}
}
```

---

## Running the Test Suite

```bash
# Run all smoke tests
pytest tests/smoke_tests/test_semantic_tag_propagator.py -v

# Run specific test category
pytest tests/smoke_tests/test_semantic_tag_propagator.py::test_deterministic_generation -v

# Run with coverage
pytest tests/smoke_tests/ --cov=economics.semantic_tag_propagator --cov-report=html

# Verify all tests pass
pytest tests/smoke_tests/ --tb=short --strict-markers
```

**Expected Output** (all tests passing):
```
tests/smoke_tests/test_semantic_tag_propagator.py::test_deterministic_generation PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_deterministic_ids PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_no_random_sources PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_enrichment_schema_compliance PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_required_fields_present PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_value_ranges PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_json_serialization PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_no_unsafe_types PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_ontology_consistency PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_task_graph_consistency PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_economics_consistency PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_no_forbidden_fields PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_no_mutations_to_inputs PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_tag_ordering PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_proposal_ordering PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_multi_video_aggregation PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_cross_video_conflict_detection PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_missing_ontology_graceful PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_missing_task_graph_graceful PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_missing_economics_hard_fail PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_partial_datapack_resilient PASSED
tests/smoke_tests/test_semantic_tag_propagator.py::test_rejected_source_proposals_filtered PASSED

====================== 22 passed in 3.45s ======================
```

---

## Acceptance Criteria

An implementation **passes the smoke test suite** if:

✅ All 22 tests pass without modifications to test code
✅ No crashes or unhandled exceptions
✅ Determinism verified (3 tests)
✅ Schema compliance verified (3 tests)
✅ JSON safety verified (2 tests)
✅ Cross-consistency verified (3 tests)
✅ Contract boundaries enforced (2 tests)
✅ Stable ordering verified (2 tests)
✅ Multi-video handling verified (2 tests)
✅ Resilience verified (5 tests)

---

**End of Smoke Test Suite**
