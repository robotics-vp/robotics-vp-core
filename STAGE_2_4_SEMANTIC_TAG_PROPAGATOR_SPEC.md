# Stage 2.4: SemanticTagPropagator - Technical Specification

**Status**: Design Complete, Implementation Pending
**Version**: 1.0
**Last Updated**: 2025-11-17

---

## Executive Summary

The **SemanticTagPropagator** is the final Stage 2 component that enriches Stage 1 datapacks with semantic metadata derived from ontology proposals (Stage 2.2) and task graph refinements (Stage 2.3). It produces **advisory-only semantic enrichments** that help the orchestrator prioritize, weight, and supervise training without mutating economic parameters, rewards, or existing datapack structures.

**Core Function**: Transform abstract ontology + task graph knowledge into concrete, per-datapack semantic tags that:
- Flag fragility, risk, and safety concerns
- Annotate affordances and efficiency opportunities
- Mark novelty and intervention potential
- Detect semantic conflicts and coherence issues
- Provide supervision hints for downstream orchestrator training

**Key Constraints**:
- **Read-only** access to Economics, Ontology, TaskGraph, and existing Datapacks
- **Advisory-only** outputs (no mutations to rewards, sampling weights, or task structure)
- **Deterministic** tag generation for equal inputs
- **Schema-compliant** enrichments that extend (never break) datapack format
- **Zero side effects** on global state

---

## 1. What SemanticTagPropagator Is

### 1.1 Purpose

The SemanticTagPropagator bridges **Stage 2 semantic understanding** (what tasks mean, how they relate, what risks they carry) with **Stage 1 execution datapacks** (the actual training episodes). It answers:

1. **"What does this datapack teach?"** → Semantic tags for content
2. **"How risky/fragile is this episode?"** → Safety and fragility annotations
3. **"What affordances does this demonstrate?"** → Action possibility tags
4. **"How novel/important is this data?"** → Novelty and intervention markers
5. **"Does this conflict with other knowledge?"** → Coherence and conflict flags
6. **"How should the orchestrator supervise this?"** → Training hints

### 1.2 Position in Pipeline

```
Stage 1 (Datapacks)          Stage 2.2 (Ontology)       Stage 2.3 (TaskGraph)
       ↓                            ↓                           ↓
       └──────────────────────→ SemanticTagPropagator ←────────┘
                                        ↓
                            SemanticEnrichmentProposal
                                        ↓
                          Enriched Datapacks (JSONL)
                                        ↓
                          Stage 3 Orchestrator (training supervision)
```

**Inputs**:
- Stage 1 datapacks (video embeddings, actions, rewards, metadata)
- Stage 2.2 OntologyUpdateProposals (affordances, relations, fragilities)
- Stage 2.3 TaskGraphRefinementProposals (dependencies, optimizations, risks)

**Outputs**:
- SemanticEnrichmentProposal objects (advisory tags per datapack)
- JSONL enrichment file (merge-ready with datapack schema)

### 1.3 What Makes It Different

Unlike Stage 2.2 and 2.3 (which propose **structural changes** to ontology/task graph), Stage 2.4 proposes **semantic annotations** for existing data. It doesn't change what tasks exist or how they relate—it adds interpretive metadata to training episodes.

**Analogy**: If datapacks are "raw footage", semantic tags are "director's commentary"—they don't change the footage, but they tell the viewer (orchestrator) what to pay attention to.

---

## 2. Inputs and Outputs

### 2.1 Inputs

#### 2.1.1 Stage 1 Datapacks (Primary Input)

**Format**: JSONL, one episode per line

```json
{
  "video_id": "drawer_demo_001",
  "episode_id": "ep_12345",
  "task": "open_drawer",
  "frames": [...],  // Video frame embeddings
  "actions": [...],  // Robot actions (joint velocities, gripper state)
  "rewards": [...],  // Per-step rewards
  "metadata": {
    "duration_sec": 12.3,
    "success": true,
    "human_demo": false,
    "objects_present": ["drawer", "handle", "vase_inside"]
  }
}
```

**Contract**:
- Read-only access
- No mutation of existing fields
- May reference `metadata` for context

#### 2.1.2 Stage 2.2 OntologyUpdateProposals (Semantic Knowledge)

**Format**: Python dataclass (from `OntologyUpdateEngine`)

```python
@dataclass
class OntologyUpdateProposal:
    proposal_id: str
    timestamp: float
    source_video: str

    new_affordances: List[AffordanceProposal]
    new_relations: List[RelationProposal]
    fragility_updates: List[FragilityProposal]

    confidence: float
    justification: str
    validation_status: str  # "pending" | "passed" | "failed"
```

**Usage**:
- Extract affordances → tag datapacks with affordance presence
- Extract fragilities → flag risky episodes
- Extract relations → tag semantic dependencies

#### 2.1.3 Stage 2.3 TaskGraphRefinementProposals (Task Structure)

**Format**: Python dataclass (from `TaskGraphRefiner`)

```python
@dataclass
class TaskGraphRefinementProposal:
    proposal_id: str
    timestamp: float
    source_analysis: str

    dependency_updates: List[DependencyProposal]
    parallelization_opportunities: List[ParallelizationProposal]
    risk_annotations: List[RiskProposal]
    efficiency_hints: List[EfficiencyProposal]

    confidence: float
    justification: str
    validation_status: str
```

**Usage**:
- Extract risk annotations → tag dangerous episodes
- Extract efficiency hints → tag optimal/suboptimal executions
- Extract dependencies → tag prerequisite violations

### 2.2 Outputs

#### 2.2.1 SemanticEnrichmentProposal (Primary Output)

**Format**: Python dataclass

```python
@dataclass
class SemanticEnrichmentProposal:
    """Advisory-only semantic tags for a datapack"""

    proposal_id: str  # Unique ID
    timestamp: float  # Generation time

    # Target datapack
    video_id: str
    episode_id: str
    task: str

    # Semantic tags (all optional, all advisory)
    fragility_tags: List[FragilityTag]
    risk_tags: List[RiskTag]
    affordance_tags: List[AffordanceTag]
    efficiency_tags: List[EfficiencyTag]
    novelty_tags: List[NoveltyTag]
    intervention_tags: List[InterventionTag]

    # Coherence analysis
    semantic_conflicts: List[SemanticConflict]
    coherence_score: float  # 0.0 = high conflict, 1.0 = fully coherent

    # Orchestrator hints
    supervision_hints: SupervisionHints

    # Metadata
    confidence: float  # Overall confidence in tags
    source_proposals: List[str]  # IDs of Stage 2.2/2.3 proposals used
    justification: str  # Human-readable explanation

    # Validation
    validation_status: str  # "pending" | "passed" | "failed"
    validation_errors: List[str]

    def to_jsonl_enrichment(self) -> dict:
        """Convert to JSONL format for merging with datapacks"""
        ...
```

#### 2.2.2 Tag Type Definitions

**FragilityTag**:
```python
@dataclass
class FragilityTag:
    object_name: str  # "vase", "glass", etc.
    fragility_level: str  # "low" | "medium" | "high" | "critical"
    damage_cost_usd: float  # From economics module
    contact_frames: List[int]  # Video frames with risky contact
    justification: str
```

**RiskTag**:
```python
@dataclass
class RiskTag:
    risk_type: str  # "collision" | "tip_over" | "entanglement" | "human_proximity"
    severity: str  # "low" | "medium" | "high" | "critical"
    affected_frames: List[int]
    mitigation_hints: List[str]  # Suggested safety measures
    justification: str
```

**AffordanceTag**:
```python
@dataclass
class AffordanceTag:
    affordance_name: str  # "graspable", "openable", "stackable"
    object_name: str
    demonstrated: bool  # Was this affordance used in episode?
    alternative_affordances: List[str]  # Other ways to achieve goal
    justification: str
```

**EfficiencyTag**:
```python
@dataclass
class EfficiencyTag:
    metric: str  # "time" | "energy" | "precision" | "success_rate"
    score: float  # 0.0 = worst, 1.0 = optimal
    benchmark: str  # What was this compared to?
    improvement_hints: List[str]  # How to improve
    justification: str
```

**NoveltyTag**:
```python
@dataclass
class NoveltyTag:
    novelty_type: str  # "state_coverage" | "action_diversity" | "failure_mode" | "edge_case"
    novelty_score: float  # 0.0 = redundant, 1.0 = maximally novel
    comparison_basis: str  # What data was this compared to?
    expected_mpl_gain: float  # From economics (E[ΔMPLᵢ])
    justification: str
```

**InterventionTag**:
```python
@dataclass
class InterventionTag:
    intervention_type: str  # "human_correction" | "failure_recovery" | "safety_override"
    frame_range: Tuple[int, int]  # When intervention occurred
    trigger: str  # What caused intervention
    learning_opportunity: str  # What should be learned
    justification: str
```

**SemanticConflict**:
```python
@dataclass
class SemanticConflict:
    conflict_type: str  # "affordance_mismatch" | "task_order_violation" | "risk_contradiction"
    conflicting_tags: List[str]  # Tag IDs in conflict
    severity: str  # "low" | "medium" | "high"
    resolution_hint: str  # How to resolve
    justification: str
```

**SupervisionHints**:
```python
@dataclass
class SupervisionHints:
    prioritize_for_training: bool  # Should orchestrator weight this higher?
    priority_level: str  # "low" | "medium" | "high" | "critical"

    suggested_weight_multiplier: float  # e.g., 1.5x for high-value data
    suggested_replay_frequency: str  # "standard" | "frequent" | "rare"

    requires_human_review: bool
    safety_critical: bool

    curriculum_stage: str  # "early" | "mid" | "late" | "advanced"
    prerequisite_tags: List[str]  # Tags that should be learned first

    justification: str
```

#### 2.2.3 JSONL Enrichment Format

**Purpose**: Merge-ready format for appending to datapack files

```jsonl
{"episode_id": "ep_12345", "enrichment": {"fragility_tags": [...], "risk_tags": [...], "affordance_tags": [...], "efficiency_tags": [...], "novelty_tags": [...], "intervention_tags": [...], "semantic_conflicts": [], "coherence_score": 0.92, "supervision_hints": {...}, "confidence": 0.87, "source_proposals": ["onto_prop_45", "task_prop_78"], "validation_status": "passed"}}
```

**Merge Strategy**:
1. Load original datapack JSONL
2. Load enrichment JSONL
3. Join on `episode_id`
4. Append `enrichment` field to each datapack entry
5. Write merged JSONL to output

**Schema Extension**:
```json
{
  "video_id": "drawer_demo_001",
  "episode_id": "ep_12345",
  "task": "open_drawer",
  "frames": [...],
  "actions": [...],
  "rewards": [...],
  "metadata": {...},

  // NEW: Semantic enrichment (optional, advisory-only)
  "enrichment": {
    "fragility_tags": [...],
    "risk_tags": [...],
    "affordance_tags": [...],
    "efficiency_tags": [...],
    "novelty_tags": [...],
    "intervention_tags": [...],
    "semantic_conflicts": [],
    "coherence_score": 0.92,
    "supervision_hints": {...},
    "confidence": 0.87,
    "source_proposals": ["onto_prop_45", "task_prop_78"],
    "validation_status": "passed"
  }
}
```

---

## 3. Contract Boundaries

### 3.1 What SemanticTagPropagator MAY Do

✅ **Read-Only Access**:
- Read Stage 1 datapacks (all fields)
- Read Stage 2.2 ontology proposals
- Read Stage 2.3 task graph proposals
- Read Economics module outputs (MPL, novelty scores, damage costs)

✅ **Advisory Proposals**:
- Generate semantic tags for datapacks
- Flag semantic conflicts and coherence issues
- Suggest supervision hints for orchestrator
- Compute confidence scores for tags

✅ **Schema Extension**:
- Add `enrichment` field to datapack schema
- Define new tag types (must be optional, backward-compatible)
- Version enrichment schema independently

✅ **Validation**:
- Check tag schema compliance
- Validate cross-consistency with ontology/task graph
- Flag forbidden field access
- Verify JSON serialization safety

### 3.2 What SemanticTagPropagator MAY NOT Do

❌ **Forbidden Mutations**:
- Modify economic parameters (MPL, wage parity, damage costs)
- Change reward values in datapacks
- Alter sampling weights or training priorities (only suggest)
- Mutate task graph structure (read-only access)
- Modify ontology affordances/relations (read-only access)

❌ **Forbidden Side Effects**:
- Write to global state
- Modify datapack files directly (only output enrichment proposals)
- Execute training or policy updates
- Trigger economics recalculations

❌ **Forbidden Overreach**:
- Make enforcement decisions (only advisory)
- Override orchestrator logic
- Directly control curriculum ordering
- Mandate data filtering or exclusion

### 3.3 Sibling Relationships

**vs. Economics (Older Sibling)**:
- Economics is **authoritative** for MPL, wage parity, damage costs
- SemanticTagPropagator **consumes** economics outputs (novelty scores, expected MPL gains)
- SemanticTagPropagator **cannot modify** economics parameters
- Economics runs first; tags reference economics results

**vs. OntologyUpdateEngine (2.2)**:
- Ontology proposals are **advisory input**
- SemanticTagPropagator **translates** ontology knowledge into datapack tags
- SemanticTagPropagator **cannot modify** ontology proposals
- If ontology proposal is rejected, tags referencing it are marked `validation_status: "failed"`

**vs. TaskGraphRefiner (2.3)**:
- Task graph proposals are **advisory input**
- SemanticTagPropagator **translates** task graph knowledge into datapack tags
- SemanticTagPropagator **cannot modify** task graph proposals
- If task graph proposal is rejected, tags referencing it are marked `validation_status: "failed"`

**vs. Datapacks (Stage 1)**:
- Datapacks are **primary input**
- SemanticTagPropagator **reads** datapacks (no mutation)
- SemanticTagPropagator **outputs** enrichment proposals (separate JSONL file)
- Merging enrichments with datapacks is **Stage 3 orchestrator's responsibility**

### 3.4 Interaction Flow

```
┌──────────────┐
│  Economics   │ (authoritative: MPL, novelty, damage costs)
└──────┬───────┘
       │ (read-only)
       ↓
┌──────────────────────────────────────────────────────────┐
│              SemanticTagPropagator                       │
│                                                          │
│  Inputs:                                                 │
│  • Stage 1 datapacks (read-only)                        │
│  • Stage 2.2 ontology proposals (read-only)             │
│  • Stage 2.3 task graph proposals (read-only)           │
│  • Economics outputs (read-only)                        │
│                                                          │
│  Outputs:                                                │
│  • SemanticEnrichmentProposal objects (advisory)        │
│  • JSONL enrichment file (merge-ready)                  │
│                                                          │
│  Constraints:                                            │
│  • No mutations to inputs                               │
│  • No side effects on global state                      │
│  • Deterministic for equal inputs                       │
│  • Schema-compliant extensions only                     │
└──────┬───────────────────────────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────────────────────────┐
│           Stage 3 Orchestrator (downstream)              │
│                                                          │
│  • Merges enrichments with datapacks                    │
│  • Uses supervision hints for training                   │
│  • Weights high-priority episodes                       │
│  • Filters safety-critical data for human review        │
└──────────────────────────────────────────────────────────┘
```

---

## 4. How Tags Flow Through the System

### 4.1 Tag Generation Pipeline

```
1. Load Inputs
   ├─ Stage 1 datapacks (JSONL)
   ├─ Stage 2.2 ontology proposals (validated only)
   ├─ Stage 2.3 task graph proposals (validated only)
   └─ Economics outputs (MPL, novelty scores)

2. Per-Datapack Processing
   For each episode in datapacks:

   2.1 Extract Context
       ├─ Task name, objects present, metadata
       ├─ Success/failure status
       └─ Duration, frame count

   2.2 Match Ontology Knowledge
       ├─ Find relevant affordance proposals (by object/task)
       ├─ Find relevant fragility proposals (by object)
       └─ Generate AffordanceTags, FragilityTags

   2.3 Match Task Graph Knowledge
       ├─ Find relevant risk annotations (by task)
       ├─ Find relevant efficiency hints (by task)
       └─ Generate RiskTags, EfficiencyTags

   2.4 Match Economics Knowledge
       ├─ Find novelty score for episode
       ├─ Find expected MPL gain
       └─ Generate NoveltyTags

   2.5 Detect Interventions
       ├─ Check metadata for human corrections
       ├─ Check for failure→recovery patterns
       └─ Generate InterventionTags

   2.6 Check Coherence
       ├─ Detect tag conflicts (e.g., "safe" + "high_risk")
       ├─ Compute coherence score
       └─ Generate SemanticConflict entries

   2.7 Generate Supervision Hints
       ├─ Combine tag priorities
       ├─ Check safety-critical flags
       ├─ Suggest weight multipliers
       └─ Assign curriculum stage

3. Create Proposal
   ├─ Assemble SemanticEnrichmentProposal
   ├─ Validate schema compliance
   ├─ Check forbidden field access
   └─ Compute overall confidence

4. Output JSONL
   ├─ Convert proposals to JSONL format
   ├─ Write enrichment file
   └─ Log validation results
```

### 4.2 Example: Drawer + Vase Episode

**Input Datapack**:
```json
{
  "video_id": "drawer_demo_001",
  "episode_id": "ep_12345",
  "task": "open_drawer",
  "metadata": {
    "success": true,
    "objects_present": ["drawer", "handle", "vase_inside"],
    "human_demo": false
  }
}
```

**Relevant Ontology Proposal (2.2)**:
```python
OntologyUpdateProposal(
    new_affordances=[
        AffordanceProposal(name="graspable", object="handle", ...),
        AffordanceProposal(name="pullable", object="drawer", ...)
    ],
    fragility_updates=[
        FragilityProposal(object="vase_inside", level="high", damage_cost=50.0, ...)
    ]
)
```

**Relevant Task Graph Proposal (2.3)**:
```python
TaskGraphRefinementProposal(
    risk_annotations=[
        RiskProposal(task="open_drawer", risk_type="collision",
                     condition="if fragile object inside", ...)
    ],
    efficiency_hints=[
        EfficiencyProposal(task="open_drawer", metric="time",
                          suggestion="slow down near fragile objects", ...)
    ]
)
```

**Economics Output**:
```python
{
    "episode_id": "ep_12345",
    "novelty_score": 0.73,  # High novelty (fragile object scenario)
    "expected_mpl_gain": 4.2  # Tier 2 (causal-novel)
}
```

**Generated Enrichment**:
```python
SemanticEnrichmentProposal(
    episode_id="ep_12345",

    fragility_tags=[
        FragilityTag(
            object_name="vase_inside",
            fragility_level="high",
            damage_cost_usd=50.0,
            contact_frames=[45, 46, 47],  # Drawer opening moments
            justification="Vase detected inside drawer; risk of tipping during pull"
        )
    ],

    risk_tags=[
        RiskTag(
            risk_type="collision",
            severity="medium",
            affected_frames=[44, 50],
            mitigation_hints=["Reduce pull velocity", "Add visual check before opening"],
            justification="Fragile object inside drawer increases collision risk"
        )
    ],

    affordance_tags=[
        AffordanceTag(
            affordance_name="graspable",
            object_name="handle",
            demonstrated=True,
            alternative_affordances=["pushable (if handle on both sides)"],
            justification="Handle grasped at frame 20"
        ),
        AffordanceTag(
            affordance_name="pullable",
            object_name="drawer",
            demonstrated=True,
            alternative_affordances=[],
            justification="Drawer pulled open successfully"
        )
    ],

    efficiency_tags=[
        EfficiencyTag(
            metric="time",
            score=0.68,  # Slower than average (good, due to fragility)
            benchmark="average_drawer_open_time",
            improvement_hints=["Maintain slow speed for safety"],
            justification="Slower execution appropriate for fragile object scenario"
        )
    ],

    novelty_tags=[
        NoveltyTag(
            novelty_type="edge_case",
            novelty_score=0.73,
            comparison_basis="all_drawer_episodes",
            expected_mpl_gain=4.2,
            justification="First episode with fragile object inside drawer"
        )
    ],

    intervention_tags=[],  # No human intervention

    semantic_conflicts=[],  # No conflicts detected
    coherence_score=0.95,  # High coherence (all tags align)

    supervision_hints=SupervisionHints(
        prioritize_for_training=True,
        priority_level="high",
        suggested_weight_multiplier=2.0,  # High novelty + safety relevance
        suggested_replay_frequency="frequent",
        requires_human_review=False,
        safety_critical=True,  # Fragile object present
        curriculum_stage="advanced",  # Requires understanding of fragility
        prerequisite_tags=["basic_drawer_open", "fragile_object_awareness"],
        justification="High-value edge case for fragility awareness training"
    ),

    confidence=0.87,
    source_proposals=["onto_prop_45", "task_prop_78"],
    validation_status="passed"
)
```

**Output JSONL**:
```jsonl
{"episode_id": "ep_12345", "enrichment": {"fragility_tags": [{"object_name": "vase_inside", "fragility_level": "high", "damage_cost_usd": 50.0, "contact_frames": [45, 46, 47], "justification": "Vase detected inside drawer; risk of tipping during pull"}], "risk_tags": [{"risk_type": "collision", "severity": "medium", "affected_frames": [44, 50], "mitigation_hints": ["Reduce pull velocity", "Add visual check before opening"], "justification": "Fragile object inside drawer increases collision risk"}], "affordance_tags": [{"affordance_name": "graspable", "object_name": "handle", "demonstrated": true, "alternative_affordances": ["pushable (if handle on both sides)"], "justification": "Handle grasped at frame 20"}, {"affordance_name": "pullable", "object_name": "drawer", "demonstrated": true, "alternative_affordances": [], "justification": "Drawer pulled open successfully"}], "efficiency_tags": [{"metric": "time", "score": 0.68, "benchmark": "average_drawer_open_time", "improvement_hints": ["Maintain slow speed for safety"], "justification": "Slower execution appropriate for fragile object scenario"}], "novelty_tags": [{"novelty_type": "edge_case", "novelty_score": 0.73, "comparison_basis": "all_drawer_episodes", "expected_mpl_gain": 4.2, "justification": "First episode with fragile object inside drawer"}], "intervention_tags": [], "semantic_conflicts": [], "coherence_score": 0.95, "supervision_hints": {"prioritize_for_training": true, "priority_level": "high", "suggested_weight_multiplier": 2.0, "suggested_replay_frequency": "frequent", "requires_human_review": false, "safety_critical": true, "curriculum_stage": "advanced", "prerequisite_tags": ["basic_drawer_open", "fragile_object_awareness"], "justification": "High-value edge case for fragility awareness training"}, "confidence": 0.87, "source_proposals": ["onto_prop_45", "task_prop_78"], "validation_status": "passed"}}
```

### 4.3 Tag Flow to Orchestrator

```
Enriched Datapack → Stage 3 Orchestrator
                         ↓
         ┌───────────────┴───────────────┐
         │                               │
    Supervision Hints              Tag-Based Filtering
         ↓                               ↓
  • Weight multiplier           • Safety-critical → human review
  • Replay frequency            • High novelty → prioritize
  • Curriculum stage            • Conflicts → investigate
         ↓                               ↓
  Training Loop                   Data Pipeline
  Prioritization                  Quality Control
```

**Orchestrator Actions** (examples):
1. **High-priority episodes** (priority_level="high") → 2x sampling weight
2. **Safety-critical episodes** (safety_critical=True) → human review queue
3. **Advanced curriculum** (curriculum_stage="advanced") → train after prerequisites
4. **High-coherence episodes** (coherence_score > 0.9) → standard training
5. **Conflicted episodes** (semantic_conflicts != []) → flag for investigation

---

## 5. Determinism and Reproducibility

### 5.1 Determinism Guarantees

**For equal inputs, SemanticTagPropagator MUST produce identical outputs.**

**Deterministic Inputs**:
- Same Stage 1 datapacks (same episodes, same order)
- Same Stage 2.2 ontology proposals (same proposal IDs, same order)
- Same Stage 2.3 task graph proposals (same proposal IDs, same order)
- Same Economics outputs (same novelty scores, MPL gains)

**Deterministic Outputs**:
- Same proposal IDs (generated from hash of inputs)
- Same tag sets (same tags, same order)
- Same confidence scores
- Same coherence scores
- Same validation status

### 5.2 Non-Deterministic Sources (Forbidden)

❌ **Avoid**:
- System timestamps (use input-derived timestamps only)
- Random number generation (no stochastic processes)
- External API calls (all knowledge from inputs)
- File system state (no directory scanning)
- Global counters (use input-derived IDs)

### 5.3 Stable Ordering

**Tag Ordering** (within each tag list):
1. Sort by `object_name` (alphabetical)
2. Then by `frame_range` (chronological)
3. Then by `confidence` (descending)

**Proposal Ordering** (across episodes):
1. Sort by `episode_id` (lexicographic)
2. Then by `timestamp` (chronological, from input datapack)

**Example**:
```python
def generate_proposals(datapacks, ontology_proposals, task_proposals, econ_outputs):
    proposals = []

    # Sort datapacks for deterministic processing
    sorted_datapacks = sorted(datapacks, key=lambda d: d['episode_id'])

    for datapack in sorted_datapacks:
        tags = generate_tags(datapack, ontology_proposals, task_proposals, econ_outputs)

        # Sort tags within each category
        tags['fragility_tags'] = sorted(tags['fragility_tags'],
                                       key=lambda t: (t.object_name, t.contact_frames[0]))
        tags['risk_tags'] = sorted(tags['risk_tags'],
                                  key=lambda t: (t.affected_frames[0], -t.confidence))
        # ... (sort other tag types)

        proposal = SemanticEnrichmentProposal(
            proposal_id=generate_deterministic_id(datapack, tags),
            timestamp=datapack['metadata']['timestamp'],  # From input
            **tags
        )
        proposals.append(proposal)

    return proposals
```

### 5.4 Reproducibility Testing

**Smoke Test**:
```python
def test_deterministic_generation():
    """Verify identical outputs for identical inputs"""
    datapacks = load_test_datapacks()
    ontology = load_test_ontology_proposals()
    task_graph = load_test_task_proposals()
    econ = load_test_economics_outputs()

    # Generate twice
    proposals_1 = propagator.generate_proposals(datapacks, ontology, task_graph, econ)
    proposals_2 = propagator.generate_proposals(datapacks, ontology, task_graph, econ)

    # Verify identical
    assert proposals_1 == proposals_2
    assert proposals_1[0].proposal_id == proposals_2[0].proposal_id
    assert proposals_1[0].fragility_tags == proposals_2[0].fragility_tags
```

---

## 6. Schema Compliance and Validation

### 6.1 Enrichment Schema

**Version**: 1.0
**Status**: Extension of Stage 1 datapack schema (backward-compatible)

**Schema Definition** (JSON Schema):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "fragility_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/FragilityTag"}
    },
    "risk_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/RiskTag"}
    },
    "affordance_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/AffordanceTag"}
    },
    "efficiency_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/EfficiencyTag"}
    },
    "novelty_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/NoveltyTag"}
    },
    "intervention_tags": {
      "type": "array",
      "items": {"$ref": "#/definitions/InterventionTag"}
    },
    "semantic_conflicts": {
      "type": "array",
      "items": {"$ref": "#/definitions/SemanticConflict"}
    },
    "coherence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "supervision_hints": {"$ref": "#/definitions/SupervisionHints"},
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "source_proposals": {
      "type": "array",
      "items": {"type": "string"}
    },
    "validation_status": {
      "type": "string",
      "enum": ["pending", "passed", "failed"]
    }
  },
  "required": ["coherence_score", "confidence", "validation_status"],
  "additionalProperties": false
}
```

### 6.2 Forbidden Fields

The enrichment schema **MUST NOT** include these fields (reserved for Economics/Stage 1):

❌ **Forbidden**:
- `rewards` (Economics owns reward computation)
- `mpl_value` (Economics owns MPL)
- `wage_parity` (Economics owns wage metrics)
- `sampling_weight` (Orchestrator owns sampling)
- `task_order` (TaskGraph owns structure)
- `affordance_definitions` (Ontology owns definitions)

**Validation**:
```python
FORBIDDEN_FIELDS = {'rewards', 'mpl_value', 'wage_parity', 'sampling_weight',
                    'task_order', 'affordance_definitions'}

def validate_enrichment_schema(enrichment: dict) -> List[str]:
    errors = []

    # Check for forbidden fields
    forbidden_found = FORBIDDEN_FIELDS.intersection(enrichment.keys())
    if forbidden_found:
        errors.append(f"Forbidden fields in enrichment: {forbidden_found}")

    # Check required fields
    required = {'coherence_score', 'confidence', 'validation_status'}
    missing = required - enrichment.keys()
    if missing:
        errors.append(f"Missing required fields: {missing}")

    # Check value ranges
    if not (0.0 <= enrichment.get('coherence_score', 0.5) <= 1.0):
        errors.append("coherence_score must be in [0.0, 1.0]")

    if not (0.0 <= enrichment.get('confidence', 0.5) <= 1.0):
        errors.append("confidence must be in [0.0, 1.0]")

    return errors
```

### 6.3 JSON Safety

**All tag types MUST be JSON-serializable.**

**Safe Types**:
- `str`, `int`, `float`, `bool`
- `List[safe_type]`
- `Dict[str, safe_type]`
- Nested combinations of above

**Unsafe Types** (forbidden):
- `numpy.ndarray` (convert to list)
- `torch.Tensor` (convert to list)
- Custom objects without `to_dict()` method
- Functions, lambdas
- File handles, sockets

**Serialization Test**:
```python
def test_json_serialization():
    proposal = generate_test_proposal()

    # Convert to dict
    enrichment_dict = proposal.to_jsonl_enrichment()

    # Verify JSON-safe
    try:
        json_str = json.dumps(enrichment_dict)
        reloaded = json.loads(json_str)
        assert enrichment_dict == reloaded
    except (TypeError, ValueError) as e:
        pytest.fail(f"Enrichment not JSON-safe: {e}")
```

---

## 7. Cross-Consistency Validation

### 7.1 Consistency with Ontology Proposals

**Rule**: Tags referencing ontology proposals MUST align with proposal content.

**Validation**:
```python
def validate_ontology_consistency(enrichment: SemanticEnrichmentProposal,
                                 ontology_proposals: List[OntologyUpdateProposal]) -> List[str]:
    errors = []

    # Check affordance tags
    for tag in enrichment.affordance_tags:
        # Find source proposal
        source_proposals = [p for p in ontology_proposals
                           if p.proposal_id in enrichment.source_proposals]

        # Verify affordance exists in source
        found = False
        for prop in source_proposals:
            if any(a.name == tag.affordance_name and a.object == tag.object_name
                   for a in prop.new_affordances):
                found = True
                break

        if not found:
            errors.append(f"AffordanceTag '{tag.affordance_name}' for '{tag.object_name}' "
                         f"not found in source ontology proposals")

    # Check fragility tags
    for tag in enrichment.fragility_tags:
        found = False
        for prop in source_proposals:
            if any(f.object == tag.object_name and f.level == tag.fragility_level
                   for f in prop.fragility_updates):
                found = True
                break

        if not found:
            errors.append(f"FragilityTag for '{tag.object_name}' "
                         f"not found in source ontology proposals")

    return errors
```

### 7.2 Consistency with Task Graph Proposals

**Rule**: Tags referencing task graph proposals MUST align with proposal content.

**Validation**:
```python
def validate_task_graph_consistency(enrichment: SemanticEnrichmentProposal,
                                   task_proposals: List[TaskGraphRefinementProposal]) -> List[str]:
    errors = []

    source_proposals = [p for p in task_proposals
                       if p.proposal_id in enrichment.source_proposals]

    # Check risk tags
    for tag in enrichment.risk_tags:
        found = False
        for prop in source_proposals:
            if any(r.task == enrichment.task and r.risk_type == tag.risk_type
                   for r in prop.risk_annotations):
                found = True
                break

        if not found:
            errors.append(f"RiskTag '{tag.risk_type}' for task '{enrichment.task}' "
                         f"not found in source task graph proposals")

    # Check efficiency tags
    for tag in enrichment.efficiency_tags:
        found = False
        for prop in source_proposals:
            if any(e.task == enrichment.task and e.metric == tag.metric
                   for e in prop.efficiency_hints):
                found = True
                break

        if not found:
            errors.append(f"EfficiencyTag metric '{tag.metric}' for task '{enrichment.task}' "
                         f"not found in source task graph proposals")

    return errors
```

### 7.3 Consistency with Economics

**Rule**: NoveltyTags MUST reference valid economics outputs.

**Validation**:
```python
def validate_economics_consistency(enrichment: SemanticEnrichmentProposal,
                                  econ_outputs: Dict[str, dict]) -> List[str]:
    errors = []

    # Check novelty tags
    for tag in enrichment.novelty_tags:
        econ_data = econ_outputs.get(enrichment.episode_id)
        if not econ_data:
            errors.append(f"No economics data for episode '{enrichment.episode_id}'")
            continue

        # Verify novelty score matches
        if abs(tag.novelty_score - econ_data['novelty_score']) > 0.01:
            errors.append(f"NoveltyTag score {tag.novelty_score} != "
                         f"economics score {econ_data['novelty_score']}")

        # Verify expected MPL gain matches
        if abs(tag.expected_mpl_gain - econ_data['expected_mpl_gain']) > 0.01:
            errors.append(f"NoveltyTag MPL gain {tag.expected_mpl_gain} != "
                         f"economics MPL gain {econ_data['expected_mpl_gain']}")

    return errors
```

### 7.4 Internal Coherence Check

**Rule**: Tags within an enrichment MUST NOT contradict each other.

**Conflict Detection**:
```python
def detect_semantic_conflicts(enrichment: SemanticEnrichmentProposal) -> List[SemanticConflict]:
    conflicts = []

    # Check: High fragility + Low risk (contradiction)
    high_fragility_objects = {t.object_name for t in enrichment.fragility_tags
                              if t.fragility_level in ['high', 'critical']}
    low_risk_tags = [t for t in enrichment.risk_tags if t.severity == 'low']

    if high_fragility_objects and low_risk_tags:
        conflicts.append(SemanticConflict(
            conflict_type="risk_fragility_mismatch",
            conflicting_tags=[f"fragility:{obj}" for obj in high_fragility_objects] +
                            [f"risk:{t.risk_type}" for t in low_risk_tags],
            severity="medium",
            resolution_hint="High fragility should correlate with higher risk severity",
            justification="Fragile objects present but risk rated as low"
        ))

    # Check: High efficiency + High novelty (unusual but not wrong)
    high_efficiency = any(t.score > 0.9 for t in enrichment.efficiency_tags)
    high_novelty = any(t.novelty_score > 0.8 for t in enrichment.novelty_tags)

    if high_efficiency and high_novelty:
        conflicts.append(SemanticConflict(
            conflict_type="efficiency_novelty_tension",
            conflicting_tags=["efficiency:high", "novelty:high"],
            severity="low",
            resolution_hint="High efficiency on novel data is unusual but possible (expert demo)",
            justification="Novel scenario executed with high efficiency"
        ))

    # Check: Safety-critical + No risk tags (likely missing data)
    if enrichment.supervision_hints.safety_critical and not enrichment.risk_tags:
        conflicts.append(SemanticConflict(
            conflict_type="safety_risk_missing",
            conflicting_tags=["supervision:safety_critical", "risk_tags:empty"],
            severity="medium",
            resolution_hint="Safety-critical episodes should have risk tags",
            justification="Marked safety-critical but no risk tags provided"
        ))

    return conflicts
```

**Coherence Score Computation**:
```python
def compute_coherence_score(enrichment: SemanticEnrichmentProposal) -> float:
    """Compute overall coherence (0.0 = high conflict, 1.0 = fully coherent)"""
    conflicts = enrichment.semantic_conflicts

    if not conflicts:
        return 1.0

    # Weight by severity
    severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.5}
    total_penalty = sum(severity_weights.get(c.severity, 0.3) for c in conflicts)

    # Normalize (cap at 1.0 total penalty)
    coherence = max(0.0, 1.0 - min(1.0, total_penalty))

    return coherence
```

---

## 8. Failure Modes and Resilience

### 8.1 Missing Ontology Proposals

**Scenario**: No ontology proposals exist for objects in datapack.

**Behavior**:
- Affordance tags: empty list
- Fragility tags: empty list (unless in economics damage_cost table)
- Confidence: reduced (e.g., 0.5 instead of 0.8)
- Justification: note missing ontology knowledge

**Graceful Degradation**:
```python
def generate_affordance_tags(datapack, ontology_proposals):
    tags = []
    objects = datapack['metadata'].get('objects_present', [])

    for obj in objects:
        # Find relevant proposals
        relevant = [p for p in ontology_proposals
                   if any(a.object == obj for a in p.new_affordances)]

        if not relevant:
            # No ontology knowledge; skip affordance tags
            continue

        # Generate tags from proposals
        for prop in relevant:
            for affordance in prop.new_affordances:
                if affordance.object == obj:
                    tags.append(AffordanceTag(
                        affordance_name=affordance.name,
                        object_name=obj,
                        demonstrated=check_demonstration(datapack, affordance),
                        alternative_affordances=[],
                        justification=f"From ontology proposal {prop.proposal_id}"
                    ))

    return tags
```

### 8.2 Missing Task Graph Proposals

**Scenario**: No task graph proposals exist for task in datapack.

**Behavior**:
- Risk tags: empty list (unless high fragility objects present)
- Efficiency tags: empty list
- Supervision hints: default to standard priority
- Confidence: reduced

### 8.3 Missing Economics Data

**Scenario**: No novelty score or MPL gain for episode.

**Behavior**:
- Novelty tags: empty list
- Supervision hints: default weight multiplier (1.0)
- Confidence: reduced
- Validation status: "failed" (economics data required)

**Hard Failure**:
```python
def validate_required_inputs(datapack, econ_outputs):
    errors = []

    episode_id = datapack['episode_id']
    if episode_id not in econ_outputs:
        errors.append(f"Missing economics data for episode {episode_id}")

    if errors:
        raise ValueError(f"Required inputs missing: {errors}")
```

### 8.4 Rejected Source Proposals

**Scenario**: Ontology/task graph proposal was rejected (validation_status="failed").

**Behavior**:
- Tags derived from rejected proposal are marked `validation_status: "failed"`
- Confidence: 0.0
- Justification: note source proposal rejection
- Do NOT include in final enrichment output

**Filtering**:
```python
def filter_validated_proposals(proposals):
    """Only use proposals that passed validation"""
    return [p for p in proposals if p.validation_status == "passed"]

def generate_proposals(datapacks, ontology_proposals, task_proposals, econ_outputs):
    # Filter to validated proposals only
    valid_ontology = filter_validated_proposals(ontology_proposals)
    valid_task = filter_validated_proposals(task_proposals)

    # Generate tags using only validated sources
    return [generate_enrichment(dp, valid_ontology, valid_task, econ_outputs)
            for dp in datapacks]
```

### 8.5 Partial Datapacks

**Scenario**: Datapack missing optional fields (e.g., `metadata.objects_present`).

**Behavior**:
- Extract what's available
- Reduce tag coverage (fewer tags)
- Mark confidence as lower
- Do NOT fail hard (graceful degradation)

**Robust Extraction**:
```python
def extract_objects(datapack) -> List[str]:
    """Safely extract objects, even if field missing"""
    metadata = datapack.get('metadata', {})
    objects = metadata.get('objects_present', [])

    if not objects:
        # Fallback: try to infer from task name
        task = datapack.get('task', '')
        objects = infer_objects_from_task(task)

    return objects
```

### 8.6 Multi-Video Aggregation

**Scenario**: Multiple videos demonstrate same task with different objects.

**Challenge**: Tag consistency across videos.

**Strategy**:
1. Generate tags per video (independent)
2. Aggregate tags across videos (union)
3. Detect cross-video conflicts
4. Resolve using confidence scores

**Aggregation**:
```python
def aggregate_tags_across_videos(enrichments: List[SemanticEnrichmentProposal]) -> SemanticEnrichmentProposal:
    """Merge tags from multiple videos for same task"""

    # Group by task
    task = enrichments[0].task
    assert all(e.task == task for e in enrichments), "Task mismatch"

    # Union of tags (deduplication by content)
    all_affordance_tags = []
    for e in enrichments:
        for tag in e.affordance_tags:
            if not any(t.affordance_name == tag.affordance_name and
                      t.object_name == tag.object_name
                      for t in all_affordance_tags):
                all_affordance_tags.append(tag)

    # Weighted average of scores
    coherence_scores = [e.coherence_score for e in enrichments]
    avg_coherence = sum(coherence_scores) / len(coherence_scores)

    # Detect cross-video conflicts
    conflicts = detect_cross_video_conflicts(enrichments)

    # Create aggregated proposal
    return SemanticEnrichmentProposal(
        proposal_id=f"aggregated_{task}_{len(enrichments)}videos",
        task=task,
        affordance_tags=all_affordance_tags,
        # ... (merge other tags)
        coherence_score=avg_coherence,
        semantic_conflicts=conflicts,
        confidence=min(e.confidence for e in enrichments),  # Conservative
        source_proposals=list(set(sum([e.source_proposals for e in enrichments], []))),
        justification=f"Aggregated from {len(enrichments)} videos"
    )
```

---

## 9. Summary of Guarantees

### 9.1 What This Spec Guarantees

✅ **Determinism**: Equal inputs → equal outputs
✅ **Schema Compliance**: All tags conform to defined schema
✅ **JSON Safety**: All outputs serializable to JSON
✅ **Contract Adherence**: No forbidden field access, no mutations
✅ **Cross-Consistency**: Tags align with source proposals
✅ **Graceful Degradation**: Missing inputs reduce coverage, not crash
✅ **Validation**: Proposals self-validate before output

### 9.2 What This Spec Does NOT Guarantee

❌ **Semantic Correctness**: Tags may misinterpret visual data
❌ **Completeness**: Missing source proposals → missing tags
❌ **Optimality**: Supervision hints are suggestions, not optimal
❌ **Conflict Resolution**: Conflicts flagged, not auto-resolved
❌ **Training Improvement**: Tags advisory; orchestrator must use them

### 9.3 Acceptance Criteria

A SemanticTagPropagator implementation is **specification-compliant** if:

1. ✅ Passes all smoke tests (determinism, schema, JSON, validation)
2. ✅ Produces no forbidden field access
3. ✅ Tags align with source proposals (cross-consistency)
4. ✅ Handles missing inputs gracefully (no crashes)
5. ✅ Outputs valid JSONL (merge-ready with datapacks)
6. ✅ Detects and flags semantic conflicts
7. ✅ Generates stable, sorted tag ordering
8. ✅ Confidence scores reflect input quality

---

## Next Steps

1. **Review this specification** with stakeholders
2. **Implement SemanticTagPropagator** per Codex guide (separate file)
3. **Run smoke tests** to verify compliance
4. **Generate sample enrichments** for drawer/vase scenario
5. **Integrate with Stage 3 orchestrator** (consumption of enrichments)
6. **Validate end-to-end**: Stage 1 → 2.1/2.2/2.3/2.4 → 3

---

**End of Stage 2.4 Specification**
