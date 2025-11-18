# Stage 2.4: SemanticTagPropagator - Pipeline Diagrams and Workflows

**Version**: 1.0
**Last Updated**: 2025-11-17

---

## 1. Full Stage 2 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: DATA LAYER                         │
│                                                                     │
│  ┌──────────────┐         ┌──────────────┐                        │
│  │ Video Demos  │────────▶│  Datapacks   │                        │
│  │  (raw data)  │         │   (JSONL)    │                        │
│  └──────────────┘         └──────┬───────┘                        │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    │ (read-only)
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: SEMANTIC UNDERSTANDING                  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Stage 2.1: Economics (MPL, Novelty, Damage Costs)         │   │
│  │ • Authoritative source for value metrics                   │   │
│  │ • Computes novelty scores, expected MPL gains              │   │
│  └─────────────────┬──────────────────────────────────────────┘   │
│                    │                                               │
│                    │ (economics outputs)                           │
│                    ↓                                               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Stage 2.2: OntologyUpdateEngine (Affordances, Fragilities) │   │
│  │ • Discovers new affordances from video                     │   │
│  │ • Infers object relations and fragilities                  │   │
│  │ • Advisory proposals only (no mutations)                   │   │
│  └─────────────────┬──────────────────────────────────────────┘   │
│                    │                                               │
│                    │ (ontology proposals)                          │
│                    ↓                                               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Stage 2.3: TaskGraphRefiner (Dependencies, Risks)          │   │
│  │ • Refines task graph structure                             │   │
│  │ • Identifies risks and optimization opportunities          │   │
│  │ • Advisory proposals only (no mutations)                   │   │
│  └─────────────────┬──────────────────────────────────────────┘   │
│                    │                                               │
│                    │ (task graph proposals)                        │
│                    ↓                                               │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Stage 2.4: SemanticTagPropagator ◀──────────────┐          │   │
│  │ • Enriches datapacks with semantic tags         │          │   │
│  │ • Combines knowledge from 2.1/2.2/2.3           │          │   │
│  │ • Generates supervision hints for orchestrator  │          │   │
│  └─────────────────┬────────────────────────────────┘          │   │
│                    │                                   ┌────────┘  │
│                    │ (enrichment proposals)            │           │
│                    │                                   │           │
│                    │  ┌────────────────────────────────┘           │
│                    │  │ (reads datapacks)                          │
│                    │  │                                            │
└────────────────────┼──┼────────────────────────────────────────────┘
                     │  │
                     ↓  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 ENRICHED DATAPACKS (JSONL OUTPUT)                   │
│                                                                     │
│  Original datapack fields + "enrichment": {...}                    │
│  • fragility_tags, risk_tags, affordance_tags                      │
│  • efficiency_tags, novelty_tags, intervention_tags                │
│  • semantic_conflicts, coherence_score                             │
│  • supervision_hints (weight, priority, curriculum stage)          │
│                                                                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: ORCHESTRATOR                             │
│                                                                     │
│  • Merges enrichments with datapacks                               │
│  • Uses supervision hints for training prioritization              │
│  • Weights high-value episodes                                     │
│  • Filters safety-critical data for human review                   │
│  • Schedules curriculum based on prerequisite tags                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. SemanticTagPropagator Internal Flow

```
INPUT PHASE
═══════════

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Datapacks   │   │   Ontology   │   │  Task Graph  │   │  Economics   │
│   (JSONL)    │   │  Proposals   │   │  Proposals   │   │   Outputs    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │   Load & Validate     │
                     │   • Check schema      │
                     │   • Filter validated  │
                     │   • Sort for order    │
                     └───────────┬───────────┘
                                 │
                                 ↓

PROCESSING PHASE (Per Datapack)
═══════════════════════════════

                     ┌───────────────────────┐
                     │  Extract Context      │
                     │  • Task, objects      │
                     │  • Success/failure    │
                     │  • Metadata           │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Match Ontology       │
                     │  • Find affordances   │
                     │  • Find fragilities   │
                     │  → AffordanceTags     │
                     │  → FragilityTags      │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Match Task Graph     │
                     │  • Find risks         │
                     │  • Find efficiencies  │
                     │  → RiskTags           │
                     │  → EfficiencyTags     │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Match Economics      │
                     │  • Novelty score      │
                     │  • Expected MPL gain  │
                     │  → NoveltyTags        │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Detect Interventions │
                     │  • Human corrections  │
                     │  • Failure recovery   │
                     │  → InterventionTags   │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Check Coherence      │
                     │  • Detect conflicts   │
                     │  • Compute score      │
                     │  → SemanticConflicts  │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Generate Hints       │
                     │  • Priority level     │
                     │  • Weight multiplier  │
                     │  • Curriculum stage   │
                     │  → SupervisionHints   │
                     └───────────┬───────────┘
                                 │
                                 ↓

VALIDATION PHASE
════════════════

                     ┌───────────────────────┐
                     │  Validate Proposal    │
                     │  • Schema compliance  │
                     │  • Forbidden fields   │
                     │  • Cross-consistency  │
                     │  • JSON safety        │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Create Proposal      │
                     │  SemanticEnrichment   │
                     │  Proposal             │
                     └───────────┬───────────┘
                                 │
                                 ↓

OUTPUT PHASE
════════════

                     ┌───────────────────────┐
                     │  Convert to JSONL     │
                     │  • to_jsonl_enrichment│
                     │  • Sort tags          │
                     │  • Serialize          │
                     └───────────┬───────────┘
                                 │
                                 ↓
                     ┌───────────────────────┐
                     │  Write Output File    │
                     │  enrichments.jsonl    │
                     └───────────────────────┘
```

---

## 3. Example Workflow: Drawer + Vase Scenario

### 3.1 Input Data

**Stage 1 Datapack**:
```json
{
  "video_id": "drawer_demo_001",
  "episode_id": "ep_12345",
  "task": "open_drawer",
  "frames": [...],  // 60 frames, 30fps
  "actions": [...],  // Joint velocities
  "rewards": [...],  // Per-step rewards
  "metadata": {
    "duration_sec": 12.3,
    "success": true,
    "human_demo": false,
    "objects_present": ["drawer", "handle", "vase_inside"]
  }
}
```

**Stage 2.2 Ontology Proposal**:
```python
OntologyUpdateProposal(
    proposal_id="onto_prop_45",
    validation_status="passed",
    new_affordances=[
        AffordanceProposal(name="graspable", object="handle", confidence=0.92),
        AffordanceProposal(name="pullable", object="drawer", confidence=0.88)
    ],
    fragility_updates=[
        FragilityProposal(object="vase_inside", level="high", damage_cost=50.0)
    ]
)
```

**Stage 2.3 Task Graph Proposal**:
```python
TaskGraphRefinementProposal(
    proposal_id="task_prop_78",
    validation_status="passed",
    risk_annotations=[
        RiskProposal(task="open_drawer", risk_type="collision",
                     condition="if fragile object inside", severity="medium")
    ],
    efficiency_hints=[
        EfficiencyProposal(task="open_drawer", metric="time",
                          suggestion="slow down near fragile objects")
    ]
)
```

**Stage 2.1 Economics Output**:
```python
{
    "episode_id": "ep_12345",
    "novelty_score": 0.73,  # High (first fragile-object scenario)
    "expected_mpl_gain": 4.2,  # Tier 2 (causal-novel)
    "tier": 2
}
```

### 3.2 Processing Steps

**Step 1: Extract Context**
```python
context = {
    "task": "open_drawer",
    "objects": ["drawer", "handle", "vase_inside"],
    "success": True,
    "duration": 12.3
}
```

**Step 2: Match Ontology → Generate Affordance/Fragility Tags**
```python
affordance_tags = [
    AffordanceTag(
        affordance_name="graspable",
        object_name="handle",
        demonstrated=True,  # Grasp detected at frame 20
        alternative_affordances=["pushable (if handle on both sides)"]
    ),
    AffordanceTag(
        affordance_name="pullable",
        object_name="drawer",
        demonstrated=True,  # Pull detected frames 22-50
        alternative_affordances=[]
    )
]

fragility_tags = [
    FragilityTag(
        object_name="vase_inside",
        fragility_level="high",
        damage_cost_usd=50.0,
        contact_frames=[45, 46, 47],  # Drawer opening moments
        justification="Vase inside drawer; risk of tipping during pull"
    )
]
```

**Step 3: Match Task Graph → Generate Risk/Efficiency Tags**
```python
risk_tags = [
    RiskTag(
        risk_type="collision",
        severity="medium",
        affected_frames=[44, 50],
        mitigation_hints=["Reduce pull velocity", "Add visual check before opening"],
        justification="Fragile object inside drawer increases collision risk"
    )
]

efficiency_tags = [
    EfficiencyTag(
        metric="time",
        score=0.68,  # Slower than average (12.3s vs 8.0s average)
        benchmark="average_drawer_open_time",
        improvement_hints=["Maintain slow speed for safety"],
        justification="Slower execution appropriate for fragile object scenario"
    )
]
```

**Step 4: Match Economics → Generate Novelty Tags**
```python
novelty_tags = [
    NoveltyTag(
        novelty_type="edge_case",
        novelty_score=0.73,
        comparison_basis="all_drawer_episodes",
        expected_mpl_gain=4.2,
        justification="First episode with fragile object inside drawer"
    )
]
```

**Step 5: Detect Interventions → No Intervention Tags**
```python
intervention_tags = []  # No human corrections detected
```

**Step 6: Check Coherence → No Conflicts**
```python
semantic_conflicts = []
coherence_score = 0.95  # High coherence

# All tags align:
# - High fragility → medium risk (consistent)
# - Slower execution → high novelty (edge case requires caution)
# - No contradictions detected
```

**Step 7: Generate Supervision Hints**
```python
supervision_hints = SupervisionHints(
    prioritize_for_training=True,
    priority_level="high",
    suggested_weight_multiplier=2.0,  # High novelty + safety relevance
    suggested_replay_frequency="frequent",
    requires_human_review=False,
    safety_critical=True,  # Fragile object present
    curriculum_stage="advanced",  # Requires fragility awareness
    prerequisite_tags=["basic_drawer_open", "fragile_object_awareness"],
    justification="High-value edge case for fragility awareness training"
)
```

**Step 8: Validate → Pass**
```python
validation_errors = []
validation_status = "passed"
confidence = 0.87  # High confidence (all source proposals validated)
```

### 3.3 Output Enrichment

**SemanticEnrichmentProposal**:
```python
SemanticEnrichmentProposal(
    proposal_id="enrich_ep_12345_drawer_vase",
    timestamp=1700000000.0,
    video_id="drawer_demo_001",
    episode_id="ep_12345",
    task="open_drawer",

    fragility_tags=[...],  # As above
    risk_tags=[...],
    affordance_tags=[...],
    efficiency_tags=[...],
    novelty_tags=[...],
    intervention_tags=[],

    semantic_conflicts=[],
    coherence_score=0.95,

    supervision_hints=SupervisionHints(...),

    confidence=0.87,
    source_proposals=["onto_prop_45", "task_prop_78"],
    justification="Drawer opening with fragile object inside (high-value edge case)",
    validation_status="passed",
    validation_errors=[]
)
```

**JSONL Output (enrichments.jsonl)**:
```jsonl
{"episode_id": "ep_12345", "enrichment": {"fragility_tags": [{"object_name": "vase_inside", "fragility_level": "high", "damage_cost_usd": 50.0, "contact_frames": [45, 46, 47], "justification": "Vase inside drawer; risk of tipping during pull"}], "risk_tags": [{"risk_type": "collision", "severity": "medium", "affected_frames": [44, 50], "mitigation_hints": ["Reduce pull velocity", "Add visual check before opening"], "justification": "Fragile object inside drawer increases collision risk"}], "affordance_tags": [{"affordance_name": "graspable", "object_name": "handle", "demonstrated": true, "alternative_affordances": ["pushable (if handle on both sides)"], "justification": "Handle grasped at frame 20"}, {"affordance_name": "pullable", "object_name": "drawer", "demonstrated": true, "alternative_affordances": [], "justification": "Drawer pulled open successfully"}], "efficiency_tags": [{"metric": "time", "score": 0.68, "benchmark": "average_drawer_open_time", "improvement_hints": ["Maintain slow speed for safety"], "justification": "Slower execution appropriate for fragile object scenario"}], "novelty_tags": [{"novelty_type": "edge_case", "novelty_score": 0.73, "comparison_basis": "all_drawer_episodes", "expected_mpl_gain": 4.2, "justification": "First episode with fragile object inside drawer"}], "intervention_tags": [], "semantic_conflicts": [], "coherence_score": 0.95, "supervision_hints": {"prioritize_for_training": true, "priority_level": "high", "suggested_weight_multiplier": 2.0, "suggested_replay_frequency": "frequent", "requires_human_review": false, "safety_critical": true, "curriculum_stage": "advanced", "prerequisite_tags": ["basic_drawer_open", "fragile_object_awareness"], "justification": "High-value edge case for fragility awareness training"}, "confidence": 0.87, "source_proposals": ["onto_prop_45", "task_prop_78"], "validation_status": "passed"}}
```

### 3.4 Merged Datapack (Final Output for Orchestrator)

```json
{
  "video_id": "drawer_demo_001",
  "episode_id": "ep_12345",
  "task": "open_drawer",
  "frames": [...],
  "actions": [...],
  "rewards": [...],
  "metadata": {
    "duration_sec": 12.3,
    "success": true,
    "human_demo": false,
    "objects_present": ["drawer", "handle", "vase_inside"]
  },

  "enrichment": {
    "fragility_tags": [
      {
        "object_name": "vase_inside",
        "fragility_level": "high",
        "damage_cost_usd": 50.0,
        "contact_frames": [45, 46, 47],
        "justification": "Vase inside drawer; risk of tipping during pull"
      }
    ],
    "risk_tags": [
      {
        "risk_type": "collision",
        "severity": "medium",
        "affected_frames": [44, 50],
        "mitigation_hints": ["Reduce pull velocity", "Add visual check before opening"],
        "justification": "Fragile object inside drawer increases collision risk"
      }
    ],
    "affordance_tags": [
      {
        "affordance_name": "graspable",
        "object_name": "handle",
        "demonstrated": true,
        "alternative_affordances": ["pushable (if handle on both sides)"],
        "justification": "Handle grasped at frame 20"
      },
      {
        "affordance_name": "pullable",
        "object_name": "drawer",
        "demonstrated": true,
        "alternative_affordances": [],
        "justification": "Drawer pulled open successfully"
      }
    ],
    "efficiency_tags": [
      {
        "metric": "time",
        "score": 0.68,
        "benchmark": "average_drawer_open_time",
        "improvement_hints": ["Maintain slow speed for safety"],
        "justification": "Slower execution appropriate for fragile object scenario"
      }
    ],
    "novelty_tags": [
      {
        "novelty_type": "edge_case",
        "novelty_score": 0.73,
        "comparison_basis": "all_drawer_episodes",
        "expected_mpl_gain": 4.2,
        "justification": "First episode with fragile object inside drawer"
      }
    ],
    "intervention_tags": [],
    "semantic_conflicts": [],
    "coherence_score": 0.95,
    "supervision_hints": {
      "prioritize_for_training": true,
      "priority_level": "high",
      "suggested_weight_multiplier": 2.0,
      "suggested_replay_frequency": "frequent",
      "requires_human_review": false,
      "safety_critical": true,
      "curriculum_stage": "advanced",
      "prerequisite_tags": ["basic_drawer_open", "fragile_object_awareness"],
      "justification": "High-value edge case for fragility awareness training"
    },
    "confidence": 0.87,
    "source_proposals": ["onto_prop_45", "task_prop_78"],
    "validation_status": "passed"
  }
}
```

### 3.5 Orchestrator Actions (Stage 3)

**Based on enrichment tags, orchestrator decides**:

1. **Training Weight**: 2.0x multiplier (high priority)
   - Episode sampled 2x more often than standard episodes

2. **Replay Frequency**: "frequent"
   - Replayed every 10 episodes (vs 50 for standard)

3. **Safety Review**: Mark as safety-critical
   - Flag for human review if failure occurs
   - Log fragility handling performance

4. **Curriculum Placement**: "advanced" stage
   - Only train after prerequisites met:
     - "basic_drawer_open" (98% success rate)
     - "fragile_object_awareness" (90% success rate)

5. **Supervision Signal**: Enhanced reward shaping
   - Penalize rapid movements near fragile objects
   - Reward slow, careful execution

---

## 4. Failure Mode Examples

### 4.1 Missing Ontology Proposal

**Scenario**: No ontology proposal for "vase_inside" object.

**Input**:
```python
# Ontology proposals empty
ontology_proposals = []
```

**Processing**:
```python
# Step 2: Match Ontology
affordance_tags = []  # No ontology knowledge
fragility_tags = []   # Cannot infer fragility without ontology

# Reduce confidence
confidence = 0.50  # Instead of 0.87
```

**Output**:
```jsonl
{"episode_id": "ep_12345", "enrichment": {"fragility_tags": [], "risk_tags": [], "affordance_tags": [], "efficiency_tags": [...], "novelty_tags": [...], "coherence_score": 0.70, "supervision_hints": {"priority_level": "medium", ...}, "confidence": 0.50, "validation_status": "passed"}}
```

**Impact**: Fewer tags, lower priority, but still processable.

### 4.2 Rejected Task Graph Proposal

**Scenario**: Task graph proposal failed validation.

**Input**:
```python
task_proposals = [
    TaskGraphRefinementProposal(
        proposal_id="task_prop_78",
        validation_status="failed",  # FAILED
        risk_annotations=[...]
    )
]
```

**Processing**:
```python
# Filter out failed proposals
valid_task_proposals = [p for p in task_proposals if p.validation_status == "passed"]
# valid_task_proposals = []  # Empty

# Step 3: Match Task Graph
risk_tags = []  # No task graph knowledge
efficiency_tags = []
```

**Output**: Same as 4.1 (graceful degradation).

### 4.3 Semantic Conflict Detected

**Scenario**: Tags contradict each other.

**Input Tags**:
```python
fragility_tags = [
    FragilityTag(object_name="vase", fragility_level="high", ...)
]

risk_tags = [
    RiskTag(risk_type="collision", severity="low", ...)  # CONFLICT: high fragility + low risk
]
```

**Coherence Check**:
```python
conflicts = [
    SemanticConflict(
        conflict_type="risk_fragility_mismatch",
        conflicting_tags=["fragility:vase", "risk:collision"],
        severity="medium",
        resolution_hint="High fragility should correlate with higher risk",
        justification="High-fragility object present but risk rated as low"
    )
]

coherence_score = 0.65  # Reduced from 1.0 due to conflict
```

**Output**:
```jsonl
{"episode_id": "ep_12345", "enrichment": {..., "semantic_conflicts": [{"conflict_type": "risk_fragility_mismatch", "severity": "medium", ...}], "coherence_score": 0.65, "supervision_hints": {"requires_human_review": true, ...}, ...}}
```

**Orchestrator Action**: Flag for human review due to conflict.

### 4.4 Missing Economics Data (Hard Failure)

**Scenario**: No economics output for episode.

**Input**:
```python
econ_outputs = {}  # Missing ep_12345
```

**Processing**:
```python
# Validation fails
errors = ["Missing economics data for episode ep_12345"]
validation_status = "failed"
```

**Output**:
```jsonl
{"episode_id": "ep_12345", "enrichment": null, "validation_status": "failed", "validation_errors": ["Missing economics data for episode ep_12345"]}
```

**Impact**: Episode excluded from training until economics data available.

---

## 5. Multi-Video Aggregation Example

**Scenario**: Three videos of "open_drawer" task with different objects.

**Inputs**:

**Video 1**: Empty drawer
```json
{"episode_id": "ep_001", "task": "open_drawer", "metadata": {"objects_present": ["drawer", "handle"]}}
```

**Video 2**: Drawer with vase
```json
{"episode_id": "ep_002", "task": "open_drawer", "metadata": {"objects_present": ["drawer", "handle", "vase_inside"]}}
```

**Video 3**: Drawer with books
```json
{"episode_id": "ep_003", "task": "open_drawer", "metadata": {"objects_present": ["drawer", "handle", "books_inside"]}}
```

**Individual Enrichments**:

**Video 1**:
```python
enrichment_1 = SemanticEnrichmentProposal(
    affordance_tags=[graspable:handle, pullable:drawer],
    fragility_tags=[],
    novelty_tags=[NoveltyTag(score=0.2)],  # Low novelty (common scenario)
    coherence_score=1.0
)
```

**Video 2**:
```python
enrichment_2 = SemanticEnrichmentProposal(
    affordance_tags=[graspable:handle, pullable:drawer],
    fragility_tags=[FragilityTag(object="vase", level="high")],
    novelty_tags=[NoveltyTag(score=0.73)],  # High novelty (edge case)
    coherence_score=0.95
)
```

**Video 3**:
```python
enrichment_3 = SemanticEnrichmentProposal(
    affordance_tags=[graspable:handle, pullable:drawer],
    fragility_tags=[],  # Books not fragile
    novelty_tags=[NoveltyTag(score=0.45)],  # Medium novelty
    coherence_score=0.98
)
```

**Aggregated Enrichment** (for task "open_drawer"):
```python
aggregated = aggregate_tags_across_videos([enrichment_1, enrichment_2, enrichment_3])

affordance_tags = [
    graspable:handle (demonstrated in 3/3 videos),
    pullable:drawer (demonstrated in 3/3 videos)
]

fragility_tags = [
    FragilityTag(object="vase", level="high", demonstrated_in="ep_002")
]

novelty_scores = [0.2, 0.73, 0.45]
avg_novelty = 0.46  # Medium overall

coherence_score = 0.98  # Min of individual scores (conservative)

cross_video_conflicts = []  # No conflicts (all consistent)
```

**Insights**:
- Core affordances (graspable, pullable) consistent across all videos
- Fragility varies by object (vase high, books low)
- Novelty peaks with edge case (vase scenario)

---

## 6. Tag Flow Decision Tree

```
For each datapack episode:

1. Load episode metadata
   ├─ Task: what action is being performed?
   ├─ Objects: what is present in the scene?
   └─ Success: did the episode succeed?

2. Match Ontology Proposals
   │
   ├─ Are objects in ontology?
   │  ├─ YES → Extract affordances
   │  │        → Extract fragilities
   │  └─ NO  → Skip affordance/fragility tags
   │
   └─ Confidence adjustment

3. Match Task Graph Proposals
   │
   ├─ Is task in task graph?
   │  ├─ YES → Extract risk annotations
   │  │        → Extract efficiency hints
   │  └─ NO  → Skip risk/efficiency tags
   │
   └─ Confidence adjustment

4. Match Economics Outputs
   │
   ├─ Is episode in economics data?
   │  ├─ YES → Extract novelty score
   │  │        → Extract expected MPL gain
   │  │        → Assign tier
   │  └─ NO  → FAIL (economics required)
   │
   └─ Confidence adjustment

5. Detect Interventions
   │
   ├─ Check metadata for "human_intervention" flag
   ├─ Check for failure→recovery patterns in rewards
   │  ├─ YES → Generate InterventionTag
   │  └─ NO  → Skip intervention tags

6. Check Coherence
   │
   ├─ Compare all tags for contradictions
   │  ├─ High fragility + Low risk? → Conflict
   │  ├─ Safety-critical + No risks? → Conflict
   │  └─ ...
   │
   ├─ Compute coherence score (0.0-1.0)
   └─ Generate SemanticConflict entries

7. Generate Supervision Hints
   │
   ├─ Combine signals:
   │  ├─ High novelty → high priority
   │  ├─ Safety-critical → human review
   │  ├─ High coherence → standard training
   │  ├─ Conflicts → investigate
   │  └─ ...
   │
   ├─ Assign curriculum stage (early/mid/late/advanced)
   ├─ Compute weight multiplier (1.0-3.0)
   └─ Set prerequisites

8. Validate
   │
   ├─ Schema compliance? (required fields, types)
   ├─ Forbidden fields? (rewards, mpl_value, etc.)
   ├─ JSON-safe? (all types serializable)
   ├─ Cross-consistent? (tags align with sources)
   │
   ├─ ALL PASS → validation_status = "passed"
   └─ ANY FAIL → validation_status = "failed"

9. Output
   │
   ├─ Create SemanticEnrichmentProposal
   ├─ Convert to JSONL format
   └─ Append to enrichments.jsonl
```

---

## 7. Stage 2 to Stage 3 Handoff

```
STAGE 2 OUTPUT
══════════════
enrichments.jsonl (one line per episode)

STAGE 3 INPUT
═════════════
1. Load datapacks.jsonl (original Stage 1 data)
2. Load enrichments.jsonl (Stage 2.4 output)

MERGE PROCESS
═════════════
for each episode in datapacks:
    enrichment = find_enrichment(episode.id)
    if enrichment:
        episode['enrichment'] = enrichment
    else:
        log_warning(f"No enrichment for {episode.id}")
        episode['enrichment'] = None

Write merged_datapacks.jsonl

ORCHESTRATOR CONSUMPTION
════════════════════════
for each episode in merged_datapacks:
    if episode.enrichment:
        # Apply supervision hints
        weight = episode.enrichment.supervision_hints.suggested_weight_multiplier
        priority = episode.enrichment.supervision_hints.priority_level
        curriculum_stage = episode.enrichment.supervision_hints.curriculum_stage

        # Filter safety-critical
        if episode.enrichment.supervision_hints.safety_critical:
            add_to_safety_review_queue(episode)

        # Filter conflicts
        if episode.enrichment.semantic_conflicts:
            add_to_investigation_queue(episode)

        # Schedule based on curriculum
        if curriculum_stage == "advanced":
            check_prerequisites(episode.enrichment.supervision_hints.prerequisite_tags)

        # Weight for training
        episode.sampling_weight = base_weight * weight
    else:
        # Default behavior (no enrichment)
        episode.sampling_weight = base_weight

Train policy with weighted sampling
```

---

**End of Pipeline Diagrams**
