# Stage 2 Pipeline Architecture

**Visual Reference for Stage 2 Semantic Layer**

---

## Full Stage 2 Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: SEMANTIC LAYER                           │
│                    (Advisory-only, no reward/RL mutation)                 │
└───────────────────────────────────────────────────────────────────────────┘

                                  ┌─────────────────┐
                                  │  UPSTREAM       │
                                  │  CONSTRAINTS    │
                                  └────────┬────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
              ▼                            ▼                            ▼
    ┌──────────────────┐         ┌─────────────────┐        ┌──────────────────┐
    │ EconomicController│         │ DatapackEngine  │        │   TaskGraph      │
    │  (econ physics)  │         │ (data physics)  │        │  (task DAG)      │
    │                  │         │                 │        │                  │
    │ • EconSignals    │         │ • DatapackSignals│       │ • TaskNode[]     │
    │ • error_urgency  │         │ • tier fractions│        │ • affordances    │
    │ • energy_urgency │         │ • coverage score│        │ • preconditions  │
    └────────┬─────────┘         └────────┬────────┘        └────────┬─────────┘
             │                            │                          │
             │                            │                          │
             └────────────────────────────┼──────────────────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   SIMA-2 Rollout Generator    │
                          │   (stubbed for testing)       │
                          │                               │
                          │ Output: {                     │
                          │   "task_type": "open_drawer"  │
                          │   "events": [...],            │
                          │   "metrics": {...}            │
                          │ }                             │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  STAGE 2.1: SemanticPrimitiveExtractor                ║
              ║  Status: ✅ COMPLETE                                  ║
              ║                                                       ║
              ║  Input:  SIMA-2 rollout dict                          ║
              ║  Output: SemanticPrimitive[]                          ║
              ║                                                       ║
              ║  @dataclass SemanticPrimitive:                        ║
              ║    primitive_id: str                                  ║
              ║    task_type: str                                     ║
              ║    tags: List[str]                                    ║
              ║    risk_level: str  # "low", "medium", "high"         ║
              ║    energy_intensity: float                            ║
              ║    success_rate: float                                ║
              ║    avg_steps: float                                   ║
              ║    source: str  # "sima2"                             ║
              ╚═══════════════════════════════════════════════════════╝
                                          │
                                          ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  STAGE 2.2: OntologyUpdateEngine                      ║
              ║  Status: 🔄 DESIGN COMPLETE → READY FOR CODEX         ║
              ║                                                       ║
              ║  Input:  SemanticPrimitive[]                          ║
              ║  Output: OntologyUpdateProposal[]                     ║
              ║  Storage: results/stage2/ontology_proposals/*.jsonl   ║
              ║                                                       ║
              ║  @dataclass OntologyUpdateProposal:                   ║
              ║    proposal_id: str                                   ║
              ║    proposal_type: ProposalType  # 9 types             ║
              ║    priority: ProposalPriority   # CRITICAL/HIGH/MED/LOW║
              ║    proposed_changes: Dict[str, Any]                   ║
              ║    rationale: str                                     ║
              ║    confidence: float                                  ║
              ║    respects_econ_constraints: bool                    ║
              ║    respects_datapack_constraints: bool                ║
              ║    respects_task_graph: bool                          ║
              ║                                                       ║
              ║  Proposal Types:                                      ║
              ║    1. ADD_AFFORDANCE                                  ║
              ║    2. ADJUST_RISK                                     ║
              ║    3. INFER_FRAGILITY                                 ║
              ║    4. ADD_OBJECT_CATEGORY                             ║
              ║    5. ADD_SEMANTIC_TAG                                ║
              ║    6. ADD_SKILL_GATE                                  ║
              ║    7. ADD_SAFETY_CONSTRAINT                           ║
              ║    8. ADD_ENERGY_HEURISTIC                            ║
              ║    9. UPDATE_OBJECT_RELATIONSHIP                      ║
              ╚═══════════════════════════════════════════════════════╝
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   Proposal Validation         │
                          │                               │
                          │ ✓ Econ constraints OK?        │
                          │ ✓ Datapack constraints OK?    │
                          │ ✓ Task graph constraints OK?  │
                          │ ✓ JSON-safe?                  │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  STAGE 2.3: TaskGraphRefiner (NEXT)                   ║
              ║  Status: ⏸️  PENDING                                   ║
              ║                                                       ║
              ║  Input:  OntologyUpdateProposal[]                     ║
              ║  Output: TaskGraphUpdate[]                            ║
              ║                                                       ║
              ║  Operations:                                          ║
              ║    • Split tasks based on skill gates                 ║
              ║    • Insert checkpoint tasks for safety               ║
              ║    • Merge redundant task nodes                       ║
              ║    • Reorder tasks based on affordance discovery      ║
              ╚═══════════════════════════════════════════════════════╝
                                          │
                                          ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  STAGE 2.4: SemanticTagPropagator (NEXT)              ║
              ║  Status: ⏸️  PENDING                                   ║
              ║                                                       ║
              ║  Input:  OntologyUpdateProposal[]                     ║
              ║  Output: Unified semantic tags                        ║
              ║                                                       ║
              ║  Operations:                                          ║
              ║    • Unify tags across VLA/SIMA/diffusion/RL         ║
              ║    • Propagate safety tags to related skills          ║
              ║    • Update cross-module vocabularies                 ║
              ╚═══════════════════════════════════════════════════════╝
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  DOWNSTREAM CONSUMERS         │
                          └───────────────┬───────────────┘
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             │                            │                            │
             ▼                            ▼                            ▼
  ┌──────────────────┐        ┌──────────────────┐       ┌──────────────────┐
  │ SemanticOrchestrator│      │  SIMA-2 Bridge   │       │ VLA/Diffusion/RL │
  │       V2          │        │                  │       │                  │
  │                   │        │ • Filter rollouts│       │ • Affordance     │
  │ • Apply proposals │        │   by skill gates │       │   constraints    │
  │ • Conflict        │        │ • Primitive      │       │ • Fragility      │
  │   resolution      │        │   selection      │       │   awareness      │
  │ • Ontology        │        │                  │       │ • Energy         │
  │   mutation        │        │                  │       │   heuristics     │
  └───────────────────┘        └──────────────────┘       └──────────────────┘
```

---

## Constraint Flow (Causality Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSTRAINT HIERARCHY                         │
│                                                                 │
│  "Older Siblings"     →     "This Stage"     →  "Younger Siblings"│
│  (Constraint Sources)       (Stage 2.2)          (Consumers)    │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────┐
│ EconomicController│  CANNOT propose:
│                   │  • price_per_unit
│  Provides:        │  • damage_cost
│  • error_urgency  │  • wage_parity
│  • energy_urgency │  • alpha/beta/gamma
│  • damage_cost    │
└─────────┬─────────┘
          │
          │  CAN consume:
          │  • error_urgency → elevate risk
          │  • energy_urgency → energy heuristics
          │  • damage_cost_total → fragility thresholds
          │
          ▼
┌───────────────────┐
│  DatapackEngine   │  CANNOT propose:
│                   │  • tier classification
│  Provides:        │  • novelty_score
│  • tier fractions │  • data_premium
│  • coverage score │
│  • tag diversity  │
└─────────┬─────────┘
          │
          │  CAN consume:
          │  • tier2_fraction → frontier focus
          │  • coverage_score → new categories
          │  • tag_diversity → tag unification
          │
          ▼
┌───────────────────┐
│    TaskGraph      │  CANNOT propose:
│                   │  • task deletion
│  Provides:        │  • dependency changes
│  • affordances    │
│  • preconditions  │
│  • objects        │
└─────────┬─────────┘
          │
          │  CAN consume:
          │  • affordances → new affordance proposals
          │  • objects_involved → object relationships
          │  • semantic_priority → skill gating
          │
          ▼
┌─────────────────────────────────────────────┐
│     STAGE 2.2: OntologyUpdateEngine         │
│                                             │
│  RESPONSIBILITIES:                          │
│  ✅ Generate proposals (advisory-only)      │
│  ✅ Validate constraint compliance          │
│  ✅ JSON-safe output                        │
│  ✅ Deterministic proposal generation       │
│                                             │
│  FORBIDDEN:                                 │
│  ❌ Mutate ontology directly                │
│  ❌ Set econ parameters                     │
│  ❌ Set data valuation logic                │
│  ❌ Delete task nodes                       │
│  ❌ Modify reward math                      │
└─────────────────┬───────────────────────────┘
                  │
                  │  Outputs:
                  │  • OntologyUpdateProposal[]
                  │
                  ▼
┌─────────────────────────────────────────────┐
│        DOWNSTREAM CONSUMERS                 │
│                                             │
│  SemanticOrchestratorV2 (Stage 2.3+):       │
│  • apply_ontology_proposals()               │
│  • validate_proposals()                     │
│  • merge_proposals()                        │
│                                             │
│  TaskGraphRefiner (Stage 2.3):              │
│  • refine_task_graph()                      │
│  • insert_checkpoints()                     │
│  • split_tasks()                            │
│                                             │
│  SIMA-2 Bridge:                             │
│  • filter_rollouts_by_gates()               │
│  • select_primitives()                      │
│                                             │
│  VLA/Diffusion/RL:                          │
│  • receive_affordance_constraints()         │
│  • receive_fragility_awareness()            │
│  • receive_energy_heuristics()              │
└─────────────────────────────────────────────┘
```

---

## Proposal Generation Flow

```
┌──────────────────────────────────────────────────────────────┐
│              PROPOSAL GENERATION PIPELINE                    │
└──────────────────────────────────────────────────────────────┘

SemanticPrimitive                  OntologyUpdateEngine
     │                                      │
     │ primitive_id: "prim_001"             │
     │ tags: ["fragile", "vase", "lift"]    │
     │ risk_level: "high"                   │
     │ energy_intensity: 0.15               │
     │ success_rate: 0.85                   │
     ▼                                      ▼

┌──────────────────────────────────────────────────────────────┐
│  1. _propose_affordances()                                   │
│                                                              │
│  IF "lift" in tags:                                          │
│    → ADD_AFFORDANCE proposal                                 │
│       proposed_changes: {                                    │
│         "affordance_type": "liftable",                       │
│         "confidence": 0.85,                                  │
│         "energy_cost_estimate": 0.15,                        │
│         "risk_level": 0.9                                    │
│       }                                                      │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  2. _propose_risk_adjustments()                              │
│                                                              │
│  IF risk_level == "high" AND econ_signals.error_urgency > 0.5:│
│    → ADJUST_RISK proposal                                    │
│       proposed_changes: {                                    │
│         "old_risk_level": 0.9,                               │
│         "new_risk_level": 1.0,  # Capped at 1.0             │
│         "adjustment_factor": 1.5,                            │
│         "trigger": "error_urgency=0.6"                       │
│       }                                                      │
│       priority: CRITICAL                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  3. _propose_fragility_inference()                           │
│                                                              │
│  IF "fragile" in tags:                                       │
│    → INFER_FRAGILITY proposal                                │
│       proposed_changes: {                                    │
│         "inferred_fragility": 0.9,                           │
│         "evidence": ["fragile", "task_type=move_vase"],      │
│         "damage_cost_estimate": 50.0                         │
│       }                                                      │
│       priority: CRITICAL                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  4. _propose_skill_gates()                                   │
│                                                              │
│  IF "fragile" in tags:                                       │
│    → ADD_SKILL_GATE proposal (for PULL skill)                │
│       proposed_changes: {                                    │
│         "gated_skill_id": 2,                                 │
│         "preconditions": ["fragility_check_passed", ...],    │
│         "safety_threshold": 0.8,                             │
│         "fallback_skill_id": 0                               │
│       }                                                      │
│       priority: HIGH                                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  5. _propose_energy_heuristics()                             │
│                                                              │
│  IF energy_intensity < 0.5 AND energy_urgency > 0.3:         │
│    → ADD_ENERGY_HEURISTIC proposal                           │
│       proposed_changes: {                                    │
│         "heuristic_type": "prefer_efficient_path",           │
│         "energy_multiplier": 0.8,                            │
│         "conditions": ["short_reach", "energy_intensity<0.15"]│
│       }                                                      │
│       priority: MEDIUM                                       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  6. _propose_semantic_tags()                                 │
│                                                              │
│  IF ("fragile" in tags) AND ("vase" in tags):                │
│    → ADD_SEMANTIC_TAG proposal                               │
│       proposed_changes: {                                    │
│         "tag": "fragile_glassware",                          │
│         "propagate_to_subtasks": True                        │
│       }                                                      │
│       priority: LOW                                          │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                     OntologyUpdateProposal[]
                     (5-7 proposals from 1 primitive)
```

---

## Validation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│               PROPOSAL VALIDATION FLOW                       │
└──────────────────────────────────────────────────────────────┘

OntologyUpdateProposal
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  validate_proposals()                                        │
│                                                              │
│  For each proposal:                                          │
│    ├─ _check_econ_constraints(proposal)                     │
│    │    └─ Reject if contains:                              │
│    │       • price_per_unit                                 │
│    │       • damage_cost                                    │
│    │       • wage_parity                                    │
│    │       • alpha/beta/gamma                               │
│    │                                                         │
│    ├─ _check_datapack_constraints(proposal)                 │
│    │    └─ Reject if contains:                              │
│    │       • tier                                           │
│    │       • novelty_score                                  │
│    │       • data_premium                                   │
│    │                                                         │
│    └─ _check_task_graph_constraints(proposal)               │
│         └─ Reject if contains:                              │
│            • delete_task                                    │
│            • modify_dependencies                            │
│                                                              │
│  Output: List[OntologyUpdateProposal] (only valid ones)     │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
Valid Proposals Only
(respects_econ_constraints = True)
(respects_datapack_constraints = True)
(respects_task_graph = True)
```

---

## Storage Format

```
results/stage2/ontology_proposals/
├── run_001_proposals.jsonl
├── run_002_proposals.jsonl
└── run_003_proposals.jsonl

Format (JSONL - one proposal per line):
{
  "proposal_id": "prop_000001_abc123",
  "proposal_type": "add_affordance",
  "priority": "medium",
  "source_primitive_id": "prim_001",
  "source": "sima2",
  "target_affordance_type": "liftable",
  "proposed_changes": {
    "affordance_type": "liftable",
    "confidence": 0.85,
    "energy_cost_estimate": 0.15,
    "risk_level": 0.9
  },
  "rationale": "Primitive 'prim_001' demonstrated 'lift' action",
  "confidence": 0.85,
  "respects_econ_constraints": true,
  "respects_datapack_constraints": true,
  "respects_task_graph": true,
  "tags": ["fragile", "vase", "lift"],
  "metadata": {}
}
```

---

## End-to-End Example

```
SIMA-2 Rollout
{
  "task_type": "move_fragile_vase",
  "events": [
    {"action": "lift", "object": "vase", "tags": ["vase", "fragile"]},
    {"action": "place", "object": "table", "tags": ["table", "place"]}
  ],
  "metrics": {"steps": 3, "success": True}
}
         │
         ▼ [Stage 2.1]
SemanticPrimitive {
  primitive_id: "prim_001",
  task_type: "move_fragile_vase",
  tags: ["vase", "fragile", "lift", "place"],
  risk_level: "high",
  energy_intensity: 0.15,
  success_rate: 0.85,
  avg_steps: 3.0
}
         │
         ▼ [Stage 2.2]
OntologyUpdateEngine.generate_proposals([prim_001])
         │
         ├─> ADD_AFFORDANCE: "liftable" (from "lift" tag)
         ├─> ADJUST_RISK: 0.9 → 1.0 (high risk + error urgency)
         ├─> INFER_FRAGILITY: 0.9 (from "fragile" tag)
         ├─> ADD_SKILL_GATE: gate skill_id=2 with safety check
         ├─> ADD_ENERGY_HEURISTIC: prefer efficient path
         └─> ADD_SEMANTIC_TAG: "fragile_glassware"
         │
         ▼
validate_proposals([...])
         │
         ├─> ✅ All proposals respect constraints
         └─> Output: 6 valid proposals
         │
         ▼ [Storage]
results/stage2/ontology_proposals/run_001_proposals.jsonl
(6 lines, one proposal per line)
         │
         ▼ [Stage 2.3 - NEXT]
SemanticOrchestratorV2.apply_ontology_proposals(proposals)
         │
         ├─> Apply INFER_FRAGILITY → update ontology.objects["vase_01"].fragility = 0.9
         ├─> Apply ADD_SKILL_GATE → insert checkpoint task before PULL skill
         └─> Apply ADD_SEMANTIC_TAG → propagate "fragile_glassware" tag
         │
         ▼
Updated Ontology + Task Graph
(ready for VLA/SIMA/RL consumption)
```

---

**End of Stage 2 Pipeline Diagrams**
