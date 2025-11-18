# Stage 2.3 Pipeline Architecture

**Visual Reference for Stage 2.3 TaskGraphRefiner**

---

## Stage 2.3 Architecture (Detailed)

```
┌────────────────────────────────────────────────────────────────────────┐
│                  STAGE 2.3: TASK GRAPH REFINER                         │
│              (Advisory-only, no DAG mutations)                         │
└────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  UPSTREAM        │
                              │  INPUTS          │
                              └────────┬─────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐   ┌──────────────────┐
    │ Stage 2.2        │    │ Stage 2.1        │   │ Constraints      │
    │ Ontology         │    │ Semantic         │   │ • EconSignals    │
    │ UpdateProposals  │    │ Primitives       │   │ • DatapackSignals│
    │                  │    │                  │   │ • TaskGraph      │
    │ Types:           │    │ Fields:          │   │ • Ontology       │
    │ • ADD_SKILL_GATE │    │ • tags           │   └──────────────────┘
    │ • INFER_FRAGILITY│    │ • risk_level     │
    │ • ADD_SAFETY_    │    │ • energy_intensity│
    │   CONSTRAINT     │    │ • success_rate   │
    │ • ADJUST_RISK    │    └──────────────────┘
    └────────┬─────────┘
             │
             └────────────────────────┐
                                      │
                                      ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  TaskGraphRefiner.generate_refinements()              ║
              ║                                                       ║
              ║  Processing Logic:                                    ║
              ║                                                       ║
              ║  1. Process OntologyUpdateProposals:                  ║
              ║     ADD_SKILL_GATE → INSERT_CHECKPOINT (mandatory)    ║
              ║     INFER_FRAGILITY → SPLIT_TASK (if high fragility) ║
              ║     ADD_SAFETY_CONSTRAINT → REORDER_TASKS (safety)    ║
              ║     ADJUST_RISK → INSERT_RECOVERY (if risk elevated)  ║
              ║                                                       ║
              ║  2. Process SemanticPrimitives (optional):            ║
              ║     Low energy → REORDER_TASKS (efficiency)          ║
              ║     High success → MERGE_TASKS (redundancy)          ║
              ║                                                       ║
              ║  3. Economic urgency-driven:                          ║
              ║     error_urgency > 0.6 → ADJUST_PRIORITY (safety)   ║
              ║     mpl_urgency > 0.5 → PARALLELIZE_TASKS (throughput)║
              ╚═══════════════════════════════════════════════════════╝
                                      │
                                      ▼
                     TaskGraphRefinementProposal[]
                                      │
                                      ▼
              ╔═══════════════════════════════════════════════════════╗
              ║  TaskGraphRefiner.validate_refinements()              ║
              ║                                                       ║
              ║  Validation Checks:                                   ║
              ║  ✓ Econ constraints (no reward math)                  ║
              ║  ✓ Datapack constraints (no data valuation)           ║
              ║  ✓ DAG topology (no cycles)                           ║
              ║  ✓ Node preservation (no deletions)                   ║
              ╚═══════════════════════════════════════════════════════╝
                                      │
                                      ▼
                     Valid TaskGraphRefinementProposal[]
                                      │
                                      ▼
              ┌──────────────────────────────────────────────┐
              │  Storage: results/stage2/task_graph_         │
              │           refinements/*.jsonl                │
              │                                              │
              │  Format:                                     │
              │  {"proposal_id": "tgr_000001_abc",           │
              │   "refinement_type": "insert_checkpoint",    │
              │   "proposed_changes": {...}}                 │
              └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  DOWNSTREAM CONSUMERS                        │
              │                                              │
              │  • SemanticOrchestratorV2                    │
              │    → apply_task_graph_refinements()          │
              │  • HRL Scheduler                             │
              │    → skill gating + checkpoint execution     │
              │  • VLA/Diffusion/RL                          │
              │    → task decomposition hints                │
              └──────────────────────────────────────────────┘
```

---

## Refinement Generation Flow (Detailed)

```
┌──────────────────────────────────────────────────────────────┐
│         REFINEMENT GENERATION PIPELINE                       │
└──────────────────────────────────────────────────────────────┘

OntologyUpdateProposal (ADD_SKILL_GATE)
     │
     │ proposal_type: ADD_SKILL_GATE
     │ target_skill_id: 2  # PULL skill
     │ proposed_changes: {
     │   "gated_skill_id": 2,
     │   "preconditions": ["fragility_check_passed"],
     │   "safety_threshold": 0.8
     │ }
     ▼

┌──────────────────────────────────────────────────────────────┐
│  1. _insert_checkpoint_from_gate()                           │
│                                                              │
│  Find task nodes with skill_id = 2 (PULL)                    │
│  For each:                                                   │
│    → INSERT_CHECKPOINT refinement                            │
│       proposed_changes: {                                    │
│         "checkpoint_task": {                                 │
│           "task_id": "checkpoint_pull_drawer",               │
│           "name": "Safety Check Before Pull Drawer",         │
│           "task_type": "checkpoint",                         │
│           "preconditions": ["fragility_check_passed"],       │
│           "postconditions": ["pull_drawer_gated_check_passed"]│
│         },                                                   │
│         "insert_before_task_id": "pull_drawer",              │
│         "mandatory": True                                    │
│       }                                                      │
│       priority: CRITICAL                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼

OntologyUpdateProposal (INFER_FRAGILITY)
     │
     │ proposal_type: INFER_FRAGILITY
     │ proposed_changes: {
     │   "inferred_fragility": 0.9  # High fragility
     │ }
     ▼

┌──────────────────────────────────────────────────────────────┐
│  2. _split_task_for_fragility()                              │
│                                                              │
│  IF fragility >= 0.7:                                        │
│    Find tasks with fragile interactions (pull/move/lift)    │
│    For each:                                                 │
│      → SPLIT_TASK refinement                                 │
│         proposed_changes: {                                  │
│           "original_task_id": "pull_drawer",                 │
│           "new_sub_tasks": [                                 │
│             {                                                │
│               "task_id": "pull_drawer_pre_check",            │
│               "name": "Pre-Check for Pull Drawer",           │
│               "task_type": "checkpoint",                     │
│             },                                               │
│             {                                                │
│               "task_id": "pull_drawer_slow",                 │
│               "name": "Pull Drawer (Slow Mode)",             │
│               "task_type": "skill",                          │
│               "metadata": {"speed_limit": 0.5}               │
│             },                                               │
│             {                                                │
│               "task_id": "pull_drawer_verify",               │
│               "name": "Verify Pull Drawer Success",          │
│               "task_type": "checkpoint",                     │
│             }                                                │
│           ],                                                 │
│           "preserve_original": False                         │
│         }                                                    │
│         priority: HIGH                                       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼

OntologyUpdateProposal (ADD_SAFETY_CONSTRAINT)
     │
     │ proposal_type: ADD_SAFETY_CONSTRAINT
     │ proposed_changes: {
     │   "constraint_type": "collision_avoidance",
     │   "applies_to_skills": [2, 5]  # PULL, MOVE
     │ }
     ▼

┌──────────────────────────────────────────────────────────────┐
│  3. _reorder_for_safety()                                    │
│                                                              │
│  Find tasks with affected skills (2, 5)                      │
│  Sort: checkpoints first, then skills                        │
│    → REORDER_TASKS refinement                                │
│       proposed_changes: {                                    │
│         "original_order": [                                  │
│           "pull_drawer",                                     │
│           "check_vase_position",                             │
│           "move_to_target"                                   │
│         ],                                                   │
│         "reordered_task_ids": [                              │
│           "check_vase_position",  # Checkpoint first         │
│           "pull_drawer",                                     │
│           "move_to_target"                                   │
│         ],                                                   │
│         "reason": "safety"                                   │
│       }                                                      │
│       priority: HIGH                                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼

Economic Urgency (error_urgency > 0.6)
     ▼

┌──────────────────────────────────────────────────────────────┐
│  4. _adjust_priority_for_safety()                            │
│                                                              │
│  Find all CHECKPOINT tasks                                   │
│  For each with priority != "high":                           │
│    → ADJUST_PRIORITY refinement                              │
│       proposed_changes: {                                    │
│         "task_id": "check_vase_position",                    │
│         "old_priority": "medium",                            │
│         "new_priority": "high",                              │
│         "reason": "error_urgency_high",                      │
│         "trigger": {"econ_signal": "error_urgency", "value": 0.6}│
│       }                                                      │
│       priority: HIGH                                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                     TaskGraphRefinementProposal[]
                     (4-8 refinements from 3 ontology proposals)
```

---

## Validation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│               REFINEMENT VALIDATION FLOW                     │
└──────────────────────────────────────────────────────────────┘

TaskGraphRefinementProposal
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  validate_refinements()                                      │
│                                                              │
│  For each refinement:                                        │
│    ├─ _check_econ_constraints(refinement)                   │
│    │    └─ Reject if contains:                              │
│    │       • price_per_unit                                 │
│    │       • damage_cost                                    │
│    │       • alpha/beta/gamma                               │
│    │                                                         │
│    ├─ _check_datapack_constraints(refinement)               │
│    │    └─ Reject if contains:                              │
│    │       • tier                                           │
│    │       • novelty_score                                  │
│    │                                                         │
│    ├─ _check_dag_topology(refinement)                       │
│    │    └─ Reject if creates cycles                         │
│    │       (Simplified check; production needs full sort)   │
│    │                                                         │
│    └─ _check_preserves_nodes(refinement)                    │
│         └─ Reject if:                                       │
│            • "delete_task" in proposed_changes              │
│            • AND type NOT in {SPLIT_TASK, MERGE_TASKS}      │
│                                                              │
│  Output: List[TaskGraphRefinementProposal] (only valid ones)│
└──────────────────────────────────────────────────────────────┘
     │
     ▼
Valid Refinements Only
(respects_econ_constraints = True)
(respects_datapack_constraints = True)
(respects_dag_topology = True)
(preserves_existing_nodes = True)
```

---

## Refinement Type Mappings

```
┌──────────────────────────────────────────────────────────────┐
│     ONTOLOGY PROPOSAL → TASK GRAPH REFINEMENT MAPPING        │
└──────────────────────────────────────────────────────────────┘

OntologyProposalType          →  RefinementType
────────────────────────────────────────────────────────────────
ADD_SKILL_GATE                →  INSERT_CHECKPOINT (mandatory)
                                 • Inserts safety checkpoint before gated skill
                                 • priority: CRITICAL
                                 • mandatory: True

INFER_FRAGILITY (high)        →  SPLIT_TASK
                                 • Splits task into: check → slow → verify
                                 • priority: HIGH
                                 • preserve_original: False

ADD_SAFETY_CONSTRAINT         →  REORDER_TASKS
                                 • Reorders: checkpoints first
                                 • reason: "safety"
                                 • priority: HIGH

ADJUST_RISK (elevated)        →  INSERT_RECOVERY
                                 • Adds recovery task after risky operation
                                 • conditional: True
                                 • priority: MEDIUM

ADD_ENERGY_HEURISTIC          →  REORDER_TASKS
                                 • Reorders: low-energy tasks first
                                 • reason: "energy_efficiency"
                                 • priority: MEDIUM


SEMANTIC PRIMITIVE → REFINEMENT MAPPING
────────────────────────────────────────────────────────────────
Low energy_intensity          →  REORDER_TASKS (efficiency)
High success_rate (>0.95)     →  MERGE_TASKS (redundancy)


ECONOMIC URGENCY → REFINEMENT MAPPING
────────────────────────────────────────────────────────────────
error_urgency > 0.6           →  ADJUST_PRIORITY (checkpoints)
mpl_urgency > 0.5             →  PARALLELIZE_TASKS (independent tasks)
energy_urgency > 0.3          →  REORDER_TASKS (low-energy first)
```

---

## End-to-End Example

```
Input: Stage 2.2 OntologyUpdateProposal
{
  "proposal_id": "ont_prop_001",
  "proposal_type": "add_skill_gate",
  "target_skill_id": 2,  # PULL skill
  "proposed_changes": {
    "gated_skill_id": 2,
    "preconditions": ["fragility_check_passed"],
    "safety_threshold": 0.8
  }
}
         │
         ▼ [Stage 2.3]
TaskGraphRefiner.generate_refinements([ont_prop_001])
         │
         ├─> Searches task_graph for nodes with skill_id=2
         │   → Finds: "pull_drawer" task
         │
         ├─> Calls _insert_checkpoint_from_gate(ont_prop_001)
         │
         └─> Generates:
             TaskGraphRefinementProposal {
               proposal_id: "tgr_000001_abc123",
               refinement_type: INSERT_CHECKPOINT,
               priority: CRITICAL,
               target_task_ids: ["pull_drawer"],
               proposed_changes: {
                 "checkpoint_task": {
                   "task_id": "checkpoint_pull_drawer",
                   "name": "Safety Check Before Pull Drawer",
                   "task_type": "checkpoint",
                   "preconditions": ["fragility_check_passed"],
                   "postconditions": ["pull_drawer_gated_check_passed"],
                   "metadata": {
                     "check_type": "skill_gate",
                     "safety_threshold": 0.8
                   }
                 },
                 "insert_before_task_id": "pull_drawer",
                 "mandatory": True
               },
               rationale: "Skill gate requires safety checkpoint before Pull Drawer"
             }
         │
         ▼
validate_refinements([refinement])
         │
         ├─> ✅ Econ constraints OK (no reward params)
         ├─> ✅ Datapack constraints OK (no tier/novelty)
         ├─> ✅ DAG topology OK (no cycles)
         └─> ✅ Node preservation OK (no deletions)
         │
         ▼ [Storage]
results/stage2/task_graph_refinements/run_001_refinements.jsonl
(1 line)
         │
         ▼ [Stage 2.4+ - NEXT]
SemanticOrchestratorV2.apply_task_graph_refinements([refinement])
         │
         ├─> Inserts checkpoint task before "pull_drawer"
         ├─> Updates task_graph with new node
         └─> Updates preconditions: "pull_drawer" now requires "pull_drawer_gated_check_passed"
         │
         ▼
Updated TaskGraph
(ready for HRL/VLA/RL consumption)
```

---

## Task Splitting Example

```
Original Task:
{
  "task_id": "pull_drawer",
  "name": "Pull Drawer",
  "task_type": "skill",
  "skill_id": 2,
  "preconditions": ["grasp_handle"],
  "postconditions": ["drawer_open"]
}

OntologyUpdateProposal (INFER_FRAGILITY):
{
  "proposed_changes": {
    "inferred_fragility": 0.9  # High fragility
  }
}
         │
         ▼ [Stage 2.3]
TaskGraphRefiner._split_task_for_fragility(ont_prop)
         │
         └─> Generates:
             TaskGraphRefinementProposal {
               refinement_type: SPLIT_TASK,
               proposed_changes: {
                 "original_task_id": "pull_drawer",
                 "new_sub_tasks": [
                   {
                     "task_id": "pull_drawer_pre_check",
                     "name": "Pre-Check for Pull Drawer",
                     "task_type": "checkpoint",
                     "preconditions": ["grasp_handle"],
                     "postconditions": ["pull_drawer_safe_to_proceed"]
                   },
                   {
                     "task_id": "pull_drawer_slow",
                     "name": "Pull Drawer (Slow Mode)",
                     "task_type": "skill",
                     "skill_id": 2,
                     "preconditions": ["pull_drawer_safe_to_proceed"],
                     "postconditions": ["drawer_open"],
                     "metadata": {"speed_limit": 0.5, "force_limit": 0.7}
                   },
                   {
                     "task_id": "pull_drawer_verify",
                     "name": "Verify Pull Drawer Success",
                     "task_type": "checkpoint",
                     "preconditions": ["drawer_open"],
                     "postconditions": ["pull_drawer_verified"]
                   }
                 ],
                 "preserve_original": False
               }
             }
         │
         ▼ [Application - NEXT STAGE]
Original task replaced by 3 sub-tasks:
1. Pre-check (checkpoint)
2. Slow execution (skill with speed/force limits)
3. Post-verify (checkpoint)

Result: Safer execution for fragile object environments
```

---

**End of Stage 2.3 Pipeline Diagrams**
