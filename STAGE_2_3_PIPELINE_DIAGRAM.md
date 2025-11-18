# Stage 2.3 – TaskGraphRefiner Pipeline Diagram

## High-Level Dataflow

```text
        SIMA-2 rollouts
                │
                ▼
  SemanticPrimitiveExtractor (Stage 2.1)
                │
                ▼
     OntologyUpdateEngine (Stage 2.2)
                │
                ▼
       Ontology + primitives

  Task graph snapshot ─────────────┐
  Episode descriptors / datapacks ─┼────► TaskGraphRefiner (Stage 2.3, advisory-only)
  Econ/semantic summaries ────────┘

        │
        ▼
  TaskGraphRefinementProposal[]
        │
        ▼
  SemanticOrchestratorV2 / tooling

Execution Flow (Conceptual)

Upstream context prepared

SIMA-2 → primitives (Stage 2.1)

Ontology proposals → accepted into ontology (Stage 2.2)

Task graph snapshot captured

Optional: episode descriptors + econ/semantic summaries

TaskGraphRefiner call

Inputs: {task_graph, ontology_snapshot, primitives, summaries}

Produces: List[TaskGraphRefinementProposal]

Downstream consumption

SemanticOrchestratorV2 ingests proposals

Operators / tools review high-priority proposals

Approved proposals are applied by a separate component that owns the DAG

Example Workflow (Fragile Drawer + Vase)

SIMA-2 + vision detect a fragile vase near a drawer.

Stage 2.2 marks the object as fragile in the ontology.

TaskGraphRefiner sees:

A composite "open drawer" task

Fragile object nearby

It proposes:

INSERT_SAFETY_CHECKPOINT before and after the "pull handle" step

TIGHTEN_PRECONDITIONS to require clearance around the vase

SemanticOrchestratorV2 receives the proposals and can:

Accept them, leading to an updated graph version

Defer or reject them based on policy or operator review

In all cases, TaskGraphRefiner only produces suggestions; it never mutates the task graph directly.


---
