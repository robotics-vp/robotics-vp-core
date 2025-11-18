# Stage 2.3 – TaskGraphRefiner Summary

## What TaskGraphRefiner Is

TaskGraphRefiner is an **advisory-only** module that looks at:

- The current task graph (DAG of skills / subtasks)
- Ontology state + SIMA-2 semantic primitives (Stage 2.1 / 2.2 outputs)
- Econ / datapack-derived signals (urgency, fragility, safety, energy)

and produces **TaskGraphRefinementProposal** objects that *suggest* refinements to the task graph.  
It never mutates the task DAG itself and never touches economics, reward weights, or datapack valuation.

## What It Can Propose

Examples of proposal types (final list is in the spec file):

- `SPLIT_TASK` – suggest splitting a composite node into ordered subtasks
- `INSERT_SAFETY_CHECKPOINT` – suggest pre- or post-conditions around risky steps
- `REORDER_FOR_EFFICIENCY` – suggest reordering sibling nodes when dependencies allow it
- `ADD_FALLBACK_BRANCH` – suggest an error-recovery or retry branch reusing existing skills
- `TIGHTEN_PRECONDITIONS` – suggest stricter preconditions when logs / primitives show failures
- `TAG_CRITICAL_PATH` – suggest marking nodes that sit on a fragile / safety-critical path

All outputs are **advisory, JSON-safe proposals** that other modules can inspect and decide whether to apply.

## Hard Boundaries (WILL NOT DO)

TaskGraphRefiner:

- ❌ Does **not** directly mutate the task DAG (no adding/removing nodes or edges)
- ❌ Does **not** set or modify economics (prices, wages, data premiums, tiers, rebates)
- ❌ Does **not** modify reward functions, objective vectors, or Phase B math
- ❌ Does **not** invent new skills or primitives
- ✅ Only emits `TaskGraphRefinementProposal` objects for downstream consumers

Econ, datapacks, and reward builder are **older siblings**; TaskGraphRefiner is a **younger sibling** that consumes their signals but never writes back to them.

## Inputs

At minimum:

- Current task graph snapshot (nodes, edges, statuses, metadata)
- Ontology state and outstanding ontology update proposals (from Stage 2.2)
- SIMA-2 semantic primitives and risk inferences (from Stage 2.1)

Optionally (for richer proposals):

- Episode descriptors derived from datapacks (energy, safety, novelty, tier)
- Econ/semantic summaries (e.g., high urgency, fragile objects, safety incidents)

## Outputs

A list of `TaskGraphRefinementProposal` objects, each including:

- `proposal_id` – stable identifier
- `proposal_type` – e.g. `SPLIT_TASK`, `INSERT_SAFETY_CHECKPOINT`, etc.
- `target_node_id` – which node the suggestion applies to
- `priority` – `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`
- `justification` – human-readable rationale (from primitives / logs / ontology)
- `payload` – JSON-safe structured details (e.g., suggested subtask ordering)

The module is **deterministic**: same inputs → same proposal list (order included).

## Contract

- Side-effect free: no graph, econ, reward, or datapack mutation
- Deterministic: proposals are stable given the same inputs
- JSON-safe: everything must serialize cleanly to JSON
- Schema-driven: all proposals conform to a small set of well-defined types

Downstream consumers (e.g. `SemanticOrchestratorV2`, UI tooling, offline analysis) decide:

- Which proposals to accept
- How to apply them to actual task graphs
- How to surface them to human operators
