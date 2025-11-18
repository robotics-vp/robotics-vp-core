```markdown
# Stage 2.3 – TaskGraphRefiner Verification Checklist

This checklist is for **design + implementation review** of Stage 2.3.

## Pre-Implementation Checks

- [ ] Task graph schema is documented (node IDs, edge types, statuses, metadata).
- [ ] Proposal types are enumerated and frozen (no ad-hoc strings).
- [ ] A clear list of forbidden modifications is agreed:
  - No direct DAG mutation (no adding/removing nodes or edges).
  - No econ changes (prices, wages, tiers, data premiums, rebates).
  - No reward changes (objective vectors, weights, Phase B behavior).
- [ ] JSON-safe helpers are available for proposal payloads.
- [ ] There is a dedicated smoke test file (e.g. `scripts/smoke_test_task_graph_refiner.py`).

## Post-Implementation Checks

### Determinism

- [ ] Same `{task_graph, ontology_snapshot, primitives, summaries}` inputs:
  - Produce the same number of proposals.
  - Produce proposals in the same order.
  - Produce identical proposal payloads.
- [ ] A deterministic random seed is used wherever sampling is involved (or no sampling is used).

### Constraint Compliance

- [ ] No proposal payload contains econ fields like:
  - `price_per_unit`, `wage`, `rebate_pct`, `data_premium`, `tier`, `w_econ`.
- [ ] No proposal references reward internals:
  - No objective vectors or reward weights.
  - No references to Phase B or RewardBuilder internals.
- [ ] Every `target_node_id` refers to an existing node in the provided graph.
- [ ] No proposal attempts to:
  - Add or remove edges.
  - Delete nodes.
  - Invent completely new skills that don’t exist in the ontology / HRL library.

### JSON Safety

- [ ] All proposals round-trip through `json.dumps` / `json.loads` without errors.
- [ ] No NaNs, infinities, or non-serializable objects appear in payloads.
- [ ] A dedicated smoke test asserts JSON round-trip for a representative proposal set.

### Smoke Tests

- [ ] `scripts/smoke_test_task_graph_refiner.py` covers at least:
  - Simple graph with a fragile object → emits `INSERT_SAFETY_CHECKPOINT`.
  - Composite task with multiple subtasks → emits `SPLIT_TASK` / ordering proposals.
  - Sibling tasks with reorderable dependencies → emits `REORDER_FOR_EFFICIENCY`.
  - Path that includes error-prone steps → emits `ADD_FALLBACK_BRANCH` or `TIGHTEN_PRECONDITIONS`.
  - Determinism: repeated calls with identical inputs yield identical proposals.
  - Constraint compliance: intentionally malformed inputs do not produce illegal proposals.
- [ ] `run_all_smokes.py` includes and executes the refiner smoke test.

## Final Sign-Off

- [ ] All refiner-related smokes pass.
- [ ] No new imports from reward / econ modules inside the refiner core.
- [ ] Code review confirms:
  - The module is advisory-only and side-effect free.
  - All contracts with Stage 2.1, 2.2, and downstream consumers are respected.
- [ ] Documentation for Stage 2.3 (spec, summary, diagrams, checklist) is up to date.
```

