# Phase 1 Sim / Synth / Physics WM: Closure Standard

## Purpose

This document defines the exact decision criteria for when Phase 1 is structurally closed, when it is still internally incomplete, and when the implementation center of gravity should begin shifting toward the Perception / Grounding WM tranche.

This is an operational complement to the Phase Exit Rule in `multi_wm_architecture_plan.md`. That rule says "advance only when blockers are honestly external." This document defines what "external" and "internal" mean concretely for Phase 1 after the Tier 2 runtime-ladder deepening.

## The Three Categories

Every finding from the Tier 1/3 verification pass falls into exactly one of these categories:

### Category A: Internal Structural Incompleteness (still Phase 1 priority)

A finding is internal incompleteness if the gap is addressable by Codex writing code in this repo without requiring external runtime, assets, GPU, data, or provider availability.

**Concrete examples that would keep Phase 1 open:**

1. **Compiler does not assemble a documented state object.** If `compile_sim_synth_physics_world_state()` silently skips physics adaptation policy, or gen2sim admission, or runtime bridge compilation — that is internal wiring, not a provider gap.

2. **A receipt that should exist does not exist.** If `gen2sim_admission.py` returns state but never emits a typed receipt that downstream consumers (replay, training corpus, economic WM) can ingest — that is a missing receipt, not missing data.

3. **The compiler does not reference Tier 2 runtime-binding truth.** If `BackendExecutionBindingState` still carries a flat "unavailable" flag without reflecting whether the deeper ladder (pack → binding → request → consumer → execution → realization) is contract-shaped or concretely realizable — that is internal. The compiler should propagate binding depth, not collapse it.

4. **Promotion has no demotion path.** If `promotion.py` can promote a helper to `promoted` via benchmark gate but has no path back to `shadow_candidate` when evidence degrades — that is implementable.

5. **Shadow execution does not thread through the adapter ladder.** If `shadow_execution.py` jumps from `PhysicsExecutionContract` directly to shadow materialization without consuming the deeper adapter ladder (runtime binding, adapter execution, adapter realization) — that is internal inconsistency between Tier 2 and Tier 3.

6. **Render providers emit state but no receipts.** If `compile_branch_render_provider_state()` produces `BranchRenderProviderState` but there is no corresponding receipt for the receipt chain — and receipt-chain completeness requires it — that is internal.

7. **Training corpus extracts binding metadata but not calibration/adaptation receipts.** If the training corpus threads runtime-binding status but not calibration quality, adaptation readiness, or shadow execution outcomes — that is a partial wiring job, not an external blocker.

8. **Branch planner has no honest fallback receipt.** If the branch planner falls back from learned to heuristic without emitting a receipt that says "fell back because benchmark gate not ready" — that is missing honesty wiring.

9. **Work-order status does not react to `binding_blocked`.** If `runtime_work_orders.py` has a `blocked_by_runtime_binding` status but it is not correctly triggered — that is a logic gap.

10. **Tests do not verify compiler round-trip.** If there is no test that compiles a full `SimSynthPhysicsWorldState` and checks all sub-objects and receipt references — that is missing test coverage for existing code.

### Category B: Honestly Externalized Blockers (Phase 1 is structurally closed despite these)

A finding is honestly external if the gap requires something this repo cannot provide by itself.

**Concrete examples that are external blockers:**

1. **No real Isaac Lab / Isaac Sim runtime on any host.** The ladder correctly names this gap. The WM can route, bind, build adapter requests, and report `binding_blocked` or `realization_blocked`. The blocker is an external install + GPU host, not missing wiring.

2. **No real Holosoma host runtime, motion corpora, or retargeting assets.** Same — the ladder is ready, the provider is not.

3. **No trained policy checkpoints for any backend.** Runtime binding correctly selects `selected_policy_ref` from candidates. There are no candidates yet. That is a training/data gap.

4. **No GPU-backed GGDS / LDM video materialization.** Render provider contracts exist. Concrete execution requires GPU and model weights.

5. **No real Unitree G1 sim assets (URDF, calibration, joint naming).** Asset contracts are typed and normalized. The assets themselves are external.

6. **Benchmark gates cannot be satisfied because no real execution has produced outcomes.** Promotion to `promoted` requires `benchmark_gate.ready`. No backend has produced outcomes to gate on. This is external.

7. **Outcome parsers exist as typed surfaces but have no real artifacts to parse.** The harvest rung of the ladder is structurally ready. No backend has produced harvestable artifacts yet.

8. **Calibration receipts score low because no evidence exists.** `build_physics_calibration_receipt()` computes quality scores. With no backend evidence, those scores are correctly low. The fix is evidence, not code.

### Category C: Ambiguous / Judgment Call

Some findings require honest judgment:

- **Randomization axes for humanoid-target hardware classes are not specified.** This is implementable (writing axis definitions is code), but the axis values depend on external knowledge of G1/R1 joint limits, contact surfaces, and operational envelopes. If the axes are genuinely placeholder because no one has the hardware knowledge yet, this is Category B. If the axes could be specified from Unitree documentation that already exists, this is Category A.

- **PyBullet adapter does not carry complete metadata.** PyBullet is the fallback, not the target. If the adapter honestly reports its limitations (no unitree claims, tabletop envelope only), minor metadata gaps in a fallback adapter should not block Phase 1 closure. If it falsely claims capabilities it doesn't have, that is Category A (anti-stub violation).

- **Inferential yield scoring does not react to backend fidelity.** If `benchmark_provenance_quality()` checks flags that are always false because no real backend has run — the interface is correct but untestable. This is Category B. If the interface doesn't even check fidelity-relevant flags, that is Category A.

## Phase 1 Closure Decision Rule

**Phase 1 is structurally closed when:**

1. The Tier 1/3 verification pass produces **zero Category A findings**, AND
2. Every Category C finding has been explicitly classified as A or B with a written rationale, AND
3. All Category A findings discovered during verification have been resolved

**Phase 1 is NOT structurally closed when:**

- Any Category A finding remains unresolved
- Any finding has not been classified (the absence of a judgment is itself a gap)
- The compiler does not assemble all documented state objects
- The receipt chain has internal breaks (state exists but no receipt, or receipt exists but nothing consumes it)
- The Tier 2 runtime ladder depth is not reflected in the compiler and downstream consumers

## When to Begin Perception WM Preparation vs. Continued Phase 1 Priority

### Continue Phase 1 priority if:

- The Tier 1/3 verification pass reveals 3+ Category A findings
- Any Category A finding affects the compiler (1.1), the receipt chain (1.3, 1.4), or promotion machinery (3.2) — these are structurally load-bearing
- Shadow execution does not thread through the adapter ladder and the gap requires nontrivial wiring (not just a parameter pass-through)

### Begin parallel Perception WM preparation if:

- The Tier 1/3 verification pass reveals ≤2 Category A findings AND those findings are isolated (e.g., a missing receipt for one subsystem, a missing test)
- All compiler assembly, receipt chain, and promotion items are Category B
- The remaining Category A items can be closed in a single focused Codex tranche without architectural changes

"Parallel preparation" means:
- Activate `codex_tranche_perception_wm_schema.md` for Codex schema/scaffolding work
- Keep Phase 1 Category A closure as the HIGHER priority
- Do not merge Perception WM schema work until Phase 1 Category A items are resolved
- Claude may draft Perception WM doctrine and review Perception schema in parallel

### Shift implementation center of gravity to Perception WM if:

- Phase 1 has zero Category A findings
- All Category C findings are classified
- The `claude_to_comment_on.md` handoff explicitly declares Phase 1 structurally closed with the Category B remainder list
- Claude reviews and confirms

This is the Phase Exit Rule applied concretely.

## Evidence Format

The Tier 1/3 verification `claude_to_comment_on.md` must include:

```
## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| [description] | A / B / C→A / C→B | [why] |

Category A count: N
Category B count: N
Category C unresolved: N

Closure recommendation: [closed / not closed / parallel prep allowed]
```
