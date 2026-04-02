# Claude Commentary Artifact

## Current Status

- **Date**: 2026-04-02
- **Branch**: `codex/multi-wm-architecture-plan`
- **Implementation center of gravity**: Phase 1 Sim / Synth / Physics WM closure
- **Active specs**:
  - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
  - `docs/economic_world_model/codex_tranche_tier1_tier3_verification.md`
  - `docs/economic_world_model/phase1_closure_standard.md`
  - `docs/economic_world_model/doctrine_runtime_ladder_reuse.md`

This file is the single current-state handoff. Historical tranche detail belongs in:
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/implementation_notes.md`

## Tranche Spec Coverage

| Area | Current state |
|------|---------------|
| Runtime layout/profile evidence | **materially improved** (prior tranche) |
| Upstream runtime-pack evidence | **materially improved** (prior tranche) |
| Selected-profile install/preflight truth | **closed on the audited path** (prior tranche) |
| Training/work-order preservation of install truth | **materially improved** (prior tranche) |
| Holosoma selected-profile false blocker | **fixed** (prior tranche) |
| Isaac partially discovered preferred-profile fallback | **fixed** (prior tranche) |
| **Promotion/demotion machinery (Tier 3.2)** | **Category A gap closed** (this tranche) |
| Render/provider lane | **unchanged** |

### Tier 3.2 detail

| Item | Status | Notes |
|------|--------|-------|
| `resolve_helper()` demotion path | **implemented** | Evidence-based demotion via `evidence_signals` param |
| `resolve_backend_selector_helper()` demotion | **implemented** | Both direct-loaded and package-loaded paths |
| `resolve_branch_planner_helper()` demotion | **implemented** | Both direct-loaded and package-loaded paths |
| Demotion triggers | **implemented** | `benchmark_gate_revoked`, `evidence_failure`, `recent_failure_rate > threshold` |
| Demoted weight assignment | **correct** | 0.25 (same as shadow_candidate) |
| Demotion reason tracing | **implemented** | `demotion_reason` field in status dict |
| Stale test expectation fix | **fixed** | `test_holosoma_binding_records_runtime_target_contract` now accepts `pack_partial` |
| Demotion tests | **7 new tests** | Covers all three resolvers, both trigger types, and no-demotion cases |

## Current Branch Truth

- Phase 1 Sim / Synth / Physics remains the active implementation center.
- The branch has not drifted upward into Perception / Grounding implementation.
- Promotion machinery now has a demotion path: `promoted` → `demoted_to_shadow` when evidence signals indicate benchmark gate revocation, evidence failure, or excessive failure rate.
- Demoted helpers get weight 0.25 (same as shadow_candidate), ensuring the compiler falls back to heuristic behavior without fully disabling the helper.
- The three downstream consumers that check `promotion_stage == "promoted"` (compiler.py:312, calibration.py:139, synthetic_branches.py:314) will correctly NOT treat a demoted helper as promoted.

## What Changed Topologically

- No new WM, ladder rung, or abstraction was introduced.
- The existing `disabled|auto|required` promotion posture now has a fourth internal state: `demoted_to_shadow`. This is not a new mode — it's a status within `auto`/`required` mode that reverses a previous promotion.
- `_check_demotion()` is a shared function used by all three resolvers.

## What Fake Readiness Was Removed

- A helper that was once promoted could previously stay promoted forever regardless of subsequent evidence. This is no longer possible.
- Demotion is triggered by explicit evidence signals, not by time or heuristic decay.

## What Was Not Changed

- `src/world_model/sim_synth_physics/render_providers.py` (Tier 3.1)
- `src/world_model/sim_synth_physics/branch_planner.py` (Tier 3.3)
- `src/world_model/sim_synth_physics/inferential.py` (Tier 3.4)
- `src/world_model/sim_synth_physics/randomization.py`, `calibration.py` (Tier 3.5)
- `src/world_model/sim_synth_physics/shadow_execution.py` (Tier 3.6)
- No Perception / Grounding implementation surfaces
- Frozen Phase B math and controller logic

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Promotion had no demotion path | A → **closed** | Implemented evidence-based demotion in all three resolvers |
| Stale test expectation (pack_ready vs pack_partial) | A → **closed** | Test updated to accept honest pack_partial from install-hardened code |
| Render providers emit state but no receipts (3.1) | unverified | Not audited this tranche |
| Branch planner fallback receipt honesty (3.3) | unverified | Not audited this tranche |
| Inferential yield reaction to fidelity (3.4) | unverified | Not audited this tranche |
| Randomization humanoid axes (3.5) | unverified | Not audited this tranche |
| Shadow execution adapter ladder threading (3.6) | unverified | Not audited this tranche |
| Gen2Sim admission receipt emission (1.3) | unverified | Not audited this tranche |

Category A count: 0 (on audited items)
Category B count: stable (unchanged from prior)
Unverified Tier 3 items: 5

Closure recommendation: **not yet closed** — 5 Tier 3 items remain unverified

## Recommendation to Claude

- **Phase 1 remains the active implementation center.**
- This tranche closed the highest-risk Category A Tier 3 item (promotion/demotion).
- The next highest-risk Tier 3 items to audit are:
  1. **3.6 Shadow execution adapter ladder threading** — likely Category A if shadow execution bypasses the Tier 2 ladder
  2. **3.1 Render provider receipt emission** — likely Category A if receipt chain requires it
  3. **3.3 Branch planner fallback receipt** — likely Category A if fallback is silent
- After auditing those three items, the remaining Tier 3 items (3.4, 3.5) are lower risk and may be Category B/C.
- Once all Tier 3 items are classified, the Phase 1 Closure Assessment table can be completed.

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`
