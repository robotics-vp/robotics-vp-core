# Advisory Purge and Promotion Plan

## Purpose

This document is the advisory counterpart to `docs/economic_world_model/heuristic_advisory_sidecar_inventory.md`.

The question here is narrower than "remove all advisory things." The real question is:

- which advisory surfaces should remain advisory
- which surfaces should graduate into canonical metadata, preconditions, work orders, or bounded authority
- how that posture should change as the repo moves from semantic/economic control-plane hardening toward lower canonical world models
- how to think about frozen Phase B math while successor layers take on more real work

## Executive Conclusion

The repo should narrow its advisory doctrine.

The old posture made sense when most of the stack was still scaffolding. It is now too blunt. After the recent wiring passes, many internal surfaces are no longer harmless previews. They already affect:

- replay selection
- curriculum weighting
- training admission
- simulation and synthetic branch ranking
- benchmark readiness
- promotion evidence

Those surfaces should not keep terminating as "advisory-only" by default.

The right doctrine now is:

1. keep external foundation-model outputs, speculative planners, and preview/report tools advisory
2. promote internal typed state and quality classes into non-advisory metadata
3. promote economically judged retrain / recollect / re-evaluate decisions into work orders
4. promote well-bounded internal selectors into bounded authority where they already affect runtime or training distribution
5. keep frozen Phase B math as the rollback anchor, but stop treating it as philosophically untouchable forever

The highest-leverage next tranche is:

- **epiplexity / inferential signal-yield / inferential work-order promotion**

That is the clearest remaining case where a valuable internal signal still behaves too much like an overlay instead of a canonical learnability class.

Status update after the recent lower-WM work:

- the inferential learnability promotion tranche has landed for replay, manifests, and work orders
- the sim / synth / physics WM now also uses inferential learnability inside agenda ranking, branch admission, and diffusion ordering
- sim/synth backend-selector and branch-planner helper lanes now have real trainer/runtime-package seams, so they should be thought of as bounded-authority WM helpers rather than advisory-shaped sidecars
- the next advisory cleanup focus should therefore move back up to the remaining live queue / curriculum / orchestration surfaces whose naming and receipts still understate their real authority

## Doctrine Update

### What should remain advisory

These surfaces should stay advisory because they are either external truth providers, previews, or intentionally non-sovereign planning layers:

- external teacher/VLA/foundation-model proposals
- preview scripts and report-only analyzers
- speculative phase shells that sit above lower canonical WMs not yet built
- ontology/task-graph proposal generators that are explicitly upstream of review or merge logic

### What should stop being "advisory-only"

These surfaces should be promoted when they are internal, typed, and already used to shape the production loop:

- learnability / epiplexity evidence
- grounding class, trace completeness, and calibration quality
- execution readiness and benchmark-gate truth
- adaptation and data-collection admission decisions
- replay weighting / queueing / curriculum policies that already change what gets trained
- bounded selection receipts that later layers need as causal training context

### Promotion taxonomy

| Class | Meaning | Typical examples | Runtime consequence |
| --- | --- | --- | --- |
| `remain_advisory` | External or preview-only output that must not become native truth | OpenVLA action proposals, preview scripts, ontology proposals | visible for context only |
| `canonical_metadata` | Non-authoritative but non-optional truth about quality, availability, or provenance | grounding class, trace completeness, teacher availability, epiplexity class | persisted into replay/datapacks/runtime packets |
| `precondition` | Explicit eligibility check for training, promotion, or execution | benchmark-ready, trace-complete, grounded-data-ready | can block or downgrade |
| `work_order` | Actionable instruction for retrain / recollect / review / rescore | adaptation admit, collect-more-data, GPU-grounding refresh | enters backlog / executor-facing artifact |
| `bounded_authority` | Controlled runtime or training influence with receipts and rollback path | queue weighting, sampler policy, selector helper, routing helper | changes ordering/selection under caps |
| `benchmark_gated_successor` | Candidate replacement layer that may eventually supersede a frozen baseline | successor reward or planner math, successor world-model head | only after repeated benchmark evidence |

## Ranked Gap Matrix

| Rank | Surface | File / path | Current behavior | Current consumers | Why it is a production problem | Recommended disposition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Epiplexity overlays and inferential signal-yield | `docs/epiplexity.md`, `src/economics/inferential_reward.py`, `src/economics/inferential_training_gate.py`, `src/orchestrator/shadow_advisory.py`, `src/rl/episode_sampling.py`, `src/orchestrator/queue_dispatch_policy.py`, `src/valuation/datapack_schema.py`, `src/valuation/datapack_repo.py` | Epiplexity is documented and stored mainly as an advisory overlay/summary. It already influences signal-yield scoring, replay weights, and adaptation budgeting, but it is not yet the canonical learnability class for training admission or promotion evidence. | shadow advisory, inferential training gate, episode sampler, queue dispatch policy, datapack repo, representation homeostasis | The stack already uses epiplexity to bias replay and budgeting, so calling it merely advisory understates its real influence. At the same time, because it stays overlay-shaped, missing learnability truth does not reliably block or downgrade downstream paths. | **Top tranche**: promote epiplexity into canonical learnability metadata plus explicit `missing` / `summary_only` / `receipt_backed` classes; use it in training admissibility, synthetic branch ranking, and promotion evidence; keep raw probe details as sidecars |
| 2 | Inferential training decisions and adaptation budgeting | `src/economics/inferential_training_gate.py`, `src/orchestrator/shadow_advisory.py`, `src/orchestrator/adaptation_budgeting.py`, `scripts/train_shadow_offline_rl.py`, `scripts/train_shadow_replay_policy.py`, `scripts/train_shadow_pricing_models.py`, `scripts/train_sac_with_ontology_logging.py` | The gate is deterministic and economically grounded, and recent work already emits work-order artifacts. But the decision still lives primarily inside shadow/advisory bundles instead of a single canonical executor-facing admission contract used everywhere. | shadow advisory, shadow/offline training scripts, queue metadata, backlog scans | Retraining and recollection decisions are too important to stay shell-shaped. If each consumer reinterprets the decision through summaries, the loop can drift on what was actually admitted or blocked. | Promote to a single canonical work-order / admission contract; make downstream trainers prefer the emitted work order over recomputation; benchmark-gate any later learned replacement |
| 3 | Runtime backbone and internal orchestrator advisory sidecars | `src/semantic/runtime_backbone.py`, `src/orchestrator/semantic_fusion_runner.py`, `src/replay/ingest.py`, `src/orchestrator/semantic_simulation.py`, `scripts/run_stage1_pipeline.py`, `scripts/bootstrap_semantic_workcell_loop.py` | The repo writes `orchestrator_advisory` sidecars beside semantic snapshots and replay metadata. These outputs increasingly contain execution preconditions and helper traces, not just speculative hints. | replay ingest, semantic fusion runner, Stage-1 pipeline, bootstrap workcell loop, orchestration trainers | For internal WM-to-WM communication, the word "advisory" is now too weak. These sidecars are part of the causal record of why selection or routing happened. Treating them as optional hints obscures their role as canonical internal context. | Split into two classes: keep preview reports advisory, but promote internal selection/readiness receipts into canonical control-plane metadata and replay-native traces |
| 4 | Queue and curriculum surfaces still labeled advisory | `src/orchestrator/queue_selection.py`, `src/rl/episode_sampling.py`, `src/rl/curriculum.py`, `src/orchestrator/queue_dispatch_policy.py` | The queue/sampler/curriculum lane now has real bounded helper packages and live training-distribution effects, but several module docs and interfaces still describe it as advisory-only. | live queue dispatch, replay selection, RL samplers, shadow training | This is now a naming and doctrine mismatch. The lane already changes what the agent trains on. Leaving it branded advisory makes it harder to reason about safety, authority, and promotion requirements. | Reclassify these as bounded-authority control-plane surfaces; keep preview scripts advisory, but not the live queue/sampler contracts |
| 4.5 | Sim/synth backend and branch helper seams | `src/world_model/sim_synth_physics/backend_selector.py`, `src/world_model/sim_synth_physics/backend_selector_runtime.py`, `src/world_model/sim_synth_physics/branch_planner.py`, `src/world_model/sim_synth_physics/branch_planner_runtime.py`, `scripts/train_sim_synth_backend_selector.py`, `scripts/train_sim_synth_branch_planner.py` | These started as early helper seams, but now have real trainer/export/runtime-package paths and affect canonical WM planning. | `SimSynthPhysicsRuntime`, coverage loop, diffusion prompt compilation, simulation agenda wrappers | If these were still described as advisory, the doctrine would lag reality again. They already influence backend/fidelity selection and branch planning inside the WM, with explicit promotion posture and receipts. | Treat as `bounded_authority`; continue improving receipt density and benchmark gates, but do not classify them as advisory-only anymore |
| 5 | Teacher / VLA / SceneTracks outputs versus their quality classes | `src/vla/teacher_runtime.py`, `src/evidence/teacher_trace.py`, `src/vla/rollout_labeler.py`, `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`, `src/evidence/scene_tracks_truth.py` | External teacher outputs remain `advisory_only`, while availability, fallback, and grounding preconditions are emitted separately. | rollout labeler, semantic fusion, replay ingest, bootstrap and Stage-1 pipelines | The current split is directionally correct, but it needs to be doctrine, not accident. The proposal/action output should remain advisory, while the quality/grounding class must be non-advisory metadata. | Keep external predictions advisory; standardize non-advisory availability, calibration, and grounding classes as training/promotion preconditions |
| 6 | Phase H and top-level portfolio shells | `src/phase_h/advisory_integration.py`, `src/phase_h/controller.py`, `src/phase_h/economic_learner.py`, `src/orchestrator/pipeline_manager.py` | These layers still assemble bounded higher-order suggestions above lower selectors and control-plane helpers. | Phase H controller, pipeline manager, shell-policy trainers | Promoting these too early would create another fake sovereign layer above still-maturing lower WMs. | Keep advisory for now, but require them to consume canonical preconditions/work orders instead of their own summary proxies |
| 7 | SIMA2 tag / ontology / task-graph proposal surfaces | `src/sima2/semantic_tag_propagator.py`, `src/sima2/ontology_proposals.py`, `src/sima2/task_graph_proposals.py`, `src/sima2/task_graph_refiner.py` | Proposal generators remain explicitly advisory and JSON-safe. | Stage-1 tag enrichment, ontology review paths | These are still pre-merge proposal layers, so treating them as authoritative would be premature. | Remain advisory; later connect them to explicit merge/reject receipts and work orders rather than direct mutation |
| 8 | Preview, logging, and analysis surfaces | `scripts/preview_orchestrated_runs.py`, `scripts/preview_stage3_sampling.py`, `scripts/analyze_stage1_datapacks_for_econ_semantics.py`, `src/logging/episode_logger.py`, `src/policies/episode_quality.py` | These are report-only or logging-only utilities. | humans, dashboards, offline analysis | They do not need authority, but they do need honest labeling so they are not mistaken for live control. | Remain advisory / report-only |

## Detailed Assessment

### 1. Epiplexity is under-deployed

Epiplexity is doing real work, but not enough work.

Today it already affects:

- `compile_signal_yield(...)` in `src/economics/inferential_reward.py`
- `InferentialTrainingGate.evaluate(...)` in `src/economics/inferential_training_gate.py`
- shadow advisory scoring in `src/orchestrator/shadow_advisory.py`
- replay weighting in `src/rl/episode_sampling.py`
- queue metadata in `src/orchestrator/queue_dispatch_policy.py`

That means it is no longer just a report metric. It is already a bounded economic and sampling input.

The gap is that the repo still describes and stores it as if it were mostly an overlay:

- `docs/epiplexity.md` still frames it as advisory
- `epiplexity_overlays.jsonl` remains the canonical portable carrier
- datapack schema fields are optional
- replay/training admission does not yet treat learnability class as canonical truth

The next correct shape is:

- keep full probe runs and detailed curves as sidecars
- promote the portable learnability summary into canonical metadata
- add explicit classes such as:
  - `missing`
  - `summary_only`
  - `portable_receipt_backed`
  - `benchmark_receipt_backed`
- let those classes affect:
  - training admission
  - adaptation work orders
  - sim/synth branch ranking
  - promotion evidence

### 2. Advisory naming is now lagging reality in the queue path

The queue/sampler path is the clearest example of doctrine lagging implementation.

Recent work already made:

- queue dispatch
- replay weighting
- sampler strategy selection
- curriculum weighting

real bounded parts of the live training distribution.

The repo should therefore stop describing those live contracts as "advisory-only" and instead describe them as:

- bounded-authority control-plane layers
- benchmark-gated helper lanes
- receipt-bearing training-distribution selectors

Preview scripts should still stay advisory. The live contracts should not.

### 3. Internal WM-to-WM surfaces should not be just advisory blobs

The new multi-WM plan only makes sense if lower and adjacent WMs talk through canonical state and receipts.

That means internal sidecars like:

- semantic runtime snapshots
- orchestrator selection summaries
- readiness / benchmark traces
- helper-trace receipts

should be treated as canonical internal communication surfaces, not as casual advisory byproducts.

This is especially important once the repo adds:

- sim / synth / physics WM
- later perception WM
- later embodiment WM
- transport bridges between those WMs

If those layers communicate through objects that are still culturally treated as "optional advisories," the topology will not stay honest.

### 4. External predictions should remain advisory, but their availability must not

The repo should keep this split:

- OpenVLA / teacher proposals remain advisory
- SceneTracks outputs may remain external-provider outputs
- external OSS perception and actuation models stay pluggable

But the following must be non-advisory:

- whether the external provider was available
- whether it used a real or fallback backend
- whether calibration was present
- what grounding class was achieved
- whether the resulting artifact is self-improvement eligible

The repo has already moved in this direction. The next pass should make it doctrine-wide and consistent.

## Recommended Top Tranche

### Tranche A: Epiplexity and inferential promotion

This should be the next advisory purge implementation pass.

Concrete goals:

1. define a canonical `learnability_class` / `inferential_class` contract that sits beside datapack/replay/runtime artifacts
2. persist that contract directly in replay rows and training manifests rather than relying on overlay joins
3. make `InferentialTrainingDecision` outputs the canonical source for:
   - retrain admission
   - collect-more-data work orders
   - review-required work orders
4. thread learnability class into:
   - synthetic branch admission
   - sim/diffusion agenda ranking
   - promotion evidence
   - benchmark reporting
5. keep detailed probe outputs and experimental analysis as sidecars

Acceptance signal:

- epiplexity stops being merely "visible to samplers"
- it becomes part of the canonical question "is this example or branch learnability-worthy enough to act on?"

### Tranche B: Internal advisory naming and contract cleanup

After Tranche A:

- rename or reclassify live queue/sampler/orchestration sidecars that are already bounded authority
- split preview/report outputs from internal canonical receipts
- make replay ingest and training docs distinguish:
  - advisory context
  - canonical metadata
  - bounded authority receipts

### Tranche C: External-provider doctrine cleanup

After Tranche B:

- standardize the split between external predictions and internal quality classes
- keep teacher/VLA outputs advisory
- require availability/calibration/grounding classes to be non-advisory metadata everywhere

## Phase B Math Posture Update

### What should not change now

Do not change in this pass:

- stable Phase B baseline world-model math
- trust-net core math
- `w_econ` lattice core math
- lambda controller equations

Those remain the rollback and comparison anchor.

### What should change philosophically

The repo should stop thinking of frozen Phase B math as sacred forever.

The better posture is:

- frozen baseline math is the current canonical anchor
- additive successor layers should keep taking over more real selection, admission, and control-plane work
- later benchmark-gated successor math may be allowed to challenge or replace parts of the frozen baseline
- replacement should happen only after repeated replay, benchmark, and governance evidence

This is a doctrine change, not a permission slip for casual edits.

### Successor adoption ladder

Any future attempt to supersede frozen Phase B math should pass through this sequence:

1. `compare_only`
2. `shadow_candidate`
3. `bounded_successor_overlay`
4. `benchmark_gated_candidate`
5. `formal_baseline_replacement`

The repo is currently still mostly in stages 1 through 3.

## Interaction With the Multi-WM Plan

This advisory purge matters to the later multi-WM topology.

If the repo builds:

- sim / synth / physics WM
- perception / grounding WM
- embodiment / actuation WM
- economic WM over those layers
- transport bridges between them

then each WM boundary needs canonical typed objects and receipts, not more soft advisory blobs.

So the advisory purge is not a side quest. It is part of making future WM-to-WM communication honest.

## Immediate Follow-On Questions

The next implementation pass should answer these concretely:

1. What exact schema should carry `learnability_class` and inferential work-order truth across datapacks, replay rows, and training manifests?
2. Which replay/training entrypoints still recompute inferential admission from summaries instead of consuming a canonical emitted work order?
3. Which internal sidecars should be renamed from `advisory` to `receipt`, `selection_summary`, or `control_plane_context` because they already function that way?
4. Which benchmark reports should begin surfacing epiplexity / learnability density as a first-class promotion metric?

## Bottom Line

The repo should not become anti-advisory.

It should become more precise about advisory scope:

- external and preview layers stay advisory
- internal typed quality, readiness, and learnability surfaces graduate out of the advisory bucket
- bounded internal selectors are named and governed as bounded authority
- frozen Phase B math stays anchored, but successor layers are allowed to earn replacement later

That is the doctrine that matches the stack as it exists now, not the stack as it existed several weeks ago.
