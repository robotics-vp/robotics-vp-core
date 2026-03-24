# Self-Improvement Preconditions Sweep

## Scope

This sweep answers one narrow question: where should this repo stop being "advisory only" and start creating real preconditions for self-improvement?

The answer is not "make everything live." The answer is:

1. keep external teachers, speculative planners, and broad pipeline managers advisory
2. promote typed evidence into eligibility checks, work orders, and replay-native traces
3. only then let downstream training or routing depend on those artifacts

The repo is already strongest where it emits packets, event rows, value targets, and governance traces. The weak point is that many of those outputs still terminate as sidecars, reports, or logs instead of becoming admission criteria for the next step.

## Executive Take

The repo already has one good exemplar of the right promotion pattern:

- `src/orchestrator/queue_selection.py`
- `src/rl/episode_sampling.py`
- `scripts/train_shadow_offline_rl.py`
- `scripts/train_shadow_replay_policy.py`
- `scripts/train_shadow_pricing_models.py`
- `scripts/train_sac_with_ontology_logging.py`

These paths do not rewrite reward math, but they do let advisory outputs exert bounded influence over selection order and sampling weights. That is the right template.

The next tranche should not promote world-model outputs directly into action authority. It should promote artifact completeness into three kinds of preconditions:

1. training-eligibility preconditions
2. adaptation/data-collection work-order preconditions
3. promotion/gating preconditions

## Current State By Area

| Area | Current state | Recommendation |
| --- | --- | --- |
| Queue dispatch and replay sampling | Already partially live and bounded | Deepen here first; use as the template |
| Adaptation budgeting | Produces decisions but does not create executable work | Promote to work-order substrate now |
| Promotion reporting | Still summary-driven; ignores most new trace artifacts | Promote to trace-aware promotion evidence now |
| Replay and dataset bridges | Strong export path, weak import/rehydration path | Promote to roundtrip trace completeness now |
| Governed video loop | Emits rich sidecars but mostly stops at logs/datapack annotations | Promote to admission and replay-ingest preconditions now |
| Evidence fusion | Emits evidence/belief state but silently skips many bad cases | Promote mismatches into explicit negative artifacts now |
| Teacher and SceneTracks plumbing | Fallback semantics are explicit but not used as eligibility classes | Promote to data-quality preconditions now |
| Phase H / top-level orchestrators | Mostly planning shells and bounded hints | Keep advisory until lower-level work orders exist |

## Where Promotion Makes Sense Now

### 1. Queue and Sampling Surfaces

These are already the cleanest "advisory to bounded influence" seams:

- `src/orchestrator/queue_selection.py`
- `src/rl/episode_sampling.py`
- `scripts/train_shadow_offline_rl.py`
- `scripts/train_shadow_replay_policy.py`
- `scripts/train_shadow_pricing_models.py`
- `scripts/train_sac_with_ontology_logging.py`

What is already good:

- `build_live_queue_selection(...)` turns episode summaries into an ordered queue artifact.
- `apply_live_queue_selection(...)` can reorder, reweight, or drop slices under bounded modes.
- training scripts already consume the queue artifact instead of just writing it.

What is still missing:

- queue decisions are still mostly driven by summary fields like priority score, tags, and replay action
- the dispatcher does not yet depend on packet/event/governance/value-target completeness

Exact next conversion:

- extend queue-entry `metadata.evidence` to carry packet/event/governance/value-target refs, not just summaries
- make promoted queue modes require trace completeness before upweight or drop authority applies
- use this layer as the first real consumer of governed-video and replay-sidecar artifacts

### 2. Adaptation Budgeting and Inferential Decisions

The repo now computes adaptation decisions but does not operationalize them:

- `src/orchestrator/shadow_advisory.py`
- `src/economics/inferential_training_gate.py`
- `src/orchestrator/adaptation_budgeting.py`
- `src/economics/inferential_reward.py`

What is already good:

- `InferentialTrainingGate.evaluate(...)` is deterministic and economically grounded
- `build_shadow_advisory_output(...)` attaches per-episode `inferential_budget_decision`
- receipt feedback can already enrich those decisions

What is missing:

- no normalized adaptation work order
- no collection work order
- no executor-facing artifact that says "this episode is admissible for retraining" versus "collect more data first"

Exact next conversion:

- keep `InferentialTrainingGate` as the judge, but stop there from being the terminal step
- add an additive work-order artifact in the shadow/advisory layer, keyed by decision ID and source refs
- thread packet refs, event refs, decision refs, and receipt refs into that work order
- make downstream trainers consume the work-order artifact instead of recomputing admission from summaries

Best module seams:

- `src/orchestrator/shadow_advisory.py`: emit `adaptation_work_orders` and `collection_work_orders`
- `src/orchestrator/adaptation_budgeting.py`: summarize admitted vs blocked work in executor-facing form
- `src/shadow_runtime/control_plane.py`: emit the originating adaptation and collect-more-data event lineage

### 3. Promotion Evidence and Authority

Promotion reporting is now a bottleneck:

- `src/regality/promotion_reporting.py`
- `src/replay/receipt_ingest.py`
- `src/replay/ingest.py`

What is already good:

- promotion reports exist
- receipt bundles exist
- replay episode metadata already contains `event_refs`, `decision_refs`, and sidecar refs

What is missing:

- `build_promotion_evidence_report(...)` still works mostly from episode summaries and receipt outcomes
- it does not join through EventSpine, DecisionLedger, or GovernanceTrace
- promotion decisions therefore still infer many things indirectly

Exact next conversion:

- use `event_refs`, `decision_refs`, `event_kinds`, and `decision_kinds` from replay metadata as first-class promotion evidence
- join governance-trace reasons into false-positive and false-negative slices
- explicitly score whether a node produced useful vetoes, holds, reweights, or collect-more-data decisions, not just whether summary outcomes correlated with deployment labels

This is the cleanest place to create authority preconditions before any wider hard-gating.

### 4. Replay Ingest and Dataset Roundtrip

Replay is close, but still incomplete as a self-improvement substrate:

- `src/replay/ingest.py`
- `src/replay/dataset.py`
- `src/dataset_bridges/sidecar_refs.py`
- `src/dataset_bridges/rlds_bridge.py`
- `src/dataset_bridges/lerobot_bridge.py`

What is already good:

- shadow ingest preserves packet/event/decision refs
- bridge exports preserve internal sidecar refs generically

What is missing:

- no reverse import glue from RLDS/LeRobot back into canonical replay metadata/provenance
- no trace-completeness classes such as "packet-complete", "event-complete", "governance-complete", "teacher-complete"
- Stage-1 governed-video artifacts are not yet canonical replay citizens

Exact next conversion:

- add replay import/rehydration from bridge exports back to canonical metadata/provenance
- define per-episode trace-completeness grades and persist them in replay metadata
- make training eligibility depend on those grades instead of ad hoc file presence
- ingest Stage-1 governed-video refs alongside shadow refs so video supervision stops living outside replay

This is a real precondition for self-improvement because later training cannot honestly depend on traces that disappear on export/import.

### 5. Governed Video Loop and Candidate Admission

The video path emits strong artifacts, but still underuses them:

- `scripts/run_stage1_pipeline.py`
- `src/world_model/governed_video_supervision.py`
- `src/world_model/governed_video_world_model.py`

What is already good:

- hypotheses are generated before rendering
- reconstruction, packet, event, decision, governance, counterfactual, and value-target sidecars are emitted
- blocked proposals are kept in the pipeline log

What is missing:

- datapack admission still hinges mostly on plausibility plus ad hoc annotation
- blocked and accepted branches are not being normalized into replay/trainable examples
- the value-ledger receipt is built through a hardcoded `/tmp` path inside `build_governed_video_supervision_bundle(...)`

Exact next conversion:

- make datapack admission require stable refs for reconstruction, runtime packet, governance trace, counterfactual eval, and value-target pack
- export both accepted and blocked branches into replay-ready records so negative branches become trainable
- remove the hardcoded temporary value-ledger path and emit a stable ledger-sidecar ref through the Stage-1 output root

This is where the repo can start creating real preconditions for governed video self-improvement without turning the video world model into a sovereign actor.

### 6. Evidence Fusion and Negative Evidence

Semantic fusion still fails too quietly:

- `src/orchestrator/semantic_fusion_runner.py`
- `src/evidence/bus.py`
- `src/evidence/belief_state.py`

What is already good:

- evidence and belief state artifacts are emitted when fusion succeeds
- teacher traces can be folded into the EvidenceBus

What is missing:

- several mismatch conditions currently log a warning and `continue`
- that means failed alignment is not replayable as a supervision target
- skipped episodes can silently vanish from later self-improvement datasets

Exact next conversion:

- emit explicit mismatch/failure evidence records instead of only warnings
- always emit a belief state, even when degraded, with failure metadata and quality class
- make training eligibility depend on whether the episode is degraded but explicit versus silently missing

This is a high-leverage precondition because silent omission is poison for self-improvement.

### 7. Teacher Runtime and SceneTracks Grounding

These modules should remain advisory in what they predict, but non-advisory in how they classify data quality:

- `src/vla/teacher_runtime.py`
- `src/vla/rollout_labeler.py`
- `src/vla/openvla_controller.py`
- `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
- `src/vision/reconstruction/four_d_reconstruction.py`

What is already good:

- teacher fallback is explicit
- teacher contract and action-envelope sidecars are emitted
- SceneTracks quality is computed and surfaced
- reconstruction sidecars expose calibration completeness

What is missing:

- OpenVLA fallback and SceneTracks stub usage are not yet converted into standard eligibility classes
- low-quality or stub-grounded artifacts can still flow downstream without a consistent "not self-improvement eligible" marker

Exact next conversion:

- define canonical grounding classes such as `grounded`, `degraded`, `stubbed`, and `unavailable`
- write those classes into teacher/runtime/reconstruction/scene-tracks metadata
- require a minimum grounding class before an episode can produce training-positive work orders
- keep external teacher outputs advisory, but make their availability and calibration non-advisory metadata

### 8. Phase H and Portfolio Control

Phase H is the clearest example of something that should not be promoted directly yet:

- `src/phase_h/advisory_integration.py`
- `src/phase_h/controller.py`
- `src/phase_h/economic_learner.py`

What is already good:

- bounded multipliers
- exploration priorities
- budget and ROI reasoning

What is missing:

- the learner still uses stubbed return measurement and emits mostly reports or bounded hints
- there is no lower-level work-order substrate for it to command safely

Exact next conversion:

- do not let Phase H directly mutate training or routing authority yet
- first let it emit budgeted work orders that target the same adaptation/collection substrate proposed above
- once that substrate exists, Phase H can prioritize among work orders rather than push raw multipliers

## Where Promotion Does Not Make Sense Yet

These modules should remain advisory until lower layers are stricter:

- `src/orchestrator/semantic_orchestrator_v2.py`
- `src/orchestrator/pipeline_manager.py`
- `src/orchestrator/economic_controller.py`
- `src/hrl/high_level_controller.py`

Reason:

- they either define upstream economics, broad planning structure, or high-level controllers that still lack packet/event/evidence-native input contracts
- promoting them first would create fake sovereignty rather than honest self-improvement

## Recommended Order Of Operations

### P0: Make emitted traces executor- and replay-usable

- adaptation/data-collection work orders from inferential decisions
- promotion reporting over EventSpine, DecisionLedger, and GovernanceTrace
- replay import/rehydration for dataset bridges
- stable Stage-1 ledger/governance/value-target refs and replay export of accepted plus blocked branches

### P1: Make degraded evidence explicit instead of silent

- semantic-fusion mismatch artifacts
- grounding classes for teacher runtime, reconstruction, and SceneTracks
- trace-completeness and grounding-completeness grades in replay metadata

### P2: Promote higher-level portfolio logic only after P0/P1 exist

- Phase H emits budgeted work orders instead of pure multipliers
- top-level orchestrators consume work-order summaries, not raw speculative guidance

## One-Sentence Rule

Do not promote another planner, teacher, or controller first. Promote the repo's ability to say, with typed artifacts, that a candidate episode is grounded enough to train on, valuable enough to adapt on, and evidenced enough to promote authority from.
