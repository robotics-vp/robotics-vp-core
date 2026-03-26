# Heuristic / Advisory / Sidecar Inventory

## Scope

This pass scanned the repo for surfaces that still materially shape runtime decisions, replay/readiness, sim/diffusion agenda generation, training corpus construction, reward-adjacent routing, benchmark eligibility, or trainer parity while remaining heuristic, advisory-only, sidecar-only, stubbed, fallback-heavy, or lightweight-only.

Ranking dimensions:

- **Production importance**: how directly the surface affects runtime/training/reward/selection paths already used in the repo.
- **Loop distortion**: how much the surface can make the stack look more “real” than it actually is if left unwired.

## Ranked Gap Matrix

| Rank | Surface | Category | Production importance | Loop distortion if left unwired | Top tranche | Status after this pass |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Local synthetic branch corpus + offline local synth trainer | `lightweight_trainer_gap` | Very high | Very high | Yes | **Wired now** |
| 2 | Stage-1 semantic seed tags + diffusion proposal routing | `heuristic` / `fallback` | Very high | High | Yes | **Wired now** |
| 3 | SceneTracks passthrough/non-stub truthiness in bootstrap + replay ingest | `fallback` | High | High | Yes | **Wired now** |
| 3a | Bootstrap workcell runtime trace completeness + grounded-data lane classification | `sidecar` / `fallback` | High | High | Yes | **Wired now** |
| 3b | Workcell `peg_in_hole` coverage-graph mapping | `heuristic` | High | High | Yes | **Wired now** |
| 4 | Shadow advisory replay sampling and queue reweighting | `heuristic` / `advisory` | High | High | Yes | **Wired now** |
| 5 | `train_vla_recap_offline.py` lightweight trainer path | `lightweight_trainer_gap` | High | Medium-high | Yes | **Wired now** |
| 6 | `train_orchestration_transformer.py` heuristic-teacher trainer | `heuristic` / `lightweight_trainer_gap` | High | Medium-high | No | Wrapped, but target remains heuristic |
| 7 | Semantic datapack/scenario selection in `semantic_policy.py` | `heuristic` | Medium-high | Medium-high | No | **Wired now** |
| 8 | `train_meta_transformer_synthetic.py` meta-transformer trainer entrypoint | `lightweight_trainer_gap` | Medium | High | Yes | **Wired now** |
| 9 | Teacher-runtime / rollout labeler semantic sidecars | `advisory` / `fallback` | Medium | Medium | No | Explicit fallback kept, benchmark-gated |
| 10 | SceneTracks runner stub/passthrough backend lane | `fallback` | Medium | Medium | No | Explicit fallback kept, benchmark-gated |

## Top Tranche Landed

### 1. Local synthetic branch corpus + offline local synth trainer

- surface: local synthetic branch collection and offline local-synth trainer
- file/path: `scripts/collect_local_synthetic_branches.py`, `scripts/train_offline_with_local_synth.py`, `src/training/synthetic_branch_corpus.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - before this pass, branch collection emitted minimal metadata and gap-label sidecars
  - `train_offline_with_local_synth.py` consumed raw NPZ branches, derived branch economics from proxy metrics, and produced ad hoc results/checkpoints outside the canonical training runtime
  - no explicit branch-corpus readiness, benchmark gate, or bounded synth-share cap existed
- current consumers:
  - `scripts/train_offline_with_local_synth.py`
  - `scripts/run_3mode_synth_ab_test.py`
  - `scripts/run_4mode_synth_ab_test.py`
  - `scripts/train_synth_lambda_controller.py`
- why it is a production problem:
  - this lane could report synthetic uplift without proving whether the source corpus was gap-labeled, semantically grounded, benchmark-eligible, or even structurally comparable to heavier trainers
  - lightweight outputs were not producing manifests/checkpoints/runtime artifacts comparable to the rest of the training stack
- recommended disposition:
  - keep the lane, but make it explicit and bounded
  - require synthetic-branch corpus metadata, execution-precondition artifacts, and benchmark-gate reports
  - cap synth share when metadata/gap labels/non-heuristic grounding are missing
  - emit canonical training artifacts via `RegalTrainingRunner`
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
  - `benchmark-gated`

### 2. Stage-1 semantic seed tags + diffusion proposal routing

- surface: keyword/rule-driven Stage-1 semantic extraction and diffusion proposal shaping
- file/path: `scripts/run_stage1_pipeline.py`, `src/orchestrator/diffusion_requests.py`, `src/diffusion/real_video_diffusion_stub.py`
- category: `heuristic`
- current behavior:
  - `extract_semantic_tags_from_video(...)` in `scripts/run_stage1_pipeline.py` still begins with deterministic seed tags, but downstream routing no longer stops there
  - `build_diffusion_prompt_from_guidance(...)`, `build_diffusion_prompt_from_coverage_gaps(...)`, and `prompt_to_diffusion_stub_input(...)` now carry structured governed hypotheses plus routing context instead of collapsing prompts to flat tag/objective rules
  - `VideoDiffusionStub.propose_augmented_clips(...)` now reranks governed hypotheses before any fallback lane and clamps confidence/novelty when routing is still heuristic or benchmark-unready
  - `run_stage1_pipeline(...)` now emits explicit benchmark-gate sidecars and downgrades unbenchmarked proposals into `shadow_stage1_datapack` work orders with tier-0 datapacks rather than treating them like normal benchmark-eligible datapacks
- current consumers:
  - `scripts/run_stage1_pipeline.py`
  - `scripts/run_orchestrated_guidance_loop.py`
  - `scripts/ingest_diffusion_responses.py`
- why it is a production problem:
  - the sim/diffusion agenda can still be shaped by keyword tags instead of evidence, branch value, or replay-backed success/regret
  - this risks generating “semantic-looking” branches that are not grounded strongly enough to drive serious training runs
- recommended disposition:
  - keep stub rendering downstream of governed hypotheses
  - keep heuristic seed-tag extraction only as a bounded bootstrap source
  - let benchmark-unready manifests stay explicit shadow/fallback lanes until learned routing and richer real video grounding are available
- disposition tag:
  - `wired now`
  - `neuralized later`
  - `benchmark-gated`

### 3. SceneTracks passthrough/non-stub truthiness in bootstrap + replay ingest

- surface: permissive `scene_tracks_non_stub` inference for passthrough paths
- file/path: `src/replay/ingest.py`, `scripts/bootstrap_semantic_workcell_loop.py`
- category: `fallback`
- current behavior:
  - replay ingest and bootstrap now share explicit truth semantics:
    - `passthrough`, `stub`, and `auto` no longer count as `scene_tracks_non_stub`
    - only `real` SceneTracks keep `scene_tracks_non_stub`, `semantic_grounding_ready`, and `semantic_grounding_non_heuristic`
  - fallback lanes still preserve backend identity and semantic density, but they no longer masquerade as grounded/non-heuristic inputs in upstream metadata
- current consumers:
  - replay import/readiness code
  - bootstrap semantic workcell loop metadata
  - downstream replay preconditions and benchmark gating
- why it is a production problem:
  - training/readiness summaries can look more production-ready than the benchmark gate would actually allow
  - this distorts corpus admission and operational dashboards
- recommended disposition:
  - keep passthrough as an explicit fallback only
  - continue preserving backend identity and density signals without promoting them to non-stub truth
- disposition tag:
  - `wired now`
  - `remain explicit fallback`
  - `benchmark-gated`

### 3a. Bootstrap workcell runtime trace completeness + grounded-data lane classification

- surface: bootstrap workcell episodes looked replay-shaped but lacked canonical runtime packet / event spine / decision ledger refs
- file/path: `scripts/bootstrap_semantic_workcell_loop.py`, `src/replay/ingest.py`
- category: `sidecar`
- current behavior:
  - the bootstrap loop now writes per-episode:
    - `*_runtime_packet_v1.json`
    - `*_event_spine_v1.json`
    - `*_decision_ledger_v1.json`
  - `metadata.json` now carries:
    - `runtime_packet_id`
    - `event_refs`
    - `decision_refs`
    - grounded-data lane facts such as `grounded_data_ready`, `grounded_data_mode`, and explicit SAM3D/GPU requirements
  - replay rollout import now discovers those refs directly instead of leaving bootstrap rows permanently trace-incomplete
- current consumers:
  - `ReplayDatasetBuilder.add_rollout_bundle(...)`
  - semantic runtime learning corpus build
  - Runpod/bootstrap readiness summaries
- why it is a production problem:
  - the bootstrap loop could generate stable replay/runtime corpora but still force `bounded_ready_count=0` because the canonical packet/event/decision substrate never existed
  - grounded-data truth about real SAM3D plus GPU dependence remained implicit even when the episodes were otherwise operationally stable
- recommended disposition:
  - keep bootstrap as a dev/regression lane
  - preserve the explicit distinction between:
    - trace-complete replay rows
    - grounded-data-ready rows
    - benchmark-eligible rows
  - keep passthrough/dev-only lanes explicit instead of letting trace completeness masquerade as benchmark readiness
- disposition tag:
  - `wired now`
  - `remain explicit fallback`
  - `benchmark-gated`

### 3b. Workcell `peg_in_hole` coverage-graph mapping

- surface: workcell rows were barely landing in the canonical task × skill × env-primitive graph
- file/path: `src/world_model/coverage_evidence_harvester.py`, `src/hrl/skill_graph.py`, `src/orchestrator/coverage_loop.py`
- category: `heuristic`
- current behavior:
  - the coverage harvester now canonicalizes env ids such as `workcell_env` into the registered `workcell` inventory
  - harvested skill ids are now canonicalized to the same ids the graph uses (`hrl:*`, `workcell:*`, etc.) instead of the old mismatched `skill:*` shape
  - the skill graph now has a built-in workcell chain for `peg_in_hole`
  - the harvester now maps workcell affordance/task evidence into that chain instead of defaulting to drawer-era skill assumptions
- current consumers:
  - `run_coverage_loop(...)`
  - workcell bootstrap coverage artifacts
  - gap-driven sim/diffusion agenda compilation
- why it is a production problem:
  - a workcell pass could execute many episodes and still report near-zero covered edges because the graph contract and harvested keys disagreed
  - this distorts coverage-based agenda generation and makes the semantic loop understate how much usable workcell evidence it actually has
- recommended disposition:
  - keep deterministic workcell skill extraction for now
  - later replace it with learned/runtime-backed skill extraction once the corpus contains richer teacher/runtime traces
- disposition tag:
  - `wired now`
  - `neuralized later`

### 4. Shadow advisory replay sampling and queue reweighting

- surface: heuristic scoring for replay sampling, queue tags, and bounded queue influence
- file/path: `src/orchestrator/shadow_advisory.py`, `src/rl/econ_regal_sampling.py`, `src/orchestrator/queue_selection.py`
- category: `advisory`
- current behavior:
  - `build_shadow_advisory_output(...)` compiles policy/pricing/data-value/regal support into advisory episode rows
  - when a semantic runtime scorer package is present, the advisory path now scores replay-native semantic runtime rows and threads bounded learned route/regret/counterfactual/authority signals into `recommend_sampling(...)`
  - `recommend_sampling(...)` still keeps the heuristic path as fallback, but learned runtime support now influences priority, queue tags, and slice weighting in a bounded way
  - `build_live_queue_selection(...)` and `apply_live_queue_selection(...)` now preserve the semantic-runtime score evidence in queue metadata instead of dropping it before the live queue lane
- current consumers:
  - `scripts/train_shadow_replay_policy.py`
  - `scripts/train_shadow_offline_rl.py`
  - `scripts/train_sac_with_ontology_logging.py`
  - `scripts/run_shadow_advisory_pass.py`
- why it is a production problem:
  - this path already affects actual replay/training selection, so a missing learned-scoring seam would keep a real control-plane lane overly heuristic
  - the remaining limitation is now scorer coverage/package availability rather than the absence of wiring
- recommended disposition:
  - keep the bounded queue lane wired
  - auto-consume replay-backed learned runtime scorers when a package is present
  - keep the current rule path only as an explicit fallback when no scorer package exists yet
- disposition tag:
  - `wired now`
  - `neuralized later`

### 5. `train_vla_recap_offline.py`

- surface: offline RECAP VLA heads trainer
- file/path: `scripts/train_vla_recap_offline.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - the direct `train_offline(...)` API still works for existing smoke and inference consumers, but it now emits:
    - recap dataset summary
    - recap feature-config artifact
    - training preconditions / benchmark-gate artifact
    - training summary
    - training job result
    - latest and best checkpoints under the original checkpoint contract expected by `src/vla/recap_inference.py`
  - the CLI path now wraps the same trainer under `RegalTrainingRunner`, registers recap artifacts/checkpoints, emits canonical runtime manifests/checkpoint registry, and projects per-episode recap rows into explicit trajectory-audit receipts instead of leaving the lane outside the runtime envelope
- current consumers:
  - `scripts/smoke_test_vla_recap_training.py`
  - `scripts/smoke_test_recap_inference.py`
  - `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json`
- why it is a production problem:
  - before this pass, RECAP head training looked like a valid trainer but emitted only local checkpoints and optional CSV metrics, so promotion/readiness tooling could not reason about it the same way it reasons about the shadow/offline/synth lanes
  - the direct path also had no honest benchmark gate, which encouraged tiny smoke corpora to look too similar to a serious recap corpus
- recommended disposition:
  - keep the direct function as a library boundary for smoke/inference code, but preserve the same artifact contract there
  - keep recap-row trajectory audits explicit as `recap_row_projection` rather than pretending the recap corpus already contains full embodied action traces
  - keep the benchmark gate conservative so small recap corpora remain runnable for dev but not promotion-ready
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
  - `benchmark-gated`

### 6. `train_orchestration_transformer.py`

- surface: orchestration transformer trainer that still uses heuristic teachers and dummy instruction tokens
- file/path: `scripts/train_orchestration_transformer.py`
- category: `heuristic`
- current behavior:
  - already wrapped with `@regal_training`
  - still trains against heuristic tool sequences
  - `create_dummy_instruction_tokens(...)` supplies placeholder language tokens rather than real instruction conditioning
- current consumers:
  - `scripts/run_stage6_train_all.py`
  - `scripts/eval_orchestration_transformer.py`
  - `scripts/train_orchestration_transformer_v1_curriculum.py`
- why it is a production problem:
  - the wrapper is heavyweight, but the supervision contract is still synthetic/heuristic
  - this can create false confidence about orchestration readiness
- recommended disposition:
  - preserve the runtime wrapper
  - replace heuristic teacher targets and dummy tokens with packet/evidence/runtime-corpus supervision
- disposition tag:
  - `neuralized later`
  - `upgraded to heavyweight parity`

### 7. Semantic datapack/scenario selection in `semantic_policy.py`

- surface: rule-based semantic scenario and datapack selection
- file/path: `src/orchestrator/semantic_policy.py`
- category: `heuristic`
- current behavior:
  - `rank_datapacks_for_intent(...)` now combines:
    - tag coverage and exact-match pressure
    - ARH-adjusted historical scenario outcomes per datapack
    - datapack quality / novelty metadata
    - benchmark/readiness support when datapack metadata carries it
    - explicit gap-fill pressure for tags not yet represented in scenario history
  - the same feature contract is now first-class in `DatapackSelectionFeatures`, and an optional bounded learned helper package can apply a capped reranking adjustment on top of the explicit prior instead of replacing it wholesale
  - `src/orchestrator/semantic_simulation.py` now merges ontology and local fallback datapacks through the same ranked pool, records a `selection_summary` into the live simulation result plus run log, and carries explicit helper-promotion state:
    - `disabled` for bootstrap
    - `auto` for shadow/helper-when-present
    - `required` when the learned helper is a runtime precondition
  - `detect_semantic_gaps(...)` still infers missing scenario tags deterministically
- current consumers:
  - `src/orchestrator/semantic_simulation.py`
- why it is a production problem:
  - before this pass, this determined which datapacks entered the simulation/training loop using little more than tag overlap plus ARH subtraction
  - that was too weak for a runtime surface that directly shapes synthetic agenda generation and future corpus construction
- recommended disposition:
  - keep the current bounded scored-selection layer live now
  - preserve the explicit prior feature contract as the bootstrap fallback and training target
  - promote the learned helper sequentially: `disabled` -> `auto` -> `required`
  - train the helper over `selection_summary` plus downstream outcome/counterfactual receipts before handing it required status
- disposition tag:
  - `wired now`
  - `neuralized later`

### 8. `train_meta_transformer_synthetic.py`

- surface: meta-transformer trainer entrypoint
- file/path: `scripts/train_meta_transformer_synthetic.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - now consumes the real meta-transformer substrate in `src/orchestrator/meta_transformer_training.py`:
    - exported `meta_transformer_runtime_dataset.json`
    - saved dataset JSON inputs
    - the real `MetaTransformerNet`
    - the existing batching/loss/eval helpers
  - synthetic generation remains available only as an explicit fallback input source when requested
  - the script now emits canonical runtime manifests/checkpoint registry/training summaries and honest corpus benchmark/precondition artifacts instead of writing an opaque random checkpoint under `results/`
- current consumers:
  - `scripts/export_semantic_runtime_learning_corpus.py`
  - future meta-transformer promotion / runtime-helper consumers
- why it is a production problem:
  - before this pass, the script looked like a trainer but bypassed the actual runtime dataset/model substrate already present in the repo
  - that created a high-distortion fake boundary because a checkpoint-shaped artifact could be produced with almost no relationship to the runtime corpus or production transformer contract
- recommended disposition:
  - keep the runtime/export dataset as the preferred source
  - keep synthetic generation explicit and benchmark-gated as a dev fallback only
  - later add a dedicated runtime-corpus density gate before promotion runs
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
  - `benchmark-gated`

### 9. Teacher-runtime / rollout-labeler semantic sidecars

- surface: external teacher contracts/envelopes and rollout labeler sidecars
- file/path: `src/vla/rollout_labeler.py`, `src/vla/teacher_runtime.py`
- category: `advisory`
- current behavior:
  - teacher contracts/action envelopes are explicit and persisted
  - VLA semantic evidence remains an external teacher sidecar rather than native truth
  - unavailable/disabled teacher states are explicit, but common
- current consumers:
  - `src/orchestrator/semantic_simulation.py`
  - rollout-labeler tests
- why it is a production problem:
  - labels and teacher semantics may still be sparse or fallback-heavy
  - downstream consumers must not mistake teacher presence for production grounding
- recommended disposition:
  - keep teacher outputs external/advisory
  - continue benchmark-gating real teacher availability
  - do not collapse teacher truth into native runtime truth
- disposition tag:
  - `remain explicit fallback`
  - `benchmark-gated`

### 10. SceneTracks runner stub/passthrough backend lane

- surface: SceneTracks backend selection with stub and passthrough options
- file/path: `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`
- category: `fallback`
- current behavior:
  - `run_scene_tracks(...)` supports `real`, `passthrough`, and `stub`
  - training eligibility is explicit and strict inside the runner
  - the runner now also emits `grounded_data_host_capabilities` and `grounded_data_host_preconditions`, so the GPU + SAM3D dependency is executable metadata rather than an implied requirement
  - Stage-1, synthetic-branch, and live runtime consumers now share the same truth helper, so passthrough/unknown artifact presence no longer gets reinterpreted upstream as non-stub grounding
  - callers can still choose fallback/passthrough modes
- current consumers:
  - `scripts/run_scene_tracks.py`
  - `scripts/bootstrap_semantic_workcell_loop.py`
  - SceneTracks tests
- why it is a production problem:
  - before the latest sweep, the runner itself was honest but several broader-stack consumers still interpreted fallback output too generously
  - grounded-data host requirements were also visible in docs/backlogs but not emitted as a reusable runtime artifact
- recommended disposition:
  - keep stub/passthrough support only as explicit fallback
  - keep every upstream/downstream metadata consumer aligned to the runner’s stricter truth semantics
  - keep real-SAM3D host readiness explicit so benchmark-grade grounded data is not inferred from sidecar existence
- disposition tag:
  - `remain explicit fallback`
  - `benchmark-gated`

## Remaining Top Follow-Ons

1. Make the shadow-advisory scorer fallback explicit in artifacts/work orders so “no scorer package available” is visible as a runtime precondition, not just a behavior branch.
2. Add a real training/export path for the new semantic datapack-selection helper so the `auto -> required` promotion path is backed by an actual corpus and scorer package, not just runtime plumbing.
3. Audit remaining vision-side sidecars that still preserve density/quality signals without yet changing runtime routing strongly enough, especially around real-SAM3D bring-up and sidecar-only grounding semantics outside the consumers already fixed.
4. Add a stricter promotion/readiness gate for meta-transformer runs so runtime-corpus density, not just script parity, controls when the lane is taken seriously.
