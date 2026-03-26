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
| 2a | Gap-ranker trainer/package + sim/gen2sim agenda ranking | `heuristic` / `lightweight_trainer_gap` | Very high | High | Yes | **Wired now, benchmark-gated** |
| 3 | SceneTracks passthrough/non-stub truthiness in bootstrap + replay ingest | `fallback` | High | High | Yes | **Wired now** |
| 3a | Bootstrap workcell runtime trace completeness + grounded-data lane classification | `sidecar` / `fallback` | High | High | Yes | **Wired now** |
| 3b | Workcell `peg_in_hole` coverage-graph mapping | `heuristic` | High | High | Yes | **Wired now** |
| 4 | Shadow advisory replay sampling and queue reweighting | `heuristic` / `advisory` | High | High | Yes | **Wired now** |
| 5 | `train_vla_recap_offline.py` lightweight trainer path | `lightweight_trainer_gap` | High | Medium-high | Yes | **Wired now** |
| 6 | `train_orchestration_transformer.py` runtime-backed trainer parity | `lightweight_trainer_gap` | High | Medium-high | Yes | **Wired now, benchmark-gated** |
| 7 | Semantic datapack/scenario selection in `semantic_policy.py` plus helper trainer/export lane | `heuristic` / `lightweight_trainer_gap` | Medium-high | Medium-high | No | **Wired now, benchmark-gated** |
| 8 | `train_meta_transformer_synthetic.py` + meta-transformer runtime package/promotion path | `lightweight_trainer_gap` | Medium-high | High | Yes | **Wired now, benchmark-gated** |
| 9 | Teacher-runtime / rollout labeler semantic sidecars | `advisory` / `sidecar` / `fallback` | Medium-high | Medium-high | No | **Wired now** |
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

### 2a. Gap-ranker trainer/package + sim/gen2sim agenda ranking

- surface: learned gap-ranker packaging and bounded use in simulation-agenda / diffusion-gap ranking
- file/path: `scripts/train_gap_ranker.py`, `src/world_model/gap_ranker_runtime.py`, `src/orchestrator/gap_agenda_ranking.py`, `src/orchestrator/semantic_simulation.py`, `src/orchestrator/diffusion_requests.py`, `src/orchestrator/coverage_loop.py`
- category: `heuristic` / `lightweight_trainer_gap`
- current behavior:
  - `scripts/train_gap_ranker.py` is now a canonical trainer lane rather than a loose standalone script:
    - emits dataset summary
    - model config
    - execution-precondition artifact
    - training summary
    - `gap_ranker_package.json`
    - runtime manifest / checkpoint registry under `RegalTrainingRunner`
  - benchmark readiness is now explicit and conservative:
    - enough fill-outcome records
    - enough positive coverage deltas
    - enough fill-method diversity
    are required before the package is treated as `promoted`
  - `src/orchestrator/gap_agenda_ranking.py` now blends:
    - heuristic coverage-gap ranking
    - learned gap-ranker ranking
    through a bounded helper weight
  - `src/orchestrator/semantic_simulation.py` and `src/orchestrator/diffusion_requests.py` now both use that same ranking helper, emit `ranking_policy`, and record helper status plus score traces on agenda items and governed diffusion prompts
  - `src/orchestrator/coverage_loop.py` now threads the configured gap-ranker helper into both agenda and diffusion compilation rather than reserving learned ranking only for later fill-path logic
- current consumers:
  - `run_coverage_loop(...)`
  - `compile_simulation_agenda(...)`
  - `build_diffusion_prompt_from_coverage_gaps(...)`
  - `scripts/train_gap_ranker.py`
- why it is a production problem:
  - before this pass, the repo already had a learned gap ranker, but the main sim/gen2sim agenda still ignored it and ranked on heuristics alone
  - that left a real self-improvement substrate stranded outside the actual branch-selection loop
  - the old trainer path also emitted only a raw checkpoint, so promotion state and fallback honesty were implicit
- recommended disposition:
  - keep the heuristic gap score as the explicit prior
  - use the learned gap ranker through `disabled` / `auto` / `required` helper semantics
  - keep benchmark-unready packages as `shadow_candidate` helpers with bounded influence only
  - later broaden the same promotion contract to the fill-path policy so every coverage-loop decision shares the same maturity semantics
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
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
  - the main advisory/training consumers now also emit:
    - `semantic_runtime_scorer_preconditions.json`
    - `semantic_runtime_scorer_work_orders.json`
    so the no-package fallback is visible as a real runtime artifact and blocking work order instead of only a hidden behavior branch
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

- surface: orchestration transformer trainer and eval path
- file/path: `scripts/train_orchestration_transformer.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - the trainer now prefers `orchestration_runtime_dataset.json` exports from the semantic runtime corpus and only falls back to synthetic/mixed data when explicitly requested or no runtime dataset is available
  - instruction conditioning is now deterministic and contract-aligned:
    - `src/orchestrator/training_dataset.py` derives stable instruction text from runtime metadata/context
    - training tensors hash that text into bounded instruction-token sequences
    - `scripts/eval_orchestration_transformer.py` now uses the same token contract instead of random placeholder tokens
  - the trainer now emits canonical runtime artifacts:
    - dataset + dataset summary
    - model config
    - execution preconditions
    - subset metrics
    - training summary
    - training job result
    - runtime manifest / checkpoint registry through `RegalTrainingRunner`
  - benchmark readiness is now honest:
    - runtime-backed corpora can train immediately
    - synthetic-only or low-density runtime corpora remain benchmark-unready
  - the remaining explicit limitation is narrower:
    - `tool_prediction_contract` is still `first_tool_only_v1`
- current consumers:
  - `scripts/run_stage6_train_all.py`
  - `scripts/eval_orchestration_transformer.py`
  - `scripts/train_orchestration_transformer_v1_curriculum.py`
- why it is a production problem:
  - before this pass, the runtime wrapper looked modern but the actual trainer still depended on synthetic teacher contracts and dummy instruction tokens
  - that made the lane appear more runtime-grounded than it really was
  - the remaining production limitation is that the supervision target is still only the first tool, not the full sequence/policy contract
- recommended disposition:
  - keep the runtime-backed trainer as the default path
  - preserve synthetic fallback only as a clearly benchmark-unready bootstrap lane
  - extend the trainer later from `first_tool_only_v1` to fuller sequence supervision once runtime receipts are dense enough
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
  - `benchmark-gated`

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
  - the same feature contract is now first-class in `DatapackSelectionFeatures`, and a bounded learned helper package can apply a capped reranking adjustment on top of the explicit prior instead of replacing it wholesale
  - the helper now also carries `DatapackSelectionContext` conditioning, so helper strength is no longer a flat scalar:
    - candidate-pool density
    - gap pressure
    - benchmark/execution-ready ratios
    - history density / cold-start pressure
    now gate the effective adjustment cap
  - `src/orchestrator/semantic_simulation.py` now merges ontology and local fallback datapacks through the same ranked pool, records a `selection_summary` into the live simulation result plus run log, and carries explicit helper-promotion state:
    - `disabled` for bootstrap
    - `auto` for shadow/helper-when-present
    - `required` when the learned helper is a runtime precondition
    - `required` now also insists on a benchmark-gated-ready helper package instead of accepting any package-shaped JSON
  - `scripts/train_datapack_selection_scorers.py` now exists and produces:
    - training dataset and dataset summary
    - model/config artifact
    - execution-precondition artifact
    - scorer package JSON
    - training summary
    - canonical runtime manifest/checkpoint registry when run under `RegalTrainingRunner`
  - `detect_semantic_gaps(...)` still infers missing scenario tags deterministically
- current consumers:
  - `src/orchestrator/semantic_simulation.py`
  - `scripts/train_datapack_selection_scorers.py`
- why it is a production problem:
  - before this pass, this determined which datapacks entered the simulation/training loop using little more than tag overlap plus ARH subtraction
  - that was too weak for a runtime surface that directly shapes synthetic agenda generation and future corpus construction
- recommended disposition:
  - keep the current bounded scored-selection layer live now
  - preserve the explicit prior feature contract as the bootstrap fallback and training target
  - promote the learned helper sequentially: `disabled` -> `auto` -> `required`
  - keep `auto` shadow-safe by clamping benchmark-unready packages
  - train the helper over `selection_summary` plus downstream outcome/counterfactual receipts before broadening the promoted/default path
- disposition tag:
  - `wired now`
  - `benchmark-gated`
  - `neuralized later`

### 8. `train_meta_transformer_synthetic.py`

- surface: meta-transformer trainer entrypoint plus runtime helper package consumption
- file/path: `scripts/train_meta_transformer_synthetic.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - now consumes the real meta-transformer substrate in `src/orchestrator/meta_transformer_training.py`:
    - exported `meta_transformer_runtime_dataset.json`
    - saved dataset JSON inputs
    - the real `MetaTransformerNet`
    - the existing batching/loss/eval helpers
  - synthetic generation remains available only as an explicit fallback input source when requested
  - the script now emits canonical runtime manifests/checkpoint registry/training summaries plus an explicit `meta_transformer_package.json` runtime artifact instead of writing an opaque random checkpoint under `results/`
  - benchmark readiness is now materially stricter:
    - runtime sample count alone is no longer enough
    - promotion now also requires bounded-ready density, semantic-grounded density, route-success density, and authority-success density from the runtime summary
  - `src/orchestrator/meta_transformer.py` now supports:
    - `helper_mode=disabled|auto|required`
    - bounded runtime loading of the trained package through `src/orchestrator/meta_transformer_runtime.py`
    - learned authority / policy-state / diffusion-conditioning / ontology-token influence with explicit `shadow_candidate` vs `promoted` stages
    - hard failure for `required` mode when the package is not benchmark-gated ready
- current consumers:
  - `scripts/export_semantic_runtime_learning_corpus.py`
  - `src/orchestrator/meta_transformer.py`
  - `src/policies/meta_advisor.py`
- why it is a production problem:
  - before this pass, the trainer existed and the runtime callout existed, but they were not connected:
    - training produced checkpoints
    - runtime kept using only the heuristic `MetaTransformer`
  - that created a high-distortion fake boundary because the lane could look architecturally complete while the trained model still had no bounded runtime effect
- recommended disposition:
  - keep the runtime/export dataset as the preferred source
  - keep synthetic generation explicit and benchmark-gated as a dev fallback only
  - keep the heuristic planner as the explicit prior
  - let the trained package influence runtime only through the bounded helper path until benchmark-gated promotion evidence is materially dense
- disposition tag:
  - `wired now`
  - `upgraded to heavyweight parity`
  - `benchmark-gated`

### 9. Teacher-runtime / rollout-labeler semantic sidecars

- surface: external teacher contracts/envelopes and rollout labeler sidecars
- file/path: `src/vla/rollout_labeler.py`, `src/vla/teacher_runtime.py`, `src/orchestrator/semantic_simulation.py`, `src/motor_backend/datapacks.py`, `src/ontology/datapack_registry.py`
- category: `advisory` / `sidecar` / `fallback`
- current behavior:
  - teacher contracts/action envelopes are explicit and persisted
  - VLA semantic evidence remains an external teacher sidecar rather than native truth
  - derived VLA-labeled datapacks now preserve:
    - teacher-runtime backend truth
    - vision-backbone truth
    - SceneTracks grounding truth
    - aggregated artifact refs
    - explicit execution preconditions
    - benchmark and future-training signals
    - bounded quality/novelty proxy scores
  - semantic simulation now enriches those labeled datapacks with semantic-fusion artifact refs and readiness instead of dropping the fusion outputs after labeling
  - datapack YAML save/load and ontology registration now preserve that metadata contract instead of collapsing back to description/tags only
- current consumers:
  - `src/orchestrator/semantic_simulation.py`
  - `src/orchestrator/semantic_policy.py`
  - datapack YAML / ontology resolution
- why it is a production problem:
  - before this pass, most teacher/runtime/grounding truth stopped at sidecars, so later selection and readiness logic saw a thin datapack object even though the labeler had already produced richer evidence
  - that made the vision lane look more disconnected and more heuristic than the selector/runtime contract actually needed it to be
- recommended disposition:
  - keep teacher outputs external/advisory
  - preserve teacher/vision/SceneTracks/fusion truth inside the datapack contract so downstream routing can use it materially
  - require explicit readiness and benchmark gating for promotion-ready labeled datapacks
  - do not collapse teacher truth into native runtime truth
- disposition tag:
  - `wired now`
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

1. Deepen orchestration supervision beyond `first_tool_only_v1` so the runtime-backed trainer learns fuller action sequences instead of only the first routing decision.
2. Continue meta-layer neuralization above the current helper path so future economic-WM and later meta-node-WM conditioning can supervise more than authority/conditioning outputs alone.
3. Extend the new gap-ranker helper contract across the remaining coverage-loop decisions, especially fill-path routing and later gen2sim validity evaluation, so the full synth agenda shares one honest promotion story.
