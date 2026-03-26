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
| 3 | SceneTracks passthrough/non-stub truthiness in bootstrap + replay ingest | `fallback` | High | High | No | Backlogged in `scripts/RUNTIME_WIRING_BACKLOG.json` |
| 4 | Shadow advisory replay sampling and queue reweighting | `heuristic` / `advisory` | High | High | No | Explicitly bounded, still heuristic |
| 5 | `train_vla_recap_offline.py` lightweight trainer path | `lightweight_trainer_gap` | High | Medium-high | No | Remains in training backlog |
| 6 | `train_orchestration_transformer.py` heuristic-teacher trainer | `heuristic` / `lightweight_trainer_gap` | High | Medium-high | No | Wrapped, but target remains heuristic |
| 7 | Semantic datapack/scenario selection in `semantic_policy.py` | `heuristic` | Medium-high | Medium-high | No | Backlogged in `scripts/RUNTIME_WIRING_BACKLOG.json` |
| 8 | `train_meta_transformer_synthetic.py` random-data trainer | `stub` / `lightweight_trainer_gap` | Medium | High | No | Remains explicit placeholder |
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
  - `_scene_tracks_rollout_metadata(...)` in `src/replay/ingest.py` treats `backend in {"real", "passthrough", "auto"}` as `scene_tracks_non_stub`
  - `scripts/bootstrap_semantic_workcell_loop.py` writes `scene_tracks_non_stub=true` for `overall_mode in {"real", "passthrough"}`
  - later benchmark gating is stricter, but early metadata/readiness summaries can still overstate grounding quality
- current consumers:
  - replay import/readiness code
  - bootstrap semantic workcell loop metadata
  - downstream replay preconditions and benchmark gating
- why it is a production problem:
  - training/readiness summaries can look more production-ready than the benchmark gate would actually allow
  - this distorts corpus admission and operational dashboards
- recommended disposition:
  - tighten early truth semantics so passthrough never appears equivalent to real grounded SceneTracks
  - keep passthrough as an explicit fallback only
- disposition tag:
  - `remain explicit fallback`
  - `benchmark-gated`

### 4. Shadow advisory replay sampling and queue reweighting

- surface: heuristic scoring for replay sampling, queue tags, and bounded queue influence
- file/path: `src/orchestrator/shadow_advisory.py`, `src/rl/econ_regal_sampling.py`, `src/orchestrator/queue_selection.py`
- category: `advisory`
- current behavior:
  - `build_shadow_advisory_output(...)` compiles policy/pricing/data-value/regal support into advisory episode rows
  - `recommend_sampling(...)` remains rule-based over coverage gap, uncertainty, datapack value, provenance, and pricing/regal flags
  - `apply_live_queue_selection(...)` can reweight or drop episodes in training-time selection
- current consumers:
  - `scripts/train_shadow_replay_policy.py`
  - `scripts/train_shadow_offline_rl.py`
  - `scripts/train_sac_with_ontology_logging.py`
  - `scripts/run_shadow_advisory_pass.py`
- why it is a production problem:
  - this path already affects actual replay/training selection, but the score is still mostly rule-based
  - it is a live control-plane seam, not just documentation
- recommended disposition:
  - keep the bounded queue lane wired
  - replace the scoring internals with replay-backed learned runtime scorers once coverage is broad enough
- disposition tag:
  - `wired now`
  - `neuralized later`

### 5. `train_vla_recap_offline.py`

- surface: offline RECAP VLA heads trainer
- file/path: `scripts/train_vla_recap_offline.py`
- category: `lightweight_trainer_gap`
- current behavior:
  - trains CPU heads over JSONL datasets and writes local metrics/checkpoints
  - does not use `RegalTrainingRunner` or emit canonical runtime manifests/checkpoint registry/receipt artifacts
- current consumers:
  - `scripts/smoke_test_vla_recap_training.py`
  - `scripts/smoke_test_recap_inference.py`
  - `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json`
- why it is a production problem:
  - this is a real training entrypoint, but its outputs are not contract-parity with heavier trainers
  - later promotion/readiness tooling cannot reason about it in the same way
- recommended disposition:
  - migrate to `RegalTrainingRunner`
  - carry recap dataset summary, receipt coverage, and checkpoint registry into the canonical runtime envelope
- disposition tag:
  - `upgraded to heavyweight parity`
  - `training backlog`

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
  - `select_datapacks_for_intent(...)` scores candidates by tag overlap, objective hints, and ARH penalties
  - `detect_semantic_gaps(...)` infers missing scenarios from set differences over tags
- current consumers:
  - `src/orchestrator/semantic_simulation.py`
- why it is a production problem:
  - this determines which datapacks and scenarios enter the simulation/training loop
  - selection remains rule-based even as other runtime surfaces are becoming packet/evidence-native
- recommended disposition:
  - replace tag-match selection with bounded learned routing over the same packet/evidence shape
- disposition tag:
  - `neuralized later`
  - `runtime backlog`

### 8. `train_meta_transformer_synthetic.py`

- surface: synthetic MetaTransformer pretraining stub
- file/path: `scripts/train_meta_transformer_synthetic.py`
- category: `stub`
- current behavior:
  - trains a tiny MLP on random features and random labels
  - writes a checkpoint under `results/` with no runtime manifest or real dataset provenance
- current consumers:
  - no production consumer path found in code search
- why it is a production problem:
  - it creates a checkpoint-shaped artifact with almost no relationship to the real semantic/runtime corpus
  - because it is a script-shaped trainer, it can be mistaken for a real pretraining path
- recommended disposition:
  - keep it quarantined as an explicit placeholder or remove it from serious training bundles
  - if retained, migrate only once it consumes the real semantic/runtime corpus
- disposition tag:
  - `remain explicit fallback`
  - `training backlog`

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
  - training eligibility is already explicit and strict inside the runner
  - callers can still choose fallback/passthrough modes
- current consumers:
  - `scripts/run_scene_tracks.py`
  - `scripts/bootstrap_semantic_workcell_loop.py`
  - SceneTracks tests
- why it is a production problem:
  - the runner itself is honest, but the broader stack still has places where fallback output is interpreted too generously
- recommended disposition:
  - keep stub/passthrough support only as explicit fallback
  - align every upstream/downstream metadata consumer with the runner’s stricter truth semantics
- disposition tag:
  - `remain explicit fallback`
  - `benchmark-gated`

## Remaining Top Follow-Ons

1. Remove permissive passthrough-as-non-stub truthiness from bootstrap/replay metadata.
2. Migrate `train_vla_recap_offline.py` to the canonical training runtime.
3. Replace bounded heuristic routing inside shadow advisory and semantic policy selection with learned runtime scorers once replay coverage is broader.
4. Either replace or quarantine `train_meta_transformer_synthetic.py` so it cannot be mistaken for a real trainer.
