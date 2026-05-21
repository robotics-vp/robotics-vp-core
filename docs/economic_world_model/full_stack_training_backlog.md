# Full-Stack Training Backlog

## Purpose

This document turns the current full-stack training situation into one honest backlog:

- rank real training lanes by production importance and dependency
- separate "runnable now" from "exists in code but not yet meaningful"
- define the recurring Runpod bundles that make sense every few weeks
- keep adjacent implementation questions visible so the stack does not jump to the next canonical world model too early

The central constraint is still the same for most learned lanes: the repo is more data-limited than GPU-limited right now. The important exception is the canonical workcell refresh/replay lane, which now assumes real SAM3D grounding rather than passthrough-only refreshes.

## Current State Snapshot

Observed in the workspace on 2026-03-26:

- `artifacts/semantic_grounding/workcell_bootstrap_20260324/replay_dataset/summary.json`
  - 2 episodes
  - 10 steps
  - 6 windows
- `artifacts/shadow_learning_smoke/replay_dataset/summary.json`
  - 3 episodes
  - 10 steps
  - 6 windows
- `artifacts/semantic_grounding/workcell_bootstrap_20260324/semantic_runtime_corpus/semantic_runtime_learning_summary.json`
  - 2 runtime rows
- `data/physics_zv_rollouts.npz`
  - 50 real episodes
- `data/synthetic_zv_rollouts.npz`
  - 50 synthetic episodes
- `data/datapacks/`
  - empty
- `results/stage1_pipeline`
  - missing
- `results/stage2_preview`
  - missing
- `results/sima2_stress`
  - missing
- `data/fill_outcomes.jsonl`
  - missing
- non-test RECAP datasets
  - missing

Consequence:

- A100 spend is still not the main blocker for most trainers.
- But the canonical workcell refresh/replay lane now assumes real SAM3D, so that first recurring corpus-refresh job should be treated as a Linux/NVIDIA A100 run rather than a CPU-only local bootstrap.
- Local passthrough refreshes still help plumbing and replay-shape validation, but they do not count as benchmark-grade replay accumulation.

## Ranked Backlog

### 1. Workcell semantic data refresh

Status:
- ready now on a real SAM3D host
- local passthrough refreshes remain dev-only

Primary entrypoint:
- `scripts/bootstrap_semantic_workcell_loop.py`

Why first:
- when run with real grounding, it increases replay density, semantic runtime rows, and coverage-loop artifacts together
- it directly feeds the higher-value learned lanes instead of training another tiny model on a smoke corpus

Internal data sources:
- repo-local workcell rollout generation
- ontology and replay artifacts already under `artifacts/` and `data/`

External data sources:
- gated SAM 3D Objects checkpoints from Hugging Face
- gated SAM 3D Body checkpoints from Hugging Face
- a Linux/NVIDIA host or prebuilt image that can actually run the real SAM3D stack

Runpod posture:
- safe to automate first
- treat 1x A100 80GB as required for the canonical recurring refresh/replay lane because real SAM3D grounding is part of the useful-data definition here
- local `backend-policy=auto|passthrough` refreshes remain useful for plumbing and corpus-shape checks only

### 2. Semantic runtime scorer training

Status:
- code exists
- not ready on current data density

Primary entrypoints:
- `scripts/export_semantic_runtime_learning_corpus.py`
- `scripts/train_semantic_runtime_scorers.py`

Dependency shape:
- low model dependency
- depends mainly on a real replay corpus

Internal data sources:
- canonical replay datasets built from shadow runs, online SAC, or bootstrap workcell loops

External data sources:
- DROID replay conversions
- future benchmark-eligible real grounding replays

Why high priority:
- this is the cleanest learned-control-plane lane after corpus refresh
- it can improve bounded reranking and authority calibration without touching frozen Phase B math

### 3. Semantic feedback adapters and WM refiner

Status:
- code exists
- blocked by sparse coverage-loop artifacts

Primary entrypoints:
- `scripts/train_semantic_feedback_adapters.py`
- `scripts/train_semantic_wm_refiner.py`

Dependency shape:
- low model dependency
- medium data dependency

Internal data sources:
- repeated `coverage_graph.json` outputs from coverage-loop runs

External data sources:
- none required initially

Why high priority:
- this is the current learned successor lane around semantic coverage, correction, and graph mutation
- it matters more than another isolated module trainer

### 4. Shadow pricing, replay BC, and offline RL

Status:
- code exists
- materially blocked by replay density

Primary entrypoints:
- `scripts/train_shadow_pricing_models.py`
- `scripts/train_shadow_replay_policy.py`
- `scripts/train_shadow_offline_rl.py`

Dependency shape:
- depends on canonical replay plus receipt-label coverage
- no additional frozen-model dependency

Internal data sources:
- replay datasets from shadow runs and online SAC

External data sources:
- none required for the first honest pass

Why below the scorer/refiner lanes:
- these become more meaningful once replay density and queue/receipt signals are larger than the current 2-3 episode datasets

### 5. VLA RECAP heads

Status:
- code exists
- blocked by missing RECAP corpus

Primary entrypoints:
- `scripts/build_vla_recap_dataset.py`
- `scripts/train_vla_recap_offline.py`

Internal data sources:
- ontology-backed episode history

External data sources:
- DROID-derived recap rows
- Bridge/OpenVLA-compatible recap exports

### 6. Perception neuralization lane

Status:
- partially real, partially still synthetic/stub-backed
- keep manual-only for now

Primary entrypoints:
- `scripts/train_vision_backbone_real.py`
- `scripts/train_sima2_segmenter.py`

Dependency shape:
- depends on missing stage roots and richer real grounding data

Internal data sources:
- `results/stage1_pipeline`
- `results/stage2_preview`
- `results/sima2_stress`

External data sources:
- real grounding frames and labels from non-stub ingestion paths

Planned fine-tuning split once those inputs are real:
- V-JEPA 2 should be tracked explicitly as the temporal perception/grounding lane for scene persistence, event continuity, and action-conditioned visual state
- prefer upstream `facebookresearch/vjepa2` bring-up and wrapper contracts over a local from-scratch reimplementation when the goal is honest progress

### 7. Gap ranker and fill-path policy

Status:
- code exists
- blocked by missing `fill_outcomes` store

Primary entrypoints:
- `scripts/train_gap_ranker.py`
- `scripts/train_fill_path_policy.py`

Internal data sources:
- `data/fill_outcomes.jsonl` from repeated coverage-loop outcomes

External data sources:
- none

### 8. Deferred future world-model lanes

Status:
- Economic WM scaffold entry is now locally preflighted (`ready_for_scaffold=true`)
- training remains backlog-only until GPU/provider and promotion-grade evidence exist

Tracked in:
- `scripts/TRAINING_MIGRATION_BACKLOG.json`

Examples:
- `train_governed_video_world_model.py`
- `train_economic_world_model_v0.py`
- `train_semantic_gap_conditioned_world_models.py`
- `train_vjepa2_sim_synth_predictor.py`
- `train_vjepa2_perception_grounding.py`

These should stay deferred until:

- non-stub SceneTracks and teacher-runtime ingestion are real
- reconstruction/calibration sidecars are richer
- governed supervision bundles are dense enough to justify long A100 runs
- the split V-JEPA 2 lanes have real stage outputs, action/context packaging, and benchmark gates instead of only narrative placement

## Explicit Non-Autonomous Lanes

Do not put these into the recurring Runpod loop unless explicitly authorized:

- `scripts/train_stable_world_model.py`
- `scripts/train_trust_net.py`
- `scripts/train_w_econ_lattice.py`
- `scripts/train_w_econ_lattice_from_J.py`
- `scripts/train_synth_lambda_controller.py`

Reason:

- repo guidance keeps the stable Phase B baseline, trust-net, `w_econ`, and lambda controller math frozen unless directly authorized

## Runpod Bundle Policy

The checked-in bundle config is:

- `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json`

The bundle readiness and launch scripts are:

- `scripts/runpod/assess_full_stack_training.py`
- `scripts/runpod/execute_training_bundle.py`
- `scripts/runpod/launch_training_bundle.py`

Current intended cadence:

1. run `workcell_data_refresh`
2. re-assess corpus density
3. only then allow `semantic_runtime_training`
4. only after repeated coverage artifacts exist, allow `semantic_feedback_training`
5. only after replay datasets are materially larger, allow `shadow_model_training`

That is the honest order. The current repo state should not auto-jump directly to shadow RL or future governed-video training.

## Weekly A100 Program From September 1, 2026

Assumed operating model:

- starting September 1, 2026, use A100-backed runs every week rather than occasional opportunistic training bursts
- execute work sub-module by sub-module inside each WM
- each weekly slot should be treated as a three-stage ladder:
  - loop runs and provider bring-up
  - training runs on the receipts/corpora produced by those loops
  - fine-tuning only for the sub-modules whose loop and training evidence already pass the relevant gates

Why this matters:

- it prevents wasting A100 time on fine-tuning lanes whose loop surfaces are still mostly fake
- it keeps the lower-WM program grounded in receipts rather than in architecture optimism
- it matches the stated goal that, by July 2027, the honest blockers should be data, GPUs, calibration, assets, and benchmarks rather than missing plumbing

Recommended weekly rotation for the first training season:

1. Sim / synth / physics WM sub-modules
2. Perception / grounding WM sub-modules
3. Embodiment / actuation WM sub-modules
4. Economic-WM consolidation lanes over the trained lower-WM outputs
5. Local meta-node neuralization and later meta-node superposition / control lanes over the stabilized lower-WM and economic-WM outputs

Within each WM, the order should stay:

1. loop-run/provider-truth sub-modules
2. corpus export and receipt quality
3. bounded helper/predictive training
4. fine-tuning and promotion candidates

The weekly A100 budget should not be spent on later transport or meta-node work until the lower-WM weekly ladders are genuinely producing benchmark-shaped receipts.

## Suggested Readiness Thresholds

The current thresholds in `scripts/runpod/FULL_STACK_TRAINING_BUNDLES.json` are intentionally conservative:

- semantic runtime training
  - at least 100 replay episodes
  - at least 5000 replay steps
- semantic feedback training
  - at least 20 coverage graphs
- shadow model training
  - at least 200 replay episodes
  - at least 50000 replay steps
- VLA recap training
  - at least 1000 RECAP rows
- gap/fill training
  - at least 1000 fill-outcome rows

If those thresholds are not met, the right move is to spend the next cycle on data accumulation, not on pretending the next model checkpoint is meaningful.

## Runpod Operating Model

The intended autonomous model is:

1. build or point Runpod at an image that already contains the desired repo state
2. attach a persistent `/workspace` volume or a network volume
3. run `scripts/runpod/launch_training_bundle.py`
4. let the pod execute `scripts/runpod/execute_training_bundle.py`
5. write receipts under `artifacts/runpod_training/<run-id>/`
6. tear the pod down at the end

Important honesty rules:

- if the image does not contain the desired code, the launcher does not fix that for you
- if the image does not contain real SAM3D repos plus authenticated checkpoints, `workcell_data_refresh` is not actually ready even if the bootstrap script itself exists
- if `runpodctl` is missing locally, launch will fail
- if a network volume is attached, the pod should be removed rather than merely stopped
- if readiness gates fail and `--force` is not passed, launch must stop before spend begins

## Adjacent Questions For Other Implementation Windows

These are not solved by the Runpod scaffolding, but they are the right adjacent priorities.

### 1. Heuristic, advisory-only, and sidecar inventory

Before the next canonical world model:

- inventory every critical `heuristic`, `advisory`, and `sidecar` surface
- rank them by production importance
- wire the top ones into actual runtime, training, and reward loops
- only then choose the next canonical WM

Why:

- too much critical functionality still survives as "named but non-authoritative" infrastructure
- if those surfaces stay detached, another canonical WM just becomes another advisory blob

### 2. World-model topology

The topology that currently makes sense is:

1. perception and grounding WM
2. embodiment and actuation WM
3. sim, synth, and physics WM
4. economic WM over those lower canonical states
5. meta-node superposition and control WM above the economic WM

This does make architectural sense if each lower WM is canonical state rather than another sidecar.

Current recommendation:

- do not pick a separate vision WM as the next major tranche
- the stronger next canonical candidate is a sim, synth, and physics WM
- the production gap is more in sim agenda compilation and gen2sim than in pure perception representation

### 3. Sim, diffusion, gen2sim, and synth

The most promising next self-improvement surface remains:

- `src/orchestrator/diffusion_requests.py`
- `src/orchestrator/semantic_simulation.py`
- `src/envs/physics/`
- the broader synth and local-branch pipeline

That is the lane where the stack stops being only good control-plane infrastructure and starts acting like a real self-improvement machine.

### 4. Stranded modules

Still likely stranded or under-wired:

- `src/vla/semantic_vla.py`
- `src/vla/recap_dataset_builder.py`
- `src/envs/physics/isaac_backend.py`
- parts of the vision backbone and synth stub paths

### 5. Meta-node neuralization

Meta-nodes are more real than before, but they are not yet sufficiently neuralized:

- they are bounded control and routing surfaces with learned layers around them
- they are not yet fully learned geometric or cybernetic objects in their own right

That means the right sequence is still:

1. wire critical heuristic, advisory, and sidecar surfaces into production loops
2. deepen sim and synth functionality
3. then come back to meta-node superposition training with a stronger lower stack

## Recommended Commands

Assess the current workspace:

```bash
python3 scripts/runpod/assess_full_stack_training.py
```

Dry-run the next ready bundle:

```bash
python3 scripts/runpod/launch_training_bundle.py \
  --bundle auto \
  --image-name robotics-vp-core:latest \
  --gpu-type "A100 SXM 80GB" \
  --dry-run
```

Launch a specific ready bundle:

```bash
python3 scripts/runpod/launch_training_bundle.py \
  --bundle workcell_data_refresh \
  --image-name robotics-vp-core:latest \
  --gpu-type "A100 SXM 80GB"
```

## Bottom Line

Today, the right recurring autonomous Runpod job is:

- run an A100-backed real-SAM3D workcell refresh/replay pass
- regenerate replay and coverage artifacts
- wait until those corpora are honestly larger
- only then spend A100 time on scorer, refiner, and shadow-model checkpoints

Anything more aggressive than that would be theater rather than an honest training program.
