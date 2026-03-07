# Shadow Learning And Replay

## What Exists Now

The repository now has an additive replay-learning layer on top of the shadow economic control plane:

- canonical replay ingestion from shadow-control-plane artifacts
- canonical replay ingestion from existing `WorkcellEnv` episode logs
- deterministic replay dataset manifests plus episode/step/window JSONL views
- a CPU-safe behavior-cloning shadow policy with shared trunk, condition-vector routing, and modular skill heads
- learned shadow pricing, data-value, and regal-support models
- advisor interfaces that can run in `heuristic_only`, `learned_only`, `heuristic_learned_residual`, or `heuristic_learned_compare_only`
- advisory-only trainer/orchestrator sidecars for sampling priority, replay tags, collect-more-data, and retrain suggestions
- CPU smoke configs now and GPU/RunPod-oriented configs/readiness checks for later

## What Is Still Heuristic

The heuristic control plane remains the baseline constitution:

- `PricingSentinel` is still the primary auditable price oracle
- typed regal nodes still own integrity/plausibility/reward-safety/pricing-truth checks
- datapack credit and governance recommendations still originate from the explicit shadow accounting path

The learned models are shadow-only and currently do one of four things:

- compare against heuristics
- emit residual corrections
- provide uncertainty/confidence
- provide anomaly support for typed regal outputs

They do not silently replace the baseline path.

## Replay Ingestion

### Supported Sources

1. Shadow control plane run directories
   Required artifact: `shadow_episode_traces.json`
   This sidecar is now emitted additively by `scripts/run_shadow_econ_control_plane.py`.

2. Existing workcell episode logs
   Source shape: `EpisodeLog` payloads from `WorkcellEnv.get_episode_log()`

### Canonical Dataset Artifacts

`scripts/build_shadow_replay_dataset.py` writes:

- `manifest.json`
- `summary.json`
- `episodes.jsonl`
- `steps.jsonl`
- `windows.jsonl`

The manifest records:

- schema version
- source adapters
- replay counts
- vector dimensions
- skill modes
- config digest
- dataset digest

## CPU Usage Now

### 1. Generate a shadow run

```bash
python3 scripts/run_shadow_econ_control_plane.py \
  --output-dir artifacts/shadow_learning/shadow_run \
  --seed 42 \
  --episodes 4 \
  --objective-profile balanced_contract
```

### 2. Build a replay dataset

```bash
python3 scripts/build_shadow_replay_dataset.py \
  --shadow-run-dir artifacts/shadow_learning/shadow_run \
  --output-dir artifacts/shadow_learning/replay_dataset
```

### 3. Train the replay BC policy

```bash
python3 scripts/train_shadow_replay_policy.py \
  --dataset-dir artifacts/shadow_learning/replay_dataset \
  --config configs/replay_policy/cpu_smoke.yaml \
  --output-dir artifacts/shadow_learning/replay_policy
```

### 4. Evaluate the replay BC policy

```bash
python3 scripts/eval_shadow_replay_policy.py \
  --dataset-dir artifacts/shadow_learning/replay_dataset \
  --checkpoint artifacts/shadow_learning/replay_policy/replay_policy_best.pt \
  --output-dir artifacts/shadow_learning/replay_policy_eval
```

### 5. Train learned shadow pricing/value/regal-support models

```bash
python3 scripts/train_shadow_pricing_models.py \
  --dataset-dir artifacts/shadow_learning/replay_dataset \
  --config configs/shadow_models/cpu_smoke.yaml \
  --output-dir artifacts/shadow_learning/shadow_models
```

### 6. Run the advisory pass

```bash
python3 scripts/run_shadow_advisory_pass.py \
  --output-dir artifacts/shadow_learning/advisory \
  --replay-dataset-dir artifacts/shadow_learning/replay_dataset \
  --policy-mode heuristic_learned_compare_only \
  --pricing-mode heuristic_learned_residual \
  --data-value-mode heuristic_learned_compare_only \
  --regal-support-mode heuristic_learned_compare_only \
  --policy-checkpoint artifacts/shadow_learning/replay_policy/replay_policy_best.pt \
  --pricing-checkpoint artifacts/shadow_learning/shadow_models/pricing_delta.pt \
  --data-value-checkpoint artifacts/shadow_learning/shadow_models/data_value.pt \
  --regal-support-checkpoint artifacts/shadow_learning/shadow_models/regal_support.pt
```

### 7. Run the learning ablations

```bash
python3 scripts/run_shadow_learning_ablations.py \
  --output-dir artifacts/shadow_learning_ablations \
  --seed 42 \
  --episodes 4
```

## GPU / RunPod Preparation

GPU-ready configs are included but intentionally not exercised in tests:

- `configs/replay_policy/gpu_full.yaml`
- `configs/shadow_models/gpu_full.yaml`

Readiness check:

```bash
python3 scripts/check_gpu_training_readiness.py
```

What this checks now:

- torch visibility
- CUDA visibility if present
- config presence
- checkpoint/output path readiness

What still remains for future GPU runs:

- larger replay corpora
- real replay ingestion beyond deterministic shadow traces
- longer schedules and checkpoint resume patterns in actual cloud jobs
- RunPod image/job wiring

## Emitted Artifacts

Common artifact directories include:

- shadow run: objective/econ/pricing/regal/ledger artifacts plus `shadow_episode_traces.json`
- replay dataset: canonical episode/step/window JSONL files
- replay policy: `replay_policy_latest.pt`, `replay_policy_best.pt`, `train_metrics.jsonl`, `train_summary.json`
- shadow models: `pricing_delta.pt`, `data_value.pt`, `regal_support.pt`, `shadow_model_train_summary.json`
- advisory pass: `shadow_advisory.json`, `shadow_advisory.md`
- ablations: per-mode `summary.json`/`summary.md` plus `shadow_learning_ablation_comparison.json`

## Known Limitations

- replay sources are still deterministic shadow/workcell sources, not real robot or broad replay corpora
- BC policy predicts vectorized abstract actions derived from current replay schemas; it does not replace live SAC/PPO stacks
- pricing/data-value labels are shadow supervisory targets derived from the existing economic story, not ground-truth deployment invoices
- regal support is augmentative only and intentionally does not override typed regal decisions
- GPU configs are future-ready scaffolding, not a claim that large-scale training has been completed here
