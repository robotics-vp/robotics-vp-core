# Training Runtime Unification And Queue Dispatch

This PR starts with a hygiene fix: the branch visibility issue for `codex/shadow-learning-replay` was checked first, the missing remote branch was pushed to `origin`, and the remote tip was confirmed to match commit `bc37097` before any new code was added.

## What changed

The shadow replay-learning entrypoints are no longer semi-detached scripts. The following now run under the canonical regal-aware training runtime:

- `scripts/train_shadow_replay_policy.py`
- `scripts/train_shadow_pricing_models.py`
- `scripts/train_shadow_offline_rl.py`

Each wrapped job now emits:

- `training_runtime_manifest.json`
- `training_runtime_summary.md`
- `checkpoint_registry.json`
- queue/advisory artifacts
- receipt-label artifacts
- recurring promotion-evidence artifacts

The runtime manifest records:

- run id and seed
- config digest
- replay manifest digest
- objective profile snapshot
- promotion policy snapshot
- source-domain coverage
- receipt-label coverage
- artifact paths
- checkpoint registry references
- runtime status and failure reason

## Queue dispatch in the real training path

`src/orchestrator/queue_selection.py` is now wired into the replay-training dispatch path through `DataPackRLSampler.dispatch_queue()`.

The bounded modes are:

- `disabled`
- `compare_only`
- `advisory_reorder`
- `bounded_reweight`
- `promoted_gate_eligible`

Current conservative defaults:

- max upweight: `2.0x`
- max downweight: `0.5x`
- no slice removal unless explicitly enabled

The three shadow training scripts now:

1. build advisory output from replay data
2. emit `live_queue_selection.json`
3. run bounded queue dispatch over replay episodes
4. train on the selected/reordered episode set
5. record the dispatch artifact in the canonical training manifest

Legacy behavior remains available:

- `sample_batch()` is unchanged for existing sampler callers
- queue dispatch only affects jobs that opt into `dispatch_queue()`
- compare-only mode logs without changing replay selection

## Receipt labels and promotion evidence

`src/replay/receipt_ingest.py` adds a unified path for:

- synthetic shadow labels
- simulated deployment labels
- future real deployment labels

Current shadow jobs use deterministic synthetic receipts when real downstream labels are absent, but the schema is shared with future deployment truth:

- deployment outcomes
- pricing acceptance/rejection
- adaptation outcomes
- datapack contribution outcomes
- deployment-style receipts

These labels now feed:

- `build_shadow_advisory_output()`
- inferential-budget demo artifacts
- recurring regal promotion evidence reports
- training runtime manifests

Promotion evidence is now a first-class recurring artifact:

- `regal_promotion_eval.json`
- `regal_promotion_eval.md`
- `promotion_evidence_<node>.json`

The report includes:

- coverage
- calibration
- baseline agreement
- disagreement slices
- false positives / false negatives
- downstream usefulness
- current maturity stage
- promotion recommendation

## SAC integration stance

SAC remains the main online RL backbone.

This PR does not replace SAC and does not promote PPO.

What is implemented now:

- `src/rl/sac_contract_aware_adapter.py`
- optional sidecar multi-head critic training for objective/econ/scalar outputs
- additive hook from `SACAgent.update()` into the sidecar
- checkpoint save/load support for the sidecar
- `configs/sac/contract_aware_smoke.yaml`

What is not changed:

- SAC actor optimization logic
- baseline scalar critic path
- reward constitution

The contract-aware path is flag-gated and optional. If the adapter is off, SAC behaves as before.

## Running it

Build or reuse replay data:

```bash
python3 scripts/build_shadow_replay_dataset.py \
  --output-dir artifacts/replay_dataset_unified \
  --generate-shadow-run \
  --episodes 3 \
  --seed 42
```

Train BC with the canonical runtime:

```bash
python3 scripts/train_shadow_replay_policy.py \
  --dataset-dir artifacts/replay_dataset_unified \
  --config configs/replay_policy/cpu_smoke.yaml \
  --output-dir artifacts/train_shadow_replay_policy_unified
```

Train pricing/value/support with the canonical runtime:

```bash
python3 scripts/train_shadow_pricing_models.py \
  --dataset-dir artifacts/replay_dataset_unified \
  --config configs/shadow_models/cpu_smoke.yaml \
  --output-dir artifacts/train_shadow_pricing_models_unified
```

Train the offline bridge with the canonical runtime:

```bash
python3 scripts/train_shadow_offline_rl.py \
  --dataset-dir artifacts/replay_dataset_unified \
  --config configs/offline_rl/cpu_smoke.yaml \
  --output-dir artifacts/train_shadow_offline_rl_unified
```

Evaluate recurring promotion evidence directly:

```bash
python3 scripts/eval_regal_promotion.py \
  --replay-dataset-dir artifacts/replay_dataset_unified \
  --output-dir artifacts/regal_promotion_eval_unified
```

## Known limits

- The default downstream labels are still synthetic unless explicit receipt labels are provided.
- Queue dispatch is bounded and reversible; it is not a global hard gate.
- The new SAC contract-aware path is a sidecar integration, not a full replacement of the scalar critic.
- Learned advisors and regals still cannot become hard gates without explicit promotion-policy coverage/calibration criteria.
