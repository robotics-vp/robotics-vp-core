# Online SAC Econ / Regal Integration

This change moves the econ / regal layer from advisory-only sidecars into the actual online SAC sampling and evidence loop, while preserving the repo contract:

- SAC remains the online backbone.
- PPO is not promoted.
- Learned advisors remain bounded and evidence-gated.
- No hard deny is applied by default.
- Objective / econ structure still survives until the compile boundary.

## What changed

The online `scripts/train_sac_with_ontology_logging.py` flow now:

1. runs real SAC updates instead of only logging transitions
2. records per-episode online rollout logs under `online_episode_logs/`
3. emits `online_episode_receipts.jsonl` with training-run receipt labels
4. rebuilds a replay dataset from those episode logs
5. feeds receipt labels into:
   - `build_shadow_advisory_output()`
   - inferential budget decisions
   - recurring promotion evidence reports
6. applies bounded queue-dispatch decisions to the live SAC replay sampler
7. optionally runs the contract-aware critic in live loss mode

## Queue dispatch in the online SAC sampler

`src/orchestrator/queue_selection.py` still owns the bounded dispatch policy, but the result now reaches the real online SAC replay sampler through `src/rl/sac.py`.

The supported influence modes are:

- `log_only`
- `advisory_reorder`
- `bounded_reweight`
- `promoted_gate_eligible`

Legacy alias:

- `compare_only` is still accepted and normalizes to `log_only`

Bounded defaults:

- max upweight: `2.0`
- max downweight: `0.5`
- no hard removal unless both config and promotion stage allow it

The live replay sampler now stores episode-level dispatch metadata on transitions and applies the bounded reweight factor during actual replay sampling. Each sampling artifact includes:

- original queue ordering
- adjusted queue ordering
- per-episode reweight factors
- reasons / evidence
- promotion stage
- influence source (`heuristic`, `learned`, or `hybrid`)

Artifacts:

- `live_queue_selection.json`
- `queue_dispatch_comparison.json`
- `queue_dispatch_history.jsonl`
- `online_sampling/online_sac_sampling.jsonl`

## Receipt labels from real rollout / training artifacts

`src/replay/receipt_ingest.py` now ingests labels from:

- synthetic shadow bundles
- simulated rollout bundles
- online training runs
- future real deployment records

Supported source domains:

- `synthetic_shadow`
- `sim_rollout`
- `training_run`
- `future_real_deployment`

Online SAC training emits `online_episode_receipts.jsonl`, and the ingest layer can also rebuild labels from:

- `training_runtime_manifest.json`
- `online_replay_dataset/`
- `online_episode_logs/`
- rollout-capture bundles

The unified receipt schema now carries:

- realized task success / failure
- realized reward
- realized vs predicted value
- adaptation benefit vs predicted benefit
- incident / failure / risk events
- pricing acceptance or rejection
- datapack usefulness realized vs predicted
- human review and override labels

## Contract-aware critic live mode

`src/rl/sac_contract_aware_adapter.py` still supports sidecar training, but it now also supports:

- `mode: live_loss`

In live mode the adapter:

- predicts objective vectors
- predicts econ vectors
- predicts compiled scalar values
- computes structured losses against the live SAC batch
- adds critic-alignment loss against the scalar SAC critic
- logs calibration and consistency metrics

The legacy scalar SAC critic is still the default path. If the adapter is disabled, behavior remains unchanged.

Configs:

- `configs/sac/contract_aware_smoke.yaml`
- `configs/sac/contract_aware_full.yaml`

Artifacts:

- `contract_aware/sac_contract_aware_metrics.jsonl`
- `contract_aware/sac_contract_aware_predictions.jsonl`

## Advisory vs operational influence

Still advisory:

- learned advisor authority beyond its promoted stage
- any hard deny / slice removal without explicit enablement
- actor optimization objective selection

Now operational, but still bounded:

- replay sampling order and weights in online SAC
- receipt-backed inferential budget evidence
- promotion evidence from actual online run outcomes
- contract-aware critic regularization when `live_loss` is enabled

## Smoke verification

Compile and targeted tests:

```bash
python3 -m compileall src scripts/train_sac_with_ontology_logging.py
pytest tests/test_online_queue_dispatch_integration.py \
  tests/test_training_run_receipt_ingest.py \
  tests/test_sac_contract_aware_live_mode.py \
  tests/test_online_promotion_reporting.py -v
```

Short online SAC run:

```bash
python3 scripts/train_sac_with_ontology_logging.py \
  --episodes 3 \
  --output-dir artifacts/online_sac_smoke \
  --queue-selection-mode bounded_reweight \
  --contract-aware-config configs/sac/contract_aware_smoke.yaml \
  --contract-aware-mode live_loss
```

Expected online artifacts:

- `online_episode_logs/`
- `online_episode_receipts.jsonl`
- `online_replay_dataset/`
- `receipt_labels/`
- `online_shadow_advisory.json`
- `regal_promotion_eval.json`
- `queue_dispatch_comparison.json`
- `online_sampling/online_sac_sampling.jsonl`
