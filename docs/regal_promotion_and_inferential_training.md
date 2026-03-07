# Regal Promotion And Inferential Training

This layer promotes the economic and regal stack into the adaptation loop without turning it into an instant hard-sovereign control plane.

## What is implemented now

- `RegalPromotionPolicy` with explicit maturity stages:
  - `compare_only`
  - `advisory`
  - `budget_gate`
  - `narrow_hard_gate`
- Promotion and demotion criteria driven by:
  - replay coverage
  - downstream labels
  - deployment receipt count
  - calibration error
  - baseline agreement
  - monotonicity and sign consistency
  - false-positive / false-negative tolerances
  - drift and residual-gain checks
- A contract-aware critic bundle:
  - objective-vector head
  - econ head
  - scalar compiled head explicitly downstream of contract compile logic
- A conservative offline RL bridge:
  - TD3+BC-style shadow path
  - behavior cloning remains the simpler fallback
  - outputs stay optional and additive
- An inferential training budget gate:
  - `adapt_now`
  - `collect_more_data`
  - `no_op`
  - `require_review`
- Advisory replay sampling and a live queue-selection shim
- Deployment label and receipt schemas ready for future live robot receipts

## Promotion logic

The regals do not become hard gates by decree. They earn authority in stages.

### Stage 0: Compare-only

- Outputs are logged and compared.
- No direct control effect.

### Stage 1: Advisory

- Outputs can influence:
  - replay tags
  - sampling priority
  - collect-more-data flags
  - retrain recommendations
  - pricing confidence annotations

### Stage 2: Budget gate

- Econ and regal signals can admit or deny adaptation spend.
- This is compute and capital allocation.
- It does not rewrite the reward function.

### Stage 3: Narrow hard gate

- Reserved for explicit, well-calibrated surfaces only.
- Current examples:
  - pricing publication suppression
  - adaptation denial on explicit integrity failure
  - datapack credit denial on broken provenance

## SAC and PPO guidance

- SAC remains the main online RL backbone.
- PPO is not promoted to system center by this PR.
- The main rethink is the critic and contract interface, not optimizer churn.
- Scalar reward remains a compiled control view, not the ontology of the stack.

## Contract-aware critic

The new critic bundle predicts structured quantities first:

- objective components
- econ-facing quantities
- compiled scalar value only after explicit contract compile

This preserves the repository direction:

- no premature scalarization
- objective and econ truth survive deeper into training
- actor improvement can still consume a scalar target for compatibility

## Inferential training gate

The gate evaluates:

- expected value gain
- compute cost
- risk cost
- uncertainty and OOD
- data quality and provenance quality
- regal failures and maturity stages

The result is a capital-allocation decision, not a reward rewrite.

## Replay and advisory wiring

Replay datasets now carry:

- explicit provenance fields
- schema compatibility summaries
- shadow artifact fingerprints
- richer adapters including rollout-capture bundles

Advisory outputs now emit:

- replay sampling recommendations
- replay actions (`upweight`, `downweight`, `holdout`, `collect_more_like_this`)
- inferential budget decisions
- live queue-selection entries

## How to run

Build a replay dataset:

```bash
python3 scripts/build_shadow_replay_dataset.py \
  --output-dir artifacts/replay_dataset_stage3 \
  --generate-shadow-run \
  --episodes 3 \
  --seed 42
```

Train the offline RL shadow bridge:

```bash
python3 scripts/train_shadow_offline_rl.py \
  --dataset-dir artifacts/replay_dataset_stage3 \
  --config configs/offline_rl/cpu_smoke.yaml \
  --output-dir artifacts/offline_rl_stage3
```

Evaluate regal promotion:

```bash
python3 scripts/eval_regal_promotion.py \
  --replay-dataset-dir artifacts/replay_dataset_stage3 \
  --promotion-policy configs/regality/promotion_default.yaml \
  --output-dir artifacts/regal_promotion_eval
```

Run the inferential budget gate demo:

```bash
python3 scripts/run_inferential_budget_gate_demo.py \
  --output-dir artifacts/inferential_budget_gate_demo \
  --generate-shadow-run \
  --episodes 3 \
  --seed 42
```

Run full verification:

```bash
python3 scripts/run_full_repo_verification.py
```

## GPU and RunPod readiness

Use:

- `configs/replay_policy/gpu_full.yaml`
- `configs/shadow_models/gpu_full.yaml`
- `configs/offline_rl/gpu_full.yaml`

And check readiness with:

```bash
python3 scripts/check_gpu_training_readiness.py
```

This validates config presence and environment visibility without requiring a GPU to be attached.

## Known limitations

- Learned advisors remain shadow, compare-only, or residual unless promotion criteria are satisfied.
- Promotion evaluation still relies on shadow labels and proxy outcomes.
- Offline RL is a conservative scaffold, not a replacement for the online SAC path.
- No real deployment receipts exist yet; receipt and label schemas are shadow-ready and future-facing.
- The live queue-selection shim is additive and advisory. It does not mutate legacy queue logic unless explicitly consumed.
