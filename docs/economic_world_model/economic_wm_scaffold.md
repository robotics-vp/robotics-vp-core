# Economic WM Scaffold

Date: 2026-05-21

## Purpose

This is the first native Economic World Model artifact layer. It turns the entry preflight into deterministic, typed Economic WM surfaces without training a model or giving the Economic WM authority over reward math.

The scaffold emits three artifacts:

- `economic_state_v1.json`: resource reservoirs, flow fields, dissipation fields, bottlenecks, and opportunity fields derived from lower-WM readiness receipts.
- `allocation_envelope_v1.json`: scaffold-only allowed/denied action and budget envelope for downstream work.
- `economic_wm_scaffold_report_v1.json`: receipt tying the state, envelope, and entry preflight together.

## Executable path

Run:

```bash
python3 scripts/economic_world_model/build_economic_wm_scaffold.py \
  --output-dir artifacts/economic_world_model/economic_wm_scaffold
```

If no preflight report is supplied, the script runs `economic_wm_entry_preflight.py`, which in turn runs the Stage-1 bridge-readiness sweep. To build from an existing preflight report:

```bash
python3 scripts/economic_world_model/build_economic_wm_scaffold.py \
  --entry-preflight-report artifacts/economic_world_model/economic_wm_entry_preflight/economic_wm_entry_preflight_report.json \
  --output-dir artifacts/economic_world_model/economic_wm_scaffold
```

## Current boundary

The scaffold is allowed to:

- define Economic WM state and allocation-envelope contracts
- expose benchmark-ready vs shadow-only replay inventory
- make GPU/provider/promotion blockers explicit
- feed later replay-feature extraction, training-row materialization, and shadow allocation evals

The scaffold is not allowed to:

- train an Economic WM
- promote a model
- mutate frozen Phase B reward, trust-net, `w_econ`, or lambda-controller math
- treat external provider or teacher outputs as native truth

## Current local posture

The current preflight-derived state is `scaffold_ready_training_blocked`:

- `ready_for_scaffold=true`
- `ready_for_training=false`
- `promotion_eligible=false`
- `authority_class=scaffold_only`
- `reward_math_mutation=false`

Training remains blocked until GPU/provider bring-up, non-stub teacher/runtime evidence, and promotion-grade benchmark evidence exist.
