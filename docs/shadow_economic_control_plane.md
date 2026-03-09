# Shadow Economic Control Plane

The shadow economic control plane is the first additive proof that this repo can run the intended economics-first loop end to end without a physical robot.

Current implemented chain:

`workcell sim episode -> ObjectiveTensor runtime builder -> ConstraintSet -> ObjectiveCompiler -> ObjectiveEconFunctor -> PricingSentinel -> ValueLedger -> shadow regal nodes -> ontology/datapack shadow updates`

This is deliberately a shadow path. It does not modify the baseline SAC/PPO reward path, does not touch the stable frozen Phase B math, and does not silently change legacy scripts when the new mode is off.

## Why It Exists

The repo telos is not “train a policy, then report business metrics later.” The target stack keeps customer objectives, economic accounting, deployment pricing, and data value under one typed contract surface.

What this PR makes real today:

- `ObjectiveTensor` can be built from deterministic runtime telemetry with deployment-ready provenance.
- `ConstraintSet` can carry hard/soft constraints, geometry hints, semantic evidence, uncertainty, and trust metadata.
- `ObjectiveProfile` loading is config-driven through contract profiles in [`config/contracts`](/Users/amarmurray/robotics-vp-core/config/contracts).
- `PricingSentinel` emits deterministic shadow pricing ticks from `EconTensor` plus uncertainty/constraint evidence.
- `ValueLedger` writes sparse JSONL receipts for each meaningful economic event.
- Meta-regal shadow nodes emit typed advisory decisions instead of vague warnings.
- Ontology and datapack records receive additive shadow metadata and sidecars.

## What Is Implemented Now

Runtime contract layer:

- [`src/objectives/runtime_builder.py`](/Users/amarmurray/robotics-vp-core/src/objectives/runtime_builder.py)
- [`src/objectives/profile_loader.py`](/Users/amarmurray/robotics-vp-core/src/objectives/profile_loader.py)
- [`src/constraints/constraint_set.py`](/Users/amarmurray/robotics-vp-core/src/constraints/constraint_set.py)

Economics layer:

- [`src/economics/pricing_sentinel.py`](/Users/amarmurray/robotics-vp-core/src/economics/pricing_sentinel.py)
- [`src/economics/value_ledger.py`](/Users/amarmurray/robotics-vp-core/src/economics/value_ledger.py)

Governance layer:

- [`src/regality/shadow_nodes.py`](/Users/amarmurray/robotics-vp-core/src/regality/shadow_nodes.py)
- [`src/regality/meta_regal.py`](/Users/amarmurray/robotics-vp-core/src/regality/meta_regal.py)

Runnable shadow path:

- [`src/shadow_runtime/demo_source.py`](/Users/amarmurray/robotics-vp-core/src/shadow_runtime/demo_source.py)
- [`src/shadow_runtime/control_plane.py`](/Users/amarmurray/robotics-vp-core/src/shadow_runtime/control_plane.py)
- [`scripts/run_shadow_econ_control_plane.py`](/Users/amarmurray/robotics-vp-core/scripts/run_shadow_econ_control_plane.py)
- [`scripts/run_shadow_econ_ablations.py`](/Users/amarmurray/robotics-vp-core/scripts/run_shadow_econ_ablations.py)

## How To Run The Golden Path

```bash
python3 scripts/run_shadow_econ_control_plane.py \
  --output-dir artifacts/shadow_econ_control_plane \
  --seed 42 \
  --episodes 2 \
  --objective-profile balanced_contract
```

Main artifacts:

- `objective_tensor.json`
- `constraint_set.json`
- `constraint_flags.json`
- `objective_compile.json`
- `econ_tensor.json`
- `pricing_ticks.jsonl`
- `regal_decisions.json`
- `value_ledger.jsonl`
- `datapack_credit_update.json`
- `summary.json`
- `summary.md`
- `ontology/` and `ontology_sidecars/`

## How To Run Ablations

```bash
python3 scripts/run_shadow_econ_ablations.py \
  --output-dir artifacts/shadow_econ_ablations \
  --seed 42 \
  --episodes 2 \
  --objective-profile balanced_contract
```

Modes:

- `Mode A`: baseline deterministic workcell path only
- `Mode B`: baseline + objective/econ/pricing shadow loop
- `Mode C`: baseline + objective/econ/pricing + meta-regal shadow control plane

Outputs:

- `mode_a_baseline/baseline_summary.json`
- `mode_b_shadow/`
- `mode_c_shadow_regal/`
- `ablation_comparison.json`
- `ablation_comparison.md`

## Current Reality vs Future Real Robot

Current:

- Source is a deterministic workcell sim adapter with scripted kitting episodes.
- Pricing is pseudo-real-time through episode plus step-window ticks.
- Regal nodes are advisory shadow governance, not live blockers for legacy training.
- Datapack/ontology updates are additive metadata and sidecars.

Future real-robot plug-in points already prepared:

- `source_domain="real_lab"` is a first-class runtime enum value.
- `ObjectiveRuntimeRecord` can be populated from a real telemetry stream instead of the workcell adapter.
- `ConstraintSet.from_runtime(...)` can take real safety envelopes and deployment constraints.
- `PricingTickInput` can be fed by real deployment event windows.
- `ObjectiveContractProfile` can be loaded from customer contract material instead of local demo configs.
- `ValueLedgerReceipt` is already shaped like a deployment receipt with hashes, schema IDs, and provenance.

## Known Limitations

- The current demo source is deterministic and scripted; it is a wiring proof, not a learned deployment policy.
- Pricing policy is explicit heuristic logic, not a learned market model.
- Constraint/evidence inputs are partly synthetic placeholders because there is no live robot telemetry yet.
- Shadow regal outputs are advisory only and intentionally do not alter baseline training behavior.
- Compiler constraints operate in normalized objective space; raw physical bounds live separately in the runtime `ConstraintSet`.
