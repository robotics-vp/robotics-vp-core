# Economic WM Provider Runbook Templates

Date: 2026-05-21

## Purpose

The teacher/provider evidence contract now names the missing evidence surfaces. This runbook compiles those requirements into manifest-shaped templates for future local, RunPod provider, GPU training, and benchmark windows.

It emits:

- `economic_wm_provider_runbook_v1.json`
- `economic_wm_provider_runbook_v1.md`
- `manifest_templates/*.manifest_template.json`

## Executable path

```bash
python3 scripts/economic_world_model/compile_economic_wm_provider_runbook.py \
  --output-dir artifacts/economic_world_model/economic_wm_provider_runbook \
  --contract artifacts/economic_world_model/economic_wm_teacher_provider_contracts/economic_wm_teacher_provider_contract_v1.json
```

If the contract is missing, the compiler can materialize the teacher/provider evidence contract first. Use `--no-run-contract` when the caller wants missing input to fail loudly.

## Current local result

Current status remains template-only:

- `authority_class=runbook_template_only`
- `launch_allowed=false`
- `provider_bringup_ready=false`
- `gpu_training_ready=false`
- `promotion_eligible=false`
- `reward_math_mutation=false`

The runbook currently compiles five templates:

1. non-stub teacher runtime invocation proof-of-life
2. external provider runtime truth receipt proof-of-life
3. promotion-grade benchmark evidence candidate
4. GPU training runtime receipt proof-of-life
5. local replay-row linkage integrity check

The first four templates are external/provider/GPU blocked. The replay-row linkage template points at existing local commands and remains a verification aid, not a promotion claim.

## Manifest discipline

Each manifest template follows `docs/agent_ergonomics/run_manifest_schema.md` fields with:

- `status=pending`
- `pod_id=null`
- no start/finish timestamps
- `task` prefixed by `[TEMPLATE ONLY]`
- guard commands for external/provider/GPU templates that fail until replaced with real commands

A manifest template is not a run. A future run becomes evidence only after the template is instantiated with a fresh `run_id`, actual commands, actual runtime identifiers, produced artifacts, timestamps, cost data, and review notes.

## Boundary

This does not run OpenVLA, V-JEPA, diffusion providers, SceneTracks, GPU training, or promotion benchmarks. It only makes the future run ledger concrete enough that a provider/GPU window can burn down a named blocker without changing frozen reward, trust-net, `w_econ`, or lambda-controller math.
