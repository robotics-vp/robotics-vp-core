# Phase 1 External GPU/Runtime Pre-Training Backlog

## Purpose

This document records the Phase 1 Sim / Synth / Physics WM items that are
now honestly external — blocked by GPU, runtime, assets, or provider reality,
not by missing internal structure. These items should be the first things
burned down when GPU/runtime resources become available, before training
runs begin.

Phase 1 internal closure is sufficient. The remaining items below are
pre-training prerequisites, not active implementation debt.

## Isaac / Unitree

| Item | Blocker | Required for | Expected timing |
|------|---------|-------------|-----------------|
| `asset::actuator_latency_profile` | No clean whole-body latency-contract artifact exists in public repos | Honest safety/watchdog gating in backend binding | GPU day-0 or hardware calibration |
| `asset::safety_watchdog_profile` | No clean safety-watchdog artifact exists in public repos | Real emergency-stop and safety-envelope receipts | GPU day-0 or hardware calibration |
| Meaningful Isaac/Unitree runtime execution | Requires Isaac Sim / Isaac Lab / GPU host | Concrete runtime receipts, episode data, adapter realization | First A100 weekly cycle |
| Concrete sim materialization | Requires GPU + Isaac Gym assets + Unitree URDF in sim | Real shadow→concrete execution promotion | First A100 weekly cycle |

### What is already done (internal)

- Robot description, whole-body joint-map, joint-limit surfaces derived from public repos
- Runtime binding, layout, pack, target, preflight, deployment contracts all emit honest truth
- Selected policy/deploy/runtime-report refs and candidate evidence survive into work orders
- Host-preflight missing components are explicit (`actuator_latency_profile`, `safety_watchdog_profile`)
- Shadow execution consumes runtime binding surfaces

### What happens on GPU day-0

1. Set up Isaac Lab environment on A100 host
2. Feed actuator latency and watchdog profiles through existing scan/binding path
3. Run concrete Isaac/Unitree adapter execution through the existing request→consumer→execution→realization chain
4. Harvest runtime outcomes and validate against selected policy refs
5. Promote from shadow to concrete execution with benchmark evidence

## Holosoma

| Item | Blocker | Required for | Expected timing |
|------|---------|-------------|-----------------|
| Meaningful runtime execution beyond repo-local evidence | Requires runtime/install/provider reality beyond local clone | Concrete execution receipts and real episode data | First A100 weekly cycle |
| Richer deploy/runtime-report/install truth | Depends on Holosoma runtime maturity | Complete runtime-target install-shape truth | Provider maturity |

### What is already done (internal)

- Repo-local runtime, model, motion, policy, and retargeting surfaces consumed
- Request→consumer→execution→realization chain wired
- Policy selection prefers model/checkpoint surfaces
- Host preflight status: `preflight_ready`

### What happens on GPU day-0

1. Run Holosoma runtime execution on A100 host
2. Validate episode outputs and motion quality
3. Harvest concrete runtime outcomes

## GGDS / LDM / Video

| Item | Blocker | Required for | Expected timing |
|------|---------|-------------|-----------------|
| GPU-backed GGDS materialization | Requires GPU + LDM/diffusion model weights | Real scene materialization beyond work-order contracts | First A100 weekly cycle |
| GPU-backed video diffusion | Requires GPU + video diffusion model weights | Real synthetic branch generation from diffusion | First A100 weekly cycle |
| Concrete NAG/LSD counterfactual execution at scale | Requires GPU + renderer + assets | Real counterfactual branch evaluation | After basic runtime bring-up |

### What is already done (internal)

- WM-owned render-provider contracts with typed materialization status
- Branch/render routing emits honest provider truth
- Work-order contracts carry preconditions and unsatisfied-precondition lists
- Foundation model bringup tracked in `scripts/FOUNDATION_MODEL_BRINGUP_BACKLOG.json`
- Non-training GPU runs tracked in `scripts/NON_TRAINING_GPU_RUN_BACKLOG.json`

## Governance Rule

These items should not be treated as reasons to reopen Phase 1 internal
architecture. Phase 1 should be reopened only when:

1. Real external runtime/assets arrive that need new internal contract surfaces
2. A downstream Phase 2/3 consumer discovers a genuine missing Phase 1 contract

Otherwise, treat Phase 1 as structurally closed and burn these items down
during the first A100 weekly cycles starting September 2026.
