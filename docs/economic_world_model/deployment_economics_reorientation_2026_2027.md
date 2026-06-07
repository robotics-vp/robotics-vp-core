# Deployment Economics Reorientation 2026-2027

## Purpose

The 2026-2027 Economic WM posture is a deployment-economics austerity layer.
It does not abandon later GPU, provider, simulator, training, RunPod, or
humanoid hardware work. It changes the near-term default: before expensive
loops exist, the repo should get better at deciding which evidence source is
worth paying for and which blocker is real.

The layer is docs-first, CPU-only, deterministic, typed, additive, and
receipt-producing.

## Near-Term Capital Discipline

For 2026-2027, default work should emphasize:

- doctrine documents that make evidence, routing, and blocker boundaries
  explicit
- typed contracts for task economics, source availability, source sufficiency,
  resource pressure, and failure-cost pressure
- deterministic routers that can be replayed from identical inputs
- receipts with stable input SHAs and decision SHAs
- replay, evidence, event, and receipt plumbing that preserves lineage
- local tests, compile checks, and touched-file lint
- honest unavailable states for missing real data, providers, calibrated assets,
  simulator assets, or hardware

This is the correct local layer while humanoid hardware is unlikely before
2028. The repo can still prepare later GPU/provider/hardware runs, but it
should not spend near-term implementation effort implying those runs are the
default habit or that they already happened.

## Deferred Expensive Work

GPU/provider/training work remains deferred until justified by at least one of:

- real data with enough grounding, calibration, and replay identity to consume
- humanoid or simulation assets with documented fit to the target evidence
  question
- calibration assets that make real/sim/geometry/video comparisons meaningful
- hardware-era requirements that cannot be answered by local deterministic
  receipts
- benchmark or promotion-grade needs with explicit costs and expected value

RunPod remains a later execution plane for GPU-backed provider bring-up,
training, and heavy validation. It is not a default near-term reflex for docs,
typed contracts, deterministic routers, local receipts, or CPU tests.

## Austerity Layer Scope

The austerity layer should answer one question before it allocates scarce
capital:

Which source gives the best usable evidence for this task under the current
economic, uncertainty, time, compute, battery, and failure-cost constraints?

The first local implementation is the deterministic representation router in
`src/deployment/representation_router.py`. It chooses among:

- `real_observation`
- `simulation`
- `geometry`
- `generated_video`
- `human_operator_input`
- `prior_replay`
- `unavailable`

The router records source availability, task economics, source-specific
sufficiency, score components, rejected-source reasons, a sufficiency/blocker
summary, the deterministic tie-break order, an input SHA, and a receipt SHA.

## Hard Boundaries

This reorientation does not authorize:

- ML training
- GPU assumptions
- provider bring-up
- hardware execution
- Unitree, ROS2, Isaac, RunPod, or humanoid runtime claims
- reward, controller, trust-net, `w_econ`, lambda-controller, or Phase-B math
  changes
- promotion claims
- fake availability

Missing data, assets, providers, calibration, or hardware must be represented
as unavailable or planning-only. A receipt that says "unavailable" is better
than a local artifact that looks like runtime proof but is not.

## Why This Helps The Economic WM

The Economic WM ultimately needs to allocate scarce evidence effort. That means
it must know when real observation is worth the cost, when prior replay is
enough, when cheap geometry is better than generated video, when simulation is
adequate, and when human operator review is the economically correct source.

This austerity layer protects capital while improving the Economic WM's ability
to decide when expensive evidence is justified. It also leaves behind typed
receipts that later GPU/provider/hardware windows can consume instead of
starting from vague roadmap intent.
