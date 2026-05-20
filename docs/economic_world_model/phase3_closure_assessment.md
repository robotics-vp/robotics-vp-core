# Phase 3 Embodiment / Actuation Closure Assessment — 2026-05-20

## Scope

This assessment classifies Phase 3 work under the user's current boundary:
**non-GPU work includes all local code, scaffolding, manifests, sidecars,
architecture contracts, tests, docs, and data-shape preparation.** The only
blocked items are work that literally requires GPU training, external provider
bring-up, native runtime execution, or hardware/calibration evidence.

## Local closure summary

| Tranche | Local status | Evidence |
|---------|--------------|----------|
| 3.1 typed canonical contracts | Closed locally | `src/world_model/embodiment_actuation/state.py`, `receipts.py` |
| 3.2 shadow compiler | Closed locally | `compiler.py` compiles from advisory embodiment outputs, registry/adapters, provider contracts, optional joint state, and source refs |
| 3.3 shadow consumers | Closed locally | `consumers.py` emits Sim/Synth, Perception, Runtime validation, and Economic receipt-bundle surfaces |
| 3.4 learned seam sockets | Closed locally | `neural_seams.py`, `training_corpus.py`, row/manifest builders, CPU-forward seam smoke |
| 3.4+ neural architecture scaffolds | Closed locally | `neural_architectures.py` emits JEPA-style, ACT-style, Diffusion Policy-style, and topology-contrastive architecture specs and finite CPU forwards |
| 3.5 provider/runtime/resource contracts | Closed locally as contracts | `provider_contracts.py` and runner sidecars expose Unitree/Holosoma/Isaac provider refs, compute/battery/thermal/latency missing evidence, and `authority_level=none` |
| Local-loop materialization | Closed locally | `sidecars.py` and `src/embodiment/runner.py` emit Phase 3 state, receipts, consumers, rows, morphology, and neural architecture manifests per episode |
| Datapack/export preservation | Closed locally at datapack-summary level | `EmbodimentProfileSummary`, validators, and token payloads preserve Phase 3 sidecar refs |
| Training backlog placement | Closed locally | `scripts/TRAINING_MIGRATION_BACKLOG.json` now includes `train_embodiment_phase34_neural_architectures.py` as GPU/provider-gated future work |

## Remaining blockers that are not local code debt

| Blocker | Why it is blocked | Required future evidence |
|---------|-------------------|--------------------------|
| GPU-backed neural training | Requires real training hardware/run budget | training run manifests, checkpoints, loss curves, heldout evals |
| Unitree / Isaac / Holosoma native provider execution | Requires provider install/runtime and/or GPU/provider plane | runtime manifests, selected policy refs, native execution logs |
| G1 hardware latency and watchdog profiles | Requires real hardware/runtime safety evidence | actuator latency profile, watchdog profile, emergency-stop/e-stop evidence |
| Sim-real drift and hardware calibration | Requires real or provider-backed rollouts | drift receipts, calibration receipts, hardware joint-limit validation |
| Policy/action promotion | Requires benchmark and demotion evidence | benchmark manifests, promotion gates, rollback/demotion tests |
| Full GR00T/V-JEPA/Diffusion/ACT model training | Requires external weights/runtime and GPU training/eval | explicit provider bring-up manifests and benchmark receipts |

## Explicit non-claims

- No native GR00T ontology or training loop has been imported.
- No V-JEPA/ACT/Diffusion/TD-MPC implementation has been vendored.
- No provider/runtime/hardware execution is claimed.
- No policy has runtime authority.
- No stable Phase B baseline, trust-net, `w_econ`, lambda-controller, or reward-path math was changed.

## Closure call

Phase 3 is **locally structurally closed** under the current no-GPU/provider
constraint. Remaining work is external evidence work, not known local substrate
debt. The next highest-leverage local work should move back to cross-WM replay,
real-video grounding, governed supervision, and dataset bridges unless new
Phase 3 gaps are discovered by an audit or by provider-season evidence.
